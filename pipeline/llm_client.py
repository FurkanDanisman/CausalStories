"""Model-agnostic LLM interface.

Every LLM call in the pipeline goes through `LLMClient.complete(...)`, which
returns a validated instance of the requested pydantic schema (structured output
via tool/JSON-mode forcing). Swap the backend by passing a different client;
nothing else in the pipeline changes.

Backends:
  * AnthropicClient -- Claude (lazy-imports `anthropic`).
  * OpenAIClient    -- GPT / any OpenAI-compatible endpoint (lazy-imports `openai`).
  * OllamaClient    -- local open-source models via Ollama's OpenAI-compatible API.

Real adapters are written but intentionally minimal; the concrete model IDs are
left to fill in when we pick a backend (see README).
"""

from __future__ import annotations

import json
import os
from typing import Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        """Return an instance of `schema` produced by the model for `prompt`.

        `task` is a short label (e.g. "extract_nodes") used for logging; backends may ignore it.
        """
        ...


def _schema_instructions(schema: Type[BaseModel]) -> str:
    """A JSON-schema hint appended to prompts for backends without native tools."""
    return (
        "Respond with a single JSON object that validates against this JSON schema. "
        "Output JSON only, no prose, no code fences.\n"
        + json.dumps(schema.model_json_schema())
    )


# ---------------------------------------------------------------------- anthropic

class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 4096):
        import anthropic  # lazy

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        # Force structured output with a single tool whose input schema == `schema`.
        tool = {
            "name": "emit",
            "description": "Return the result.",
            "input_schema": schema.model_json_schema(),
        }
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return schema.model_validate(block.input)
        raise RuntimeError("Anthropic response contained no tool_use block")


# ------------------------------------------------------------------------- openai

class OpenAIClient:
    """Any OpenAI-compatible chat endpoint. Point `base_url` at a cluster server
    (vLLM / TGI / llama.cpp / Ollama all expose this API). Local servers ignore
    the api key, so we default it to "EMPTY" when a base_url is given."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        import openai  # lazy

        kwargs = {"timeout": 120.0, "max_retries": 3}   # a hung connection retries instead of waiting forever
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["api_key"] = api_key or "EMPTY"
        elif api_key:
            kwargs["api_key"] = api_key
        self._client = openai.OpenAI(**kwargs)
        self.model = model

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        import re
        last = None
        for attempt in range(5):                     # retries above temperature 0 (else the same bad answer)
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature if attempt == 0 else max(temperature, 0.7),
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},   # vLLM: no thinking (Qwen 3.5)
                messages=[{"role": "user", "content": prompt + "\n\n" + _schema_instructions(schema)}],
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.choices[0].message.content or "").strip())
            try:
                return schema.model_validate_json(text)
            except Exception as e:
                last = e
        raise last


class OpenRouterClient(OpenAIClient):
    """Hosted open models (e.g. google/gemma-4-31b-it) through OpenRouter's OpenAI-compatible
    API. Key in OPENROUTER_API_KEY. Strips code fences and retries once on bad JSON."""

    def __init__(self, model: str = "google/gemma-4-31b-it", api_key: str | None = None):
        import os
        super().__init__(model=model, base_url="https://openrouter.ai/api/v1",
                         api_key=api_key or os.environ.get("OPENROUTER_API_KEY"))

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        import re
        last = None
        for attempt in range(5):                     # retries above temperature 0, else the same bad answer repeats
            resp = self._client.chat.completions.create(
                model=self.model, temperature=temperature if attempt == 0 else max(temperature, 0.7),
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},          # no thinking (e.g. Qwen 3.5)
                messages=[{"role": "user", "content": prompt + "\n\n" + _schema_instructions(schema)}])
            try:                                        # OpenRouter reports the call's cost in usage
                from pipeline.kev_client import log_cost
                u = resp.usage
                log_cost(getattr(u, "cost", None) or (getattr(u, "model_extra", None) or {}).get("cost"), "generator")
            except Exception:
                pass
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            try:
                return schema.model_validate_json(text)
            except Exception as e:                      # one more try on malformed JSON
                last = e
        raise last


# ------------------------------------------------------------------------- ollama

class OllamaClient(OpenAIClient):
    """Local open-source models through Ollama's OpenAI-compatible endpoint."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        super().__init__(model=model, base_url=base_url)


# ------------------------------------------------- huggingface transformers (in-process)

def _extract_json(text: str) -> dict | None:
    """Best-effort: parse the whole thing, else the first balanced {...} block."""
    import json

    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    break
        start = text.find("{", start + 1)
    return None


class VLLMClient:
    """Runs a local open-source model in-process via vLLM, with schema-guided JSON
    decoding (reliable structured output). Best backend for a GPU node with local
    weights. Load ONE model per process — vLLM does not cleanly free GPU memory
    in-process, so orchestrate multiple models as separate processes.

    Env knobs: VLLM_TP (tensor-parallel GPUs, default 1), VLLM_GPU_UTIL (0.90),
    VLLM_MAX_LEN (8192)."""

    def __init__(self, model: str, max_tokens: int = 2048):
        # Use vLLM's native PyTorch sampler instead of FlashInfer, whose sampler
        # JIT-compiles a CUDA kernel at runtime (needs nvcc). On clusters where the
        # CUDA toolkit is module-gated (e.g. Alliance/Killarney) nvcc isn't on the
        # node and FlashInfer's sampler crashes: "Could not find nvcc". We decode
        # greedily, so the native sampler is all we need.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM  # lazy

        self.max_tokens = max_tokens
        self.llm = LLM(
            model=model,
            tensor_parallel_size=int(os.environ.get("VLLM_TP", "1")),
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.90")),
            max_model_len=int(os.environ.get("VLLM_MAX_LEN", "8192")),
            dtype="bfloat16",
            trust_remote_code=True,
        )

    def _sampling_params(self, schema_json: dict, max_tokens: int, temperature: float):
        from vllm import SamplingParams
        try:  # vLLM v1 (>=0.11 / Alliance 0.25): structured outputs
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                  structured_outputs=StructuredOutputsParams(json=schema_json))
        except Exception:  # older vLLM: guided decoding
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                  guided_decoding=GuidedDecodingParams(json=schema_json))

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        # Guided decoding guarantees schema-valid JSON only if generation COMPLETES;
        # if it hits max_tokens (truncation) or loops, the JSON is invalid. Retry with
        # a bigger budget, then with a concise nudge + a little temperature to break a
        # repetition loop. Small models (Gemma/Llama) otherwise return empty ~20-35%.
        schema_json = schema.model_json_schema()
        attempts = [
            (prompt, self.max_tokens, temperature),
            (prompt, self.max_tokens * 2, temperature),
            (prompt + "\n\nBe concise: output ONLY the minimal valid JSON, no repetition.",
             self.max_tokens * 2, max(temperature, 0.3)),
        ]
        last_err = None
        for p, mt, temp in attempts:
            sp = self._sampling_params(schema_json, mt, temp)
            out = self.llm.chat([{"role": "user", "content": p}], sampling_params=sp)
            try:
                return schema.model_validate_json(out[0].outputs[0].text)
            except Exception as e:
                last_err = e
        raise last_err


class HFClient:
    """Runs an open-source model in-process via transformers — for a cluster batch
    job with no server. No tool-forcing, so we prompt for JSON and parse (1 retry)."""

    def __init__(self, model: str, max_new_tokens: int = 1024):
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map="auto")
        self.max_new_tokens = max_new_tokens

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        base = prompt + "\n\n" + _schema_instructions(schema)
        content = base
        for _ in range(2):
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True)
            inputs = self.tok(text, return_tensors="pt").to(self.model.device)
            gen_kwargs = {"max_new_tokens": self.max_new_tokens}
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            with self._torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            reply = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            obj = _extract_json(reply)
            if obj is not None:
                try:
                    return schema.model_validate(obj)
                except Exception:
                    pass
            content = base + "\n\nYour previous reply was not valid JSON. Output ONLY the JSON object."
        raise RuntimeError(f"HF model returned no valid JSON for task={task}")
