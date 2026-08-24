"""Model-agnostic LLM interface.

Every LLM call in the pipeline goes through `LLMClient.complete(...)`, which
returns a validated instance of the requested pydantic schema (structured output
via tool/JSON-mode forcing). Swap the backend by passing a different client;
nothing else in the pipeline changes.

Backends:
  * MockLLMClient   -- no network, canned answers keyed by `task`. Lets the whole
                       pipeline run + be tested with zero API keys.
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

from . import mock_data

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        """Return an instance of `schema` produced by the model for `prompt`.

        `task` is a short label (e.g. "extract_nodes") used for logging and by the
        mock to route to canned data; real backends may ignore it.
        """
        ...


def _schema_instructions(schema: Type[BaseModel]) -> str:
    """A JSON-schema hint appended to prompts for backends without native tools."""
    return (
        "Respond with a single JSON object that validates against this JSON schema. "
        "Output JSON only, no prose, no code fences.\n"
        + json.dumps(schema.model_json_schema())
    )


# --------------------------------------------------------------------------- mock

class MockLLMClient:
    """Deterministic, offline. Returns hand-authored answers for the demo example
    so the end-to-end pipeline (including eval) produces meaningful numbers."""

    # marker string that must appear in the prompt for the fixture to apply
    _MARKERS = {"keating": "Keating"}

    def __init__(self, fixture: str = "keating"):
        self.fixture = fixture

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        marker = self._MARKERS.get(self.fixture, "")
        applies = marker in prompt
        payload = mock_data.answer(self.fixture, task) if applies else None
        if payload is None:
            # Unknown task: return an empty-but-valid instance so plumbing survives.
            return schema.model_validate(mock_data.empty_for(schema))
        return schema.model_validate(payload)


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

        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["api_key"] = api_key or "EMPTY"
        elif api_key:
            kwargs["api_key"] = api_key
        self._client = openai.OpenAI(**kwargs)
        self.model = model

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt + "\n\n" + _schema_instructions(schema)}],
        )
        return schema.model_validate_json(resp.choices[0].message.content)


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

    def __init__(self, model: str, max_tokens: int = 1024):
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

    def complete(self, *, task: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        from vllm import SamplingParams

        schema_json = schema.model_json_schema()
        try:  # vLLM v1 (>=0.11 / Alliance 0.25): structured outputs
            from vllm.sampling_params import StructuredOutputsParams
            sp = SamplingParams(temperature=temperature, max_tokens=self.max_tokens,
                                structured_outputs=StructuredOutputsParams(json=schema_json))
        except Exception:  # older vLLM: guided decoding
            from vllm.sampling_params import GuidedDecodingParams
            sp = SamplingParams(temperature=temperature, max_tokens=self.max_tokens,
                                guided_decoding=GuidedDecodingParams(json=schema_json))
        out = self.llm.chat([{"role": "user", "content": prompt}], sampling_params=sp)
        return schema.model_validate_json(out[0].outputs[0].text)


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
