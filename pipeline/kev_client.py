"""Client for a Kev decision-model server (github.com/jaredpalmer/kev).

Kev answers typed questions about one text (the "state") with calibrated
probabilities; it does not generate text. Start a server with
    python -m kev.serve --run jaredpalmer/kev-27b --port 8009
and point KevClient at it. Only the standard library is used.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

JEV_URL = "https://api.typesafe.ai"
def log_cost(usd, kind):
    """Append one call's OpenRouter cost to $COST_LOG (one line per call) when that is set."""
    path = os.environ.get("COST_LOG")
    if path and usd:
        with open(path, "a") as f:
            f.write(f"{kind}\t{usd}\n")


JEV_OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"   # Jev on OpenRouter (decisions endpoint)


class KevClient:
    """Also talks to TypeSafe's hosted Jev (same System One API): base_url=JEV_URL, key in
    TYPESAFE_API_KEY, model in KEV_MODEL (e.g. jev-latest)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8009", timeout: float = 600.0,
                 api_key: str | None = None, model: str | None = None):
        self.timeout = timeout
        self.requests = 0
        if "openrouter.ai" in base_url:     # same request/answer format; OpenRouter key; newest Jev
            self.url = base_url
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            self.model = model or os.environ.get("KEV_MODEL", "~typesafe/jev-latest")
        else:
            self.url = base_url.rstrip("/") + "/v1/systemone"
            self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
            self.model = model or os.environ.get("KEV_MODEL", "kev-latest")

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """Like _ask; a request too large for the server (max_tokens_exceeded, or a server error such as
        running out of GPU memory on a long request) is split in halves and the answers merged. The
        questions themselves are unchanged."""
        try:
            return self._ask(state, questions)
        except RuntimeError as e:
            if not ("max_tokens_exceeded" in str(e) or "HTTP 500" in str(e)) or len(questions) < 2:
                raise
            print(f"[split] request of {len(questions)} questions failed ({str(e)[:60]}); asking in two halves", flush=True)
            keys = list(questions)
            half = len(keys) // 2
            out = self.ask(state, {k: questions[k] for k in keys[:half]})
            out.update(self.ask(state, {k: questions[k] for k in keys[half:]}))
            return out

    def _ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        if self.model.startswith("llm:"):                # a generator LLM used as the decision model
            return self._ask_llm(state, questions)
        return self._ask_systemone(state, questions)

    def _ask_llm(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """"Same LLM" adapter: each choice question becomes a lettered multiple-choice prompt to a
        chat model on OpenRouter; the option probabilities are read from the logprobs of the one
        answer token (renormalised over the option letters). Same answer shape as System One."""
        import math
        from concurrent.futures import ThreadPoolExecutor
        model = self.model[4:]
        url = os.environ.get("LLM_URL", "https://openrouter.ai/api/v1/chat/completions")   # local: vLLM server
        key = os.environ.get("OPENROUTER_API_KEY") if "openrouter.ai" in url else "EMPTY"

        def one(item):
            qid, q = item
            opts = list(q["criteria"].items())
            letters = [chr(65 + i) for i in range(len(opts))]
            prompt = (f"Text:\n{state}\n\nQuestion: {q.get('instructions', '')}\nOptions:\n" +
                      "\n".join(f"{L}) {k}: {d}" for L, (k, d) in zip(letters, opts)) +
                      "\n\nAnswer with the single letter of the best option and nothing else.")
            body = {"model": model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                    "messages": [{"role": "user", "content": prompt}]}
            if "openrouter.ai" in url:
                body["reasoning"] = {"enabled": False}
            data = json.dumps(body).encode()
            for attempt in range(6):
                req = urllib.request.Request(url, data=data, headers={"authorization": f"Bearer {key}",
                                                                      "content-type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        resp = json.loads(r.read())
                    log_cost((resp.get("usage") or {}).get("cost"), "decision-llm")
                    ch = resp["choices"][0]
                    lp = {}
                    for t in ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs", []):
                        tok = t["token"].strip().rstrip(")").upper()
                        if tok in letters and tok not in lp:
                            lp[tok] = t["logprob"]
                    if not lp:                                # no logprobs: use the answer letter itself
                        tok = (ch["message"].get("content") or "").strip()[:1].upper()
                        lp = {tok: 0.0} if tok in letters else {letters[0]: 0.0}
                    z = sum(math.exp(v) for v in lp.values())
                    probs = {k: math.exp(lp[L]) / z if L in lp else 0.0 for L, (k, _) in zip(letters, opts)}
                    best = max(probs, key=probs.get)
                    self.requests += 1
                    return qid, {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                        time.sleep(2 ** attempt); continue
                    raise RuntimeError(f"LLM decision HTTP {e.code}: {e.read()[:300]!r}") from e
                except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, IndexError):
                    if attempt < 5:
                        time.sleep(2 ** attempt); continue
                    raise

        with ThreadPoolExecutor(4) as ex:
            return dict(ex.map(one, questions.items()))

    def _ask_systemone(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """questions: {qid: {"type": "choice"|"noul"|"score", "instructions": ..., "criteria": ...}}.
        Returns Kev's "answers" dict, keyed by the same qids."""
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode()
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        for attempt in range(5):                       # hosted API: retry rate limits / server errors
            req = urllib.request.Request(self.url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.requests += 1
                    resp = json.loads(r.read())
                    log_cost((resp.get("usage") or {}).get("cost"), "decision")
                    return resp["answers"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Kev/Jev HTTP {e.code}: {e.read()[:500]!r}") from e
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt < 4:                         # dropped connection: retry
                    time.sleep(2 ** attempt)
                    continue
                raise
