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


class KevClient:
    """Also talks to TypeSafe's hosted Jev (same System One API): base_url=JEV_URL, key in
    TYPESAFE_API_KEY, model in KEV_MODEL (e.g. jev-latest)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8009", timeout: float = 600.0,
                 api_key: str | None = None, model: str | None = None):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout
        self.requests = 0
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.model = model or os.environ.get("KEV_MODEL", "kev-latest")

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
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
                    return json.loads(r.read())["answers"]
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
