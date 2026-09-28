"""Client for a Kev decision-model server (github.com/jaredpalmer/kev).

Kev answers typed questions about one text (the "state") with calibrated
probabilities; it does not generate text. Start a server with
    python -m kev.serve --run jaredpalmer/kev-27b --port 8009
and point KevClient at it. Only the standard library is used.
"""

from __future__ import annotations

import json
import urllib.request


class KevClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8009", timeout: float = 600.0):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout
        self.requests = 0

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """questions: {qid: {"type": "choice"|"noul"|"score", "instructions": ..., "criteria": ...}}.
        Returns Kev's "answers" dict, keyed by the same qids."""
        body = json.dumps({"state": state, "model": "kev-latest", "questions": questions}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self.requests += 1
            return json.loads(r.read())["answers"]
