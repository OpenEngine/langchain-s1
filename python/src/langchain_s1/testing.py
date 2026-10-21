"""In-process mock of a TypeSafe-compatible System One endpoint, for tests and examples.

Pass an instance as ``S1Client(transport=...)``; no network or credentials are used.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx

AnswerFn = Callable[[Dict[str, Any]], Mapping[str, Any]]


def default_answer(question: Mapping[str, Any]) -> Dict[str, Any]:
    """A deterministic, contract-valid answer for a single question."""
    qtype = question.get("type")
    if qtype == "noul":
        return {"type": "noul", "noul": 0.9}
    if qtype == "choice":
        names = list(question["criteria"])
        probs = {n: 0.1 / max(len(names) - 1, 1) for n in names[1:]}
        probs[names[0]] = 0.9 if len(names) > 1 else 1.0
        return {
            "type": "choice",
            "choice": names[0],
            "confidence": probs[names[0]],
            "probabilities": probs,
        }
    if qtype == "score":
        levels = list(question["criteria"])
        top = len(levels) - 1
        return {
            "type": "score",
            "score": float(top),
            "confidence": 1.0,
            "legend": {str(i): d for i, d in enumerate(levels)},
            "probabilities": {str(i): 1.0 if i == top else 0.0 for i in range(len(levels))},
        }
    raise ValueError(f"unsupported question type {qtype!r}")


class MockSystemOne(httpx.MockTransport):
    """Serves ``POST <path>`` with ``{model, answers, usage}`` responses.

    Args:
        answer: ``answer(request_body) -> {question_name: answer}``. Defaults to
            :func:`default_answer` for every question.
        path: Only this URL path is served; others return 404.
        required_headers: Headers that must be present with exact values, else 401.
        model: Model reported in responses; defaults to the requested model.
    """

    def __init__(
        self,
        answer: Optional[AnswerFn] = None,
        *,
        path: str = "/v1/systemone",
        required_headers: Optional[Mapping[str, str]] = None,
        model: Optional[str] = None,
    ) -> None:
        self.answer = answer or (
            lambda body: {k: default_answer(q) for k, q in body["questions"].items()}
        )
        self.path = path
        self.required_headers = dict(required_headers or {})
        self.model = model
        self.requests: List[httpx.Request] = []
        super().__init__(self._handle)

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method != "POST" or request.url.path != self.path:
            return httpx.Response(404, json={"detail": "Not Found"})
        for key, value in self.required_headers.items():
            if request.headers.get(key) != value:
                return httpx.Response(401, json={"detail": f"missing or invalid {key}"})
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": self.model or body["model"],
                "answers": dict(self.answer(body)),
                "usage": {"input_tokens": len(request.content), "output_tokens": len(body["questions"])},
            },
        )
