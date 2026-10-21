"""HTTP client for TypeSafe-compatible System One endpoints."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from typing import Any, Dict, Mapping, Optional, Union

import httpx

from langchain_s1.errors import (
    S1APIError,
    S1ConfigurationError,
    S1RequestError,
    S1ResponseError,
    S1TimeoutError,
)
from langchain_s1.types import Question, S1Result, coerce_questions, parse_response

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
SYSTEM_ONE_PATH = "/systemone"
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

State = Union[str, Mapping[str, Any], list]


class S1Client:
    """Sends typed decision requests to any TypeSafe-compatible System One endpoint.

    Args:
        model: Model identifier sent with every request. Not restricted.
        base_url: API root; ``/systemone`` is appended. Defaults to TypeSafe's ``/v1`` root.
        endpoint_url: Full endpoint URL. Overrides ``base_url`` when set.
        api_key: Optional bearer token. Omit for local or header-authenticated endpoints.
        headers: Extra headers sent with every request (e.g. custom auth).
        timeout: Per-attempt timeout in seconds, or an ``httpx.Timeout``.
        max_retries: Retries after the first attempt for connection errors, timeouts,
            and retryable statuses (408, 429, 500, 502, 503, 504). ``0`` disables retries.
        retry_backoff: Base delay in seconds for exponential backoff between retries.
        transport: Optional ``httpx`` transport for custom integrations or testing.
            ``httpx.MockTransport`` serves both sync and async requests.
        async_transport: Optional async transport, if different from ``transport``.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Union[float, httpx.Timeout] = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise S1ConfigurationError("model must be a non-empty string")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise S1ConfigurationError("max_retries must be a non-negative integer")
        if retry_backoff < 0:
            raise S1ConfigurationError("retry_backoff must be non-negative")
        if endpoint_url:
            url = endpoint_url
        elif base_url:
            url = base_url.rstrip("/") + SYSTEM_ONE_PATH
        else:
            raise S1ConfigurationError("either base_url or endpoint_url must be set")
        self.model = model
        self.endpoint_url = url
        self.timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._api_key = api_key
        self._extra_headers = dict(headers or {})
        self._transport = transport
        if async_transport is None and isinstance(transport, httpx.AsyncBaseTransport):
            async_transport = transport
        self._async_transport = async_transport
        self._sync_client: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        # Never include credentials or header values.
        return f"S1Client(model={self.model!r}, endpoint_url={self.endpoint_url!r})"

    # -- request building -------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._extra_headers)
        return headers

    def build_request_body(
        self, state: State, questions: Mapping[str, Union[Question, Mapping[str, Any]]]
    ) -> Dict[str, Any]:
        if not isinstance(state, (str, Mapping, list, tuple)):
            raise S1ConfigurationError(
                f"state must be a string, object, or array, got {type(state).__name__}"
            )
        typed = coerce_questions(questions)
        return {
            "model": self.model,
            "state": list(state) if isinstance(state, tuple) else state,
            "questions": {name: q.to_dict() for name, q in typed.items()},
        }

    def _delay(self, attempt: int, response: Optional[httpx.Response]) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(max(float(retry_after), 0.0), 60.0)
                except ValueError:
                    pass
        if self.retry_backoff == 0:
            return 0.0
        return self.retry_backoff * (2**attempt) * (0.5 + random.random() / 2)

    def _handle(
        self, response: httpx.Response, questions: Mapping[str, Question], attempts: int
    ) -> S1Result:
        if response.status_code >= 400:
            try:
                body: Any = response.json()
            except ValueError:
                body = response.text
            detail = body.get("detail", body) if isinstance(body, dict) else body
            raise S1APIError(
                f"System One endpoint {self.endpoint_url} returned HTTP "
                f"{response.status_code}: {detail}",
                status_code=response.status_code,
                body=body,
                attempts=attempts,
            )
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise S1ResponseError(f"response body is not valid JSON: {e}") from e
        return parse_response(data, questions)

    def _failure(self, error: httpx.TransportError, attempts: int) -> S1RequestError:
        cls = S1TimeoutError if isinstance(error, httpx.TimeoutException) else S1RequestError
        what = "timed out" if cls is S1TimeoutError else f"failed ({type(error).__name__}: {error})"
        return cls(
            f"System One request to {self.endpoint_url} {what} after {attempts} attempt(s)",
            attempts=attempts,
        )

    # -- sync -------------------------------------------------------------

    def _client(self) -> httpx.Client:
        with self._lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(timeout=self.timeout, transport=self._transport)
            return self._sync_client

    def decide(
        self, state: State, questions: Mapping[str, Union[Question, Mapping[str, Any]]]
    ) -> S1Result:
        """Ask the endpoint one or more named questions about ``state``."""
        typed = coerce_questions(questions)
        body = self.build_request_body(state, typed)
        client = self._client()
        for attempt in range(self.max_retries + 1):
            response: Optional[httpx.Response] = None
            try:
                response = client.post(self.endpoint_url, json=body, headers=self._headers())
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise self._failure(e, attempt + 1) from e
            else:
                if (
                    response.status_code not in RETRYABLE_STATUS_CODES
                    or attempt == self.max_retries
                ):
                    return self._handle(response, typed, attempt + 1)
            time.sleep(self._delay(attempt, response))
        raise AssertionError("unreachable")  # pragma: no cover

    # -- async ------------------------------------------------------------

    async def adecide(
        self, state: State, questions: Mapping[str, Union[Question, Mapping[str, Any]]]
    ) -> S1Result:
        """Async variant of :meth:`decide`. Cancellation propagates immediately."""
        typed = coerce_questions(questions)
        body = self.build_request_body(state, typed)
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._async_transport
        ) as client:
            for attempt in range(self.max_retries + 1):
                response: Optional[httpx.Response] = None
                try:
                    response = await client.post(
                        self.endpoint_url, json=body, headers=self._headers()
                    )
                except httpx.TransportError as e:
                    if attempt == self.max_retries:
                        raise self._failure(e, attempt + 1) from e
                else:
                    if (
                        response.status_code not in RETRYABLE_STATUS_CODES
                        or attempt == self.max_retries
                    ):
                        return self._handle(response, typed, attempt + 1)
                await asyncio.sleep(self._delay(attempt, response))
        raise AssertionError("unreachable")  # pragma: no cover

    def close(self) -> None:
        with self._lock:
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None

    def __enter__(self) -> "S1Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
