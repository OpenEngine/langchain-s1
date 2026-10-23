"""HTTP client for TypeSafe-compatible System One endpoints."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
import weakref
from typing import Any, Dict, Mapping, Optional, Union
from urllib.parse import urlsplit, urlunsplit

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

MAX_ERROR_BODY_CHARS = 2048

State = Union[str, Mapping[str, Any], list]


def redact_url(url: str) -> str:
    """Drop userinfo, query string, and fragment so URLs are safe to log."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
    except ValueError:
        return "<invalid url>"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


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
            ``httpx.MockTransport`` serves both sync and async requests. A sync-only
            transport requires ``async_transport`` for async calls.
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
        self._safe_url = redact_url(url)
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
        # AsyncClients are bound to the event loop that created them.
        self._async_clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        # Never include credentials or header values.
        return f"S1Client(model={self.model!r}, endpoint_url={self._safe_url!r})"

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
        """Build the JSON body sent to the endpoint.

        Args:
            state: The state to ask about (a string, object, or array).
            questions: Named questions (typed objects or contract-shaped dicts).

        Returns:
            The ``{"model", "state", "questions"}`` request body.

        Raises:
            S1ConfigurationError: If ``state`` is not a string, object, or array, or a
                question is malformed.
            S1UnsupportedFeatureError: If a question uses an unsupported type or field.
        """
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
            # The body stays off the message (which reaches tracing) and is truncated.
            text = response.text
            body: Any
            if len(text) > MAX_ERROR_BODY_CHARS:
                body = text[:MAX_ERROR_BODY_CHARS] + "...[truncated]"
            else:
                try:
                    body = response.json()
                except ValueError:
                    body = text
            raise S1APIError(
                f"System One endpoint {self._safe_url} returned HTTP "
                f"{response.status_code}; see .body for the response detail",
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
            f"System One request to {self._safe_url} {what} after {attempts} attempt(s)",
            attempts=attempts,
        )

    def _settle(
        self,
        attempt: int,
        questions: Mapping[str, Question],
        response: Optional[httpx.Response],
        error: Optional[httpx.TransportError],
    ) -> Union[S1Result, float]:
        """Return the final result, raise the final error, or return the retry delay."""
        last = attempt == self.max_retries
        if error is not None:
            if last:
                raise self._failure(error, attempt + 1) from error
        elif response is not None and (
            response.status_code not in RETRYABLE_STATUS_CODES or last
        ):
            return self._handle(response, questions, attempt + 1)
        return self._delay(attempt, response)

    # -- sync -------------------------------------------------------------

    def _client(self) -> httpx.Client:
        with self._lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(timeout=self.timeout, transport=self._transport)
            return self._sync_client

    def decide(
        self, state: State, questions: Mapping[str, Union[Question, Mapping[str, Any]]]
    ) -> S1Result:
        """Ask the endpoint one or more named questions about ``state``.

        All questions are sent in a single request. Connection errors, timeouts, and
        retryable statuses (408, 429, 5xx) are retried with exponential backoff, up to
        ``max_retries`` retries.

        Args:
            state: The state to ask about (a string, object, or array).
            questions: Named questions (typed objects or contract-shaped dicts).

        Returns:
            A validated :class:`~langchain_s1.types.S1Result` keyed by question name.

        Raises:
            S1ConfigurationError: If ``state`` or a question is malformed.
            S1TimeoutError: If every attempt timed out.
            S1RequestError: If the request failed after all attempts.
            S1APIError: If the endpoint returned a non-success status.
            S1ResponseError: If the response does not match the contract.
        """
        typed = coerce_questions(questions)
        body = self.build_request_body(state, typed)
        client = self._client()
        for attempt in range(self.max_retries + 1):
            response: Optional[httpx.Response] = None
            error: Optional[httpx.TransportError] = None
            try:
                response = client.post(self.endpoint_url, json=body, headers=self._headers())
            except httpx.TransportError as e:
                error = e
            outcome = self._settle(attempt, typed, response, error)
            if isinstance(outcome, S1Result):
                return outcome
            time.sleep(outcome)
        raise AssertionError("unreachable")  # pragma: no cover

    # -- async ------------------------------------------------------------

    def _aclient(self) -> httpx.AsyncClient:
        if self._transport is not None and self._async_transport is None:
            raise S1ConfigurationError(
                "transport does not support async requests; pass async_transport "
                "(e.g. an httpx.AsyncBaseTransport) to use adecide/ainvoke/abatch"
            )
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self.timeout, transport=self._async_transport)
            self._async_clients[loop] = client
        return client

    async def adecide(
        self, state: State, questions: Mapping[str, Union[Question, Mapping[str, Any]]]
    ) -> S1Result:
        """Async variant of :meth:`decide`.

        Uses a per-event-loop ``httpx.AsyncClient`` that is created on first use and
        cleaned up when the loop goes away. Cancellation propagates immediately.
        Arguments, return value, and errors are the same as :meth:`decide`.
        """
        typed = coerce_questions(questions)
        body = self.build_request_body(state, typed)
        client = self._aclient()
        for attempt in range(self.max_retries + 1):
            response: Optional[httpx.Response] = None
            error: Optional[httpx.TransportError] = None
            try:
                response = await client.post(self.endpoint_url, json=body, headers=self._headers())
            except httpx.TransportError as e:
                error = e
            outcome = self._settle(attempt, typed, response, error)
            if isinstance(outcome, S1Result):
                return outcome
            await asyncio.sleep(outcome)
        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        """Close the async client for the running event loop. No-op if none was created."""
        client = self._async_clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    def close(self) -> None:
        """Close the shared sync client, if one was created. Safe to call twice."""
        with self._lock:
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None

    def __enter__(self) -> "S1Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> "S1Client":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
