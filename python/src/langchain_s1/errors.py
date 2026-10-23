"""Exceptions raised by langchain-s1."""

from __future__ import annotations

from typing import Any, Optional


class S1Error(Exception):
    """Base class for all langchain-s1 errors."""


class S1ConfigurationError(S1Error, ValueError):
    """Invalid client, question, decision, or gate configuration."""


class S1UnsupportedFeatureError(S1Error):
    """A feature outside the supported text-based System One contract was requested or returned."""


class S1InferenceError(S1Error):
    """An inference request failed. Gates apply their ``on_error`` policy to these errors."""


class S1RequestError(S1InferenceError):
    """The request could not be completed (connection failure, exhausted retries)."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class S1TimeoutError(S1RequestError):
    """The request timed out on every attempt."""


class S1APIError(S1InferenceError):
    """The endpoint returned a non-success HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: Any = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.attempts = attempts


class S1ResponseError(S1InferenceError, ValueError):
    """The endpoint returned a response that does not match the System One contract."""

    def __init__(self, message: str, *, path: Optional[str] = None) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path


class S1UncertainDecisionError(S1Error):
    """A gate policy returned ``uncertain`` and the gate is configured to raise."""

    def __init__(self, message: str, *, decision: Any) -> None:
        super().__init__(message)
        self.decision = decision
