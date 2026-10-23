"""``S1Decision``: System One questions as a LangChain runnable."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

from langchain_core.runnables import Runnable, RunnableConfig

from langchain_s1.client import S1Client, State
from langchain_s1.errors import S1ConfigurationError
from langchain_s1.types import Question, S1Result, coerce_questions


class S1Decision(Runnable[Any, S1Result]):
    """Ask typed System One questions about a selected view of the runnable input.

    Args:
        client: The :class:`S1Client` to send requests through.
        questions: Named questions; multiple questions are sent in one request.
        select_state: Maps the runnable input to the ``state`` sent to the endpoint
            (a string, object, or array). Only what this returns leaves the process.
        name: Optional run name used in tracing.

    Returns an :class:`S1Result` whose answers are keyed by question name.
    Supports ``invoke``, ``ainvoke``, ``batch``, ``abatch``, and composition with ``|``.
    """

    def __init__(
        self,
        client: S1Client,
        questions: Mapping[str, Union[Question, Mapping[str, Any]]],
        *,
        select_state: Callable[[Any], State],
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(client, S1Client):
            raise S1ConfigurationError("client must be an S1Client")
        if not callable(select_state):
            raise S1ConfigurationError("select_state must be a callable")
        self.client = client
        self.questions: Dict[str, Question] = coerce_questions(questions)
        self.select_state = select_state
        self.name = name or "S1Decision"

    @property
    def InputType(self) -> Any:  # noqa: N802 - LangChain API
        return Any

    @property
    def OutputType(self) -> Any:  # noqa: N802 - LangChain API
        return S1Result

    def _decide(self, input: Any) -> S1Result:
        return self.client.decide(self.select_state(input), self.questions)

    async def _adecide(self, input: Any) -> S1Result:
        return await self.client.adecide(self.select_state(input), self.questions)

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> S1Result:
        return self._call_with_config(self._decide, input, config, run_type="chain")

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> S1Result:
        return await self._acall_with_config(self._adecide, input, config, run_type="chain")

    def __repr__(self) -> str:
        return f"S1Decision(client={self.client!r}, questions={list(self.questions)!r})"

