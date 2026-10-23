"""``s1_gate``: run or skip an existing runnable based on an S1 decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Collection, Dict, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    adispatch_custom_event,
    dispatch_custom_event,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import coerce_to_runnable
from langchain_core.runnables.config import patch_config

from langchain_s1.errors import (
    S1ConfigurationError,
    S1InferenceError,
    S1UncertainDecisionError,
)
from langchain_s1.types import ChoiceAnswer, NoulAnswer, S1Result

GATE_EVENT = "s1_gate"
_FALLBACKS = ("run", "skip", "raise")


class GateAction(str, Enum):
    """What a gate policy decided for a given S1 result."""

    RUN = "run"
    SKIP = "skip"
    UNCERTAIN = "uncertain"


Policy = Callable[[S1Result], Union[GateAction, str]]


@dataclass(frozen=True)
class GateOutcome:
    """The gate's resolved action, passed to ``skip_output`` and recorded in tracing.

    ``reason`` is ``"policy"``, ``"uncertain"``, or ``"inference_error"``.
    ``decision`` is ``None`` when inference failed.
    """

    action: str
    reason: str
    decision: Optional[S1Result] = None
    policy_result: Optional[GateAction] = None
    error: Optional[BaseException] = None

    def to_trace(self) -> Dict[str, Any]:
        """Return a JSON-safe view of the outcome, recorded in the ``s1_gate`` event."""
        return {
            "action": self.action,
            "reason": self.reason,
            "policy_result": self.policy_result.value if self.policy_result else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "error": f"{type(self.error).__name__}: {self.error}" if self.error else None,
        }


def _resolve(raw: Union[GateAction, str]) -> GateAction:
    try:
        return GateAction(raw)
    except ValueError:
        raise S1ConfigurationError(
            f"gate policy must return 'run', 'skip', or 'uncertain', got {raw!r}"
        ) from None


class S1Gate(Runnable[Any, Any]):
    """Runnable that consults an S1 decision before running a wrapped node.

    The wrapped node is invoked at most once per gate invocation. Only the
    inference request is retried (by the :class:`S1Client`); the node is never
    replayed because of inference or tracing failures.

    Args:
        node: The existing runnable (or callable) to gate.
        decision: An :class:`~langchain_s1.S1Decision` (or any runnable returning
            ``S1Result``) consulted before each invocation.
        policy: Maps the S1 result to ``"run"``, ``"skip"``, or ``"uncertain"``.
        skip_output: Called as ``skip_output(input, outcome)`` when the node is
            skipped; must match the node's output contract.
        on_uncertain: ``"run"``, ``"skip"``, or ``"raise"`` when the policy is
            uncertain.
        on_error: ``"run"``, ``"skip"``, or ``"raise"`` when inference fails after
            retries.
        name: Optional run name for tracing.
    """

    def __init__(
        self,
        node: Union[Runnable[Any, Any], Callable[..., Any]],
        decision: Runnable[Any, S1Result],
        *,
        policy: Policy,
        skip_output: Callable[[Any, GateOutcome], Any],
        on_uncertain: str = "raise",
        on_error: str = "raise",
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(decision, Runnable):
            raise S1ConfigurationError("decision must be a Runnable (e.g. S1Decision)")
        if not callable(policy):
            raise S1ConfigurationError("policy must be a callable mapping S1Result -> GateAction")
        if not callable(skip_output):
            raise S1ConfigurationError(
                "skip_output must be a callable (input, outcome) -> output matching the node's "
                "output contract"
            )
        for label, value in (("on_uncertain", on_uncertain), ("on_error", on_error)):
            if value not in _FALLBACKS:
                raise S1ConfigurationError(f"{label} must be one of {_FALLBACKS}, got {value!r}")
        self.node = coerce_to_runnable(node)
        self.decision = decision
        self.policy = policy
        self.skip_output = skip_output
        self.on_uncertain = on_uncertain
        self.on_error = on_error
        self.name = name or f"s1_gate({self.node.get_name()})"

    # -- shared logic -----------------------------------------------------

    def _from_result(self, result: Any) -> GateOutcome:
        if not isinstance(result, S1Result):
            raise S1ConfigurationError(
                f"decision must return an S1Result, got {type(result).__name__}"
            )
        verdict = _resolve(self.policy(result))
        if verdict is GateAction.UNCERTAIN:
            return GateOutcome(
                action=self.on_uncertain, reason="uncertain", decision=result, policy_result=verdict
            )
        return GateOutcome(
            action=verdict.value, reason="policy", decision=result, policy_result=verdict
        )

    def _from_error(self, error: S1InferenceError) -> GateOutcome:
        return GateOutcome(action=self.on_error, reason="inference_error", error=error)

    def _finish(self, input: Any, outcome: GateOutcome) -> Any:
        if outcome.action == "raise":
            if outcome.error is not None:
                raise outcome.error
            raise S1UncertainDecisionError(
                f"{self.get_name()}: S1 decision was uncertain and on_uncertain='raise'",
                decision=outcome.decision,
            )
        return self.skip_output(input, outcome)

    # -- sync -------------------------------------------------------------

    def _run(
        self, input: Any, run_manager: CallbackManagerForChainRun, config: RunnableConfig
    ) -> Any:
        try:
            result = self.decision.invoke(
                input, patch_config(config, callbacks=run_manager.get_child("s1_decision"))
            )
        except S1InferenceError as e:
            outcome = self._from_error(e)
        else:
            outcome = self._from_result(result)
        dispatch_custom_event(GATE_EVENT, outcome.to_trace(), config=config)
        if outcome.action == "run":
            return self.node.invoke(
                input, patch_config(config, callbacks=run_manager.get_child("s1_gated_node"))
            )
        return self._finish(input, outcome)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        return self._call_with_config(self._run, input, config, run_type="chain")

    # -- async ------------------------------------------------------------

    async def _arun(
        self, input: Any, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig
    ) -> Any:
        try:
            result = await self.decision.ainvoke(
                input, patch_config(config, callbacks=run_manager.get_child("s1_decision"))
            )
        except S1InferenceError as e:
            outcome = self._from_error(e)
        else:
            outcome = self._from_result(result)
        await adispatch_custom_event(GATE_EVENT, outcome.to_trace(), config=config)
        if outcome.action == "run":
            return await self.node.ainvoke(
                input, patch_config(config, callbacks=run_manager.get_child("s1_gated_node"))
            )
        return self._finish(input, outcome)

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        return await self._acall_with_config(self._arun, input, config, run_type="chain")


def s1_gate(
    node: Union[Runnable[Any, Any], Callable[..., Any]],
    decision: Runnable[Any, S1Result],
    *,
    policy: Policy,
    skip_output: Callable[[Any, GateOutcome], Any],
    on_uncertain: str = "raise",
    on_error: str = "raise",
    name: Optional[str] = None,
) -> S1Gate:
    """Wrap ``node`` so it only runs when ``policy(decision(input))`` says so.

    Args:
        node: The existing runnable (or callable) to gate.
        decision: An :class:`~langchain_s1.S1Decision` (or any runnable returning ``S1Result``).
        policy: Maps the S1 result to ``"run"``, ``"skip"``, or ``"uncertain"``.
        skip_output: Called as ``skip_output(input, outcome)`` when the node is skipped. Must
            return a value compatible with the node's output (e.g. ``{}`` for a LangGraph node).
        on_uncertain: ``"run"``, ``"skip"``, or ``"raise"`` when the policy returns uncertain.
        on_error: ``"run"``, ``"skip"``, or ``"raise"`` when inference fails after retries.
        name: Optional run name for tracing.
    """
    return S1Gate(
        node,
        decision,
        policy=policy,
        skip_output=skip_output,
        on_uncertain=on_uncertain,
        on_error=on_error,
        name=name,
    )


def noul_policy(question: str, *, run_at: float, skip_below: float) -> Policy:
    """Run when P(yes) >= ``run_at``, skip when P(yes) < ``skip_below``, else uncertain.

    Args:
        question: The noul question to read from the result.
        run_at: P(yes) at or above which the node runs.
        skip_below: P(yes) strictly below which the node is skipped.

    Returns:
        A policy callable for :func:`s1_gate`.

    Raises:
        S1ConfigurationError: If ``0 <= skip_below <= run_at <= 1`` does not hold.
    """
    if not 0.0 <= skip_below <= run_at <= 1.0:
        raise S1ConfigurationError("noul_policy requires 0 <= skip_below <= run_at <= 1")

    def policy(result: S1Result) -> GateAction:
        answer = result[question]
        if not isinstance(answer, NoulAnswer):
            raise S1ConfigurationError(f"noul_policy: answer {question!r} is not a noul answer")
        if answer.noul >= run_at:
            return GateAction.RUN
        if answer.noul < skip_below:
            return GateAction.SKIP
        return GateAction.UNCERTAIN

    return policy


def choice_policy(
    question: str,
    *,
    run: Collection[str],
    skip: Collection[str],
    min_confidence: float = 0.0,
) -> Policy:
    """Map choices to run/skip. Unlisted choices or low confidence are uncertain.

    Args:
        question: The choice question to read from the result.
        run: Choices that make the node run.
        skip: Choices that make the node skip.
        min_confidence: Minimum answer confidence; below it the policy is uncertain.

    Returns:
        A policy callable for :func:`s1_gate`.

    Raises:
        S1ConfigurationError: If a choice is listed in both ``run`` and ``skip``.
    """
    overlap = set(run) & set(skip)
    if overlap:
        raise S1ConfigurationError(f"choice_policy: choices in both run and skip: {sorted(overlap)}")

    def policy(result: S1Result) -> GateAction:
        answer = result[question]
        if not isinstance(answer, ChoiceAnswer):
            raise S1ConfigurationError(
                f"choice_policy: answer {question!r} is not a choice answer"
            )
        if answer.confidence < min_confidence:
            return GateAction.UNCERTAIN
        if answer.choice in run:
            return GateAction.RUN
        if answer.choice in skip:
            return GateAction.SKIP
        return GateAction.UNCERTAIN

    return policy
