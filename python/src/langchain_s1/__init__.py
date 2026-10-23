"""A langchain wrapper for s1 decisions."""

from langchain_s1.client import DEFAULT_BASE_URL, S1Client
from langchain_s1.decision import S1Decision
from langchain_s1.errors import (
    S1APIError,
    S1ConfigurationError,
    S1Error,
    S1InferenceError,
    S1RequestError,
    S1ResponseError,
    S1TimeoutError,
    S1UncertainDecisionError,
    S1UnsupportedFeatureError,
)
from langchain_s1.gate import (
    GATE_EVENT,
    GateAction,
    GateOutcome,
    S1Gate,
    choice_policy,
    noul_policy,
    s1_gate,
)
from langchain_s1.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    S1Result,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "DEFAULT_BASE_URL",
    "GATE_EVENT",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "GateAction",
    "GateOutcome",
    "NoulAnswer",
    "NoulQuestion",
    "S1APIError",
    "S1Client",
    "S1ConfigurationError",
    "S1Decision",
    "S1Error",
    "S1Gate",
    "S1InferenceError",
    "S1RequestError",
    "S1ResponseError",
    "S1Result",
    "S1TimeoutError",
    "S1UncertainDecisionError",
    "S1UnsupportedFeatureError",
    "ScoreAnswer",
    "ScoreQuestion",
    "Usage",
    "choice_policy",
    "noul_policy",
    "s1_gate",
]
