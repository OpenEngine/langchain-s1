"""Typed questions and answers for the TypeSafe System One contract.

Request:  ``{model, state, questions}``
Response: ``{model, answers, usage}``

Only text-based ``noul``, ``choice``, and ``score`` questions are supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from langchain_s1.errors import (
    S1ConfigurationError,
    S1ResponseError,
    S1UnsupportedFeatureError,
)

SUPPORTED_TYPES = ("noul", "choice", "score")
# Instructions may be a string, object, or array per the contract.
Instructions = Union[str, Mapping[str, Any], Sequence[Any]]


def _check_instructions(instructions: Any) -> None:
    if isinstance(instructions, str):
        if not instructions:
            raise S1ConfigurationError("question instructions must not be empty")
    elif not isinstance(instructions, (Mapping, list, tuple)):
        raise S1ConfigurationError(
            "question instructions must be a string, object, or array, "
            f"got {type(instructions).__name__}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoulQuestion:
    """A yes/no question. The answer is the probability of *yes*."""

    instructions: Instructions
    criteria: Optional[Mapping[str, str]] = None
    type: str = field(default="noul", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if self.criteria is not None:
            unknown = set(self.criteria) - {"true", "false"}
            if unknown:
                raise S1ConfigurationError(
                    f"noul criteria only accepts 'true' and 'false' keys, got {sorted(unknown)}"
                )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"type": "noul", "instructions": _jsonable(self.instructions)}
        if self.criteria is not None:
            data["criteria"] = dict(self.criteria)
        return data


@dataclass(frozen=True)
class ChoiceQuestion:
    """Select one option. ``criteria`` maps choice names to descriptions (1-255 choices)."""

    instructions: Instructions
    criteria: Mapping[str, str]
    type: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if not isinstance(self.criteria, Mapping) or not 1 <= len(self.criteria) <= 255:
            raise S1ConfigurationError("choice criteria must be a mapping with 1-255 choices")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "instructions": _jsonable(self.instructions),
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class ScoreQuestion:
    """Rate on a scale. ``criteria`` lists level descriptions from level 0 upward (1-10 levels)."""

    instructions: Instructions
    criteria: Sequence[str]
    type: str = field(default="score", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if isinstance(self.criteria, (str, Mapping)) or not 1 <= len(self.criteria) <= 10:
            raise S1ConfigurationError("score criteria must be a list of 1-10 level descriptions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "instructions": _jsonable(self.instructions),
            "criteria": list(self.criteria),
        }


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


def coerce_question(name: str, question: Union[Question, Mapping[str, Any]]) -> Question:
    """Convert a question object or contract-shaped dict into a typed question."""
    if isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
        return question
    if not isinstance(question, Mapping):
        raise S1ConfigurationError(
            f"question {name!r} must be a question object or mapping, got {type(question).__name__}"
        )
    qtype = question.get("type")
    if qtype not in SUPPORTED_TYPES:
        raise S1UnsupportedFeatureError(
            f"question {name!r} has unsupported type {qtype!r}; "
            f"supported types are {', '.join(SUPPORTED_TYPES)}"
        )
    extra = set(question) - {"type", "instructions", "criteria"}
    if extra:
        raise S1UnsupportedFeatureError(
            f"question {name!r} has unsupported fields {sorted(extra)}"
        )
    if "instructions" not in question:
        raise S1ConfigurationError(f"question {name!r} is missing 'instructions'")
    kwargs = {"instructions": question["instructions"]}
    if "criteria" in question:
        kwargs["criteria"] = question["criteria"]
    cls = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}[qtype]
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise S1ConfigurationError(f"question {name!r}: {e}") from e


def coerce_questions(
    questions: Mapping[str, Union[Question, Mapping[str, Any]]],
) -> Dict[str, Question]:
    if not isinstance(questions, Mapping) or not questions:
        raise S1ConfigurationError("questions must be a non-empty mapping of name -> question")
    out: Dict[str, Question] = {}
    for name, q in questions.items():
        if not isinstance(name, str) or not name:
            raise S1ConfigurationError(f"question names must be non-empty strings, got {name!r}")
        out[name] = coerce_question(name, q)
    return out


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoulAnswer:
    """``noul`` is the probability that the answer is *yes* (0-1). It is not a confidence."""

    noul: float
    type: str = field(default="noul", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "noul", "noul": self.noul}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    type: str = field(default="choice", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class ScoreAnswer:
    """``score`` is the probability-weighted average level."""

    score: float
    confidence: float
    probabilities: Dict[str, float]
    legend: Dict[str, str] = field(default_factory=dict)
    type: str = field(default="score", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "confidence": self.confidence,
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
        }


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> Dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass(frozen=True)
class S1Result(Mapping[str, Answer]):
    """A validated System One response. Indexing by question name returns its typed answer."""

    model: str
    answers: Dict[str, Answer]
    usage: Usage

    def __getitem__(self, name: str) -> Answer:
        try:
            return self.answers[name]
        except KeyError:
            raise KeyError(
                f"no answer named {name!r}; available: {sorted(self.answers)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "usage": self.usage.to_dict(),
        }


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise S1ResponseError(f"expected a probability number, got {value!r}", path=path)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise S1ResponseError(f"probability must be within [0, 1], got {value!r}", path=path)
    return float(value)


def _probabilities(value: Any, allowed: Sequence[str], path: str) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise S1ResponseError(f"expected an object, got {value!r}", path=path)
    unknown = [k for k in value if k not in allowed]
    if unknown:
        raise S1ResponseError(
            f"unexpected keys {unknown}; expected a subset of {list(allowed)}", path=path
        )
    return {k: _probability(v, f"{path}.{k}") for k, v in value.items()}


def _require(obj: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise S1ResponseError(f"missing required field {key!r}", path=path)
    return obj[key]


def _parse_answer(name: str, question: Question, raw: Any) -> Answer:
    path = f"answers.{name}"
    if not isinstance(raw, Mapping):
        raise S1ResponseError(f"expected an answer object, got {raw!r}", path=path)
    atype = _require(raw, "type", path)
    if atype not in SUPPORTED_TYPES:
        raise S1UnsupportedFeatureError(
            f"{path}: endpoint returned unsupported answer type {atype!r}"
        )
    if atype != question.type:
        raise S1ResponseError(
            f"answer type {atype!r} does not match question type {question.type!r}", path=path
        )

    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=_probability(_require(raw, "noul", path), f"{path}.noul"))

    confidence = _probability(_require(raw, "confidence", path), f"{path}.confidence")

    if isinstance(question, ChoiceQuestion):
        choices = list(question.criteria)
        choice = _require(raw, "choice", path)
        if not isinstance(choice, str) or choice not in question.criteria:
            raise S1ResponseError(
                f"choice {choice!r} is not one of {choices}", path=f"{path}.choice"
            )
        probabilities = _probabilities(
            _require(raw, "probabilities", path), choices, f"{path}.probabilities"
        )
        return ChoiceAnswer(choice=choice, confidence=confidence, probabilities=probabilities)

    levels = [str(i) for i in range(len(question.criteria))]
    score = _require(raw, "score", path)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise S1ResponseError(f"expected a number, got {score!r}", path=f"{path}.score")
    if not 0 <= score <= len(levels) - 1:
        raise S1ResponseError(
            f"score {score!r} is outside the level range [0, {len(levels) - 1}]",
            path=f"{path}.score",
        )
    probabilities = _probabilities(
        _require(raw, "probabilities", path), levels, f"{path}.probabilities"
    )
    legend_raw = raw.get("legend", {})
    if not isinstance(legend_raw, Mapping) or not all(
        isinstance(v, str) for v in legend_raw.values()
    ):
        raise S1ResponseError(f"expected an object of strings, got {legend_raw!r}", path=f"{path}.legend")
    return ScoreAnswer(
        score=float(score),
        confidence=confidence,
        probabilities=probabilities,
        legend={str(k): v for k, v in legend_raw.items()},
    )


def _token_count(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise S1ResponseError(f"expected a non-negative integer, got {value!r}", path=path)
    return value


def parse_response(data: Any, questions: Mapping[str, Question]) -> S1Result:
    """Validate a raw response body against the questions that were asked."""
    if not isinstance(data, Mapping):
        raise S1ResponseError(f"expected a JSON object response, got {type(data).__name__}")
    model = _require(data, "model", "response")
    if not isinstance(model, str):
        raise S1ResponseError(f"expected a string, got {model!r}", path="model")
    answers_raw = _require(data, "answers", "response")
    if not isinstance(answers_raw, Mapping):
        raise S1ResponseError(f"expected an object, got {answers_raw!r}", path="answers")
    missing: List[str] = [name for name in questions if name not in answers_raw]
    if missing:
        raise S1ResponseError(
            f"response is missing answers for questions {missing}; got {sorted(answers_raw)}",
            path="answers",
        )
    answers = {name: _parse_answer(name, q, answers_raw[name]) for name, q in questions.items()}
    usage_raw = _require(data, "usage", "response")
    if not isinstance(usage_raw, Mapping):
        raise S1ResponseError(f"expected an object, got {usage_raw!r}", path="usage")
    usage = Usage(
        input_tokens=_token_count(_require(usage_raw, "input_tokens", "usage"), "usage.input_tokens"),
        output_tokens=_token_count(
            _require(usage_raw, "output_tokens", "usage"), "usage.output_tokens"
        ),
    )
    return S1Result(model=model, answers=answers, usage=usage)
