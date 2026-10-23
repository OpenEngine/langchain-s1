# Python: API reference

Generated from the docstrings in `python/src/langchain_s1` on every build — it always
matches the released `langchain-s1` package. Modules are ordered as in
[the architecture](../architecture.md): the contract, the transport, the runnables.

## The contract

Questions, answers, results, and response validation.

::: langchain_s1.types
    options:
        members:
            - NoulQuestion
            - ChoiceQuestion
            - ScoreQuestion
            - coerce_question
            - coerce_questions
            - NoulAnswer
            - ChoiceAnswer
            - ScoreAnswer
            - Usage
            - S1Result
            - parse_response

## The transport

The HTTP client: request building, auth, retries, redaction.

::: langchain_s1.client
    options:
        members:
            - DEFAULT_BASE_URL
            - S1Client
            - redact_url

## Runnables

The LangChain integration: the decision runnable and the gate.

::: langchain_s1.decision
    options:
        members:
            - S1Decision

::: langchain_s1.gate
    options:
        members:
            - GATE_EVENT
            - GateAction
            - GateOutcome
            - S1Gate
            - s1_gate
            - noul_policy
            - choice_policy

## Errors

The exception hierarchy (see [Errors](errors.md) for when each one fires).

::: langchain_s1.errors
    options:
        members:
            - S1Error
            - S1ConfigurationError
            - S1UnsupportedFeatureError
            - S1InferenceError
            - S1RequestError
            - S1TimeoutError
            - S1APIError
            - S1ResponseError
            - S1UncertainDecisionError

## Testing

The in-process mock endpoint (see [Testing](testing.md)).

::: langchain_s1.testing
    options:
        members:
            - MockSystemOne
            - default_answer
