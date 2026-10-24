import asyncio
import json

import httpx
import pytest

from langchain_s1 import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    S1APIError,
    S1Client,
    S1ConfigurationError,
    S1RequestError,
    S1ResponseError,
    S1TimeoutError,
    S1UnsupportedFeatureError,
    ScoreAnswer,
    ScoreQuestion,
)
from langchain_s1.testing import MockSystemOne

QUESTIONS = {
    "is_question": NoulQuestion("Is the user asking a question?"),
    "route": ChoiceQuestion(
        "Which team should handle this?", {"billing": "Payments", "support": "Everything else"}
    ),
    "urgency": ScoreQuestion("How urgent is this?", ["not urgent", "somewhat", "very"]),
}


def raw_transport(handler):
    calls = []

    def wrapped(request):
        calls.append(request)
        return handler(request, len(calls))

    t = httpx.MockTransport(wrapped)
    t.calls = calls
    return t


def ok_body(answers, model="m"):
    return {"model": model, "answers": answers, "usage": {"input_tokens": 3, "output_tokens": 1}}


def respond(answers, status=200):
    return raw_transport(lambda req, n: httpx.Response(status, json=ok_body(answers)))


# -- configuration ----------------------------------------------------------


def test_default_base_url_appends_systemone():
    assert S1Client("jev-latest").endpoint_url == "https://api.typesafe.ai/v1/systemone"


def test_custom_base_url_and_endpoint_override():
    assert S1Client("x", base_url="http://localhost:8080/api/").endpoint_url == (
        "http://localhost:8080/api/systemone"
    )
    c = S1Client("x", base_url="http://ignored", endpoint_url="http://h:1/custom/decide")
    assert c.endpoint_url == "http://h:1/custom/decide"


def test_local_endpoint_without_api_key_sends_no_auth():
    mock = MockSystemOne(path="/systemone")
    c = S1Client("local-model", base_url="http://127.0.0.1:9000", transport=mock)
    result = c.decide("hi", QUESTIONS)
    assert result.model == "local-model"
    assert "authorization" not in mock.requests[0].headers


def test_api_key_and_custom_headers():
    mock = MockSystemOne(
        path="/gw/s1", required_headers={"x-api-token": "tok", "authorization": "Bearer k"}
    )
    c = S1Client(
        "m",
        endpoint_url="https://gw.internal/gw/s1",
        api_key="k",
        headers={"X-Api-Token": "tok"},
        transport=mock,
    )
    c.decide("hi", QUESTIONS)
    unauth = S1Client("m", endpoint_url="https://gw.internal/gw/s1", transport=mock, max_retries=0)
    with pytest.raises(S1APIError) as e:
        unauth.decide("hi", QUESTIONS)
    assert e.value.status_code == 401


def test_repr_hides_credentials():
    c = S1Client("m", api_key="sk-secret", headers={"X-Key": "hdr-secret"})
    assert "secret" not in repr(c)


def test_timeout_is_applied_per_request():
    mock = MockSystemOne()
    S1Client("m", timeout=1.5, transport=mock).decide("s", QUESTIONS)
    assert mock.requests[0].extensions["timeout"]["read"] == 1.5


@pytest.mark.parametrize("kwargs", [{"model": ""}, {"model": "m", "max_retries": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(S1ConfigurationError):
        S1Client(**kwargs)


# -- contract ---------------------------------------------------------------


def test_request_body_follows_contract():
    mock = MockSystemOne()
    S1Client("jev-latest", transport=mock).decide({"messages": ["hi"]}, QUESTIONS)
    body = mock.bodies[0]
    assert body == {
        "model": "jev-latest",
        "state": {"messages": ["hi"]},
        "questions": {
            "is_question": {"type": "noul", "instructions": "Is the user asking a question?"},
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments", "support": "Everything else"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat", "very"],
            },
        },
    }


def test_mixed_answers_preserve_structure():
    answers = {
        "is_question": {"type": "noul", "noul": 0.25},
        "route": {
            "type": "choice",
            "choice": "support",
            "confidence": 0.7,
            "probabilities": {"billing": 0.3, "support": 0.7},
        },
        "urgency": {
            "type": "score",
            "score": 1.4,
            "confidence": 0.5,
            "legend": {"0": "not urgent", "1": "somewhat", "2": "very"},
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        },
    }
    result = S1Client("m", transport=respond(answers)).decide("s", QUESTIONS)
    assert list(result) == ["is_question", "route", "urgency"]
    assert result["is_question"] == NoulAnswer(noul=0.25)
    assert isinstance(result["route"], ChoiceAnswer)
    assert result["route"].probabilities == {"billing": 0.3, "support": 0.7}
    assert isinstance(result["urgency"], ScoreAnswer)
    assert result["urgency"].legend["2"] == "very"
    assert result.usage.input_tokens == 3 and result.usage.output_tokens == 1
    assert result.to_dict() == ok_body(answers)


def test_dict_questions_accepted():
    mock = MockSystemOne()
    result = S1Client("m", transport=mock).decide(
        "s", {"q": {"type": "noul", "instructions": "ok?", "criteria": {"true": "yes"}}}
    )
    assert isinstance(result["q"], NoulAnswer)


@pytest.mark.parametrize(
    "answers,match",
    [
        ({}, r"missing answers for questions \['q'\]"),
        ({"q": {"type": "choice", "choice": "a", "confidence": 1, "probabilities": {}}}, "does not match"),
        ({"q": {"type": "noul", "noul": 1.2}}, r"answers.q.noul: probability must be within"),
        ({"q": {"type": "noul", "noul": "yes"}}, "expected a probability number"),
        ({"q": {"type": "noul", "noul": True}}, "expected a probability number"),
        ({"q": {"type": "noul"}}, "missing required field 'noul'"),
    ],
)
def test_invalid_noul_answers(answers, match):
    c = S1Client("m", transport=respond(answers))
    with pytest.raises(S1ResponseError, match=match):
        c.decide("s", {"q": NoulQuestion("?")})


def test_invalid_choice_and_score_answers():
    choice_q = {"q": ChoiceQuestion("?", {"a": "A", "b": "B"})}
    bad_choice = {"q": {"type": "choice", "choice": "c", "confidence": 1, "probabilities": {}}}
    with pytest.raises(S1ResponseError, match=r"choice 'c' is not one of \['a', 'b'\]"):
        S1Client("m", transport=respond(bad_choice)).decide("s", choice_q)
    bad_probs = {
        "q": {"type": "choice", "choice": "a", "confidence": 1, "probabilities": {"a": -0.1}}
    }
    with pytest.raises(S1ResponseError, match=r"probabilities.a"):
        S1Client("m", transport=respond(bad_probs)).decide("s", choice_q)

    score_q = {"q": ScoreQuestion("?", ["low", "high"])}
    bad_score = {"q": {"type": "score", "score": 3, "confidence": 1, "probabilities": {}}}
    with pytest.raises(S1ResponseError, match="outside the level range"):
        S1Client("m", transport=respond(bad_score)).decide("s", score_q)
    bad_level = {"q": {"type": "score", "score": 1, "confidence": 1, "probabilities": {"5": 1}}}
    with pytest.raises(S1ResponseError, match="unexpected keys"):
        S1Client("m", transport=respond(bad_level)).decide("s", score_q)


def test_malformed_envelopes():
    q = {"q": NoulQuestion("?")}
    ans = {"q": {"type": "noul", "noul": 0.5}}
    cases = [
        (httpx.Response(200, text="not json"), "not valid JSON"),
        (httpx.Response(200, json=[1]), "expected a JSON object"),
        (httpx.Response(200, json={"answers": ans, "usage": {}}), "missing required field 'model'"),
        (httpx.Response(200, json={"model": "m", "answers": ans}), "'usage'"),
        (
            httpx.Response(
                200,
                json={"model": "m", "answers": ans, "usage": {"input_tokens": -1, "output_tokens": 0}},
            ),
            "usage.input_tokens",
        ),
    ]
    for response, match in cases:
        c = S1Client("m", transport=raw_transport(lambda req, n, r=response: r))
        with pytest.raises(S1ResponseError, match=match):
            c.decide("s", q)


def test_unsupported_features():
    with pytest.raises(S1UnsupportedFeatureError, match="unsupported type 'box'"):
        S1Client("m").decide("s", {"where": {"type": "box", "instructions": "find the cat"}})
    with pytest.raises(S1UnsupportedFeatureError, match="unsupported fields"):
        S1Client("m").decide("s", {"q": {"type": "noul", "instructions": "?", "assets": {}}})
    returned_box = {"q": {"type": "box", "box": {}, "confidence": None}}
    with pytest.raises(S1UnsupportedFeatureError, match="unsupported answer type 'box'"):
        S1Client("m", transport=respond(returned_box)).decide("s", {"q": NoulQuestion("?")})


def test_validation_error_detail_is_surfaced():
    detail = [{"loc": ["body", "model"], "msg": "unknown model", "type": "value_error"}]
    t = raw_transport(lambda req, n: httpx.Response(422, json={"detail": detail}))
    with pytest.raises(S1APIError, match="HTTP 422") as e:
        S1Client("m", transport=t).decide("s", QUESTIONS)
    assert e.value.body == {"detail": detail}
    assert "unknown model" not in str(e.value)  # body stays off the traced message
    assert len(t.calls) == 1  # not retried


# -- retries and timeouts ---------------------------------------------------


def flaky(fail_times, failure):
    answers = {"q": {"type": "noul", "noul": 0.5}}

    def handler(req, n):
        if n <= fail_times:
            if isinstance(failure, int):
                return httpx.Response(failure, json={"detail": "busy"})
            raise failure
        return httpx.Response(200, json=ok_body(answers))

    return raw_transport(handler)


@pytest.mark.parametrize("failure", [503, 429, httpx.ConnectError("refused")])
def test_bounded_retries_recover(failure):
    t = flaky(2, failure)
    result = S1Client("m", transport=t, max_retries=2, retry_backoff=0).decide(
        "s", {"q": NoulQuestion("?")}
    )
    assert result["q"].noul == 0.5
    assert len(t.calls) == 3


def test_retries_exhausted():
    t = flaky(10, 503)
    with pytest.raises(S1APIError) as e:
        S1Client("m", transport=t, max_retries=2, retry_backoff=0).decide(
            "s", {"q": NoulQuestion("?")}
        )
    assert e.value.status_code == 503 and e.value.attempts == 3
    assert len(t.calls) == 3

    t = flaky(10, httpx.ConnectError("refused"))
    with pytest.raises(S1RequestError, match="after 2 attempt"):
        S1Client("m", transport=t, max_retries=1, retry_backoff=0).decide(
            "s", {"q": NoulQuestion("?")}
        )


def test_timeout_raises_after_retries():
    t = flaky(10, httpx.ReadTimeout("slow"))
    with pytest.raises(S1TimeoutError, match="timed out after 3 attempt") as e:
        S1Client("m", transport=t, max_retries=2, retry_backoff=0).decide(
            "s", {"q": NoulQuestion("?")}
        )
    assert e.value.attempts == 3


def test_async_retries_and_timeout():
    t = flaky(1, 500)
    c = S1Client("m", transport=t, retry_backoff=0)
    result = asyncio.run(c.adecide("s", {"q": NoulQuestion("?")}))
    assert result["q"].noul == 0.5 and len(t.calls) == 2

    t = flaky(10, httpx.ReadTimeout("slow"))
    c = S1Client("m", transport=t, max_retries=0)
    with pytest.raises(S1TimeoutError):
        asyncio.run(c.adecide("s", {"q": NoulQuestion("?")}))
    assert len(t.calls) == 1


def test_async_cancellation_propagates():
    async def main():
        started = asyncio.Event()

        async def handler(request):
            started.set()
            await asyncio.sleep(60)

        c = S1Client("m", transport=httpx.MockTransport(handler))
        task = asyncio.ensure_future(c.adecide("s", {"q": NoulQuestion("?")}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(main(), 5))


def test_state_serialized_as_json():
    mock = MockSystemOne()
    S1Client("m", transport=mock).decide(["a", {"b": 1}], {"q": NoulQuestion("?")})
    assert json.loads(mock.requests[0].content)["state"] == ["a", {"b": 1}]


# -- credential and payload hygiene -----------------------------------------


def test_url_credentials_redacted_from_messages_and_repr():
    url = "https://user:hunter2@gw.example:8443/s1?api_key=sk-leak#frag"
    t = raw_transport(lambda req, n: httpx.Response(500, json={"detail": "x"}))
    c = S1Client("m", endpoint_url=url, transport=t, max_retries=0)
    assert c.endpoint_url == url
    assert "hunter2" not in repr(c) and "sk-leak" not in repr(c)
    with pytest.raises(S1APIError) as e:
        c.decide("s", QUESTIONS)
    assert "hunter2" not in str(e.value) and "sk-leak" not in str(e.value)
    assert "https://gw.example:8443/s1" in str(e.value)

    t = flaky(10, httpx.ConnectError("refused"))
    c = S1Client("m", endpoint_url=url, transport=t, max_retries=0)
    with pytest.raises(S1RequestError) as e:
        c.decide("s", QUESTIONS)
    assert "hunter2" not in str(e.value) and "sk-leak" not in str(e.value)


def test_transport_error_text_is_redacted():
    proxy = "http://user:hunter2@proxy.example:8080"
    t = flaky(10, httpx.ProxyError(f"proxy connection failed for {proxy}"))
    c = S1Client("m", transport=t, max_retries=0)
    with pytest.raises(S1RequestError) as e:
        c.decide("s", QUESTIONS)
    assert "hunter2" not in str(e.value)
    assert "http://proxy.example:8080" in str(e.value)

    def echo(request):
        raise httpx.TransportError(f"internal detail {proxy} leaked")

    c = S1Client("m", transport=httpx.MockTransport(echo), max_retries=0)
    with pytest.raises(S1RequestError) as e:
        c.decide("s", QUESTIONS)
    assert "hunter2" not in str(e.value)
    assert "http://proxy.example:8080" in str(e.value)


def test_large_error_body_truncated():
    t = raw_transport(lambda req, n: httpx.Response(400, text="E" * 100_000))
    with pytest.raises(S1APIError) as e:
        S1Client("m", transport=t).decide("s", QUESTIONS)
    assert len(e.value.body) < 3000 and e.value.body.endswith("[truncated]")
    assert "EEEE" not in str(e.value)


# -- async transport and connection reuse -----------------------------------


class SyncOnly(httpx.BaseTransport):
    def handle_request(self, request):
        return httpx.Response(200, json=ok_body({"q": {"type": "noul", "noul": 0.5}}))


def test_sync_only_transport_refuses_async():
    c = S1Client("m", transport=SyncOnly())
    assert c.decide("s", {"q": NoulQuestion("?")})["q"].noul == 0.5
    with pytest.raises(S1ConfigurationError, match="async_transport"):
        asyncio.run(c.adecide("s", {"q": NoulQuestion("?")}))


def test_async_client_reused_within_loop():
    t = respond({"q": {"type": "noul", "noul": 0.5}})

    async def main():
        async with S1Client("m", transport=t) as c:
            await asyncio.gather(*(c.adecide("s", {"q": NoulQuestion("?")}) for _ in range(5)))
            first = c._aclient()
            await c.adecide("s", {"q": NoulQuestion("?")})
            assert c._aclient() is first
        assert first.is_closed

    asyncio.run(main())
    asyncio.run(main())  # a new loop gets a fresh client
