import json

import httpx
import pytest

from ragdemia.llm import LLMError, OpenAICompatibleLLM


def transport(handler):
    return httpx.MockTransport(handler)


def test_generate_sends_bearer_key_and_parses_reply():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello [1]."}}]})

    llm = OpenAICompatibleLLM(
        "llama-3.1-8b-instant", "https://api.example/v1", "sk-test", transport=transport(handler)
    )
    assert llm.generate([{"role": "user", "content": "hi"}]) == "Hello [1]."
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "llama-3.1-8b-instant" and seen["body"]["stream"] is False


def test_stream_parses_sse_deltas():
    chunks = ["Byte", "-pair", " encoding [1]."]
    body = (
        "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks)
        + "data: [DONE]\n\n"
    )

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM("m", "https://api.example/v1", transport=transport(handler))
    assert "".join(llm.stream([{"role": "user", "content": "x"}])) == "Byte-pair encoding [1]."


@pytest.mark.parametrize("status,msg", [(401, "API key"), (429, "rate limit"), (500, "error 500")])
def test_errors_are_explained(status, msg):
    llm = OpenAICompatibleLLM(
        "m",
        "https://api.example/v1",
        transport=transport(lambda r: httpx.Response(status, text="nope")),
    )
    with pytest.raises(LLMError, match=msg):
        llm.generate([{"role": "user", "content": "x"}])


def test_error_message_includes_provider_reason():
    body = {
        "error": {
            "message": "The model `x` has been decommissioned",
            "code": "model_decommissioned",
        }
    }
    llm = OpenAICompatibleLLM(
        "x", "https://api.example/v1", transport=transport(lambda r: httpx.Response(400, json=body))
    )
    with pytest.raises(LLMError, match="decommissioned.*model_decommissioned"):
        llm.generate([{"role": "user", "content": "x"}])


def test_whitespace_around_pasted_key_is_removed():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatibleLLM(
        "m", "https://api.example/v1", " gsk_abc123\r\n", transport=transport(handler)
    )
    llm.generate([{"role": "user", "content": "x"}])
    assert seen["auth"] == "Bearer gsk_abc123"


def test_retired_model_switches_to_an_available_one():
    calls = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "whisper-large-v3"},
                        {"id": "llama-3.3-70b-versatile"},
                        {"id": "qwen/qwen3-32b"},
                    ]
                },
            )
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "llama-3.1-8b-instant":
            return httpx.Response(
                404, json={"error": {"message": "does not exist", "code": "model_not_found"}}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"answer from {model}"}}]}
        )

    llm = OpenAICompatibleLLM(
        "llama-3.1-8b-instant", "https://api.example/v1", transport=transport(handler)
    )
    assert llm.generate([{"role": "user", "content": "x"}]) == "answer from llama-3.3-70b-versatile"
    assert llm.name == "openai:llama-3.3-70b-versatile"
    # later calls go straight to the new model
    llm.generate([{"role": "user", "content": "y"}])
    assert calls == ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-3.3-70b-versatile"]


def test_retired_model_switch_also_works_when_streaming():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})
        if json.loads(request.content)["model"] == "old":
            return httpx.Response(
                404, json={"error": {"message": "gone", "code": "model_decommissioned"}}
            )
        body = (
            f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    llm = OpenAICompatibleLLM("old", "https://api.example/v1", transport=transport(handler))
    assert "".join(llm.stream([{"role": "user", "content": "x"}])) == "ok"
