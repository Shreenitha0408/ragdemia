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
