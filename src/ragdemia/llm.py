"""LLM backends behind one tiny interface: ``generate(messages)`` and ``stream(messages)``.

* ``OllamaLLM`` - talks HTTP to a local Ollama server (default). Ollama runs a 4-bit
  quantised GGUF build of the model through llama.cpp, so a 7-8B model fits in ~5 GB of
  VRAM.  No API key, no per-token cost, documents never leave the machine.
* ``HFLocalLLM`` - loads a model in-process with Hugging Face ``transformers``.  Useful
  when Ollama is not installed (CI, notebooks, servers without root).
* ``OpenAICompatibleLLM`` - any OpenAI-protocol endpoint (Groq, OpenRouter, vLLM); used by
  the public demo.
* ``EchoLLM`` - deterministic stub for tests.

Messages use the OpenAI/Ollama chat format: ``[{"role": "system"|"user"|"assistant",
"content": str}]``.  The model's chat template (applied by Ollama or by
``tokenizer.apply_chat_template``) turns this into the special-token format the model was
instruction-tuned on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Protocol

import httpx

Message = dict[str, str]


class LLM(Protocol):
    name: str

    def generate(self, messages: list[Message]) -> str: ...

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        num_ctx: int = 8192,
        timeout: float = 300.0,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.options = {
            "temperature": temperature,
            "num_predict": max_new_tokens,
            "num_ctx": num_ctx,
            "seed": 42,  # reproducible eval runs
        }
        self.timeout = timeout

    def _payload(self, messages: list[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": self.options,
        }

    def _raise_helpful(self, exc: Exception) -> None:
        raise LLMError(
            f"Could not reach Ollama at {self.base_url} ({exc}). Is it running? "
            f"Start it with `ollama serve` and pull the model with `ollama pull {self.model}`."
        ) from exc

    def generate(self, messages: list[Message]) -> str:
        try:
            r = httpx.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, False),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            self._raise_helpful(exc)
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")
        r.raise_for_status()
        return r.json()["message"]["content"]

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=self._payload(messages, True),
                timeout=self.timeout,
            ) as r:
                if r.status_code == 404:
                    raise LLMError(
                        f"Model '{self.model}' not found. Run `ollama pull {self.model}`."
                    )
                r.raise_for_status()
                for line in r.iter_lines():  # Ollama streams newline-delimited JSON
                    if not line:
                        continue
                    data = json.loads(line)
                    if token := data.get("message", {}).get("content"):
                        yield token
                    if data.get("done"):
                        break
        except httpx.HTTPError as exc:
            self._raise_helpful(exc)


class HFLocalLLM:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .embeddings import resolve_device

        self.name = f"hf:{model}"
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # fp16 on GPU; bf16 on CPU halves RAM (a 3B model needs ~6 GB instead of ~12 GB).
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _inputs(self, messages: list[Message]):
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return {k: v.to(self.device) for k, v in ids.items()}

    def _gen_kwargs(self) -> dict:
        kw = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if self.temperature > 0:
            kw.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            kw.update(do_sample=False)
        return kw

    def generate(self, messages: list[Message]) -> str:
        import torch

        inputs = self._inputs(messages)
        torch.manual_seed(42)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs())
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages: list[Message]) -> Iterator[str]:
        from threading import Thread

        import torch
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        torch.manual_seed(42)
        thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **self._gen_kwargs(), "streamer": streamer},
        )
        thread.start()
        yield from streamer
        thread.join()


class OpenAICompatibleLLM:
    """Any endpoint speaking the OpenAI chat-completions protocol.

    Used for the public demo (Groq's free tier serves open models such as Llama 3.1 8B in about
    a second), but equally for OpenRouter, a self-hosted vLLM server, or Ollama's ``/v1``.
    Streaming uses the protocol's own Server-Sent Events: ``data: {json}`` lines carrying
    ``choices[0].delta.content``, terminated by ``data: [DONE]``.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = f"openai:{model}"
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.params = {"temperature": temperature, "max_tokens": max_new_tokens, "seed": 42}
        self.client = httpx.Client(timeout=timeout, transport=transport)

    def _payload(self, messages: list[Message], stream: bool) -> dict:
        return {"model": self.model, "messages": messages, "stream": stream, **self.params}

    def _check(self, r: httpx.Response) -> None:
        if r.status_code == 401:
            raise LLMError("The LLM API rejected the API key (401). Check the configured key.")
        if r.status_code == 429:
            raise LLMError("The LLM API rate limit was reached. Please try again in a minute.")
        if r.status_code >= 400:
            r.read()
            # OpenAI-style errors: {"error": {"message": ..., "code": ...}}; fall back to raw text.
            detail = r.text[:300]
            try:
                err = r.json().get("error", {})
                detail = err.get("message") or detail
                if err.get("code"):
                    detail = f"{detail} (code: {err['code']})"
            except ValueError:
                pass
            raise LLMError(
                f"LLM API error {r.status_code} for model '{self.model}': "
                f"{detail or r.reason_phrase or 'no details returned'}"
            )

    def generate(self, messages: list[Message]) -> str:
        try:
            r = self.client.post(
                self.url, json=self._payload(messages, False), headers=self.headers
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach the LLM API at {self.url} ({exc}).") from exc
        self._check(r)
        return r.json()["choices"][0]["message"]["content"] or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            with self.client.stream(
                "POST", self.url, json=self._payload(messages, True), headers=self.headers
            ) as r:
                self._check(r)
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or [{}]
                    if token := (choices[0].get("delta") or {}).get("content"):
                        yield token
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach the LLM API at {self.url} ({exc}).") from exc


class EchoLLM:
    """Test double: returns a canned response (or echoes the last user message)."""

    def __init__(self, response: str | None = None) -> None:
        self.name = "echo"
        self.response = response
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self.response if self.response is not None else messages[-1]["content"]

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield from self.generate(messages).split(" ")


def build_llm(cfg) -> LLM:
    """Factory from ``LLMConfig``."""
    if cfg.backend == "ollama":
        return OllamaLLM(cfg.model, cfg.base_url, cfg.temperature, cfg.max_new_tokens, cfg.num_ctx)
    if cfg.backend == "hf":
        return HFLocalLLM(cfg.hf_model, cfg.temperature, cfg.max_new_tokens)
    if cfg.backend == "openai":
        import os

        return OpenAICompatibleLLM(
            cfg.model,
            cfg.base_url,
            os.environ.get(cfg.api_key_env),
            cfg.temperature,
            cfg.max_new_tokens,
        )
    raise ValueError(f"Unknown LLM backend '{cfg.backend}' (expected 'ollama', 'hf' or 'openai')")
