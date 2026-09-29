"""Offline stand-in for the openai client, for tests and for developing without cluster access.

    fake = FakeOpenAI(chat_handler=lambda payload: "some answer")
    client = DHClient(cfg, openai_client=fake)

A chat handler receives the request payload and returns either a string (the answer), a dict
with "content" / "tool_calls" / "finish_reason", or raises an exception to simulate errors.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


class _Response:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data


def _usage(prompt: int, completion: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 0}}


class _Completions:
    def __init__(self, owner: "FakeOpenAI"):
        self.owner = owner

    def create(self, **payload: Any) -> _Response:
        self.owner.chat_calls.append(payload)
        out = self.owner.chat_handler(payload)
        if isinstance(out, str):
            out = {"content": out}
        message = {"role": "assistant", "content": out.get("content"), "tool_calls": out.get("tool_calls")}
        if "reasoning" in out:
            message["reasoning"] = out["reasoning"]
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in payload["messages"]) // 4
        return _Response({
            "id": "chatcmpl-fake", "model": payload["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": out.get("finish_reason", "tool_calls" if out.get("tool_calls") else "stop")}],
            "usage": _usage(prompt_tokens, len(out.get("content") or "") // 4),
        })


class _Embeddings:
    def __init__(self, owner: "FakeOpenAI"):
        self.owner = owner

    def create(self, model: str, input: list[str] | str) -> _Response:
        texts = [input] if isinstance(input, str) else list(input)
        self.owner.embedding_calls.append({"model": model, "input": texts})
        # Deterministic pseudo-vectors so identical texts get identical embeddings.
        data = []
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(f"{model}|{text}".encode()).hexdigest()[:8], 16)
            vec = np.random.default_rng(seed).standard_normal(self.owner.embedding_dim)
            data.append({"object": "embedding", "index": i, "embedding": vec.tolist()})
        data.reverse()  # the real API does not guarantee order either; callers must sort by index
        return _Response({"object": "list", "model": model, "data": data,
                          "usage": _usage(sum(len(t) for t in texts) // 4, 0)})


class _Chat:
    def __init__(self, owner: "FakeOpenAI"):
        self.completions = _Completions(owner)


class FakeOpenAI:
    def __init__(self, chat_handler: Callable[[dict], Any] | None = None, embedding_dim: int = 32):
        self.chat_handler = chat_handler or (lambda payload: "ok")
        self.embedding_dim = embedding_dim
        self.chat_calls: list[dict] = []
        self.embedding_calls: list[dict] = []
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
