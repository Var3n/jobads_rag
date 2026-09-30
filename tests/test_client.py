import json

import numpy as np
import pytest
from pydantic import BaseModel

from hisrag.config import load_config
from hisrag.llm import DHClient, FakeOpenAI, StructuredOutputError


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"]["cache_db"] = str(tmp_path / "cache.sqlite")
    cfg["paths"]["usage_log"] = str(tmp_path / "usage.jsonl")
    cfg["api"]["requests_per_minute"] = None
    return cfg


class Position(BaseModel):
    surface: str
    lemma: str
    is_position: bool


def test_chat_sets_thinking_flag_and_caches(cfg):
    fake = FakeOpenAI(lambda p: "Hallo")
    client = DHClient(cfg, openai_client=fake)

    first = client.chat([{"role": "user", "content": "Hi"}])
    second = client.chat([{"role": "user", "content": "Hi"}])

    assert first.content == second.content == "Hallo"
    assert not first.from_cache and second.from_cache
    assert len(fake.chat_calls) == 1
    assert fake.chat_calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    totals = client.usage.summary()["default"]
    assert totals["requests"] == 1 and totals["cache_hits"] == 1


def test_truncated_answers_are_cached_per_max_tokens(cfg):
    fake = FakeOpenAI(lambda p: {"content": "abgeschnit", "finish_reason": "length"})
    client = DHClient(cfg, openai_client=fake)
    client.chat([{"role": "user", "content": "x"}], max_tokens=100)
    again = client.chat([{"role": "user", "content": "x"}], max_tokens=100)
    assert len(fake.chat_calls) == 1 and again.from_cache and again.finish_reason == "length"
    client.chat([{"role": "user", "content": "x"}], max_tokens=200)  # more tokens: a new request
    assert len(fake.chat_calls) == 2


def test_chat_json_prompt_mode_parses_fenced_json(cfg):
    answer = '```json\n{"surface": "Köchin", "lemma": "Köchin", "is_position": true}\n```'
    fake = FakeOpenAI(lambda p: answer)
    client = DHClient(cfg, openai_client=fake)

    result = client.chat_json([{"role": "user", "content": "Köchin"}], Position, mode="prompt")

    assert result == Position(surface="Köchin", lemma="Köchin", is_position=True)
    assert "JSON-Schema" in fake.chat_calls[0]["messages"][0]["content"]


def test_chat_json_repairs_invalid_answer_once(cfg):
    answers = iter(['{"surface": "Köchin"}', '{"surface": "Köchin", "lemma": "Köchin", "is_position": true}'])
    client = DHClient(cfg, openai_client=FakeOpenAI(lambda p: next(answers)))
    assert client.chat_json([{"role": "user", "content": "x"}], Position, mode="prompt").lemma == "Köchin"


def test_repaired_answers_come_from_the_cache_on_rerun(cfg):
    answers = iter(['{"surface": "Köchin"}', '{"surface": "Köchin", "lemma": "Köchin", "is_position": true}'])
    fake = FakeOpenAI(lambda p: next(answers))
    for _ in range(2):  # a second client on the same cache, as in a re-run of a step
        client = DHClient(cfg, openai_client=fake)
        assert client.chat_json([{"role": "user", "content": "x"}], Position, mode="prompt").lemma == "Köchin"
    assert len(fake.chat_calls) == 2


def test_chat_json_gives_up_after_repairs(cfg):
    client = DHClient(cfg, openai_client=FakeOpenAI(lambda p: "kein json"))
    with pytest.raises(StructuredOutputError):
        client.chat_json([{"role": "user", "content": "x"}], Position, mode="prompt")


def test_chat_json_auto_falls_back_to_prompt_on_400(cfg):
    class BadRequest(Exception):
        status_code = 400

    def handler(payload):
        if "response_format" in payload:
            raise BadRequest("json_schema not supported")
        return '{"surface": "a", "lemma": "a", "is_position": false}'

    cfg["llm"]["structured_output"] = "auto"
    client = DHClient(cfg, openai_client=FakeOpenAI(handler))
    client.chat_json([{"role": "user", "content": "x"}], Position)
    assert client.structured_mode == "prompt"


def test_tool_calls_round_trip(cfg):
    call = {"id": "call_1", "type": "function",
            "function": {"name": "count_ads", "arguments": json.dumps({"decade": 1870})}}
    client = DHClient(cfg, openai_client=FakeOpenAI(lambda p: {"content": None, "tool_calls": [call]}))
    result = client.chat([{"role": "user", "content": "?"}], tools=[{"type": "function", "function": {"name": "count_ads"}}])
    assert result.finish_reason == "tool_calls"
    assert result.message["tool_calls"][0]["function"]["name"] == "count_ads"


def test_embed_orders_by_index_applies_prefix_and_normalizes(cfg):
    fake = FakeOpenAI(embedding_dim=16)
    client = DHClient(cfg, openai_client=fake)
    texts = [f"Anzeige {i}" for i in range(10)]

    vecs = client.embed(texts, "embeddinggemma-300m", kind="query", batch_size=4)
    again = client.embed(texts[3:4], "embeddinggemma-300m", kind="query")

    assert vecs.shape == (10, 16)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vecs[3], again[0], atol=1e-6)  # same text → same row, despite reversed API order
    assert fake.embedding_calls[0]["input"][0].startswith("task: search result | query: ")


def test_embed_uses_separate_query_and_passage_models(cfg):
    fake = FakeOpenAI()
    client = DHClient(cfg, openai_client=fake)
    client.embed(["a"], "jina-embeddings-v3", kind="query")
    client.embed(["a"], "jina-embeddings-v3", kind="passage")
    assert [c["model"] for c in fake.embedding_calls] == ["jina-embeddings-v3-query", "jina-embeddings-v3-passage"]


def test_map_can_collect_exceptions(cfg):
    client = DHClient(cfg, openai_client=FakeOpenAI())

    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return x * 10

    out = client.map(fn, [1, 2, 3], return_exceptions=True)
    assert out[0] == 10 and isinstance(out[1], ValueError) and out[2] == 30


def test_leading_blank_lines_from_reasoning_mode_are_stripped(cfg):
    client = DHClient(cfg, openai_client=FakeOpenAI(lambda p: "\n\nJa, das ist eine Berufsbezeichnung."))
    assert client.chat([{"role": "user", "content": "?"}], thinking=True).content.startswith("Ja")


def test_per_request_timeout_and_retries_are_not_part_of_cache_key(cfg):
    fake = FakeOpenAI(lambda p: "ok")
    client = DHClient(cfg, openai_client=fake)
    client.chat([{"role": "user", "content": "x"}], timeout_s=1800, max_retries=0)
    again = client.chat([{"role": "user", "content": "x"}])
    assert again.from_cache and len(fake.chat_calls) == 1
    assert fake.options[0]["max_retries"] == 0 and fake.options[0]["timeout"].read == 1800
