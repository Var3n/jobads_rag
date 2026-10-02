import json

import pytest

from hisrag.agent.loop import Agent, check_citations
from hisrag.agent.tools import Tools
from hisrag.llm import DHClient, FakeOpenAI
from test_index import built, cfg  # noqa: F401  (fixtures: a two-newspaper index with fake vectors)


def call(name, args, id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def make_agent(built, handler):  # noqa: F811
    cfg, *_ = built
    cfg["paths"]["agent_log"] = str(cfg.path("index_dir").parent / "logs" / "agent.jsonl")
    fake = FakeOpenAI(chat_handler=handler, embedding_dim=32)
    client = DHClient(cfg, openai_client=fake)
    return Agent(cfg, client, Tools(cfg, client=client)), fake


def rounds(payload) -> int:
    return sum(m["role"] == "assistant" for m in payload["messages"])


def test_tool_rounds_then_answer_with_checked_citations(built):  # noqa: F811
    def handler(payload):
        n = rounds(payload)
        if n == 0:  # two calls in one round
            return {"tool_calls": [call("expand_concept", {"term": "Lehrer"}, "a"),
                                   call("search_ads", {"query": "Krakau", "mode": "keyword"}, "b")]}
        if n == 1:
            assert [m["tool_call_id"] for m in payload["messages"] if m["role"] == "tool"] == ["a", "b"]
            return {"tool_calls": [call("aggregate", {"group_by": "decade"})]}
        return "Lehrerstellen in Krakau [w3]; dazu erfunden [wrz_18990101_001_region_0001]."

    agent, fake = make_agent(built, handler)
    a = agent.ask("Lehrer in Krakau?")
    assert a.stopped == "answer" and a.steps == 3 and a.tools_used() == "expand_concept → search_ads → aggregate"
    assert a.trace[1]["result"]["total_matches"] == 1
    assert fake.chat_calls[0]["messages"][0]["role"] == "system" and "tools" in fake.chat_calls[0]
    assert a.cited == ["wrz_18990101_001_region_0001"]  # w3 is a test ID, not in the wrz ID format
    assert a.unknown_ids == ["wrz_18990101_001_region_0001"]
    log = [json.loads(l) for l in agent.cfg.path("agent_log").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["question"] == "Lehrer in Krakau?" and len(log[-1]["trace"]) == 3


def test_step_limit_forces_an_answer_without_tools(built):  # noqa: F811
    def handler(payload):
        if "tools" not in payload:
            return "Was ich gefunden habe."
        return {"tool_calls": [call("search_ads", {"query": f"Runde {rounds(payload)}"})]}

    agent, fake = make_agent(built, handler)
    a = agent.ask("Endlos?", max_steps=3, log=False)
    assert a.stopped == "max_steps" and a.steps == 4 and len(a.trace) == 3
    assert a.answer == "Was ich gefunden habe."
    assert "höchste Zahl" in fake.chat_calls[-1]["messages"][-1]["content"]


def test_bad_tool_arguments_go_back_to_the_model(built):  # noqa: F811
    def handler(payload):
        if rounds(payload) == 0:
            return {"tool_calls": [call("search_ads", {"query": "x", "k": 999})]}
        tool_msg = payload["messages"][-1]
        assert "ungültige Argumente" in tool_msg["content"]
        return "Korrigiert."

    agent, _ = make_agent(built, handler)
    assert agent.ask("?", log=False).answer == "Korrigiert."


def test_api_failure_keeps_the_trace(built):  # noqa: F811
    def handler(payload):
        if rounds(payload) == 0:
            return {"tool_calls": [call("expand_concept", {"term": "Lehrer"})]}
        raise RuntimeError("503")

    agent, _ = make_agent(built, handler)
    a = agent.ask("?", log=False)
    assert a.stopped == "error" and "503" in a.answer and len(a.trace) == 1


def test_check_citations():
    trace = [{"result": {"results": [{"ad_id": "wrz_18620412_017_region_0162"}]}}]
    cited, unknown = check_citations("A [wrz_18620412_017_region_0162], B [nfp_19000101_002_region_0003], "
                                     "A again [wrz_18620412_017_region_0162]", trace)
    assert cited == ["wrz_18620412_017_region_0162", "nfp_19000101_002_region_0003"]
    assert unknown == ["nfp_19000101_002_region_0003"]


def test_pilot_writes_a_review_csv(built, monkeypatch):  # noqa: F811
    import pandas as pd

    from hisrag.agent.__main__ import PILOT_QUESTIONS, run_pilot

    cfg, *_ = built
    cfg["paths"]["agent_log"] = str(cfg.path("index_dir").parent / "logs" / "agent.jsonl")

    def handler(payload):
        if rounds(payload) == 0:
            return {"tool_calls": [call("aggregate", {"group_by": "decade"})]}
        return "Antwort."

    fake = FakeOpenAI(chat_handler=handler, embedding_dim=32)
    monkeypatch.setattr("hisrag.llm.DHClient", lambda c: DHClient(c, openai_client=fake))
    report = run_pilot(cfg, [False, True])
    rows = pd.read_csv(report["review_csv"])
    assert len(rows) == 2 * len(PILOT_QUESTIONS) and set(rows["tools_used"]) == {"aggregate"}
    assert report["per_thinking"]["False"]["answers"] == len(PILOT_QUESTIONS)


def test_an_answer_without_tools_is_sent_back_once(built):  # noqa: F811
    def handler(payload):
        users = [m["content"] for m in payload["messages"] if m["role"] == "user"]
        if len(users) == 1:
            return {"content": "Geschätzt: viele.", "reasoning": "hmm"}
        if rounds(payload) == 1:
            return {"tool_calls": [call("aggregate", {"group_by": "none"})], "reasoning": "jetzt Tools"}
        return {"content": "Gezählt: 5.", "reasoning": "fertig"}

    agent, _ = make_agent(built, handler)
    a = agent.ask("Wie viele?", log=False)
    assert a.answer == "Gezählt: 5." and len(a.trace) == 1 and a.steps == 3
    assert a.reasoning_chars == len("hmm") + len("jetzt Tools") + len("fertig")
    assert a.settings["prompt_version"] == "v2"
