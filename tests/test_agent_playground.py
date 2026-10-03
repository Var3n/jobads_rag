import json
from types import SimpleNamespace

import pytest

from hisrag.agent import feedback as F
from hisrag.agent.loop import Agent
from hisrag.agent.playground import Playground
from hisrag.agent.render import basis_line, link_citations, tool_summary, trace_markdown
from hisrag.agent.tools import Tools
from hisrag.llm import DHClient, FakeOpenAI
from test_index import built, cfg  # noqa: F401  (fixtures: a two-newspaper index with fake vectors)

ID = "wrz_18620412_017_region_0162"


@pytest.fixture
def logs(cfg, tmp_path):  # noqa: F811
    cfg["paths"].update(agent_log=str(tmp_path / "logs" / "agent.jsonl"),
                        ratings_log=str(tmp_path / "logs" / "ratings.jsonl"))
    return cfg


def test_ratings_join_the_logged_answers(logs, monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_USER", "anna")
    F.append_jsonl(logs.path("agent_log"), {
        "time": "t0", "id": "a1", "user": "anna", "question": "Q?", "answer": "A.", "settings": {"thinking": True},
        "steps": 2, "stopped": "answer", "seconds": 3.0, "trace": [{"tool": "aggregate"}], "cited": [],
        "unknown_ids": []})
    F.append_jsonl(logs.path("agent_log"), {  # a pilot record of prompt v1: no id, no user
        "time": "t1", "question": "Q2?", "answer": "B.", "settings": {"thinking": False}, "steps": 1,
        "stopped": "answer", "seconds": 1.0, "trace": [], "cited": [], "unknown_ids": []})
    F.rate("a1", "teilweise", None, "fehlt: 1860er", cfg=logs)
    F.rate("a1", "richtig", True, " passt ", cfg=logs)  # the later rating of the same user wins
    with pytest.raises(ValueError):
        F.rate("a1", "gut", cfg=logs)
    log = F.interactions(logs)
    assert len(log) == 2 and set(log["prompt_version"]) == {"v1"}
    row = log[log["answer_id"] == "a1"].iloc[0]
    assert (row["verdict"], row["citations_fit"], row["comment"], row["rated_by"]) == ("richtig", True, "passt", "anna")
    assert row["tools"] == "aggregate"


def test_a_cut_line_does_not_hide_the_log(logs):
    path = logs.path("ratings_log")
    F.append_jsonl(path, {"answer_id": "a1"})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"answer_id": "a2", "verd')
    assert F.read_jsonl(path) == [{"answer_id": "a1"}]


def test_render():
    trace = [{"step": 1, "tool": "aggregate", "args": {"group_by": "decade"}, "result": {"n_ads": 136, "groups_total": 7}},
             {"step": 1, "tool": "search_ads", "args": {"query": "x"},
              "result": {"results": [{"ad_id": ID}, {"ad_id": "b"}], "total_matches": 9}},
             {"step": 2, "tool": "get_ad", "args": {"ad_ids": [ID]}, "result": {"ads": [{"ad_id": ID, "text": "t"}]}},
             {"step": 2, "tool": "search_ads", "args": {"query": "y", "k": 99},
              "result": {"error": "ungültige Argumente: k: …"}}]
    a = SimpleNamespace(trace=trace, steps=3, seconds=41.6, cited=[ID], stopped="answer")
    assert basis_line(a) == ("4 Tool-Aufrufe in 3 Anfragen, 42 s · gezählt: n = 136 · "
                             "2 Anzeigen gesehen, 1 vollständig gelesen, 1 zitiert")
    md = trace_markdown(trace)
    assert "- Runde 1: `aggregate`" in md and "2 Treffer von 9" in md and "**Fehler:**" in md
    assert tool_summary("expand_concept", {"term": "Koch", "positions": [{"lemma": "Koch", "n_ads": 5}],
                                           "requirements": []}) == "Koch: Lemmata Koch (5); Tags "  # v1/v2 format
    text = f"Belegt [{ID}] und erfunden [wrz_18990101_001_region_0001]."
    assert link_citations(text, {ID: "http://iiif/x"}) == (
        f"Belegt [[{ID}](http://iiif/x)] und erfunden [~~wrz_18990101_001_region_0001~~ (unbekannt)].")


def test_playground_asks_shows_and_saves_a_rating(built, tmp_path, monkeypatch):  # noqa: F811
    cfg, *_ = built
    cfg["paths"].update(agent_log=str(tmp_path / "l" / "agent.jsonl"), ratings_log=str(tmp_path / "l" / "ratings.jsonl"))
    monkeypatch.setenv("JUPYTERHUB_USER", "anna")

    def handler(payload):
        if sum(m["role"] == "assistant" for m in payload["messages"]) == 0:
            return {"tool_calls": [{"id": "c", "type": "function",
                                    "function": {"name": "aggregate", "arguments": json.dumps({"group_by": "none"})}}]}
        return "Fünf Anzeigen."

    fake = FakeOpenAI(chat_handler=handler, embedding_dim=32)
    client = DHClient(cfg, openai_client=fake)
    pg = Playground(agent=Agent(cfg, client, Tools(cfg, client=client)))
    pg._ask()
    assert pg.current is None and "Bitte" in pg.status.value  # no question yet
    pg.question.value = "Wie viele Anzeigen?"
    pg._ask()
    assert pg.current.answer == "Fünf Anzeigen." and pg.rating.layout.display == "flex"
    assert pg.cited_ads(["w3", "nope"]).keys() == {"w3"}
    pg._save()
    assert "Bitte zuerst" in pg.rating_status.value
    pg.verdict.value, pg.citations_fit.value, pg.comment.value = "falsch", False, "Zahl fehlt"
    pg._save()
    assert pg.rating_status.value.endswith("gespeichert")
    row = F.interactions(cfg).iloc[0]
    assert (row["answer_id"], row["user"], row["verdict"], row["comment"]) == (pg.current.id, "anna", "falsch",
                                                                              "Zahl fehlt")
    pg._new()
    assert pg.current is None and pg.question.value == "" and pg.rating.layout.display == "none"


def test_sitzung_without_widgets(built, tmp_path, monkeypatch, capsys):  # noqa: F811
    from hisrag.agent.playground import Sitzung

    cfg, *_ = built
    cfg["paths"].update(agent_log=str(tmp_path / "l" / "agent.jsonl"), ratings_log=str(tmp_path / "l" / "ratings.jsonl"))
    monkeypatch.setenv("JUPYTERHUB_USER", "ben")

    def handler(payload):
        if sum(m["role"] == "assistant" for m in payload["messages"]) == 0:
            return {"tool_calls": [{"id": "c", "type": "function",
                                    "function": {"name": "search_ads",
                                                 "arguments": json.dumps({"query": "Krakau", "mode": "keyword"})}}]}
        return "Eine Lehrerstelle in Krakau."

    fake = FakeOpenAI(chat_handler=handler, embedding_dim=32)
    client = DHClient(cfg, openai_client=fake)
    s = Sitzung(agent=Agent(cfg, client, Tools(cfg, client=client)))
    with pytest.raises(ValueError):
        s.bewerte("richtig")
    displayed = []
    monkeypatch.setattr("hisrag.agent.playground.display", lambda obj: displayed.append(obj))
    s.frage("Lehrer in Krakau?")
    html_parts = [d.data for d in displayed if type(d).__name__ == "HTML"]
    assert "<details><summary><b>Tool-Aufrufe (1)</b>" in html_parts[0] and "search_ads" in html_parts[0]
    s.bewerte("teilweise", belege=None, kommentar="Quelle prüfen")
    row = F.interactions(cfg).iloc[0]
    assert (row["user"], row["verdict"], row["citations_fit"], row["comment"]) == ("ben", "teilweise", None,
                                                                                   "Quelle prüfen")
