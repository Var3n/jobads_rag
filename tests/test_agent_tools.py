import json

import pytest

from hisrag.agent.tools import Tools
from test_index import built, cfg  # noqa: F401  (fixtures: a two-newspaper index with fake vectors)


@pytest.fixture
def tools(built):  # noqa: F811
    cfg, client, *_ = built
    return Tools(cfg, client=client)


def test_specs_are_plain_json_schema(tools):
    specs = tools.specs()
    assert [s["function"]["name"] for s in specs] == ["search_ads", "get_ad", "aggregate", "expand_concept"]
    text = json.dumps(specs, ensure_ascii=False)
    assert "anyOf" not in text and "$ref" not in text and '"title"' not in text
    search = specs[0]["function"]["parameters"]
    assert search["required"] == ["query"]
    assert search["properties"]["filters"]["properties"]["year_from"]["type"] == "integer"
    assert "Haushalt/Dienstboten" in search["properties"]["filters"]["properties"]["position_categories"]["items"]["enum"]


def test_call_reports_errors_to_the_model(tools):
    assert "unbekanntes Tool" in tools.call("delete_all", "{}")["error"]
    assert "ungültige Argumente" in tools.call("search_ads", '{"query": "x", "k": 500}')["error"]
    assert "ungültige Argumente" in tools.call("search_ads", "{not json")["error"]
    assert "must occur" in tools.call("search_ads", {"query": "-Krakau", "mode": "keyword"})["error"]


def test_search_ads(tools):
    r = tools.call("search_ads", json.dumps({"query": "Wirtschafterin", "mode": "keyword", "k": 1}))
    assert r["total_matches"] == 2 and r["total_countable"] == 2 and len(r["results"]) == 1
    hit = r["results"][0]
    assert set(hit) >= {"ad_id", "date", "label", "positions", "text", "printings"}
    r = tools.call("search_ads", {"query": "krakau*", "mode": "keyword"})
    assert r["expanded"] == {"krakau*": ["krakau", "krakauer"]}
    r = tools.call("search_ads", {"query": "Lehrer", "filters": {"year_from": 1870, "labels": ["job_offer"]}})
    assert {h["ad_id"] for h in r["results"]} == {"w3", "w5", "n1"} and "total_matches" not in r


def test_get_ad(tools):
    r = tools.call("get_ad", {"ad_ids": ["w3", "w2", "nope"]})["ads"]
    assert r[0]["pay"] == {"min": 600.0, "max": 600.0, "currency": "fl", "standard": "öW", "period": "jahr"}
    assert r[0]["positions_modern"] == ["Lehrer"] and r[0]["amounts"][0]["component"] == "gehalt"
    assert r[1]["printings"]["count"] == 2 and r[1]["countable"] is False
    assert r[2] == {"ad_id": "nope", "error": "nicht gefunden"}


def test_aggregate_count_and_groups(tools):
    r = tools.call("aggregate", {"measure": "count", "group_by": "decade"})
    assert r["n_ads"] == 5  # w2 is a reprint, w6 a death-register entry
    assert [(x["group"], x["n"]) for x in r["rows"]] == [(1860, 1), (1870, 1), (1880, 1), (1890, 1), (1900, 1)]
    r = tools.call("aggregate", {"group_by": "position_lemma"})
    assert r["rows"][0] == {"group": "Lehrer", "n": 2} and "note_groups" in r
    r = tools.call("aggregate", {"group_by": "requirement_value", "dimension": "sprachkenntnisse"})
    assert r["rows"] == [{"group": "Französisch", "n": 1}]
    assert tools.call("aggregate", {"group_by": "benefit"})["rows"] == [{"group": "kost", "n": 1}]


def test_aggregate_share_and_keyword(tools):
    r = tools.call("aggregate", {"measure": "share", "group_by": "none", "filters": {"labels": ["job_offer"]},
                                 "subset": {"position_lemmas": ["Lehrer"]}})
    assert r["rows"] == [{"group": "alle", "n_base": 4, "n": 2, "share_pct": 50.0}] and r["n_subset"] == 2
    r = tools.call("aggregate", {"group_by": "newspaper", "filters": {"keyword": "Wirthschafterin"}})
    assert {x["group"]: x["n"] for x in r["rows"]} == {"wrz": 1, "nfp": 1}  # the countable printing of w1/w2
    assert "share" in tools.call("aggregate", {"measure": "share"})["error"]


def test_aggregate_pay_keeps_currencies_apart(tools):
    r = tools.call("aggregate", {"measure": "pay", "group_by": "none"})
    assert r["n_ads_with_pay"] == 1
    assert r["rows"] == [{"group": "alle", "currency": "fl", "standard": "öW", "period": "jahr", "n": 1,
                          "p25": 600.0, "median": 600.0, "p75": 600.0, "min": 600.0, "max": 600.0}]


def test_expand_concept(tools):
    def expand(term):
        return tools.call("expand_concept", {"terms": [term]})["results"][0]

    both = tools.call("expand_concept", {"terms": ["Lehrer", "französisch", ""]})["results"]
    assert [r["term"] for r in both] == ["Lehrer", "französisch", ""] and both[2]["error"] == "leerer Begriff"
    r = expand("Lehrerin")
    assert r["positions"][0]["lemma"] == "Lehrer" and r["positions"][0]["n_ads"] == 2  # the lemma: Lehrer and Lehrerin
    r = expand("Lehrer")
    assert r["positions"][0]["n_ads"] == 2 and r["positions"][0]["modern"][:2] in (["Lehrer", "Lehrerin"],
                                                                                    ["Lehrerin", "Lehrer"])
    r = expand("französisch")
    assert r["requirements"] == [{"tag": "sprachkenntnisse:Französisch", "n_ads": 1}]
    r = expand("Haushalt")
    assert r["categories"] == ["Haushalt/Dienstboten"]
    assert "Kein Eintrag" in expand("Astronaut")["hint"]
    assert "ungültige Argumente" in tools.call("expand_concept", {"terms": []})["error"]


def test_search_ads_returns_the_full_text(tools, monkeypatch):
    import pandas as pd
    long = pd.DataFrame([{"ad_id": "x", "date": None, "label": "job_offer", "position_modern": None,
                          "text": "Köchin gesucht. " * 200, "dup_cluster_size": 1, "quality_warning": None}])
    monkeypatch.setattr(tools.index, "semantic", lambda *a, **k: long)
    assert tools.call("search_ads", {"query": "Köchin"})["results"][0]["text"] == "Köchin gesucht. " * 200


def test_keyword_without_hits_says_how_to_loosen(tools):
    r = tools.call("search_ads", {"query": "Haus Rothschild", "mode": "keyword"})
    assert r["total_matches"] == 0 and "weniger" in r["hint"]
    assert "hint" not in tools.call("search_ads", {"query": "Krakau OR Rothschild", "mode": "keyword"})


def test_aggregate_all_requirement_tags(tools):
    r = tools.call("aggregate", {"group_by": "requirement_value"})
    assert r["rows"] == [{"group": "sprachkenntnisse:Französisch", "n": 1}]
