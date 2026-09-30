import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hisrag.config import load_config
from hisrag.eval import retrieval as E
from hisrag.llm import DHClient, FakeOpenAI


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"].update(cache_db=str(tmp_path / "cache.sqlite"), usage_log=str(tmp_path / "usage.jsonl"),
                        embeddings_dir=str(tmp_path / "emb"))
    cfg["api"]["requests_per_minute"] = None
    return cfg


def docs_frame():
    texts = ["Eine Wirthschafterin wird gesucht, welche gut kochen kann.",
             "Ein Commis für ein Specereiwarengeschäft wird aufgenommen.",
             "Lehrerstelle an der Volksschule mit 600 fl. Gehalt.",
             "Eine Wirthschafterin wird gesucht, welche gut kochen kann.",   # reprint of the first
             "Ein Diener mit guten Zeugnissen sucht eine Stelle."]
    return pd.DataFrame({"ad_id": [f"a{i}" for i in range(5)], "dup_cluster_id": ["c0", "c1", "c2", "c0", "c4"],
                         "decade": [1860, 1860, 1870, 1860, 1900], "label": "job_offer",
                         "countable": [True, True, True, False, True], "raw": texts, "enriched": texts})


def test_tokens_fold_historical_spelling_and_drop_stopwords():
    assert E.tokens("Wirthschafterin") == E.tokens("Wirtschafterin")
    assert E.tokens("Correspondenz") == E.tokens("Korrespondenz")
    assert "der" not in E.tokens("der Lehrer")


def test_bm25_ranks_matching_ad_first():
    bm = E.BM25(docs_frame()["raw"].tolist())
    assert bm.search("Handlungsgehilfe Gemischtwarengeschäft Commis")[0] == 1
    assert set(bm.search("Wirtschafterin kochen")[:2]) == {0, 3}


def test_rank_fusion_and_hit_rank_with_reprints():
    fused = E.rrf(np.array([3, 1, 2]), np.array([1, 2, 3]))
    assert fused[0] == 1
    clusters = docs_frame()["dup_cluster_id"].to_numpy()
    assert E.hit_rank(np.array([2, 3, 1]), clusters, "c0") == 2   # a3 is a printing of a0
    assert E.hit_rank(np.array([2, 1]), clusters, "c4") is None
    m = E.metrics(pd.Series([1, 3, None], dtype="Int32"))
    assert m["recall@1"] == pytest.approx(1 / 3, abs=0.001) and m["recall@5"] == pytest.approx(2 / 3, abs=0.001)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3, abs=0.001)


def test_sample_targets_takes_countable_ads_from_every_decade():
    t = E.sample_targets(docs_frame(), n=3, min_chars=10)
    assert t["countable"].all() and set(t["decade"]) == {1860, 1870, 1900}


def test_enriched_fields_are_modern_german():
    r = SimpleNamespace(position_modern=["Köchin"], requirements=[{"value": "ledig"}, {"value": "ledig"}],
                        pay_min=200.0, pay_max=250.0, pay_currency="fl", pay_period="jahr")
    assert E._fields(r) == "Stelle: Köchin\nAnforderungen: ledig\nLohn: 200–250 fl jährlich"
    empty = SimpleNamespace(position_modern=None, requirements=None, pay_min=float("nan"), pay_max=float("nan"),
                            pay_currency=None, pay_period=None)
    assert E._fields(empty) == ""


def test_queries_embeddings_and_scoring_end_to_end(cfg, monkeypatch):
    def chat(payload):
        n = len(re.findall(r"^\d+\. \(", payload["messages"][-1]["content"], re.M))
        return json.dumps({"items": [{"i": i, "query": "Wirtschafterin die kochen kann"} for i in range(1, n + 1)]})
    fake = FakeOpenAI(chat, embedding_dim=16)
    client = DHClient(cfg, openai_client=fake)
    docs = docs_frame()
    monkeypatch.setattr(E, "CHUNK", 2)

    queries, stats = E.write_queries(client, docs[docs["ad_id"] == "a0"], progress=False)
    assert queries["query"].tolist() == ["Wirtschafterin die kochen kann"] and queries["leakage"][0] == 0.5

    first = E.embed_documents(client, docs, "raw", "bge-m3", cfg, progress=False)
    again = E.embed_documents(client, docs, "raw", "bge-m3", cfg, progress=False)
    assert (first["embedded"], again["embedded"], again["skipped_existing"]) == (5, 0, 3)
    vectors = E.load_vectors(cfg, "raw", "bge-m3", docs)
    assert vectors.shape == (5, 16) and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert E.load_vectors(cfg, "enriched", "bge-m3", docs) is None

    results = E.evaluate(queries, docs, cfg, client, ["bge-m3", "embeddinggemma-300m"], progress=False)
    methods = set(zip(results["method"], results["variant"]))
    assert {("bm25", "raw"), ("bm25", "enriched"), ("bge-m3", "raw"), ("hybrid:bge-m3", "raw")} == methods
    assert results.set_index(["method", "variant"]).loc[("bm25", "raw"), "rank"] == 1
    table = E.summary(results)
    assert table.loc[("bm25", "raw"), "recall@1"] == 1.0
