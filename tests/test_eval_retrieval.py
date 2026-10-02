import json
import re
from pathlib import Path
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
    texts = ["Eine Wirthschafterin wird gesucht, welche gut kochen kann und in einem herrschaftlichen Hause gedient hat.",
             "Ein Commis für ein Specereiwarengeschäft wird aufgenommen, der mit der Buchhaltung vertraut ist.",
             "Lehrerstelle an der Volksschule mit 600 fl. Gehalt und freier Wohnung im Schulhause zu besetzen.",
             "Eine Wirthschafterin wird gesucht, welche gut kochen kann und in einem herrschaftlichen Hause gedient hat.",
             "Bewerber um diese Stellen haben ihre Gesuche bis Ende des Monats einzubringen."]
    return pd.DataFrame({"ad_id": [f"a{i}" for i in range(5)], "dup_cluster_id": ["c0", "c1", "c2", "c0", "c4"],
                         "decade": [1860, 1860, 1870, 1860, 1900], "label": "job_offer",
                         "countable": [True, True, True, False, True], "lang": "de",
                         "has_position": [True, True, True, True, False], "raw": texts, "enriched": texts})


def test_tokens_fold_historical_spelling_and_drop_stopwords():
    assert E.tokens("Wirthschafterin") == E.tokens("Wirtschafterin")
    assert E.tokens("Correspondenz") == E.tokens("Korrespondenz")
    assert "der" not in E.tokens("der Lehrer")


def test_bm25_ranks_matching_ad_first():
    bm = E.BM25(docs_frame()["raw"].tolist())
    assert bm.search("Handlungsgehilfe Gemischtwarengeschäft Commis")[0] == 1
    assert set(bm.search("Wirtschafterin kochen")[:2]) == {0, 3}


def test_rank_fusion():
    assert E.rrf(np.array([3, 1, 2]), np.array([1, 2, 3]))[0] == 1


def test_seeds_are_countable_german_ads_with_a_position():
    s = E.sample_seeds(docs_frame(), n=3, min_chars=50)
    assert set(s["ad_id"]) <= {"a0", "a1", "a2"}       # a3: reprint, a4: notice tail without a position
    assert set(s["decade"]) == {1860, 1870}


def test_enriched_fields_are_modern_german():
    r = SimpleNamespace(position_modern=["Köchin"], requirements=[{"value": "ledig"}, {"value": "ledig"}],
                        pay_min=200.0, pay_max=250.0, pay_currency="fl", pay_period="jahr")
    assert E._fields(r) == "Stelle: Köchin\nAnforderungen: ledig\nLohn: 200–250 fl jährlich"
    empty = SimpleNamespace(position_modern=None, requirements=pd.NA, pay_min=float("nan"), pay_max=float("nan"),
                            pay_currency=None, pay_period=None)
    assert E._fields(empty) == ""


def test_question_vectors_are_stored_and_reused(cfg):
    fake = FakeOpenAI(embedding_dim=8)
    client = DHClient(cfg, openai_client=fake)
    first = E.query_vectors(client, cfg, ["Köchinnen", "Lehrer"], "bge-m3")
    again = E.query_vectors(client, cfg, ["Lehrer", "Köchinnen"], "bge-m3")
    assert len(fake.embedding_calls) == 1 and np.allclose(first[::-1], again)
    E.query_vectors(client, cfg, ["Lehrer", "Ärzte"], "bge-m3")
    assert fake.embedding_calls[-1]["input"] == ["Ärzte"]  # only the new question is embedded


def test_scores_precision_ndcg_recall_and_seed():
    runs = pd.DataFrame({"query_id": 0, "method": "m", "variant": "raw", "rank": [1, 2, 3],
                         "ad_id": ["x", "y", "z"], "cluster": ["cx", "cy", "cz"]})
    judgments = pd.DataFrame({"query_id": 0, "ad_id": ["x", "y", "z", "w"], "cluster": ["cx", "cy", "cz", "cw"],
                              "grade": [0, 2, 1, 2]}).astype({"grade": "Int8"})
    questions = pd.DataFrame({"query_id": [0], "seed_cluster_id": ["cy"]})
    s = E.scores(runs, judgments, questions).iloc[0]
    assert s["p@10"] == pytest.approx(0.1) and s["recall"] == pytest.approx(0.5) and s["seed_found"]
    dcg = 3 / np.log2(3) + 1 / np.log2(4)
    ideal = 3 + 3 / np.log2(3) + 1 / np.log2(4)
    assert s["ndcg@10"] == pytest.approx(dcg / ideal)


def test_questions_embeddings_runs_and_judging_end_to_end(cfg, monkeypatch):
    def chat(payload):
        user = payload["messages"][-1]["content"]
        if user.startswith("Fragen:"):   # judge: the reprint pair and the Wirtschafterin ads are relevant
            ads = re.findall(r"^(\d+)\. \[F\d+\] (.*)$", user, re.M)
            return json.dumps({"items": [{"i": int(i), "grade": 2 if "Wirthschafterin" in t else 0} for i, t in ads]})
        n = len(re.findall(r"^\d+\. \(", user, re.M))
        return json.dumps({"items": [{"i": i, "question": "Stellen für Wirtschafterinnen in herrschaftlichen Haushalten",
                                      "criterion": "Die Anzeige sucht eine Wirtschafterin."} for i in range(1, n + 1)]})
    client = DHClient(cfg, openai_client=FakeOpenAI(chat, embedding_dim=16))
    docs = docs_frame()
    monkeypatch.setattr(E, "CHUNK", 2)

    questions, _ = E.write_questions(client, docs[docs["ad_id"] == "a0"], progress=False)
    assert questions.loc[0, "question"].startswith("Stellen für") and questions.loc[0, "seed_cluster_id"] == "c0"

    first = E.embed_documents(client, docs, "raw", "bge-m3", cfg, progress=False)
    again = E.embed_documents(client, docs, "raw", "bge-m3", cfg, progress=False)
    assert (first["embedded"], again["embedded"], again["skipped_existing"]) == (5, 0, 3)
    assert E.load_vectors(cfg, "raw", "bge-m3", docs).shape == (5, 16)
    assert E.load_vectors(cfg, "enriched", "bge-m3", docs) is None

    runs = E.run_methods(questions, docs, cfg, client, ["bge-m3", "embeddinggemma-300m"], progress=False)
    methods = set(zip(runs["method"], runs["variant"]))
    assert methods == {("bm25", "raw"), ("bm25", "enriched"), ("bge-m3", "raw"), ("hybrid:bge-m3", "raw")}
    for _, g in runs.groupby(["method", "variant"]):
        assert g["cluster"].is_unique                           # a0 and its reprint a3 count once

    pooled = E.pool(runs, questions, docs)
    judgments, stats = E.judge(client, pooled, progress=False)
    assert stats["failed_batches"] == [] and judgments["grade"].notna().all()
    s = E.summary(E.scores(runs, judgments, questions))
    assert s.loc[("bm25", "raw"), "seed_found"] == 1.0 and s.loc[("bm25", "raw"), "recall"] == 1.0

    # the command itself, pilot and full: files written and the report printable as JSON
    import hisrag.eval.__main__ as cli
    import hisrag.llm
    cfg["paths"]["eval_dir"] = str(Path(cfg["paths"]["cache_db"]).parent / "eval")
    Path(cfg["paths"]["eval_dir"]).mkdir()
    questions.to_parquet(Path(cfg["paths"]["eval_dir"]) / "queries.parquet")
    monkeypatch.setattr(E, "documents", lambda c: docs)
    monkeypatch.setattr(hisrag.llm, "DHClient", lambda c: client)
    for pilot in (True, False):
        report = cli.run_score(cfg, pilot, ["bge-m3"])
        json.dumps(report, ensure_ascii=False, default=str)
    out = Path(cfg["paths"]["eval_dir"])
    assert (out / "judgments_pilot.csv").exists() and (out / "scores.parquet").exists()
    assert "bm25/raw" in report["summary"]
