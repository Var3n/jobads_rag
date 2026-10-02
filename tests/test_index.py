import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hisrag.config import load_config
from hisrag.data import derived_dir, write_partitioned
from hisrag.eval import retrieval as E
from hisrag.index import AdIndex, Filters
from hisrag.index import build as B
from hisrag.index.search import parse_keywords
from hisrag.llm import DHClient, FakeOpenAI
from hisrag.normalize import clean as C
from hisrag.normalize import positions as P
from hisrag.normalize import requirements as R
from hisrag.normalize import salary as S

ADS = pd.DataFrame([
    # ad_id, newspaper, year, label, cluster, text
    ("w1", "wrz", 1862, "job_offer", "w1", "Eine Wirthschafterin wird gesucht, welche gut kochen kann."),
    ("w2", "wrz", 1863, "job_offer", "w1", "Eine Wirthschafterin wird gesucht, welche gut kochen kann."),
    ("w3", "wrz", 1875, "job_offer", "w3", "Lehrerstelle in Krakau an der k. k. Volksschule, 600 fl. Gehalt, freie Wohnung."),
    ("w4", "wrz", 1880, "job_search", "w4", "Ein Commis aus Krakauer Hause sucht Stelle als Correspondent."),
    ("w5", "wrz", 1890, "job_offer", "w5", "Lehrerin für Clavier und französische Sprache gesucht."),
    ("w6", "wrz", 1880, "job_offer", "w6", "Sterbefall: Johann Huber, Wirthschaftsbesitzer."),  # death register
    ("n1", "nfp", 1900, "job_offer", "n1", "Wirtschafterin für ein Landgut in Mähren gesucht, Kost und Wohnung."),
], columns=["ad_id", "newspaper", "year", "label", "cluster", "text"])
SEARCHABLE = 6


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"].update(ads_dir=str(tmp_path / "ads"), derived_dir=str(tmp_path / "derived"),
                        embeddings_dir=str(tmp_path / "emb"), index_dir=str(tmp_path / "index"),
                        cache_db=str(tmp_path / "cache.sqlite"), usage_log=str(tmp_path / "usage.jsonl"))
    cfg["api"]["requests_per_minute"] = None
    cfg["index"]["dims"] = 16
    write_clean_table(cfg)
    return cfg


def keys(ids):
    sub = ADS.set_index("ad_id").loc[ids].reset_index()
    return sub[["ad_id", "newspaper", "year"]]


def write_clean_table(cfg):
    a = ADS.assign(date=[dt.date(y, 3, 1) for y in ADS["year"]], decade=ADS["year"] // 10 * 10, page=1,
                   iiif_link="http://iiif/x", heading_text=None)
    write_partitioned(a, cfg.path("ads_dir"))
    death = ADS["ad_id"] == "w6"
    write_partitioned(keys(ADS["ad_id"]).assign(text_norm=ADS["text"], lang="de", n_flags=death.astype(int),
                                                flag_pc_repetition=False, flag_pc_expanded=False,
                                                flag_too_short=False, flag_pc_unsupported=False,
                                                flag_death_register=death),
                      derived_dir("ad_text", cfg))
    size = ADS.groupby("cluster")["ad_id"].transform("size")
    write_partitioned(keys(ADS["ad_id"]).assign(dup_cluster_id=ADS["cluster"], dup_cluster_size=size,
                                                is_canonical=ADS["ad_id"] != "w2",
                                                run_first_date=dt.date(1862, 3, 1), run_last_date=dt.date(1863, 3, 1)),
                      derived_dir("ad_dups", cfg))
    pos = keys(["w1", "w3", "w5"]).assign(source="span", span_start=0, span_end=6, surface="x", key="k",
                                          extracted_gender=None, term=["Wirthschafterin", "Lehrer", "Lehrerin"],
                                          lemma=["Wirtschafter", "Lehrer", "Lehrer"],
                                          modern=["Wirtschafterin", "Lehrer", "Lehrerin"],
                                          gender_form=["f", "m", "f"],
                                          category=["Hauswirtschaft", "Unterricht", "Unterricht"], confidence="high")
    write_partitioned(pos, derived_dir("ad_positions", cfg), P.AD_POSITIONS_SCHEMA)
    req = keys(["w5", "w5"]).assign(column="language", span_start=0, span_end=5, phrase="französische Sprache",
                                    dimension="sprachkenntnisse", group="Qualifikation",
                                    value=["Französisch", None], detail=None, in_vocab=True)
    write_partitioned(req, derived_dir("ad_requirements", cfg), R.AD_REQUIREMENTS_SCHEMA)
    sal = keys(["w3"]).assign(span_start=40, span_end=47, phrase="600 fl.", amount_min=600.0, amount_max=600.0,
                              currency="fl", standard="öW", standard_source="date", component="gehalt",
                              period="jahr", period_source="assumed", parsed_by="rules")
    write_partitioned(sal, derived_dir("ad_salary", cfg), S.AD_SALARY_SCHEMA)
    pay = S.ad_pay(sal, pd.DataFrame({"ad_id": ["n1"], "newspaper": "nfp", "year": [1900], "phrase": ["Kost"]}))
    write_partitioned(pay, derived_dir("ad_pay", cfg), S.AD_PAY_SCHEMA)
    C.write(C.build(cfg), derived_dir("ad_clean", cfg))


def store_step8_vectors(cfg, client, n):
    """The step-8 store: full-length vectors of the first n searchable ads, in one chunk."""
    docs = E.documents(cfg).head(n)
    vectors = client.embed(docs["enriched"].tolist(), cfg["index"]["model"], kind="passage")
    path = B._vector_dir(cfg) / "chunk-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"ad_id": docs["ad_id"].tolist(), "vector": pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.ravel()), vectors.shape[1])}), path)
    return docs, vectors


@pytest.fixture
def built(cfg):
    fake = FakeOpenAI(embedding_dim=32)
    client = DHClient(cfg, openai_client=fake)
    docs, vectors = store_step8_vectors(cfg, client, 4)
    report = B.build(cfg, client=client, progress=False)
    return cfg, client, fake, docs, vectors, report


def test_build_reuses_stored_vectors_and_embeds_the_rest(built):
    cfg, client, fake, docs, vectors, report = built
    assert report["total_rows"] == SEARCHABLE  # the death-register entry is left out
    assert list(docs["ad_id"]) == ["n1", "w1", "w2", "w3"]  # stored; w4 and w5 are not
    assert report["wrz"]["ads"] == 5 and report["wrz"]["embedding"]["embedded"] == 2
    assert "embedding" not in report["nfp"]
    assert len(fake.embedding_calls) == 2  # the stored chunk, then the two missing ads
    row = B.connect(cfg).open_table(B.TABLE).search().where("ad_id = 'w1'").to_pandas().iloc[0]
    assert len(row["vector"]) == 16
    assert np.allclose(row["vector"], E.truncate(vectors[1:2], 16)[0], atol=1e-6)
    assert list(row["position_categories"]) == ["Hauswirtschaft"]


def test_rebuilding_one_newspaper_replaces_only_its_rows(built):
    cfg, client, fake, *_ = built
    calls = len(fake.embedding_calls)
    report = B.build(cfg, ["nfp"], client, progress=False)
    assert report["total_rows"] == SEARCHABLE
    assert len(fake.embedding_calls) == calls  # the new vectors were stored and are found again


def test_build_without_vectors_needs_a_client(cfg):
    with pytest.raises(RuntimeError, match="no stored"):
        B.build(cfg, ["wrz"], progress=False)


def test_semantic_search_finds_the_ad_of_its_own_vector_once_per_cluster(built):
    cfg, client, fake, docs, vectors, _ = built
    index = AdIndex(cfg, client)
    hits = index.semantic(vector=E.truncate(vectors[1:2], 16)[0], k=10)
    assert hits["ad_id"].iloc[0] in {"w1", "w2"} and hits["score"].iloc[0] == pytest.approx(1, abs=1e-5)
    assert hits["dup_cluster_id"].is_unique and len(hits) == SEARCHABLE - 1  # w1/w2 are one ad
    assert index.semantic("Wirtschafterin", k=3)["ad_id"].notna().all()  # embeds the question


def test_filters(built):
    cfg, client, *_ = built
    index = AdIndex(cfg, client)
    found = lambda f: set(index.semantic("x", f, k=10)["ad_id"])
    assert found(Filters(newspapers=["nfp"])) == {"n1"}
    assert found(Filters(year_from=1870, year_to=1885)) == {"w3", "w4"}
    assert found(Filters(labels=["job_search"])) == {"w4"}
    assert found(Filters(position_categories=["Unterricht"], position_gender=["f"])) == {"w5"}
    assert found(Filters(requirement_tags=["sprachkenntnisse:Französisch"])) == {"w5"}
    assert found(Filters(has_pay=True)) == {"w3"}
    assert found(Filters(benefits=["kost"])) == {"n1"}
    assert found(Filters(countable_only=True, where="year < 1870")) == {"w1"}
    with pytest.raises(ValueError, match="unknown benefit"):
        Filters(benefits=["auto"]).sql()
    assert "'O''Brien'" in Filters(position_lemmas=["O'Brien"]).sql()


def test_parse_keywords():
    parts = parse_keywords('"k. k. Statthalterei" Krakau* -Wirthschafterin')
    assert [(p.occur, p.kind, p.words) for p in parts] == [
        ("MUST", "phrase", ["k", "k", "stattalterei"]), ("MUST", "prefix", ["krakau"]),
        ("MUST_NOT", "word", ["wirtskhafterin"])]


def test_keyword_search(built):
    cfg, client, *_ = built
    index = AdIndex(cfg, client)
    found = lambda q, f=None: set(index.keyword(q, f, k=None)["ad_id"])
    assert found("Wirtschafterin") in ({"w1", "n1"}, {"w2", "n1"})  # historical spelling, one per cluster
    assert found("Wirtschafterin", Filters(newspapers=["nfp"])) == {"n1"}
    assert found("Krakau") == {"w3"}                  # exact word form
    hits = index.keyword("krakau*", k=None)
    assert set(hits["ad_id"]) == {"w3", "w4"} and hits.attrs["expanded"] == {"krakau*": ["krakau", "krakauer"]}
    assert found("Korrespondent") == {"w4"} and found("Klavier") == {"w5"}
    assert found('"k. k. Volksschule"') == {"w3"} and found('"Volksschule k. k."') == set()
    assert found("gesucht -Clavier") == {"w1", "n1"} or found("gesucht -Clavier") == {"w2", "n1"}
    assert found("Lehrer* Krakau") == {"w3"}          # all words must occur
    assert found("Zahnarzt*") == set()
    assert found("Sterbefall") == set()               # death register is not searchable
    with pytest.raises(ValueError, match="must occur"):
        index.keyword("-Krakau")


def test_ann_index_from_a_row_count_on(cfg):
    cfg["index"]["ann_min_rows"] = 1
    client = DHClient(cfg, openai_client=FakeOpenAI(embedding_dim=32))
    report = B.build(cfg, client=client, progress=False)
    assert "vector" in report["indexes"]
    assert len(AdIndex(cfg, client).semantic("Lehrer", k=3)) == 3


def test_check_reproduces_the_evaluation_runs(cfg, monkeypatch):
    from hisrag.index.__main__ import run_check

    fake = FakeOpenAI(embedding_dim=32)
    monkeypatch.setattr("hisrag.llm.DHClient", lambda c: DHClient(c, openai_client=fake))
    client = DHClient(cfg, openai_client=fake)
    docs, _ = store_step8_vectors(cfg, client, SEARCHABLE)
    B.build(cfg, client=client, progress=False)
    cfg["paths"]["eval_dir"] = str(cfg.path("index_dir").parent / "eval")
    cfg.path("eval_dir").mkdir()
    questions = pd.DataFrame({"query_id": [0, 1], "question": ["Köchin", "Lehrer in Galizien"]})
    questions.to_parquet(cfg.path("eval_dir") / "queries.parquet")
    runs = E.run_methods(questions, docs, cfg, client, ["qwen3-embedding-8b@16"], variants=("enriched",),
                         bm25=False, hybrids=False, progress=False)
    runs.to_parquet(cfg.path("eval_dir") / "runs.parquet")
    report = run_check(cfg)
    assert report["mean_overlap_with_eval_top10"] == 1.0 and report["questions_identical"] == 2
