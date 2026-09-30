import json
import re

import pandas as pd
import pytest

from hisrag.config import load_config
from hisrag.llm import DHClient, FakeOpenAI
from hisrag.normalize import requirements as R


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"]["cache_db"] = str(tmp_path / "cache.sqlite")
    cfg["paths"]["usage_log"] = str(tmp_path / "usage.jsonl")
    cfg["api"]["requests_per_minute"] = None
    return cfg


ANSWERS = {
    "deutscher": [{"dimension": "unterrichtssprache", "value": "Deutsch"}],
    "ledig": [{"dimension": "familienstand", "value": "ledig"}],
    "buchhaltung und stenographie": [{"dimension": "fachkenntnisse", "value": "Buchhaltung"},
                                     {"dimension": "fachkenntnisse", "value": "Stenografie"}],
    "welches fähig ist": [],
    "gebürtige steirerin": [{"dimension": "herkunft", "value": "Steiermark"},
                            {"dimension": "religion", "value": "steirisch"}],  # outside the closed list
}


def handler(payload):
    user = payload["messages"][-1]["content"]
    found = re.findall(r'^(\d+)\. \[(\w+)\] "(.*?)" — Kontext', user, re.M)
    return json.dumps({"items": [{"i": int(i), "tags": ANSWERS[s.lower()]} for i, _, s in found]})


def spans_frame():
    rows = [("a", "language", "mit deutscher Unterrichtssprache", 4, 13), ("b", "background", "Ein lediger Mann", 4, 11),
            ("c", "background", "ledig, 30 J.", 0, 5), ("d", "job_specific", "Buchhaltung und Stenographie gesucht", 0, 28),
            ("e", "job_specific", "welches fähig ist", 0, 17), ("f", "background", "gebürtige Steirerin", 0, 19)]
    df = pd.DataFrame(rows, columns=["ad_id", "column", "text", "start", "end"])
    df["phrase"] = [t[s:e] for t, s, e in zip(df["text"], df["start"], df["end"])]
    df["newspaper"], df["year"] = "wrz", 1880
    return df


def test_vocabulary_loads_and_versions_by_content(tmp_path):
    v = R.Vocabulary()
    assert "sprachkenntnisse" in v.dimensions and v.group_of["arbeitshaltung"] == "Eigenschaften"
    assert {"böhmisch", "tschechisch"} <= v.closed_values("sprachkenntnisse")
    assert v.closed_values("fachkenntnisse") is None
    changed = tmp_path / "v.yaml"
    changed.write_text(R.VOCAB_PATH.read_text(encoding="utf-8") + "\n# edit\n", encoding="utf-8")
    assert R.Vocabulary(changed).version != v.version


def test_schema_restricts_dimensions():
    _, _, Batch = R.build_models(R.Vocabulary())
    with pytest.raises(Exception):
        Batch.model_validate({"items": [{"i": 1, "tags": [{"dimension": "hobby", "value": "Schach"}]}]})


def test_collect_phrases_groups_per_column_and_key():
    p = R.collect_phrases(spans_frame()).set_index("key")
    assert p.loc["background|ledig", "count"] == 1 and p.loc["background|lediger", "count"] == 1
    assert p.loc["language|deutscher", "context"].startswith("…mit [deutscher]")


def test_full_mapping_to_dictionary_and_ad_rows(cfg):
    spans = spans_frame()
    spans.loc[1, ["start", "end"]] = (4, 9)  # "ledig" in both background rows → one phrase, two mentions
    spans["phrase"] = [t[s:e] for t, s, e in zip(spans["text"], spans["start"], spans["end"])]
    mapper = R.RequirementMapper()
    phrases = R.collect_phrases(spans)
    results, stats = mapper.run(DHClient(cfg, openai_client=FakeOpenAI(handler)), phrases, batch_size=10, progress=False)
    assert stats["failed_batches"] == []

    d = mapper.to_dictionary(phrases, results, "qwen").set_index("key")
    assert not d.loc["job_specific|welches fähig ist", "is_informative"]
    religion = [t for t in d.loc["background|gebürtige steirerin", "tags"] if t["dimension"] == "religion"][0]
    assert religion["in_vocab"] is False and religion["group"] == "Person"

    ad = R.ad_requirements(spans, d.reset_index())
    assert sorted(ad.loc[ad["value"] == "ledig", "ad_id"]) == ["b", "c"]
    assert ad.loc[ad["ad_id"] == "d", "value"].tolist() == ["Buchhaltung", "Stenografie"]
    assert "e" not in set(ad["ad_id"])
    report = R.summarize(d.reset_index(), ad)
    assert report["mentions_by_dimension"]["fachkenntnisse"] == 2
    assert report["values_outside_closed_lists"] == {"religion: steirisch": 1}


def test_pilot_sample_takes_frequent_and_random_per_column():
    phrases = pd.DataFrame({"key": [f"c{c}|p{i}" for c in range(2) for i in range(30)],
                            "column": [f"c{c}" for c in range(2) for _ in range(30)],
                            "phrase_key": [f"p{i:02d}" for _ in range(2) for i in range(30)],
                            "count": [30 - i for _ in range(2) for i in range(30)]})
    s = R.pilot_sample(phrases, per_column=10)
    assert s.groupby("column").size().tolist() == [10, 10]
    assert {"c0|p0", "c0|p4"} <= set(s["key"])


def test_frequent_phrases_get_contexts_from_different_ads_and_years():
    rows = [(f"ad{i}", "language", f"Text {year}: der deutschen Sprache mächtig", 11, 32, year)
            for i, year in enumerate([1860, 1870, 1880, 1890, 1900])]
    rows.append(("ad0", "language", "Text 1860: der deutschen Sprache mächtig", 11, 32, 1860))  # same ad twice
    rows.append(("x", "language", "nur einmal: italienisch", 12, 23, 1900))
    spans = pd.DataFrame(rows, columns=["ad_id", "column", "text", "start", "end", "year"])
    spans["phrase"] = [t[s:e] for t, s, e in zip(spans["text"], spans["start"], spans["end"])]
    p = R.collect_phrases(spans).set_index("key")

    frequent = p.loc["language|der deutschen sprache", "context"].split(R.CONTEXT_SEPARATOR)
    assert p.loc["language|der deutschen sprache", "count"] == 5  # distinct ads
    assert [re.search(r"\d{4}", c).group() for c in frequent] == ["1860", "1880", "1900"]
    assert R.CONTEXT_SEPARATOR not in p.loc["language|italienisch", "context"]
    msgs = R.RequirementMapper().messages(p.reset_index())
    assert "Kontexte:" in msgs[1]["content"] and '"italienisch" — Kontext:' in msgs[1]["content"]


@pytest.mark.parametrize("detail, phrase, kept", [
    ("in Wort und Schrift", "Kenntnis der deutschen Sprache", False),   # from the context, not the phrase
    ("Unterstufe", "böhmischer Unter", False),                          # misread truncation
    ("Hauptfach", "Deutsch und Französisch", False),
    ("vollkommen", "vollkommene Kenntniß der italienischen Sprache", True),
    ("200 fl. C. M.", "eine Caution von 200 fl. C. M. in Hypothek", True),
    ("Korrespondenz", "deutsche Correspondenz", True),                 # modernized spelling
    ("Alter 4 bis 6 Jahre", "Kinder im Alter von 4 bis 6 Jahren", True),
    # audit of the full run: historical spellings and cue words are grounded ...
    ("tüchtig, korrekt, sehr leserlich", "eine tüchtige, correcte sehr leserliche Hand- und Zifferschrift", True),
    ("14 Jahre", "14jährigen äußern und innern Forstpraxis", True),
    ("nahe Schottenfeld", "welcher ganz nähe der Schottenfelder Realschule wohnt", True),
    ("bevorzugt", "mehrjährige Tätigkeit in einem Appreturbetrieb, möglichst als Meister", True),
    ("abgeschlossen", "Absolvirung eines klinischen psychiatrischen Curses", True),
    # ... details taken from the context still are not
    ("bevorzugt", "mit technischen Vorkenntnissen", False),
    ("Forstwirthe", "Nachweisung der mit entsprechendem Erfolge abgelegten Staatsprüfung", False),
    ("Klavier", "Privat-Unterricht ertheilen zu können", False),
    ("teilweise", "absolvirt", False),
])
def test_detail_grounding(detail, phrase, kept):
    assert R.detail_is_grounded(detail, phrase) is kept


@pytest.mark.parametrize("value, phrase, grounded", [
    ("Kaution", "welches auch Caution zu leisten im Stande", True),
    ("technische Hochschule", "die Studien an einer technischen Hochschule", True),
    ("Buchführung", "ger", False),                      # fragment tagged from the context
    ("Latein", "Deutsch und Französisch oder Deutsch", False),
])
def test_value_grounding(value, phrase, grounded):
    assert R.value_is_grounded(value, phrase) is grounded


def test_ungrounded_details_are_dropped_from_dictionary(cfg):
    def handler(payload):
        return json.dumps({"items": [{"i": 1, "tags": [
            {"dimension": "sprachkenntnisse", "value": "Deutsch", "detail": "in Wort und Schrift"}]}]})
    phrases = pd.DataFrame({"key": ["language|kenntnis der deutschen sprache"], "column": ["language"],
                            "phrase_key": ["kenntnis der deutschen sprache"], "count": [53],
                            "surface": ["Kenntnis der deutschen Sprache"], "context": ["…"]})
    mapper = R.RequirementMapper()
    results, _ = mapper.run(DHClient(cfg, openai_client=FakeOpenAI(handler)), phrases, progress=False)
    d = mapper.to_dictionary(phrases, results, "qwen")
    assert d["tags"][0][0]["detail"] is None and mapper.details_dropped == 1
    assert d["tags"][0][0]["detail_raw"] == "in Wort und Schrift"
