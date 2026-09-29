import json
import re

import pandas as pd
import pytest

from hisrag.config import load_config
from hisrag.llm import DHClient, FakeOpenAI
from hisrag.normalize import positions as P


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"]["cache_db"] = str(tmp_path / "cache.sqlite")
    cfg["paths"]["usage_log"] = str(tmp_path / "usage.jsonl")
    cfg["api"]["requests_per_minute"] = None
    return cfg


def entry(term, lemma, gender="m", category="Erziehung/Unterricht"):
    return {"term": term, "lemma": lemma, "modern": term, "gender_form": gender, "category": category}


ANSWERS = {
    "Unterlehrer": [entry("Unterlehrer", "Unterlehrer")],
    "Lehrerin": [entry("Lehrerin", "Lehrer", "f")],
    "eine": [],
    "Schulleiters- und Lehrerstellen": [entry("Schulleiter", "Schulleiter"), entry("Lehrer", "Lehrer")],
}


def fake_llm(skip=()):
    def handler(payload):
        user = payload["messages"][-1]["content"]
        items = [{"i": int(i), "entries": ANSWERS[s], "confidence": "high"}
                 for i, s in re.findall(r'^(\d+)\. "(.*?)" — Kontext', user, re.M) if s not in skip or "\n2." not in user]
        return json.dumps({"items": items})
    return FakeOpenAI(handler)


@pytest.mark.parametrize("raw, key", [
    ("Unterlehrer⸗", "unterlehrer"), ("Lehrerinſtelle.", "lehrerinstelle"),
    ("1. Brückenmeiſterpoſten", "brückenmeisterposten"), ("a) Lehrſtelle", "lehrstelle"),
])
def test_form_key(raw, key):
    assert P.form_key(raw) == key


def test_collect_forms_merges_variants_and_adds_headings():
    text = "Die Unterlehrer⸗, reſp. Lehrerinſtelle an der Volksſchule."
    spans = pd.DataFrame({"ad_id": ["a", "a", "b"], "text": [text, text, text], "start": [4, 26, 4],
                          "end": [16, 40, 16], "form": ["Unterlehrer⸗", "Lehrerinſtelle", "Unterlehrer⸗"],
                          "gender": ["male", "female", None]})
    headings = pd.DataFrame({"ad_id": ["c", "d"], "heading": ["Unterlehrerſtelle.", "Schulleiters- und Lehrerſtellen."],
                             "ad_text": ["Im Schulbezirke …", None]})
    forms = P.collect_forms(spans, headings).set_index("key")
    assert forms.loc["unterlehrer", "count"] == 2
    assert json.loads(forms.loc["unterlehrer", "extracted_gender"]) == {"male": 1, "unknown": 1}
    assert "[Unterlehrer-]" in forms.loc["unterlehrer", "context"]
    assert forms.loc["unterlehrerstelle", "n_headings"] == 1 and forms.loc["unterlehrerstelle", "n_spans"] == 0
    assert forms.loc["schulleiters- und lehrerstellen", "context"] == "Überschrift"


def forms_frame(surfaces):
    return pd.DataFrame({"key": [P.form_key(s) for s in surfaces], "surface": surfaces, "context": "…",
                         "count": 1, "n_spans": 1, "n_headings": 0, "extracted_gender": "{}"})


def test_run_retries_skipped_items_and_builds_dictionary(cfg):
    fake = fake_llm(skip={"eine"})
    client = DHClient(cfg, openai_client=fake)
    forms = forms_frame(["Unterlehrer", "eine", "Lehrerin", "Schulleiters- und Lehrerstellen"])
    results, stats = P.run(client, forms, batch_size=4, progress=False)

    assert stats["failed_batches"] == [] and len(results) == 4
    assert len(fake.chat_calls) == 2  # one batch + one retry for the skipped form
    d = P.to_dictionary(forms, results, "qwen").set_index("key")
    assert not d.loc["eine", "is_position"] and d.loc["lehrerin", "entries"][0]["lemma"] == "Lehrer"
    assert len(d.loc["schulleiters- und lehrerstellen", "entries"]) == 2
    assert d.loc["unterlehrer", "prompt_version"] == P.PROMPT_VERSION


def test_ad_positions_explodes_entries_and_skips_non_positions(cfg):
    forms = forms_frame(["Unterlehrer", "eine", "Schulleiters- und Lehrerstellen"])
    results, _ = P.run(DHClient(cfg, openai_client=fake_llm()), forms, batch_size=10, progress=False)
    dictionary = P.to_dictionary(forms, results, "qwen")
    spans = pd.DataFrame({"ad_id": ["a", "a"], "newspaper": "wrz", "year": 1880, "start": [0, 12], "end": [11, 16],
                          "form": ["Unterlehrer⸗", "eine"], "gender": ["male", "neutral"]})
    headings = pd.DataFrame({"ad_id": ["b"], "newspaper": "wrz", "year": 1880, "heading": ["Schulleiters- und Lehrerſtellen."]})
    ap = P.ad_positions(spans, headings, dictionary)
    assert sorted(zip(ap["ad_id"], ap["source"], ap["lemma"])) == [
        ("a", "span", "Unterlehrer"), ("b", "heading", "Lehrer"), ("b", "heading", "Schulleiter")]
    assert ap.loc[ap["ad_id"] == "a", "extracted_gender"].item() == "male"


def test_consistency_report_finds_conflicting_categories():
    d = pd.DataFrame({"count": [5, 2], "entries": [[entry("Lehrer", "Lehrer")],
                                                    [entry("Lehrers", "Lehrer", category="Sonstiges")]]})
    rep = P.consistency_report(d)
    assert rep["lemmas_with_conflicting_category"] == 1
    assert rep["examples"]["Lehrer"] == {"Erziehung/Unterricht": 5, "Sonstiges": 2}


def test_harmonize_categories_uses_count_weighted_majority():
    d = pd.DataFrame({"count": [280, 3, 10], "entries": [
        [entry("Köchin", "Koch", "f", "Haushalt/Dienstboten")],
        [entry("Hotelköchin", "Koch", "f", "Gastgewerbe")],
        [entry("Lehrer", "Lehrer")],
    ]})
    fixed, changed = P.harmonize_categories(d)
    assert changed == 1
    assert fixed["entries"][1][0]["category"] == "Haushalt/Dienstboten"
    assert fixed["entries"][2][0]["category"] == "Erziehung/Unterricht"


def test_harmonize_keeps_categories_of_generic_titles():
    d = pd.DataFrame({"count": [93, 54, 18], "entries": [
        [entry("Adjunct", "Adjunct", category="Justiz/Recht")],
        [entry("Adjunct", "Adjunct", category="Öffentliche Verwaltung")],
        [entry("Adjunct", "Adjunct", category="Industrie/Technik")],
    ]})
    fixed, changed = P.harmonize_categories(d)
    assert changed == 0
    assert [e[0]["category"] for e in fixed["entries"]] == ["Justiz/Recht", "Öffentliche Verwaltung", "Industrie/Technik"]


def test_unparseable_batch_is_split_until_it_works(cfg):
    def handler(payload):
        user = payload["messages"][-1]["content"]
        found = re.findall(r'^(\d+)\. "(.*?)" — Kontext', user, re.M)
        if len(found) > 2:
            return '{"items": [{"i": 1, "entries": ['   # breaks off mid-JSON
        return json.dumps({"items": [{"i": int(i), "entries": ANSWERS[s], "confidence": "high"} for i, s in found]})

    fake = FakeOpenAI(handler)
    forms = forms_frame(["Unterlehrer", "eine", "Lehrerin", "Schulleiters- und Lehrerstellen"])
    results, stats = P.run(DHClient(cfg, openai_client=fake), forms, batch_size=4, progress=False)
    assert stats["failed_batches"] == [] and len(results) == 4
    assert results["lehrerin"].entries[0].lemma == "Lehrer"
