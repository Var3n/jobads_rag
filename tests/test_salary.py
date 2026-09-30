import datetime as dt
import json
import re

import pandas as pd
import pytest

from hisrag.config import load_config
from hisrag.llm import DHClient, FakeOpenAI
from hisrag.normalize import salary as S

D1855, D1870 = dt.date(1855, 5, 1), dt.date(1870, 5, 1)


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"]["cache_db"] = str(tmp_path / "cache.sqlite")
    cfg["paths"]["usage_log"] = str(tmp_path / "usage.jsonl")
    cfg["api"]["requests_per_minute"] = None
    return cfg


@pytest.mark.parametrize("text, date, expected", [
    ("600 fl. (Amtsblatt Nr. 38.)", D1870, (600, 600, "fl", "öW", "date")),
    ("735 fl. 5. W., Wohnung und Holz", D1870, (735, 735, "fl", "öW", "stated")),   # OCR of "ö. W."
    ("500 fl. C. M. jährlich", D1870, (500, 500, "fl", "CM", "stated")),
    ("500 fl. jährlich", D1855, (500, 500, "fl", "CM", "date")),
    ("von 226 fl. 80 kr. öst. W.", D1870, (226.8, 226.8, "fl", "öW", "stated")),
    ("30 kr. C. M. und voller Verpflegung", D1855, (0.5, 0.5, "fl", "CM", "stated")),  # 60 kr. = 1 fl. CM
    ("78½ kr. (Amtsblatt", D1870, (0.785, 0.785, "fl", "öW", "date")),
    ("1400—1600 K und 180 K Quartiergeld", D1870, (1400, 1600, "K", None, None)),
    ("K 400 Quartiergeld", D1870, (400, 400, "K", None, None)),
    ("von fl. 1200.— , mit dem Anspruche", D1870, (1200, 1200, "fl", "öW", "date")),
    ("1.200 fl. jährlich", D1870, (1200, 1200, "fl", "öW", "date")),
    ("2 K 20 h ist bei dem", D1870, (2.2, 2.2, "K", None, None)),
    ("50, resp. 40 fl., oder falls", D1870, (50, 50, "fl", "öW", "date")),              # list sharing its currency
])
def test_parse_amount(text, date, expected):
    a = S.parse_amount(text, date)
    assert (a.amount_min, a.amount_max, a.currency, a.standard, a.standard_source) == pytest.approx(expected)


def test_currency_written_before_the_span():
    a = S.parse_amount("160— bis 180— Küchenmädchen", None, before="Küchenmädchen, Lohn Fr.")
    assert (a.amount_min, a.amount_max, a.currency) == (160, 180, "Fr")


@pytest.mark.parametrize("text, not_money", [
    ("840 Gramm , der vorschriftsmäßigen", True), ("25 pCt. Activitätszulage", True), ("20 Stunden wöchentlich", True),
    ("600 fl. Gehalt", False),
])
def test_numbers_that_are_no_money(text, not_money):
    assert S.is_not_money(text) is not_money


@pytest.mark.parametrize("before, after, span, expected", [
    ("Jahresgehalt", " ö. W., Wohnung und Holz.", "735 fl.", ("gehalt", "jahr", "stated")),
    ("diese mit 300 fl. Gehalt,", " Quartiergeld und Amtskleidung", "42 fl.", ("quartiergeld", "jahr", "assumed")),
    ("jede mit 600 fl. Gehalt,", " Functionszulage, Naturalquartier", "50 fl.", ("zulage", "jahr", "assumed")),
    ("Finanzdirection zu Linz, Gehalt", " , Quartiergeld. (Amtsblatt", "1050 fl.", ("gehalt", "jahr", "assumed")),
    ("Gehalt", " Activitätszulage und Amtsleidung", "300 fl,", ("gehalt", "jahr", "assumed")),
    ("eine Quartiergeldentschädigung von jährlich", " , so wie", "450 fl.", ("quartiergeld", "jahr", "stated")),
    ("gegen Erlag einer Dienstcaution von", " zu besetzen.", "400 fl.", ("kaution", None, None)),
    ("mit dem Taggelde von", " vom Präsidium", "90 kr.", ("taggeld", "tag", "stated")),
    ("in der XX. Diätenclasse mit dem Gehalte von", " und", "735 fl.", ("gehalt", "jahr", "assumed")),
    ("Assistenten mit jährlichen 500 fl., 450 fl.,", " und 300 fl. Gehalt erledigt", "350 fl.", ("gehalt", "jahr", "stated")),
    ("Jahresgehalt der II. Kategorie", " , Funktionszulage jährlicher 200 K", "1800 K", ("gehalt", "jahr", "stated")),
    ("mit einem Honorar monatlicher", " , die Naturalverpflegung", "100 K", ("remuneration", "monat", "stated")),
])
def test_classify(before, after, span, expected):
    assert S.classify(before, after, span) == expected


def spans_frame():
    rows = [  # (ad_id, date, text, span text)
        ("a", D1870, "Kanzlistenstelle, Gehalt 600 fl. ö. W., Activitätszulage 150 fl. und Naturalwohnung.", ["600 fl.", "150 fl."]),
        ("b", D1855, "Amtsdiener mit 300 fl., eventuell 250 fl. Gehalt, gegen Caution von 400 fl.", ["300 fl.", "250 fl.", "400 fl."]),
        ("c", D1870, "Taggeld zweihundert Kreuzer.", ["zweihundert Kreuzer"]),
        ("d", D1870, "Brotportion von 840 Gramm täglich.", ["840 Gramm"]),
    ]
    out = []
    for ad_id, date, text, parts in rows:
        pos = 0
        for p in parts:
            start = text.index(p, pos)
            out.append({"ad_id": ad_id, "newspaper": "wrz", "year": date.year, "date": date, "text": text,
                        "start": start, "end": start + len(p), "phrase": p})
            pos = start + len(p)
    return pd.DataFrame(out)


def test_spans_to_rows_llm_leftovers_and_pay(cfg):
    spans = spans_frame()
    rows = S.parse_spans(spans)
    assert rows["parsed_by"].fillna("-").tolist() == ["rules"] * 5 + ["-", "not_money"]
    assert rows["component"].tolist()[:5] == ["gehalt", "zulage", "gehalt", "gehalt", "kaution"]

    left = S.leftovers(rows)
    assert len(left) == 1 and "[zweihundert Kreuzer]" in left["key"].iloc[0]

    def handler(payload):
        found = re.findall(r"^(\d+)\. ", payload["messages"][-1]["content"], re.M)
        return json.dumps({"items": [{"i": int(i), "is_amount": True, "amount_min": 2.0, "amount_max": 2.0,
                                      "currency": "fl"} for i in found]})
    results, stats = S.run_llm(DHClient(cfg, openai_client=FakeOpenAI(handler)), left, progress=False)
    assert stats["failed_batches"] == []
    rows = S.assign_standard_by_date(S.apply_llm(rows, results), spans["date"])
    llm = rows[rows["parsed_by"] == "llm"].iloc[0]
    assert (llm["amount_min"], llm["currency"], llm["standard"], llm["period"]) == (2.0, "fl", "öW", "tag")

    benefits = pd.DataFrame({"ad_id": ["a", "e"], "newspaper": "wrz", "year": [1870, 1870],
                             "phrase": ["Naturalwohnung", "Kost und Wohnung"]})
    pay = S.ad_pay(rows, benefits).set_index("ad_id")
    assert (pay.loc["a", "pay_min"], pay.loc["a", "pay_max"], pay.loc["a", "pay_standard"]) == (600, 600, "öW")
    assert pay.loc["a", "benefit_wohnung"] and pay.loc["a", "benefit_zulagen"]
    assert (pay.loc["b", "pay_min"], pay.loc["b", "pay_max"], pay.loc["b", "pay_amounts"]) == (250, 300, 2)  # Kaution left out
    assert pd.isna(pay.loc["e", "pay_min"]) and pay.loc["e", "benefit_kost"] and pay.loc["e", "benefit_wohnung"]
    report = S.summarize(rows, pay)
    assert report["parsed_by_llm"] == 1 and report["ads_with_pay"] == 3
