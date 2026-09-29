import pandas as pd
import pytest

from hisrag.data import query
from hisrag.ingest.extractions import read_extractions_csv, write_ads

SPAN_COLS = ["salary", "salary_importance", "salary_period", "unspecific_salary", "verpflegung", "position",
             "activity", "attitude towards work", "background", "interpersonal", "job specific", "language",
             "gender"]


def row(i, metadata, region, label, text, box, **spans):
    r = {"Unnamed: 0": i, "metadata": metadata, "region_id": region, "reading_order": i, "text": text.upper(),
         "min_x": box[0], "max_x": box[1], "min_y": box[2], "max_y": box[3], "post_corrected": text,
         "prediction_label": label, "iiif_link": f"https://iiif/{metadata}/{region}"}
    r.update({c: None for c in SPAN_COLS})
    r.update({k.replace("_", " ") if k in ("job_specific", "attitude_towards_work") else k: v for k, v in spans.items()})
    return r


@pytest.fixture
def csv(tmp_path):
    rows = [
        row(0, "wrz_18700408_013", "region_0001", "heading", "Köchinſtelle.", (100, 900, 1000, 1050)),
        row(1, "wrz_18700408_013", "region_0002", "job_offer", "Eine Köchin wird geſucht, 10 fl.",
            (110, 890, 1040, 1300),
            position="[(5, 11, 'Köchin')]", gender="[(5, 11, 'female')]", salary="[(26, 32, '10 fl.')]"),
        # other column: must not get the heading
        row(2, "wrz_18700408_013", "region_0003", "job_search", "Ein Diener ſucht Stelle.", (1500, 2300, 1040, 1200),
            position="[(4, 10, 'Diener')]", gender="[(4, 10, 'male')]"),
        # same column but far below: must not get the heading
        row(3, "wrz_18700408_013", "region_0004", "job_offer", "Gärtner geſucht.", (100, 900, 2000, 2100)),
        # copied page file: exact duplicate of region_0002, to be dropped
        row(4, "wrz_18700408_013 (Kopie)", "region_0002", "job_offer", "Eine Köchin wird geſucht, 10 fl.",
            (110, 890, 1040, 1300)),
        # misaligned gender: positions kept, gender dropped
        row(5, "wrz_19120101_002", "region_0001", "job_offer", "Lehrer und Lehrerin",
            (0, 10, 0, 10), position="[(0, 6, 'Lehrer'), (11, 19, 'Lehrerin')]", gender="[(0, 6, 'male')]",
            job_specific="[(0, 6, 'Lehrer')]"),
    ]
    rows[4]["iiif_link"] = rows[1]["iiif_link"]
    path = tmp_path / "x.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_read_parses_ids_spans_and_dates(csv):
    df, report = read_extractions_csv(csv)
    ad = df.set_index("ad_id").loc["wrz_18700408_013_region_0002"]

    assert report["rows_in"] == 6 and report["rows_out"] == 5
    assert report["duplicates_dropped"] == ["wrz_18700408_013 (Kopie)"]
    assert (ad["newspaper"], str(ad["date"]), ad["page"], ad["year"], ad["decade"]) == ("wrz", "1870-04-08", 13, 1870, 1870)
    assert ad["positions"] == [{"start": 5, "end": 11, "text": "Köchin", "gender": "female"}]
    assert ad["salary"] == [{"start": 26, "end": 32, "text": "10 fl."}]
    assert ad["text"][26:32] == "10 fl." and ad["text_ocr"].startswith("EINE")


def test_misaligned_gender_keeps_positions(csv):
    df, report = read_extractions_csv(csv)
    ps = df.set_index("ad_id").loc["wrz_19120101_002_region_0001", "positions"]
    assert [p["text"] for p in ps] == ["Lehrer", "Lehrerin"]
    assert all(p["gender"] is None for p in ps) and report["positions_without_gender"] == 2


def test_heading_links_only_to_ad_directly_below_in_same_column(csv):
    df, report = read_extractions_csv(csv)
    linked = df.set_index("ad_id")["heading_text"].dropna()
    assert linked.to_dict() == {"wrz_18700408_013_region_0002": "Köchinſtelle."}
    assert report["headings_linked"] == 1


def test_write_and_query_round_trip(csv, tmp_path):
    from hisrag.config import load_config

    df, _ = read_extractions_csv(csv)
    out = tmp_path / "ads"
    write_ads(df, out)
    write_ads(df, out)  # re-running replaces partitions instead of duplicating rows

    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"]["ads_dir"] = str(out)
    got = query("SELECT newspaper, year, count(*) n FROM ads GROUP BY ALL ORDER BY year", cfg=cfg)
    assert got.to_dict("records") == [{"newspaper": "wrz", "year": 1870, "n": 4}, {"newspaper": "wrz", "year": 1912, "n": 1}]
    pos = query("SELECT positions[1].gender g FROM ads WHERE ad_id = 'wrz_18700408_013_region_0002'", cfg=cfg)
    assert pos["g"].iloc[0] == "female"
