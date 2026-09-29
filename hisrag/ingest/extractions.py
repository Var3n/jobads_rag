"""Step 1: extraction CSV (one per newspaper) → one clean row per region, written as Parquet.

The CSV schema is the same for all newspapers. Span columns hold Python-literal lists of
(start, end, text) tuples whose offsets point into `post_corrected`; `gender` is aligned
one-to-one with `position`, so it is stored on each position span.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# CSV column → output column. `gender` is folded into `positions`.
SPAN_COLUMNS = {
    "position": "positions",
    "salary": "salary",
    "salary_importance": "salary_importance",
    "salary_period": "salary_period",
    "unspecific_salary": "unspecific_salary",
    "verpflegung": "verpflegung",
    "activity": "activity",
    "attitude towards work": "attitude_towards_work",
    "background": "background",
    "interpersonal": "interpersonal",
    "job specific": "job_specific",
    "language": "language",
}

SPAN = pa.struct([("start", pa.int32()), ("end", pa.int32()), ("text", pa.string())])
POSITION_SPAN = pa.struct([("start", pa.int32()), ("end", pa.int32()), ("text", pa.string()),
                           ("gender", pa.string())])

SCHEMA = pa.schema(
    [
        ("ad_id", pa.string()),
        ("newspaper", pa.string()),
        ("date", pa.date32()),
        ("year", pa.int16()),
        ("decade", pa.int16()),
        ("page", pa.int16()),
        ("region_id", pa.string()),
        ("reading_order", pa.int32()),
        ("label", pa.string()),
        ("text", pa.string()),          # post-corrected text; all span offsets refer to it
        ("text_ocr", pa.string()),      # raw OCR
        ("min_x", pa.int32()), ("max_x", pa.int32()), ("min_y", pa.int32()), ("max_y", pa.int32()),
        ("iiif_link", pa.string()),
        ("heading_text", pa.string()),  # heading directly above this ad, if any (often the job title)
        ("heading_ad_id", pa.string()),
        ("heading_gap_px", pa.int32()),
        ("positions", pa.list_(POSITION_SPAN)),
        *[(name, pa.list_(SPAN)) for col, name in SPAN_COLUMNS.items() if col != "position"],
        ("metadata_raw", pa.string()),
        ("source_row", pa.int64()),
    ]
)

METADATA_RE = re.compile(r"^(?P<newspaper>[a-z]+)_(?P<date>\d{8})_(?P<page>\d{3})")

# A heading counts as the title of an ad when it sits in the same column (≥50 % horizontal
# overlap) and the ad starts at most this far below it. On the Wiener Zeitung sample the
# median gap is -6 px (touching); larger gaps are usually a different ad.
HEADING_MAX_GAP_PX = 80
HEADING_MIN_OVERLAP = 0.5


def parse_spans(value: object) -> list[dict]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [{"start": int(s), "end": int(e), "text": t} for s, e, t in ast.literal_eval(value)]


def _positions(position: object, gender: object) -> list[dict]:
    spans = parse_spans(position)
    genders = parse_spans(gender)
    if len(genders) != len(spans) or any((g["start"], g["end"]) != (s["start"], s["end"])
                                         for g, s in zip(genders, spans)):
        genders = [{"text": None}] * len(spans)  # misaligned: keep positions, drop gender
    return [{**s, "gender": g["text"]} for s, g in zip(spans, genders)]


def read_extractions_csv(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Parse one extraction CSV into the SCHEMA columns. Returns (frame, report)."""
    raw = pd.read_csv(path)
    report: dict = {"csv": str(path), "rows_in": len(raw)}

    meta = raw["metadata"].str.extract(METADATA_RE)
    bad_meta = meta["newspaper"].isna()
    if bad_meta.any():
        raise ValueError(f"Unparseable metadata values: {raw.loc[bad_meta, 'metadata'].head().tolist()}")
    report["metadata_irregular"] = raw.loc[raw["metadata"] != raw["metadata"].str.extract(
        r"^([a-z]+_\d{8}_\d{3})")[0], "metadata"].tolist()

    df = pd.DataFrame({
        "newspaper": meta["newspaper"],
        "date": pd.to_datetime(meta["date"], format="%Y%m%d").dt.date,
        "page": meta["page"].astype(int),
        "region_id": raw["region_id"],
        "reading_order": raw["reading_order"],
        "label": raw["prediction_label"],
        "text": raw["post_corrected"],
        "text_ocr": raw["text"],
        "min_x": raw["min_x"], "max_x": raw["max_x"], "min_y": raw["min_y"], "max_y": raw["max_y"],
        "iiif_link": raw["iiif_link"],
        "metadata_raw": raw["metadata"],
        "source_row": raw.iloc[:, 0] if raw.columns[0].startswith("Unnamed") else np.arange(len(raw)),
    })
    df["ad_id"] = df["newspaper"] + "_" + meta["date"] + "_" + meta["page"] + "_" + df["region_id"]
    df["year"] = pd.to_datetime(meta["date"], format="%Y%m%d").dt.year
    df["decade"] = df["year"] // 10 * 10

    df["positions"] = [_positions(p, g) for p, g in zip(raw["position"], raw["gender"])]
    for col, name in SPAN_COLUMNS.items():
        if col != "position":
            df[name] = raw[col].map(parse_spans)

    # Exact duplicate regions (e.g. a copied page file): keep the one with regular metadata.
    df["_irregular"] = df["metadata_raw"].isin(report["metadata_irregular"])
    df = df.sort_values("_irregular", kind="stable")
    dup = df.duplicated(subset=["ad_id"]) | df.duplicated(subset=["iiif_link", "text"])
    report["duplicates_dropped"] = df.loc[dup, "metadata_raw"].tolist()
    df = df.loc[~dup].drop(columns="_irregular").sort_values("source_row").reset_index(drop=True)

    df = link_headings(df)
    report.update(summarize(df))
    return df, report


def link_headings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each heading region to the ad directly below it in the same column."""
    df = df.copy()
    df["heading_text"] = None
    df["heading_ad_id"] = None
    df["heading_gap_px"] = pd.array([pd.NA] * len(df), dtype="Int32")

    is_heading = df["label"] == "heading"
    ads_by_page = {key: grp for key, grp in df.loc[~is_heading].groupby(["newspaper", "date", "page"])}
    for idx, h in df.loc[is_heading].iterrows():
        ads = ads_by_page.get((h["newspaper"], h["date"], h["page"]))
        if ads is None:
            continue
        overlap = (np.minimum(ads["max_x"], h["max_x"]) - np.maximum(ads["min_x"], h["min_x"])) \
            / max(h["max_x"] - h["min_x"], 1)
        gap = ads["min_y"] - h["max_y"]
        cand = ads[(overlap >= HEADING_MIN_OVERLAP) & (ads["min_y"] >= h["min_y"]) & (gap <= HEADING_MAX_GAP_PX)]
        if cand.empty:
            continue
        target = (cand["min_y"] - h["max_y"]).idxmin()
        if pd.notna(df.at[target, "heading_ad_id"]):  # two headings above one ad: keep the closer
            if df.at[target, "heading_gap_px"] <= gap[target]:
                continue
        df.at[target, "heading_text"] = h["text"]
        df.at[target, "heading_ad_id"] = h["ad_id"]
        df.at[target, "heading_gap_px"] = int(gap[target])
    return df


def summarize(df: pd.DataFrame) -> dict:
    headings = int((df["label"] == "heading").sum())
    return {
        "rows_out": len(df),
        "labels": df["label"].value_counts().to_dict(),
        "years": f"{df['year'].min()}–{df['year'].max()}",
        "rows_per_decade": {int(k): int(v) for k, v in df["decade"].value_counts().sort_index().items()},
        "headings": headings,
        "headings_linked": int(df["heading_ad_id"].notna().sum()),
        "spans": {name: int(df[name].map(len).sum()) for name in SPAN_COLUMNS.values()},
        "positions_without_gender": int(sum(p["gender"] is None for ps in df["positions"] for p in ps)),
    }


def write_ads(df: pd.DataFrame, out_dir: Path | str) -> None:
    """Write as a Hive-partitioned Parquet dataset (newspaper=…/year=…).

    Re-running replaces only the partitions present in `df`, so newspapers can be ingested one
    CSV at a time.
    """
    table = pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA, preserve_index=False)
    ds.write_dataset(
        table, out_dir, format="parquet",
        partitioning=ds.partitioning(pa.schema([("newspaper", pa.string()), ("year", pa.int16())]), flavor="hive"),
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
    )
