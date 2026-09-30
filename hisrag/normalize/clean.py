"""Step 7: one clean table, one row per ad, joining steps 1–6.

`ad_clean` is what the index (step 9) and the agent tools (step 10) read: the text, the quality flags,
the repeated-printing cluster, the normalized positions, the requirement tags and the pay. Two columns
encode how an ad may be used:

  searchable  not a death-register entry and not a bare heading (headings are attached to the ad below);
  countable   searchable, the canonical printing of its cluster, and not text invented by post-correction
              (flag_pc_unsupported). Counts of ads use `countable`; such ads are still searchable with a warning.

Pure SQL over the DuckDB views, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds

from hisrag.config import Config
from hisrag.data import PARTITIONING, connect

REQUIRED = ("ads", "ad_text", "ad_dups", "ad_positions", "ad_requirements", "ad_pay", "ad_salary")
BENEFITS = ("wohnung", "kost", "kleidung", "heizung", "licht", "deputat", "quartiergeld", "zulagen")

SQL = f"""
WITH pos AS (
    SELECT ad_id,
           list_sort(list(DISTINCT term)) AS position_terms,
           list_sort(list(DISTINCT lemma)) AS position_lemmas,
           list_sort(list(DISTINCT modern)) AS position_modern,
           list_sort(list(DISTINCT category)) AS position_categories,
           CASE WHEN bool_and(gender_form = 'f') THEN 'f'
                WHEN bool_and(gender_form = 'm') THEN 'm'
                WHEN bool_or(gender_form = 'f') AND bool_or(gender_form = 'm') THEN 'mixed'
                WHEN bool_and(gender_form IN ('m/f', 'n')) THEN 'neutral' END AS position_gender
    FROM ad_positions GROUP BY ad_id
),
req AS (
    SELECT ad_id,
           list_sort(list(DISTINCT {{'group': "group", 'dimension': dimension, 'value': value, 'detail': detail}})) AS requirements,
           list_sort(list(DISTINCT dimension)) AS requirement_dimensions
    FROM ad_requirements GROUP BY ad_id
),
amounts AS (
    SELECT ad_id,
           list({{'amount_min': amount_min, 'amount_max': amount_max, 'currency': currency, 'standard': standard,
                  'component': component, 'period': period}} ORDER BY span_start) AS salary_amounts
    FROM ad_salary WHERE amount_min IS NOT NULL GROUP BY ad_id
)
SELECT a.ad_id, a.newspaper, a.year::SMALLINT AS year, a.date, a.decade, a.page, a.label, a.iiif_link,
       a.heading_text, a.text, t.text_norm, t.lang,
       t.n_flags, t.flag_pc_repetition, t.flag_pc_expanded, t.flag_pc_unsupported, t.flag_too_short,
       t.flag_death_register,
       d.dup_cluster_id, d.dup_cluster_size, d.is_canonical, d.run_first_date, d.run_last_date,
       NOT t.flag_death_register AND a.label <> 'heading' AS searchable,
       NOT t.flag_death_register AND a.label <> 'heading' AND d.is_canonical AND NOT t.flag_pc_unsupported AS countable,
       CASE WHEN t.flag_pc_unsupported THEN 'Text weicht stark vom OCR ab (von der Nachkorrektur ergänzt oder erfunden)'
            WHEN t.flag_pc_repetition THEN 'Nachkorrektur hat Text wiederholt'
            WHEN t.flag_too_short THEN 'Sehr kurzer Text' END AS quality_warning,
       pos.position_terms, pos.position_lemmas, pos.position_modern, pos.position_categories, pos.position_gender,
       req.requirements, req.requirement_dimensions,
       p.pay_min, p.pay_max, p.pay_currency, p.pay_standard, p.pay_period, coalesce(p.pay_amounts, 0) AS pay_amounts,
       amounts.salary_amounts,
       {", ".join(f"coalesce(p.benefit_{b}, false) AS benefit_{b}" for b in BENEFITS)}
FROM ads a
JOIN ad_text t USING (ad_id)
JOIN ad_dups d USING (ad_id)
LEFT JOIN pos USING (ad_id)
LEFT JOIN req USING (ad_id)
LEFT JOIN ad_pay p USING (ad_id)
LEFT JOIN amounts USING (ad_id)
"""


def build(cfg: Config | None = None) -> pa.Table:
    with connect(cfg) as con:
        views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
        missing = [t for t in REQUIRED if t not in views]
        if missing:
            raise RuntimeError(f"step 7 needs the tables of steps 1–6; missing: {', '.join(missing)}")
        result = con.execute(SQL).arrow()
        return result.read_all() if isinstance(result, pa.RecordBatchReader) else result  # newer DuckDB streams


def write(table: pa.Table, out_dir: Path | str) -> None:
    ds.write_dataset(table, out_dir, format="parquet", partitioning=PARTITIONING,
                     existing_data_behavior="delete_matching", basename_template="part-{i}.parquet")


def summarize(table: pa.Table) -> dict:
    with duckdb.connect() as con:
        con.register("c", table)
        one = lambda sql: con.execute(sql).fetchone()[0]
        n = one("SELECT count(*) FROM c")
        countable = one("SELECT count(*) FROM c WHERE countable")
        share = lambda cond: round(100 * one(f"SELECT avg(({cond})::int) FROM c WHERE countable"), 1)
        return {
            "ads": n, "searchable": one("SELECT count(*) FROM c WHERE searchable"), "countable": countable,
            "pct_countable_with_position": share("position_lemmas IS NOT NULL"),
            "pct_countable_with_requirements": share("requirements IS NOT NULL"),
            "pct_countable_with_pay": share("pay_min IS NOT NULL"),
            "countable_by_label": dict(con.execute(
                "SELECT label, count(*) FROM c WHERE countable GROUP BY 1 ORDER BY 2 DESC").fetchall()),
        }
