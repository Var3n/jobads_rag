"""python -m hisrag.normalize STEP

  text    step 2: normalized text, quality metrics and flags → derived/ad_text
  dedup   step 3: clusters of repeated printings → derived/ad_dups (needs `text`)
  positions [--pilot] [--thinking] [--batch-size N] [--workers N]
          step 4: position dictionary via the LLM → derived/position_dict, derived/ad_positions.
          --pilot runs ~80 forms and writes a review CSV instead (with --thinking also a
          20-form comparison with reasoning).
"""

import argparse
import json

from hisrag.config import load_config
from hisrag.data import derived_dir, query, write_partitioned


def run_text(cfg) -> dict:
    from hisrag.normalize.quality import SCHEMA, assess, summarize

    ads = query("SELECT ad_id, newspaper, year, label, text, text_ocr, heading_text FROM ads", cfg=cfg)
    q = assess(ads)
    out = derived_dir("ad_text", cfg)
    write_partitioned(q, out, SCHEMA)
    return {**summarize(q, ads["label"]), "written_to": str(out)}


def run_dedup(cfg) -> dict:
    from hisrag.normalize.dedup import SCHEMA, deduplicate

    regions = query("""
        SELECT ad_id, a.newspaper, a.year, a.date, t.text_norm, t.n_flags, t.ocr_support,
               a.label <> 'heading' AND NOT t.flag_death_register AND NOT t.flag_too_short
                   AND NOT t.flag_pc_unsupported AS eligible
        FROM ads a JOIN ad_text t USING (ad_id)""", cfg=cfg)
    dups, report = deduplicate(regions)
    out = derived_dir("ad_dups", cfg)
    write_partitioned(dups, out, SCHEMA)
    return {**report, "written_to": str(out)}


def _position_inputs(cfg):
    spans = query("""
        SELECT a.ad_id, a.newspaper, a.year, a.text, u.p.start AS start, u.p."end" AS "end",
               u.p.text AS form, u.p.gender AS gender
        FROM ads a, UNNEST(a.positions) AS u(p)""", cfg=cfg)
    headings = query("""
        SELECT ad_id, newspaper, year, heading_text AS heading, text AS ad_text
        FROM ads WHERE heading_text IS NOT NULL
        UNION ALL
        SELECT ad_id, newspaper, year, text AS heading, NULL AS ad_text
        FROM ads WHERE label = 'heading' AND ad_id NOT IN (SELECT heading_ad_id FROM ads WHERE heading_ad_id IS NOT NULL)
        """, cfg=cfg)
    return spans, headings


PILOT_THINKING_FORMS = 20
PILOT_THINKING_BATCH = 5


def _estimate(stats: dict, n_forms: int, batch_size: int, client) -> float:
    """Minutes for all forms: rounds of `max_workers` parallel requests, each as long as a pilot round."""
    rounds = -(-n_forms // batch_size) / client.max_workers
    return round(stats["seconds"] * rounds / 60, 1)


def _pilot(cfg, client, forms, batch_size, with_reasoning=False) -> dict:
    """80 forms without reasoning (written right away); with --thinking also 20 of them with reasoning.

    The reasoning stage is small on purpose: at ~10 tokens/s per request it takes minutes per
    request. If it fails or is interrupted, the no-reasoning results are already on disk.
    """
    import sys

    from hisrag.normalize import positions as P

    fmt = lambda r: "; ".join(f"{e.term} | {e.lemma} | {e.modern} | {e.gender_form} | {e.category}"
                              for e in r.entries) if r.entries else "—"
    out = derived_dir("pilot", cfg)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "positions_pilot.csv"
    sample = P.pilot_sample(forms)
    review = sample[["key", "surface", "count", "context"]].copy()

    print(f"[1/{2 if with_reasoning else 1}] {len(sample)} forms without reasoning, {batch_size} per request …", file=sys.stderr)
    off, stats_off = P.run(client, sample, batch_size=batch_size, thinking=False)
    review["no_reasoning"] = [fmt(off[k]) if k in off else "FAILED" for k in review["key"]]
    review.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"      done in {stats_off['seconds']} s → {csv}", file=sys.stderr)
    report = {"forms_total": len(forms), "pilot_forms": len(sample), "no_reasoning": stats_off,
              # pilot batches ran in parallel, so their wall time is about one request's duration
              "estimated_full_run_minutes_no_reasoning": _estimate(stats_off, len(forms), batch_size, client),
              "review_csv": str(csv)}

    if not with_reasoning:
        report["usage"] = client.usage.summary()
        return report
    subset = sample.iloc[:: max(len(sample) // PILOT_THINKING_FORMS, 1)].head(PILOT_THINKING_FORMS)
    print(f"[2/2] {len(subset)} of them with reasoning, {PILOT_THINKING_BATCH} per request "
          "(minutes per request; Ctrl+C keeps the results above) …", file=sys.stderr)
    try:
        on, stats_on = P.run(client, subset, batch_size=PILOT_THINKING_BATCH, thinking=True)
    except KeyboardInterrupt:
        report["with_reasoning"] = "interrupted"
    else:
        review["with_reasoning"] = [fmt(on[k]) if k in on else "" for k in review["key"]]
        review["differs"] = [bool(w) and w != n for n, w in zip(review["no_reasoning"], review["with_reasoning"])]
        review.to_csv(csv, index=False, encoding="utf-8-sig")
        report["with_reasoning"] = stats_on
        report["agreement_on_reasoning_subset"] = P.compare_runs({k: off[k] for k in on if k in off}, on)
        report["estimated_full_run_minutes_with_reasoning"] = _estimate(stats_on, len(forms), PILOT_THINKING_BATCH, client)
    report["usage"] = client.usage.summary()
    return report


def run_positions(cfg, pilot=False, thinking=False, batch_size=25, workers=None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hisrag.llm import DHClient
    from hisrag.normalize import positions as P

    spans, headings = _position_inputs(cfg)
    forms = P.collect_forms(spans, headings)
    client = DHClient(cfg)
    if workers:
        client.max_workers = workers

    if pilot:
        return _pilot(cfg, client, forms, batch_size, with_reasoning=thinking)

    results, stats = P.run(client, forms, batch_size=batch_size, thinking=thinking)
    dictionary = P.to_dictionary(forms, results, client.model)
    consistency = P.consistency_report(dictionary)
    dictionary, harmonized = P.harmonize_categories(dictionary)
    d_out = derived_dir("position_dict", cfg)
    d_out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(dictionary, schema=P.DICT_SCHEMA, preserve_index=False),
                   d_out / "part-0.parquet")
    # Unlinked headings help build the dictionary but belong to no ad.
    ap = P.ad_positions(spans, headings[headings["ad_text"].notna()], dictionary)
    write_partitioned(ap, derived_dir("ad_positions", cfg), P.AD_POSITIONS_SCHEMA)
    return {
        **stats, "usage": client.usage.summary(),
        "forms_normalized": int(dictionary["is_position"].notna().sum()),
        "forms_that_are_positions": int(dictionary["is_position"].fillna(False).sum()),
        "position_mentions": len(ap), "ads_with_position": int(ap["ad_id"].nunique()),
        "consistency_before_harmonizing": consistency,
        "entries_with_harmonized_category": harmonized,
        "top_lemmas": ap["lemma"].value_counts().head(15).to_dict(),
        "categories": ap["category"].value_counts().to_dict(),
    }


STEPS = {"text": run_text, "dedup": run_dedup, "positions": run_positions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalization steps")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--pilot", action="store_true", help="positions: pilot run with review CSV")
    parser.add_argument("--thinking", action="store_true", help="positions: enable reasoning")
    parser.add_argument("--batch-size", type=int, default=25, help="positions: forms per request")
    parser.add_argument("--workers", type=int, default=None, help="positions: parallel requests")
    args = parser.parse_args()
    cfg = load_config()
    if args.step == "positions":
        report = run_positions(cfg, pilot=args.pilot, thinking=args.thinking,
                               batch_size=args.batch_size, workers=args.workers)
    else:
        report = STEPS[args.step](cfg)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
