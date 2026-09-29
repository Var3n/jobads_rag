"""python -m hisrag.normalize STEP

  text    step 2: normalized text, quality metrics and flags → derived/ad_text
  dedup   step 3: clusters of repeated printings → derived/ad_dups (needs `text`)
  positions [--pilot] [--thinking] [--batch-size N] [--workers N]
          step 4: position dictionary via the LLM → derived/position_dict, derived/ad_positions.
          --pilot runs ~80 forms with and without reasoning and writes a review CSV instead.
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
        sample = P.pilot_sample(forms)
        off, stats_off = P.run(client, sample, batch_size=batch_size, thinking=False)
        on, stats_on = P.run(client, sample, batch_size=batch_size, thinking=True)
        fmt = lambda r: "; ".join(f"{e.term} | {e.lemma} | {e.modern} | {e.gender_form} | {e.category}"
                                  for e in r.entries) if r.entries else "—"
        review = sample[["key", "surface", "count", "context"]].copy()
        review["no_reasoning"] = [fmt(off[k]) if k in off else "FAILED" for k in review["key"]]
        review["with_reasoning"] = [fmt(on[k]) if k in on else "FAILED" for k in review["key"]]
        review["differs"] = review["no_reasoning"] != review["with_reasoning"]
        out = derived_dir("pilot", cfg)
        out.mkdir(parents=True, exist_ok=True)
        review.to_csv(out / "positions_pilot.csv", index=False, encoding="utf-8-sig")
        usage = client.usage.summary()
        per_form = {m: s["seconds"] / max(s["forms"], 1) for m, s in (("no_reasoning", stats_off), ("with_reasoning", stats_on))}
        return {
            "forms_total": len(forms), "pilot_forms": len(sample),
            "no_reasoning": stats_off, "with_reasoning": stats_on,
            "agreement": P.compare_runs(off, on), "usage": usage,
            "estimated_full_run_minutes": {m: round(v * len(forms) / 60, 1) for m, v in per_form.items()},
            "review_csv": str(out / "positions_pilot.csv"),
        }

    results, stats = P.run(client, forms, batch_size=batch_size, thinking=thinking)
    dictionary = P.to_dictionary(forms, results, client.model)
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
        "consistency": P.consistency_report(dictionary),
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
