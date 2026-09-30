"""python -m hisrag.normalize STEP

  text    step 2: normalized text, quality metrics and flags → derived/ad_text
  dedup   step 3: clusters of repeated printings → derived/ad_dups (needs `text`)
  positions [--pilot] [--thinking] [--batch-size N] [--workers N]
          step 4: position dictionary via the LLM → derived/position_dict, derived/ad_positions.
          --pilot runs ~80 forms and writes a review CSV instead (with --thinking also a
          20-form comparison with reasoning).
  requirements [--pilot] [--batch-size N] [--workers N]
          step 5: requirement tags via the LLM, vocabulary from vocab/requirements.yaml →
          derived/requirement_dict, derived/ad_requirements. --pilot maps 20 phrases per column
          and writes a review CSV instead. Tags whose value is not in the phrase are then checked
          again without context; --verify-pilot checks 120 of them and writes a review CSV instead.
  salary [--pilot]
          step 6: amounts from the `salary` spans by rules (LLM for the few leftovers) →
          derived/ad_salary (one row per amount), derived/ad_pay (main pay and benefits per ad).
          --pilot writes 200 rule-parsed rows and 40 LLM leftovers to a review CSV instead.
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


def _requirement_spans(cfg):
    from hisrag.normalize.requirements import COLUMNS

    selects = " UNION ALL ".join(
        f"""SELECT a.ad_id, a.newspaper, a.year, '{c}' AS "column", a.text,
                   u.s.start AS start, u.s."end" AS "end", u.s.text AS phrase
            FROM ads a JOIN ad_text t USING (ad_id), UNNEST(a.{c}) AS u(s)
            WHERE NOT t.flag_death_register""" for c in COLUMNS)
    return query(selects, cfg=cfg)


VERIFY_PILOT_TAGS = 120


def run_requirements(cfg, pilot=False, verify_pilot=False, batch_size=40, workers=None) -> dict:
    import sys

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hisrag.llm import DHClient
    from hisrag.normalize import requirements as R

    spans = _requirement_spans(cfg)
    phrases = R.collect_phrases(spans)
    mapper = R.RequirementMapper()
    client = DHClient(cfg)
    if workers:
        client.max_workers = workers
    base = {"vocabulary_version": mapper.vocab.version, "phrases_total": len(phrases),
            "phrases_by_column": phrases["column"].value_counts().to_dict()}

    if pilot:
        sample = R.pilot_sample(phrases)
        print(f"{len(sample)} phrases, {batch_size} per request …", file=sys.stderr)
        results, stats = mapper.run(client, sample, batch_size=batch_size)
        review = sample[["column", "surface", "count", "context"]].copy()
        review["tags"] = [R.format_tags(results[k]) if k in results else "FAILED" for k in sample["key"]]
        kept = mapper.to_dictionary(sample, results, client.model)
        review["tags_after_checks"] = [R.format_tag_dicts(t) if t is not None else "FAILED" for t in kept["tags"]]
        out = derived_dir("pilot", cfg)
        out.mkdir(parents=True, exist_ok=True)
        review.to_csv(out / "requirements_pilot.csv", index=False, encoding="utf-8-sig")
        return {**base, "pilot": stats, "review_csv": str(out / "requirements_pilot.csv"),
                "estimated_full_run_minutes": _estimate(stats, len(phrases), batch_size, client),
                "usage": client.usage.summary()}

    results, stats = mapper.run(client, phrases, batch_size=batch_size)
    dictionary = mapper.to_dictionary(phrases, results, client.model)
    candidates = R.verification_candidates(dictionary)
    verifier = R.TagVerifier(mapper.vocab)

    if verify_pilot:
        top = candidates.nlargest(VERIFY_PILOT_TAGS // 2, "count")
        rest = candidates.drop(top.index)
        sample = pd.concat([top, rest.sample(min(VERIFY_PILOT_TAGS - len(top), len(rest)), random_state=0)])
        print(f"{len(sample)} of {len(candidates)} tags to check …", file=sys.stderr)
        checked, vstats = verifier.run(client, sample)
        review = sample.drop(columns="key").assign(
            stated=[checked[k].stated if k in checked else "FAILED" for k in sample["key"]])
        out = derived_dir("pilot", cfg)
        out.mkdir(parents=True, exist_ok=True)
        review.to_csv(out / "requirements_verify_pilot.csv", index=False, encoding="utf-8-sig")
        return {**base, "tags_to_check": len(candidates), "pilot": vstats,
                "rejected_in_pilot": int((review["stated"] == False).sum()),  # noqa: E712
                "review_csv": str(out / "requirements_verify_pilot.csv"),
                "estimated_full_check_minutes": _estimate(vstats, len(candidates), R.VERIFY_BATCH_SIZE, client),
                "usage": client.usage.summary()}

    checked, vstats = verifier.run(client, candidates)
    dictionary = R.apply_verification(dictionary, checked)
    d_out = derived_dir("requirement_dict", cfg)
    d_out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(dictionary, schema=R.DICT_SCHEMA, preserve_index=False),
                   d_out / "part-0.parquet")
    ad_req = R.ad_requirements(spans, dictionary)
    write_partitioned(ad_req, derived_dir("ad_requirements", cfg), R.AD_REQUIREMENTS_SCHEMA)
    return {**base, **stats, "verification": vstats, **R.summarize(dictionary, ad_req),
            "details_dropped_as_ungrounded": mapper.details_dropped, "usage": client.usage.summary()}


SALARY_PILOT_ROWS = 200
SALARY_PILOT_LEFTOVERS = 40


def run_salary(cfg, pilot=False, workers=None) -> dict:
    import pandas as pd

    from hisrag.llm import DHClient
    from hisrag.normalize import salary as S

    spans = query("""
        SELECT a.ad_id, a.newspaper, a.year, a.date, a.text, u.s.start AS start, u.s."end" AS "end", u.s.text AS phrase
        FROM ads a JOIN ad_text t USING (ad_id), UNNEST(a.salary) AS u(s)
        WHERE NOT t.flag_death_register""", cfg=cfg)
    benefits = query(" UNION ALL ".join(f"""
        SELECT a.ad_id, a.newspaper, a.year, u.s.text AS phrase
        FROM ads a JOIN ad_text t USING (ad_id), UNNEST(a.{c}) AS u(s)
        WHERE NOT t.flag_death_register""" for c in ("verpflegung", "unspecific_salary")), cfg=cfg)
    rows = S.parse_spans(spans)
    left = S.leftovers(rows)
    client = DHClient(cfg)
    if workers:
        client.max_workers = workers

    if pilot:
        left = left.sample(min(SALARY_PILOT_LEFTOVERS, len(left)), random_state=0).sort_values("key")
        results, stats = S.run_llm(client, left)
        sample = rows[rows["parsed_by"] == "rules"].sample(min(SALARY_PILOT_ROWS, len(rows)), random_state=0)
        llm = rows[rows["snippet"].isin(left["key"])].drop_duplicates("snippet")
        llm = S.apply_llm(llm, results).assign(parsed_by=lambda d: d["parsed_by"].fillna("llm: no amount"))
        cols = ["year", "snippet", "amount_min", "amount_max", "currency", "standard", "standard_source",
                "component", "period", "period_source", "parsed_by"]
        out = derived_dir("pilot", cfg)
        out.mkdir(parents=True, exist_ok=True)
        pd.concat([sample, llm])[cols].to_csv(out / "salary_pilot.csv", index=False, encoding="utf-8-sig")
        return {"salary_spans": len(rows), "parsed_by": rows["parsed_by"].value_counts(dropna=False).to_dict(),
                "leftovers_distinct": len(S.leftovers(rows)), "pilot": stats,
                "review_csv": str(out / "salary_pilot.csv"), "usage": client.usage.summary()}

    results, stats = S.run_llm(client, left)
    rows = S.apply_llm(rows, results)
    rows = S.assign_standard_by_date(rows, spans["date"])
    write_partitioned(rows, derived_dir("ad_salary", cfg), S.AD_SALARY_SCHEMA)
    pay = S.ad_pay(rows, benefits)
    write_partitioned(pay, derived_dir("ad_pay", cfg), S.AD_PAY_SCHEMA)
    return {**S.summarize(rows, pay), "llm": stats, "usage": client.usage.summary()}


STEPS = {"text": run_text, "dedup": run_dedup, "positions": run_positions, "requirements": run_requirements,
         "salary": run_salary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalization steps")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--pilot", action="store_true", help="positions: pilot run with review CSV")
    parser.add_argument("--verify-pilot", action="store_true",
                        help="requirements: check a sample of tags without context, write a review CSV")
    parser.add_argument("--thinking", action="store_true", help="positions: enable reasoning")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="forms per request (positions: 25, requirements: 40)")
    parser.add_argument("--workers", type=int, default=None, help="positions: parallel requests")
    args = parser.parse_args()
    cfg = load_config()
    if args.step == "positions":
        report = run_positions(cfg, pilot=args.pilot, thinking=args.thinking,
                               batch_size=args.batch_size or 25, workers=args.workers)
    elif args.step == "requirements":
        report = run_requirements(cfg, pilot=args.pilot, verify_pilot=args.verify_pilot,
                                  batch_size=args.batch_size or 40, workers=args.workers)
    elif args.step == "salary":
        report = run_salary(cfg, pilot=args.pilot, workers=args.workers)
    else:
        report = STEPS[args.step](cfg)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
