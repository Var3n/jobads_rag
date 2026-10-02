"""python -m hisrag.eval STEP   (step 8: retrieval comparison)

  queries [--pilot] [--n 300]
          research questions: the LLM writes a question and a relevance criterion for a stratified sample of
          seed ads → data/eval/queries.parquet. --pilot writes 30 to data/eval/queries_pilot.csv for review.
  embed [--pilot] [--models a,b] [--variant raw|enriched|all]
          embed all searchable ads per model (resumable, chunks of 1024) → data/embeddings/<variant>/<model>/.
          --pilot embeds only the first chunk per model and estimates the full run.
  score [--pilot] [--models a,b]
          top 10 of BM25, each fully embedded model and the hybrids for every question; the LLM judges the
          pooled ads → data/eval/runs.parquet, judgments.parquet, scores.parquet; prints the measures.
          --pilot judges 20 questions and writes data/eval/judgments_pilot.csv for review.
  score --extend --models qwen3-embedding-8b@1024,... [--variant enriched|raw|all]
          add methods to the finished run (model@dims = stored vectors cut to their first dims): only
          (question, ad) pairs no method had found are judged, then every method is rescored.
"""

import argparse
import json
import math

from hisrag.config import load_config

PILOT_QUESTIONS = 30
PILOT_JUDGED_QUESTIONS = 20


def _models(cfg, arg: str | None) -> list[str]:
    return arg.split(",") if arg else list(cfg["embeddings"]["models"])


def run_queries(cfg, pilot: bool, n: int) -> dict:
    from hisrag.eval import retrieval as E
    from hisrag.llm import DHClient

    docs = E.documents(cfg)
    seeds = E.sample_seeds(docs, n=n)
    if pilot:
        seeds = seeds.sample(min(PILOT_QUESTIONS, len(seeds)), random_state=1).sort_values("ad_id")
    client = DHClient(cfg)
    questions, stats = E.write_questions(client, seeds)
    out = cfg.path("eval_dir")
    out.mkdir(parents=True, exist_ok=True)
    if pilot:
        path = out / "queries_pilot.csv"
        questions[["decade", "label", "question", "criterion", "leakage", "raw"]].to_csv(
            path, index=False, encoding="utf-8-sig")
    else:
        path = out / "queries.parquet"
        questions.to_parquet(path, index=False)
    return {"searchable_ads": len(docs), "questions": len(questions), **stats,
            "mean_verbatim_share": round(float(questions["leakage"].mean()), 2),
            "by_decade": questions["decade"].value_counts().sort_index().to_dict(),
            "written_to": str(path), "usage": client.usage.summary()}


def run_embed(cfg, pilot: bool, models: list[str], variants: list[str]) -> dict:
    from hisrag.eval import retrieval as E
    from hisrag.llm import DHClient

    docs = E.documents(cfg)
    client = DHClient(cfg)
    n_chunks = math.ceil(len(docs) / E.CHUNK)
    report = {"searchable_ads": len(docs), "chunks": n_chunks}
    for v in variants:
        for m in models:
            print(f"{v}/{m} …", flush=True)
            try:
                stats = E.embed_documents(client, docs, v, m, cfg, chunks=[0] if pilot else None)
            except Exception as exc:  # one model failing must not stop the others; re-run resumes
                report[f"{v}/{m}"] = f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"
                continue
            if pilot and stats["embedded"]:
                stats["estimated_full_minutes"] = round(stats["seconds"] * n_chunks / 60, 1)
            report[f"{v}/{m}"] = stats
    report["usage"] = client.usage.summary()
    return report


def run_score(cfg, pilot: bool, models: list[str]) -> dict:
    import pandas as pd

    from hisrag.eval import retrieval as E
    from hisrag.llm import DHClient

    out = cfg.path("eval_dir")
    questions = pd.read_parquet(out / "queries.parquet")
    if pilot:
        questions = questions.sample(min(PILOT_JUDGED_QUESTIONS, len(questions)), random_state=2)
    docs = E.documents(cfg)
    client = DHClient(cfg)
    runs = E.run_methods(questions, docs, cfg, client, models)
    pooled = E.pool(runs, questions, docs)
    judgments, stats = E.judge(client, pooled)
    report = {"questions": len(questions), "methods": int(runs.groupby(["method", "variant"]).ngroups),
              "pooled_pairs": len(pooled), "pooled_per_question": round(len(pooled) / len(questions), 1),
              "judging": stats,
              # str keys: numpy int8 grades cannot be JSON keys
              "grades": {str(k): int(v) for k, v in judgments["grade"].value_counts(dropna=False).items()}}
    if pilot:
        found_by = (runs.assign(m=runs["method"] + "/" + runs["variant"])
                        .groupby(["query_id", "ad_id"])["m"].agg(lambda m: ", ".join(sorted(m))).rename("found_by"))
        review = (pooled.merge(judgments, on=["query_id", "ad_id"]).join(found_by, on=["query_id", "ad_id"])
                        [["query_id", "question", "criterion", "grade", "reason", "raw", "found_by"]])
        review.to_csv(out / "judgments_pilot.csv", index=False, encoding="utf-8-sig")
        report["review_csv"] = str(out / "judgments_pilot.csv")
    else:
        per_query = E.scores(runs, judgments, questions)
        runs.to_parquet(out / "runs.parquet", index=False)
        judgments.to_parquet(out / "judgments.parquet", index=False)
        per_query.to_parquet(out / "scores.parquet", index=False)
        report["summary"] = {f"{m}/{v}": row.to_dict() for (m, v), row in E.summary(per_query).iterrows()}
    report["usage"] = client.usage.summary()
    return report


def run_extend(cfg, models: list[str], variants: tuple[str, ...]) -> dict:
    """Add methods to a finished run: judge only the (question, ad) pairs no method had found before, then
    rescore every method against the larger pool."""
    import pandas as pd

    from hisrag.eval import retrieval as E
    from hisrag.llm import DHClient

    out = cfg.path("eval_dir")
    questions = pd.read_parquet(out / "queries.parquet")
    runs = pd.read_parquet(out / "runs.parquet")
    judgments = pd.read_parquet(out / "judgments.parquet")
    docs = E.documents(cfg)
    client = DHClient(cfg)
    new_runs = E.run_methods(questions, docs, cfg, client, models, variants=variants, bm25=False, hybrids=False)
    replaced = set(zip(new_runs["method"], new_runs["variant"]))
    runs = pd.concat([runs[[(m, v) not in replaced for m, v in zip(runs["method"], runs["variant"])]], new_runs],
                     ignore_index=True)
    pooled = E.pool(new_runs, questions, docs)
    judged = set(zip(judgments["query_id"], judgments["ad_id"]))
    new_pairs = pooled[[(q, a) not in judged for q, a in zip(pooled["query_id"], pooled["ad_id"])]].reset_index(drop=True)
    new_judgments, stats = E.judge(client, new_pairs) if len(new_pairs) else (judgments.iloc[:0], {})
    judgments = pd.concat([judgments, new_judgments], ignore_index=True)
    per_query = E.scores(runs, judgments, questions)
    runs.to_parquet(out / "runs.parquet", index=False)
    judgments.to_parquet(out / "judgments.parquet", index=False)
    per_query.to_parquet(out / "scores.parquet", index=False)
    return {"added_methods": sorted(f"{m}/{v}" for m, v in replaced), "new_pairs_judged": len(new_pairs),
            "judging": stats,
            "new_grades": {str(k): int(v) for k, v in new_judgments["grade"].value_counts(dropna=False).items()},
            "summary": {f"{m}/{v}": row.to_dict() for (m, v), row in E.summary(per_query).iterrows()},
            "usage": client.usage.summary()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: retrieval comparison")
    parser.add_argument("step", choices=["queries", "embed", "score"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--n", type=int, default=300, help="queries: number of research questions")
    parser.add_argument("--models", default=None, help="comma-separated model names (default: all configured)")
    parser.add_argument("--variant", default=None, choices=["raw", "enriched", "all"],
                        help="text variant (embed: default raw; score --extend: default enriched)")
    parser.add_argument("--extend", action="store_true",
                        help="score: add --models (e.g. qwen3-embedding-8b@1024) to the finished run")
    args = parser.parse_args()
    cfg = load_config()
    variants = lambda default: ("raw", "enriched") if args.variant == "all" else (args.variant or default,)
    if args.step == "queries":
        report = run_queries(cfg, args.pilot, args.n)
    elif args.step == "embed":
        report = run_embed(cfg, args.pilot, _models(cfg, args.models), list(variants("raw")))
    elif args.extend:
        if not args.models:
            parser.error("score --extend needs --models")
        report = run_extend(cfg, _models(cfg, args.models), variants("enriched"))
    else:
        report = run_score(cfg, args.pilot, _models(cfg, args.models))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
