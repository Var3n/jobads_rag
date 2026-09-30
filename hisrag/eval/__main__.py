"""python -m hisrag.eval STEP   (step 8: retrieval comparison)

  queries [--pilot] [--n 300]
          test questions: the LLM writes a modern search question for a stratified sample of countable
          ads → data/eval/queries.parquet. --pilot writes 30 to data/eval/queries_pilot.csv for review.
  embed [--pilot] [--models a,b] [--variant raw|enriched|all]
          embed all searchable ads per model (resumable, chunks of 1024) → data/embeddings/<variant>/<model>/.
          --pilot embeds only the first chunk per model and estimates the full run.
  score [--models a,b]
          rank all searchable ads for every question with BM25, each embedded model and the hybrids →
          data/eval/retrieval_results.parquet; prints recall@k and MRR per method.
"""

import argparse
import json
import math

from hisrag.config import load_config


def _models(cfg, arg: str | None) -> list[str]:
    return arg.split(",") if arg else list(cfg["embeddings"]["models"])


def run_queries(cfg, pilot: bool, n: int) -> dict:
    from hisrag.eval import retrieval as E
    from hisrag.llm import DHClient

    docs = E.documents(cfg)
    targets = E.sample_targets(docs, n=n)
    if pilot:
        targets = targets.sample(min(30, len(targets)), random_state=1).sort_values("ad_id")
    client = DHClient(cfg)
    queries, stats = E.write_queries(client, targets)
    out = cfg.path("eval_dir")
    out.mkdir(parents=True, exist_ok=True)
    if pilot:
        path = out / "queries_pilot.csv"
        queries[["decade", "label", "query", "leakage", "raw"]].to_csv(path, index=False, encoding="utf-8-sig")
    else:
        path = out / "queries.parquet"
        queries.to_parquet(path, index=False)
    return {"searchable_ads": len(docs), "questions": len(queries), **stats,
            "mean_verbatim_share": round(float(queries["leakage"].mean()), 2),
            "by_decade": queries["decade"].value_counts().sort_index().to_dict(),
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


def run_score(cfg, models: list[str]) -> dict:
    import pandas as pd

    from hisrag.eval import retrieval as E
    from hisrag.llm import DHClient

    out = cfg.path("eval_dir")
    queries = pd.read_parquet(out / "queries.parquet")
    docs = E.documents(cfg)
    client = DHClient(cfg)
    results = E.evaluate(queries, docs, cfg, client, models)
    results.to_parquet(out / "retrieval_results.parquet", index=False)
    table = E.summary(results)
    return {"questions": len(queries), "searchable_ads": len(docs),
            "methods": {f"{m}/{v}": row.to_dict() for (m, v), row in table.iterrows()},
            "written_to": str(out / "retrieval_results.parquet"), "usage": client.usage.summary()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: retrieval comparison")
    parser.add_argument("step", choices=["queries", "embed", "score"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--n", type=int, default=300, help="queries: number of test questions")
    parser.add_argument("--models", default=None, help="comma-separated model names (default: all configured)")
    parser.add_argument("--variant", default="raw", choices=["raw", "enriched", "all"], help="embed: text variant")
    args = parser.parse_args()
    cfg = load_config()
    if args.step == "queries":
        report = run_queries(cfg, args.pilot, args.n)
    elif args.step == "embed":
        variants = ["raw", "enriched"] if args.variant == "all" else [args.variant]
        report = run_embed(cfg, args.pilot, _models(cfg, args.models), variants)
    else:
        report = run_score(cfg, _models(cfg, args.models))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
