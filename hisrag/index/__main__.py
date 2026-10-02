"""python -m hisrag.index STEP   (step 9: search index)

  build [--newspapers wrz,…]
          (re)build the rows of these newspapers (default: all in ad_clean) in data/index/, then the
          full-text, filter and (from index.ann_min_rows on) vector indexes. Vectors come from the step-8
          store; ads without one are embedded first. Prints build times and size.
  check   semantic search through the index for the step-8 questions must find what the evaluation found
          for the same model, variant and dimensions (runs.parquet); also times both search modes.
"""

import argparse
import json
import time

from hisrag.config import load_config


def run_build(cfg, names: list[str] | None) -> dict:
    from hisrag.index import build as B
    from hisrag.llm import DHClient

    report = B.build(cfg, names, DHClient(cfg))
    report["config"] = cfg["index"]
    return report


def run_check(cfg) -> dict:
    import numpy as np
    import pandas as pd

    from hisrag.eval import retrieval as E
    from hisrag.index import AdIndex
    from hisrag.llm import DHClient

    ix = cfg["index"]
    out = cfg.path("eval_dir")
    questions = pd.read_parquet(out / "queries.parquet")
    runs = pd.read_parquet(out / "runs.parquet")
    method = f"{ix['model']}@{ix['dims']}"
    runs = runs[(runs["method"] == method) & (runs["variant"] == ix["variant"])]
    if runs.empty:
        return {"error": f"runs.parquet has no {method}/{ix['variant']}; run `score --extend --models {method}`"}
    client = DHClient(cfg)
    index = AdIndex(cfg, client)
    # the stored question vectors of step 8: a fresh embedding differs in the last digits
    qv = E.truncate(E.query_vectors(client, cfg, questions["question"].tolist(), ix["model"]), ix["dims"])
    expected = runs.groupby("query_id")["cluster"].apply(set)
    overlap, seconds = [], []
    for qid, v in zip(questions["query_id"], qv):
        t0 = time.monotonic()
        hits = index.semantic(vector=v, k=E.DEPTH)
        seconds.append(time.monotonic() - t0)
        want = expected.get(qid, set())
        overlap.append(len(set(hits["dup_cluster_id"]) & want) / max(len(want), 1))
    t0 = time.monotonic()
    words = ["Köchin", "Wirthschafterin", '"k. k. Statthalterei"', "Lehrer* Krakau", "Gouvernante französisch"]
    counts = {w: len(index.keyword(w, k=None)) for w in words}
    keyword_ms = (time.monotonic() - t0) / len(words) * 1000
    t0 = time.monotonic()
    index.semantic("Welche Sprachkenntnisse wurden von Gouvernanten verlangt?")
    return {"questions": len(questions), "method": f"{method}/{ix['variant']}",
            "mean_overlap_with_eval_top10": round(float(np.mean(overlap)), 3),
            "questions_identical": int(sum(o == 1 for o in overlap)),
            "semantic_ms_mean": round(1000 * float(np.mean(seconds)), 1),
            "semantic_with_query_embedding_ms": round(1000 * (time.monotonic() - t0)),
            "keyword_all_matches": counts, "keyword_ms_mean": round(keyword_ms, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9: search index")
    parser.add_argument("step", choices=["build", "check"])
    parser.add_argument("--newspapers", default=None, help="build: comma-separated newspapers (default: all)")
    args = parser.parse_args()
    cfg = load_config()
    if args.step == "build":
        report = run_build(cfg, args.newspapers.split(",") if args.newspapers else None)
    else:
        report = run_check(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
