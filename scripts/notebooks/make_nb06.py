import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 06 · Retrieval comparison (step 8)

~300 research questions written by the LLM from seed ads; every method returned its top 10 (one printing per ad); the
LLM judged every pooled ad blind to the method (2 relevant, 1 partly, 0 not). Methods: BM25, five embedding models, and
BM25 + model hybrids, each on the ad text (`raw`) and on the ad plus its normalized fields (`enriched`).

Measures per question, averaged: **nDCG@10** (gains 0/1/3, ranks matter), **p@10** (share of the top 10 graded 2),
**p@10_lenient** (graded ≥ 1), **recall** (relevant ads found / all relevant ads in the pool), **seed_found** (the seed ad
or a reprint among the top 10). Intervals are 95 % bootstrap intervals over questions; differences between methods are
**paired** (same questions), which is what decides whether a gap is real."""),
    code("""import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import pandas as pd
import pyarrow.parquet as pq
from hisrag.config import load_config

cfg = load_config()
EVAL = cfg.path("eval_dir")
scores = pd.read_parquet(EVAL / "scores.parquet")
questions = pd.read_parquet(EVAL / "queries.parquet")
judgments = pd.read_parquet(EVAL / "judgments.parquet")
runs = pd.read_parquet(EVAL / "runs.parquet")
scores["name"] = scores["method"] + "/" + scores["variant"]
scores = scores.merge(questions[["query_id", "decade", "label"]], on="query_id")
MEASURES = ["ndcg@10", "p@10", "p@10_lenient", "recall", "seed_found"]
rng = np.random.default_rng(0)
B = 2000

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlelocation": "left",
    "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
})


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    \"\"\"95 % bootstrap interval of the mean over questions.\"\"\"
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    means = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


wide = scores.pivot_table(index="query_id", columns="name", values="ndcg@10")
print(f"{scores.query_id.nunique()} questions, {scores.name.nunique()} methods, "
      f"{len(judgments)} judged pairs ({len(judgments) / scores.query_id.nunique():.0f} per question)")
judgments["grade"].value_counts(normalize=True).sort_index().round(3)"""),
    md("## All methods, with 95 % intervals"),
    code("""rows = []
for name, g in scores.groupby("name"):
    row = {"method": name}
    for m in MEASURES:
        row[m] = g[m].mean()
    lo, hi = boot_ci(g["ndcg@10"])
    row["ndcg_lo"], row["ndcg_hi"] = lo, hi
    rows.append(row)
table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 0.28 * len(table) + 1))
y = np.arange(len(table))[::-1]
color = [SERIES[0] if not m.startswith(("hybrid", "bm25")) else SERIES[1] if m.startswith("hybrid") else INK_2
         for m in table["method"]]
ax.errorbar(table["ndcg@10"], y, xerr=[table["ndcg@10"] - table["ndcg_lo"], table["ndcg_hi"] - table["ndcg@10"]],
            fmt="none", ecolor=GRID, elinewidth=3)
ax.scatter(table["ndcg@10"], y, c=color, s=36, zorder=3)
ax.set_yticks(y, table["method"])
ax.set_xlabel("nDCG@10 (mean over questions, 95 % interval)")
ax.set_title("Methods by nDCG@10")
for c, label in [(SERIES[0], "embedding model"), (SERIES[1], "hybrid with BM25"), (INK_2, "BM25")]:
    ax.scatter([], [], c=c, s=36, label=label)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
table.round(3)"""),
    md("""## Paired differences to the best method
Mean difference in nDCG@10 per question (best − other) with a paired bootstrap interval. If the interval contains 0,
the two methods are not distinguishable on these questions."""),
    code("""best = table.loc[0, "method"]


def paired(a: str, b: str, measure: str = "ndcg@10") -> dict:
    w = scores.pivot_table(index="query_id", columns="name", values=measure)[[a, b]].dropna()
    d = (w[a] - w[b]).to_numpy()
    lo, hi = boot_ci(d)
    return {"vs": b, "diff": d.mean(), "lo": lo, "hi": hi, "better_in": (d > 0).mean(), "worse_in": (d < 0).mean(),
            "distinguishable": not (lo <= 0 <= hi)}


pd.DataFrame([paired(best, m) for m in table["method"][1:]]).round(3)"""),
    md("""## Does the enriched text help?
Per model: enriched − raw, paired."""),
    code("""names = set(table["method"])
models = sorted(m for m in {n.rsplit("/", 1)[0] for n in names} if {f"{m}/enriched", f"{m}/raw"} <= names)
pd.DataFrame([{"model": m, **{k: v for k, v in paired(f"{m}/enriched", f"{m}/raw").items() if k != "vs"}}
              for m in models]).round(3)"""),
    md("""## Stable across decades and kinds of ad?
nDCG@10 of the leading methods by decade group and by kind of seed ad. Small groups (`n`) are uncertain."""),
    code("""top = list(table["method"][:4]) + ["bm25/enriched"]
s = scores[scores["name"].isin(top)].copy()
s["period"] = pd.cut(s["decade"], [0, 1869, 1889, 1918, 2000], labels=["1850–60s", "1870–80s", "1890–1910s", "after 1918"])
by_period = s.pivot_table(index="period", columns="name", values="ndcg@10", observed=True)[top].round(3)
by_period["n"] = s.groupby("period", observed=True)["query_id"].nunique()
by_label = s.pivot_table(index="label", columns="name", values="ndcg@10")[top].round(3)
by_label["n"] = s.groupby("label")["query_id"].nunique()
display(by_period)
by_label"""),
    md("""## Cost at scale
Vector size per model (from the stored embeddings) and storage for 18 million ads as float32. Embedding times per 50,000
ads are from the embedding pilot on DHinfra (2026-10-02)."""),
    code("""minutes = {"embeddinggemma-300m": 1.7, "bge-m3": 2.2, "jina-embeddings-v3": 2.9,
           "jina-embeddings-v4-text-retrieval": 13.0, "qwen3-embedding-8b": 21.2}
cost = []
for m, t in minutes.items():
    f = cfg.path("embeddings_dir") / "raw" / m / "chunk-00000.parquet"
    dim = pq.read_table(f)["vector"].type.list_size if f.exists() else None
    best_ndcg = table.loc[table["method"].str.startswith(m + "/"), "ndcg@10"].max()
    cost.append({"model": m, "dimensions": dim, "best ndcg@10": round(best_ndcg, 3), "min per 50k ads": t,
                 "hours for 18M ads": round(t * 18e6 / 5e4 / 60, 1),
                 "GB for 18M ads": round(18e6 * dim * 4 / 1e9) if dim else None})
pd.DataFrame(cost).sort_values("best ndcg@10", ascending=False)"""),
    md("""## Shortened qwen3-embedding-8b vectors
Added with `score --extend --models qwen3-embedding-8b@2048,...`: the stored vectors cut to their first dimensions
(Matryoshka). Each against the full model and against embeddinggemma, paired; storage for 18 million ads as float32.
Embedding time does not change: the model still reads every text in full."""),
    code("""full, gemma = "qwen3-embedding-8b/enriched", "embeddinggemma-300m/enriched"
cut = sorted([m for m in table["method"] if m.startswith("qwen3-embedding-8b@") and m.endswith("/enriched")],
             key=lambda m: -int(m.split("@")[1].split("/")[0]))
if not cut:
    print("no shortened variants yet: run python -m hisrag.eval score --extend --models qwen3-embedding-8b@1024,...")
else:
    rows = []
    for m in [full] + cut + [gemma]:
        dims = int(m.split("@")[1].split("/")[0]) if "@" in m else (4096 if m == full else 768)
        row = {"method": m, "dimensions": dims, "ndcg@10": scores.loc[scores["name"] == m, "ndcg@10"].mean(),
               "GB for 18M ads": round(18e6 * dims * 4 / 1e9)}
        if m != full:
            d = paired(full, m)
            row |= {"full − this": d["diff"], "lo": d["lo"], "hi": d["hi"], "distinguishable": d["distinguishable"]}
        if m != gemma:
            d = paired(m, gemma)
            row |= {"this − gemma": d["diff"], "lo ": d["lo"], "hi ": d["hi"], "beats gemma": d["lo"] > 0}
        rows.append(row)
    shortened = pd.DataFrame(rows).round(3)
    js = scores[scores["label"] == "job_search"].pivot_table(index="name", values="ndcg@10")["ndcg@10"]
    shortened["ndcg@10 job searches"] = shortened["method"].map(js).round(3)
    display(shortened)"""),
    md("""## Side by side: one question, the top 5 of two methods
Grades in brackets. Re-run with another `qid`."""),
    code("""def side_by_side(qid: int, a: str = best, b: str = "embeddinggemma-300m/enriched", k: int = 5):
    q = questions.set_index("query_id").loc[qid]
    print(q["question"], "\\n  criterion:", q["criterion"], "\\n")
    g = judgments.set_index(["query_id", "ad_id"])["grade"]
    from hisrag.data import query
    texts = query("SELECT ad_id, text_norm FROM ad_clean WHERE ad_id IN (SELECT unnest(?))",
                  [list(runs.loc[runs["query_id"] == qid, "ad_id"].unique())]).set_index("ad_id")["text_norm"]
    for name in (a, b):
        m, v = name.rsplit("/", 1)
        r = runs[(runs["query_id"] == qid) & (runs["method"] == m) & (runs["variant"] == v)].sort_values("rank").head(k)
        print(f"--- {name}")
        for row in r.itertuples():
            print(f"  [{g.get((qid, row.ad_id), '?')}] {texts.get(row.ad_id, '')[:160]}")


side_by_side(int(questions["query_id"].iloc[0]))"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/06_retrieval_comparison.ipynb")
