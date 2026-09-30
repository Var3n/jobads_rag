"""Step 8: which retrieval finds a known ad best?

There is no evaluation set yet, so the LLM writes one: for a stratified sample of countable ads it
writes the search question a researcher would ask in modern German to find that ad again (known-item
queries). Each method ranks all searchable ads; a hit is the target ad or any printing of it
(same `dup_cluster_id`). Compared: BM25 on spelling-folded words, every configured embedding model,
and a hybrid of BM25 and each model (reciprocal rank fusion). Two document texts: the ad as printed
("raw") and the ad plus the normalized fields of steps 4–6 ("enriched").
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from hisrag.config import Config
from hisrag.data import query
from hisrag.llm.client import DHClient
from hisrag.normalize.batching import run_batches
from hisrag.normalize.text import fold_spelling, normalize_text

VARIANTS = ("raw", "enriched")
CHUNK = 1024          # documents per stored embedding file; the pilot embeds chunk 0
TOP_K = 100           # ranks kept per query and method
RRF_K = 60
KS = (1, 5, 10, 50)

# ------------------------------------------------------------------ documents


def documents(cfg: Config | None = None) -> pd.DataFrame:
    """All searchable ads in a fixed order (ad_id), with both text variants."""
    d = query("""SELECT ad_id, dup_cluster_id, decade, label, countable, heading_text, text_norm,
                        position_modern, requirements, pay_min, pay_max, pay_currency, pay_period
                 FROM ad_clean WHERE searchable ORDER BY ad_id""", cfg=cfg)
    head = d["heading_text"].map(normalize_text)
    d["raw"] = [f"{h}\n{t}" if h else t for h, t in zip(head, d["text_norm"])]
    d["enriched"] = [f"{raw}\n{_fields(r)}".rstrip() for raw, r in zip(d["raw"], d.itertuples())]
    return d[["ad_id", "dup_cluster_id", "decade", "label", "countable", "raw", "enriched"]]


def _listish(x) -> list:
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else []  # None, NaN or pd.NA for a missing list


def _fields(r) -> str:
    """The normalized fields in modern German, appended to the ad for the "enriched" variant."""
    parts = []
    if positions := [p for p in _listish(r.position_modern) if p]:
        parts.append("Stelle: " + ", ".join(positions))
    if reqs := _listish(r.requirements):
        values = list(dict.fromkeys(t["value"] for t in reqs if t["value"]))[:15]
        parts.append("Anforderungen: " + ", ".join(values))
    if r.pay_min == r.pay_min and r.pay_min is not None:  # not NaN
        amount = f"{r.pay_min:g}" + (f"–{r.pay_max:g}" if r.pay_max != r.pay_min else "")
        period = {"jahr": "jährlich", "monat": "monatlich", "woche": "wöchentlich", "tag": "täglich",
                  "stunde": "pro Stunde"}.get(r.pay_period, "")
        parts.append(f"Lohn: {amount} {r.pay_currency} {period}".rstrip())
    return "\n".join(parts)


# ------------------------------------------------------------------ test questions


def sample_targets(docs: pd.DataFrame, n: int = 300, seed: int = 0, min_chars: int = 80) -> pd.DataFrame:
    """Countable ads, spread over decades in proportion to the square root of their size, so thin
    decades are represented without dominating."""
    pool = docs[docs["countable"] & (docs["raw"].str.len() >= min_chars)]
    sizes = pool.groupby("decade").size()
    weights = np.sqrt(sizes)
    alloc = (n * weights / weights.sum()).round().astype(int).clip(lower=1)
    parts = [pool[pool["decade"] == dec].sample(min(k, sizes[dec]), random_state=seed) for dec, k in alloc.items()]
    return pd.concat(parts).sort_values("ad_id").reset_index(drop=True)


class QueryItem(BaseModel):
    i: int
    query: str


class QueryBatch(BaseModel):
    items: list[QueryItem]


QUERY_PROMPT = """Du hilfst, eine Suchmaschine für historische Stellenanzeigen der Wiener Zeitung (1850–1950) zu testen.
Zu jeder Anzeige schreibst du eine Suchanfrage, wie eine Historikerin sie stellen würde, die sich an diese Anzeige ungefähr erinnert und sie wiederfinden will.
Regeln:
- Modernes Deutsch mit modernen Begriffen und moderner Schreibung, 6 bis 15 Wörter, als Suchanfrage oder kurze Frage.
- Nenne 2 bis 4 inhaltliche Merkmale der Anzeige: Stelle oder Tätigkeit, Anforderungen, Bedingungen, Art der Anzeige (Angebot, Gesuch, Vermittlung), Ort höchstens grob (Wien, Böhmen, Land).
- Keine Jahreszahlen oder Daten, keine Namen von Personen oder Firmen, keine Adressen, keine genauen Beträge.
- Übernimm keine seltenen Wörter oder Wendungen wörtlich aus der Anzeige, sondern umschreibe sie so, wie jemand heute sucht.
- Die Anfrage muss zu dieser Anzeige passen und sie von anderen Anzeigen derselben Art unterscheiden können.
Antworte mit einem JSON-Objekt {"items": [...]} mit genau einem Element pro Anzeige, "i" = Nummer der Anzeige."""


def _query_messages(batch: pd.DataFrame) -> list[dict]:
    ads = "\n\n".join(f"{n}. ({r.label}) {r.raw[:1500]}" for n, r in enumerate(batch.itertuples(), 1))
    return [{"role": "system", "content": QUERY_PROMPT}, {"role": "user", "content": "Anzeigen:\n\n" + ads}]


def write_queries(client: DHClient, targets: pd.DataFrame, *, batch_size: int = 10,
                  progress: bool = True) -> tuple[pd.DataFrame, dict]:
    rows = targets.assign(key=targets["ad_id"])
    results, stats = run_batches(client, rows, job="eval_queries", batch_size=batch_size, progress=progress,
                                 build_messages=_query_messages, result_model=QueryBatch,
                                 empty_item=lambda: QueryItem(i=1, query=""))
    out = targets[["ad_id", "dup_cluster_id", "decade", "label", "raw"]].copy()
    out["query"] = [results[a].query.strip() if a in results else "" for a in out["ad_id"]]
    out["leakage"] = [verbatim_share(q, t) for q, t in zip(out["query"], out["raw"])]
    out = out[out["query"] != ""].reset_index(drop=True)
    out.insert(0, "query_id", range(len(out)))
    return out, stats


def verbatim_share(q: str, text: str) -> float:
    """Share of the query's longer words (≥ 6 letters) that occur verbatim in the ad: high values
    make the question easy for keyword search."""
    words = [w for w in re.findall(r"[^\W\d_]{6,}", q.lower())]
    if not words:
        return 0.0
    t = text.lower()
    return round(sum(w in t for w in words) / len(words), 2)


# ------------------------------------------------------------------ BM25

_STOP = set("""der die das den dem des ein eine einer eines einem einen und oder mit von zu zur zum im in an am auf
für bei als auch aus nach wird werden ist sind hat haben sich nicht nur wie so wo welche welcher welches sucht gesucht
suche stelle anzeige inserat""".split())
_SUFFIX = re.compile(r"(?:ern|em|en|er|es|e|n|s)$")


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", fold_spelling(text))
    out = []
    for w in words:
        if len(w) < 2 or w in _STOP:
            continue
        stem = _SUFFIX.sub("", w) if len(w) > 5 else w
        out.append(stem)
    return out


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, t in enumerate(texts):
            tf = Counter(tokens(t))
            lengths[i] = sum(tf.values())
            for term, c in tf.items():
                postings.setdefault(term, []).append((i, c))
        self.norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1))
        self.index = {t: (np.array([p[0] for p in ps]), np.array([p[1] for p in ps], dtype=np.float32))
                      for t, ps in postings.items()}

    def search(self, q: str, k: int = TOP_K) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokens(q)):
            if term not in self.index:
                continue
            docs, tf = self.index[term]
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += idf * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return _top(scores, k)


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


# ------------------------------------------------------------------ embeddings (stored in chunks, resumable)


def _chunk_path(cfg: Config, variant: str, model: str, n: int) -> Path:
    return cfg.path("embeddings_dir") / variant / model / f"chunk-{n:05d}.parquet"


def embed_documents(client: DHClient, docs: pd.DataFrame, variant: str, model: str, cfg: Config, *,
                    chunks: list[int] | None = None, progress: bool = True) -> dict:
    """Embed the documents chunk by chunk; existing chunks with the same ads are skipped."""
    n_chunks = math.ceil(len(docs) / CHUNK)
    done, seconds, embedded = 0, 0.0, 0
    for c in chunks if chunks is not None else range(n_chunks):
        part = docs.iloc[c * CHUNK:(c + 1) * CHUNK]
        path = _chunk_path(cfg, variant, model, c)
        if path.exists() and pq.read_table(path, columns=["ad_id"])["ad_id"].to_pylist() == part["ad_id"].tolist():
            done += 1
            continue
        t0 = time.monotonic()
        vectors = client.embed(part[variant].tolist(), model, kind="passage", progress=False)
        seconds += time.monotonic() - t0
        embedded += len(part)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({"ad_id": part["ad_id"].tolist(),
                          "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), vectors.shape[1])})
        pq.write_table(table, path)
        if progress:
            print(f"  {variant}/{model} chunk {c + 1}/{n_chunks}: {len(part)} ads in {time.monotonic() - t0:.0f} s",
                  flush=True)
    return {"chunks": n_chunks, "skipped_existing": done, "embedded": embedded, "seconds": round(seconds, 1)}


def load_vectors(cfg: Config, variant: str, model: str, docs: pd.DataFrame) -> np.ndarray | None:
    """Vectors in the order of `docs`, or None if the model is not fully embedded."""
    d = cfg.path("embeddings_dir") / variant / model
    files = sorted(d.glob("chunk-*.parquet")) if d.exists() else []
    if len(files) != math.ceil(len(docs) / CHUNK):
        return None
    table = pa.concat_tables(pq.read_table(f) for f in files)
    if table["ad_id"].to_pylist() != docs["ad_id"].tolist():
        return None
    flat = table["vector"].combine_chunks()
    return flat.values.to_numpy().reshape(len(flat), -1).astype(np.float32)


def dense_search(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int = TOP_K) -> list[np.ndarray]:
    out = []
    for i in range(0, len(query_vectors), 64):
        scores = query_vectors[i:i + 64] @ doc_vectors.T
        out += [_top(s, k) for s in scores]
    return out


def rrf(*rankings: np.ndarray, k: int = TOP_K) -> np.ndarray:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
    return np.array(sorted(score, key=score.get, reverse=True)[:k])


# ------------------------------------------------------------------ scoring


def hit_rank(ranking: np.ndarray, clusters: np.ndarray, target: str) -> int | None:
    """1-based rank of the first result in the target's printing cluster, None if not in the top k."""
    hits = np.nonzero(clusters[ranking] == target)[0]
    return int(hits[0]) + 1 if len(hits) else None


def metrics(ranks: pd.Series) -> dict:
    r = ranks.astype("float")
    out = {f"recall@{k}": round(float((r <= k).mean()), 3) for k in KS}
    out["mrr"] = round(float((1 / r).fillna(0).mean()), 3)
    return out


def evaluate(queries: pd.DataFrame, docs: pd.DataFrame, cfg: Config, client: DHClient,
             models: list[str], progress: bool = True) -> pd.DataFrame:
    """One row per (query, method) with the rank of the target (None = not in the top 100)."""
    clusters = docs["dup_cluster_id"].to_numpy()
    runs: dict[str, list[np.ndarray]] = {}
    bm25 = {v: BM25(docs[v].tolist()) for v in VARIANTS}
    for v in VARIANTS:
        runs[f"bm25/{v}"] = [bm25[v].search(q) for q in queries["query"]]
    for model in models:
        qv = None
        for v in VARIANTS:
            vectors = load_vectors(cfg, v, model, docs)
            if vectors is None:
                if progress:
                    print(f"  skip {v}/{model}: not fully embedded", flush=True)
                continue
            if qv is None:
                qv = client.embed(queries["query"].tolist(), model, kind="query")
            runs[f"{model}/{v}"] = dense_search(vectors, qv)
            runs[f"hybrid:{model}/{v}"] = [rrf(a, b) for a, b in zip(runs[f"bm25/{v}"], runs[f"{model}/{v}"])]
    rows = []
    for method, rankings in runs.items():
        name, variant = method.rsplit("/", 1)
        for (_, q), ranking in zip(queries.iterrows(), rankings):
            rows.append({"query_id": q["query_id"], "method": name, "variant": variant,
                         "rank": hit_rank(ranking, clusters, q["dup_cluster_id"])})
    return pd.DataFrame(rows).astype({"rank": "Int32"})


def summary(results: pd.DataFrame) -> pd.DataFrame:
    return (results.groupby(["method", "variant"])["rank"].apply(metrics).unstack()
                   .sort_values("mrr", ascending=False))
