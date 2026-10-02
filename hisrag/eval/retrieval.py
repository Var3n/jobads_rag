"""Step 8: which retrieval answers research questions best?

There is no evaluation set yet, so the LLM builds one in two steps (pooling, as in TREC):

1. From a stratified sample of seed ads it writes research questions a historian would put to the
   collection, each with a relevance criterion; the seed ad is one of several relevant ads.
2. Every method returns its top 10 for every question; the LLM judges each pooled ad against the
   question and criterion without knowing which method found it (2 relevant, 1 partly, 0 not).

Compared: BM25 on spelling-folded words, every configured embedding model, and a hybrid of BM25 and
each model (reciprocal rank fusion), each on the ad as printed ("raw") and on the ad plus the
normalized fields of steps 4–6 ("enriched"). Measures: precision@10 (grade 2), nDCG@10 and recall
against all relevant ads found in the pool.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

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
DEPTH = 10            # results per method and question that are pooled and judged
RRF_K = 60

# ------------------------------------------------------------------ documents


def documents(cfg: Config | None = None) -> pd.DataFrame:
    """All searchable ads in a fixed order (ad_id), with both text variants."""
    d = query("""SELECT ad_id, dup_cluster_id, decade, label, countable, lang, heading_text, text_norm,
                        position_lemmas IS NOT NULL AS has_position,
                        position_modern, requirements, pay_min, pay_max, pay_currency, pay_period
                 FROM ad_clean WHERE searchable ORDER BY ad_id""", cfg=cfg)
    head = d["heading_text"].map(normalize_text)
    d["raw"] = [f"{h}\n{t}" if h else t for h, t in zip(head, d["text_norm"])]
    d["enriched"] = [f"{raw}\n{_fields(r)}".rstrip() for raw, r in zip(d["raw"], d.itertuples())]
    return d[["ad_id", "dup_cluster_id", "decade", "label", "countable", "lang", "has_position", "raw", "enriched"]]


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


# ------------------------------------------------------------------ research questions


def sample_seeds(docs: pd.DataFrame, n: int = 300, seed: int = 0, min_chars: int = 120) -> pd.DataFrame:
    """Countable German ads that name a position and are long enough to be about something (no notice
    tails, no loan or sale ads), spread over decades in proportion to the square root of their size."""
    pool = docs[docs["countable"] & docs["has_position"] & (docs["lang"] == "de")
                & (docs["raw"].str.len() >= min_chars)]
    sizes = pool.groupby("decade").size()
    weights = np.sqrt(sizes)
    alloc = (n * weights / weights.sum()).round().astype(int).clip(lower=1)
    parts = [pool[pool["decade"] == dec].sample(min(k, sizes[dec]), random_state=seed) for dec, k in alloc.items()]
    return pd.concat(parts).sort_values("ad_id").reset_index(drop=True)


class QuestionItem(BaseModel):
    i: int
    question: str
    criterion: str


class QuestionBatch(BaseModel):
    items: list[QuestionItem]


QUESTION_PROMPT = """Du hilfst, eine Suchmaschine für historische Stellenanzeigen der Wiener Zeitung (1850–1950) zu testen. Historikerinnen und Wirtschaftshistoriker stellen ihr Forschungsfragen in modernem Deutsch.
Zu jeder Anzeige schreibst du eine solche Forschungsfrage, für die diese Anzeige EINER VON VIELEN relevanten Treffern wäre, und ein Relevanzkriterium.

Die Frage:
- fragt nach einer Gruppe von Anzeigen, nicht nach dieser einen: nach einem Beruf oder einer Berufsgruppe und einem Aspekt (Anforderungen, Bedingungen, Lohn oder Naturalleistungen, Art der Anzeige, Lebensumstände der Bewerber), z. B. "Welche Sprachkenntnisse wurden von Gouvernanten verlangt?", "Stellengesuche von Gärtnern, die bei einer Herrschaft unterkommen wollten", "Lehrerstellen an Volksschulen mit freier Wohnung";
- verwendet heutige Begriffe und eigene Worte; seltene oder auffällige Wörter der Anzeige werden nicht übernommen;
- enthält keine Jahreszahlen oder Daten (der Zeitraum wird getrennt gefiltert), keine Namen von Personen oder Firmen, keine Adressen und keine Orte unterhalb eines Kronlands oder Bundeslands (Wien ist erlaubt);
- hat 5 bis 15 Wörter.
Das Kriterium sagt in einem Satz, was JEDE relevante Anzeige erfüllen muss (z. B. "Die Anzeige bietet eine Lehrerstelle an einer Volksschule an und nennt eine freie Wohnung oder Dienstwohnung."). Es ist genau so allgemein wie die Frage: es nennt keine Einzelheiten der Beispielanzeige, die nicht in der Frage stehen (keinen bestimmten Ort, keine bestimmte Institution, keinen engeren Beruf, keinen Betrag). Fragt die Frage nach Handwerkern, verlangt das Kriterium einen Handwerker, nicht einen Maurer.
Antworte mit einem JSON-Objekt {"items": [...]} mit genau einem Element pro Anzeige, "i" = Nummer der Anzeige."""


def _question_messages(batch: pd.DataFrame) -> list[dict]:
    ads = "\n\n".join(f"{n}. ({r.label}) {r.raw[:1500]}" for n, r in enumerate(batch.itertuples(), 1))
    return [{"role": "system", "content": QUESTION_PROMPT}, {"role": "user", "content": "Anzeigen:\n\n" + ads}]


def write_questions(client: DHClient, seeds: pd.DataFrame, *, batch_size: int = 10,
                    progress: bool = True) -> tuple[pd.DataFrame, dict]:
    rows = seeds.assign(key=seeds["ad_id"])
    results, stats = run_batches(client, rows, job="eval_questions", batch_size=batch_size, progress=progress,
                                 build_messages=_question_messages, result_model=QuestionBatch,
                                 empty_item=lambda: QuestionItem(i=1, question="", criterion=""))
    out = seeds[["ad_id", "dup_cluster_id", "decade", "label", "raw"]].rename(
        columns={"ad_id": "seed_ad_id", "dup_cluster_id": "seed_cluster_id"})
    out["question"] = [results[a].question.strip() if a in results else "" for a in seeds["ad_id"]]
    out["criterion"] = [results[a].criterion.strip() if a in results else "" for a in seeds["ad_id"]]
    out["leakage"] = [verbatim_share(q, t) for q, t in zip(out["question"], out["raw"])]
    out = out[out["question"] != ""].reset_index(drop=True)
    out.insert(0, "query_id", range(len(out)))
    return out, stats


def verbatim_share(q: str, text: str) -> float:
    """Share of the question's longer words (≥ 6 letters) that occur verbatim in the seed ad."""
    words = re.findall(r"[^\W\d_]{6,}", q.lower())
    if not words:
        return 0.0
    t = text.lower()
    return round(sum(w in t for w in words) / len(words), 2)


# ------------------------------------------------------------------ BM25

_STOP = set("""der die das den dem des ein eine einer eines einem einen und oder mit von zu zur zum im in an am auf
für bei als auch aus nach wird werden ist sind hat haben sich nicht nur wie so wo welche welcher welches wurden wurde
gab gibt suchten sucht gesucht suche anzeige anzeigen inserat inserate""".split())
_SUFFIX = re.compile(r"(?:ern|em|en|er|es|e|n|s)$")


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", fold_spelling(text)):
        if len(w) < 2 or w in _STOP:
            continue
        out.append(_SUFFIX.sub("", w) if len(w) > 5 else w)
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

    def search(self, q: str, k: int = DEPTH) -> np.ndarray:
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


def dense_search(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int = DEPTH) -> list[np.ndarray]:
    out = []
    for i in range(0, len(query_vectors), 64):
        scores = query_vectors[i:i + 64] @ doc_vectors.T
        out += [_top(s, k) for s in scores]
    return out


def rrf(*rankings: np.ndarray, k: int = DEPTH) -> np.ndarray:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
    return np.array(sorted(score, key=score.get, reverse=True)[:k])


def query_vectors(client: DHClient, cfg: Config, texts: list[str], model: str) -> np.ndarray:
    """Question embeddings, stored per model: embeddings are not in the response cache, and tiny float
    differences between calls would change the top 10 and with it the pool to judge."""
    path = cfg.path("embeddings_dir") / "questions" / f"{model}.parquet"
    stored = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["text", "vector"])
    known = dict(zip(stored["text"], stored["vector"]))
    missing = [t for t in dict.fromkeys(texts) if t not in known]
    if missing:
        for t, v in zip(missing, client.embed(missing, model, kind="query")):
            known[t] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": list(known), "vector": [np.asarray(v, dtype=np.float32) for v in known.values()]}
                     ).to_parquet(path, index=False)
    return np.stack([np.asarray(known[t], dtype=np.float32) for t in texts])


def split_model(name: str) -> tuple[str, int | None]:
    """"qwen3-embedding-8b@1024" → ("qwen3-embedding-8b", 1024): the stored vectors cut to their first
    1024 dimensions (Matryoshka-trained models keep most of their quality there)."""
    base, _, dims = name.partition("@")
    return base, int(dims) if dims else None


def truncate(vectors: np.ndarray, dims: int | None) -> np.ndarray:
    if not dims:
        return vectors
    v = np.ascontiguousarray(vectors[:, :dims])
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


def run_methods(questions: pd.DataFrame, docs: pd.DataFrame, cfg: Config, client: DHClient,
                models: list[str], *, variants: tuple[str, ...] = VARIANTS, bm25: bool = True,
                hybrids: bool = True, progress: bool = True) -> pd.DataFrame:
    """One row per (question, method, variant, rank) with the ad found there (top DEPTH). Model names may
    carry a dimension ("qwen3-embedding-8b@1024"); hybrids need bm25."""
    runs: dict[str, list[np.ndarray]] = {}
    hybrid_depth = 50  # the hybrid fuses deeper lists than it returns
    for v in variants if bm25 else ():
        bm = BM25(docs[v].tolist())
        runs[f"bm25/{v}"] = [bm.search(q, hybrid_depth) for q in questions["question"]]
    loaded: dict[tuple[str, str], np.ndarray] = {}  # full vectors of the last base model, reused per dimension
    for model in models:
        base, dims = split_model(model)
        qv = None
        for v in variants:
            if (base, v) not in loaded:
                loaded.clear()
                full = load_vectors(cfg, v, base, docs)
                if full is None:
                    if progress:
                        print(f"  skip {v}/{model}: not fully embedded", flush=True)
                    continue
                loaded[(base, v)] = full
            if qv is None:
                qv = truncate(query_vectors(client, cfg, questions["question"].tolist(), base), dims)
            runs[f"{model}/{v}"] = dense_search(truncate(loaded[(base, v)], dims), qv, hybrid_depth)
            if hybrids and bm25:
                runs[f"hybrid:{model}/{v}"] = [rrf(a, b) for a, b in zip(runs[f"bm25/{v}"], runs[f"{model}/{v}"])]
    ad_ids, clusters = docs["ad_id"].to_numpy(), docs["dup_cluster_id"].to_numpy()
    rows = []
    for method, rankings in runs.items():
        name, variant = method.rsplit("/", 1)
        for qid, ranking in zip(questions["query_id"], rankings):
            # one result per printing cluster, as the index will show it: reprints must not count twice
            seen, kept = set(), []
            for doc in ranking:
                if clusters[doc] not in seen:
                    seen.add(clusters[doc])
                    kept.append(doc)
            for rank, doc in enumerate(kept[:DEPTH], 1):
                rows.append({"query_id": qid, "method": name, "variant": variant, "rank": rank,
                             "ad_id": ad_ids[doc], "cluster": clusters[doc]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ relevance judgments


class Judgment(BaseModel):
    i: int
    reason: str  # before the grade, so the model states what it sees before it grades
    grade: Literal[0, 1, 2]


class JudgmentBatch(BaseModel):
    items: list[Judgment]


JUDGE_PROMPT = """Du beurteilst, ob historische Stellenanzeigen der Wiener Zeitung (1850–1950) für eine Forschungsfrage relevant sind.
Zur Frage gibt es ein Relevanzkriterium. Für jede Anzeige schreibst du zuerst "reason": höchstens 12 Wörter dazu, was in der Anzeige für oder gegen das Kriterium spricht; dann "grade":
- 2: die Anzeige erfüllt das Kriterium klar;
- 1: teilweise oder am Rande. Immer 1 (nicht 0), wenn der Beruf passt, aber der gefragte Aspekt fehlt, oder wenn der Aspekt passt, aber zu einem verwandten Beruf;
- 0: nicht relevant.
Frage und Kriterium verwenden heutige Begriffe, die Anzeigen historische: zeitgenössische Entsprechungen zählen als Erfüllung (eine Köchin, Magd oder ein Stubenmädchen ist eine Hausgehilfin; ein Commis ist ein Handelsangestellter; "mit guten Zeugnissen versehen" sind gute Zeugnisse).
Beurteile nur den Text der Anzeige. OCR-Fehler, alte Schreibung, Abkürzungen und Bruchstücke sind normal; die Anzeige darf in einer anderen Sprache sein.
Antworte mit einem JSON-Objekt {"items": [...]} mit genau einem Element pro Anzeige, "i" = Nummer der Anzeige."""


def pool(runs: pd.DataFrame, questions: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Distinct (question, ad) pairs found by any method, sorted by question."""
    p = runs[["query_id", "ad_id", "cluster"]].drop_duplicates().sort_values(["query_id", "ad_id"])
    p = p.merge(questions[["query_id", "question", "criterion"]], on="query_id")
    p = p.merge(docs[["ad_id", "raw"]], on="ad_id")
    p["key"] = p["query_id"].astype(str) + "|" + p["ad_id"]
    return p.reset_index(drop=True)


def _judge_messages(batch: pd.DataFrame) -> list[dict]:
    q = batch.iloc[0]  # one question per batch (judge() groups by question)
    ads = "\n\n".join(f"{n}. {r.raw[:1200]}" for n, r in enumerate(batch.itertuples(), 1))
    return [{"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Frage: {q['question']}\nKriterium: {q['criterion']}\n\nAnzeigen:\n\n{ads}"}]


def judge(client: DHClient, pooled: pd.DataFrame, *, batch_size: int = 21,
          progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """One question per request: in the pilot a request mixing questions, or simply a long one, could
    lose track and grade a whole run of fitting ads 0."""
    results, stats = run_batches(client, pooled, job="eval_judge", batch_size=batch_size, progress=progress,
                                 group_by="query_id", build_messages=_judge_messages, result_model=JudgmentBatch,
                                 empty_item=lambda: Judgment(i=1, reason="", grade=0))
    out = pooled[["query_id", "ad_id", "cluster"]].copy()
    out["grade"] = [results[k].grade if k in results else None for k in pooled["key"]]
    out["reason"] = [results[k].reason if k in results else None for k in pooled["key"]]
    return out.astype({"grade": "Int8"}), stats


# ------------------------------------------------------------------ measures


def scores(runs: pd.DataFrame, judgments: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Per (question, method, variant): precision@10 (grade 2), nDCG@10 (gains 0/1/3), recall against all
    ads judged relevant for the question in the pool, and whether the seed ad (or a reprint) was found."""
    r = runs.merge(judgments[["query_id", "ad_id", "grade"]], on=["query_id", "ad_id"], how="left")
    r["grade"] = r["grade"].fillna(0).astype(int)
    gain = {0: 0.0, 1: 1.0, 2: 3.0}
    # relevance per printing cluster: a method can return only one printing of an ad
    per_cluster = (judgments.assign(grade=judgments["grade"].fillna(0).astype(int))
                            .groupby(["query_id", "cluster"])["grade"].max().reset_index())
    relevant = per_cluster[per_cluster["grade"] == 2].groupby("query_id").size()
    ideal = (per_cluster.assign(g=per_cluster["grade"].map(gain))
                        .sort_values("g", ascending=False).groupby("query_id")["g"]
                        .apply(lambda g: sum(v / math.log2(i + 2) for i, v in enumerate(g.head(DEPTH)))))
    seeds = questions.set_index("query_id")["seed_cluster_id"]
    rows = []
    for (qid, method, variant), g in r.groupby(["query_id", "method", "variant"]):
        g = g.sort_values("rank")
        dcg = sum(gain[x] / math.log2(rank + 1) for x, rank in zip(g["grade"], g["rank"]))
        n_rel = int(relevant.get(qid, 0))
        rows.append({"query_id": qid, "method": method, "variant": variant,
                     "p@10": (g["grade"] == 2).sum() / DEPTH,
                     "p@10_lenient": (g["grade"] >= 1).sum() / DEPTH,
                     "ndcg@10": dcg / ideal[qid] if ideal.get(qid, 0) > 0 else 0.0,
                     "recall": (g["grade"] == 2).sum() / n_rel if n_rel else np.nan,
                     "seed_found": bool((g["cluster"] == seeds[qid]).any())})
    return pd.DataFrame(rows)


def summary(per_query: pd.DataFrame) -> pd.DataFrame:
    return (per_query.groupby(["method", "variant"])
                     .agg(**{"p@10": ("p@10", "mean"), "p@10_lenient": ("p@10_lenient", "mean"),
                             "ndcg@10": ("ndcg@10", "mean"), "recall": ("recall", "mean"),
                             "seed_found": ("seed_found", "mean")})
                     .round(3).sort_values("ndcg@10", ascending=False))
