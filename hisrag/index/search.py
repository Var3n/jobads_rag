"""Step 9: searching the index, in two modes.

  semantic(question)  the question embedded like the step-8 winner (query prefix from config.yaml, cut to
                      index.dims): finds ads by meaning, in modern German, across historical wording.
  keyword(words)      exact words in the folded text, ranked by BM25: for names, places and fixed terms. All
                      words must occur; "…" is a phrase, word* a prefix (expanded from the index's word list),
                      -word excludes. Historical spellings match their modern form (Wirthschafterin,
                      Correspondent, Clavier), inflected forms do not ("Krakau" misses "Krakauer": krakau*).

Both take `Filters` and return one ad per printing cluster (the first printing found), best first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from hisrag.config import Config, load_config
from hisrag.eval.retrieval import truncate
from hisrag.index.build import TABLE, WORD, connect
from hisrag.normalize.clean import BENEFITS
from hisrag.normalize.text import fold_spelling

DISPLAY = ["ad_id", "newspaper", "date", "year", "label", "position_modern", "text", "iiif_link", "quality_warning",
           "countable", "dup_cluster_id", "dup_cluster_size", "run_first_date", "run_last_date"]
MAX_EXPANSIONS = 100  # words a prefix may stand for (the most frequent ones)


def _lit(value) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(int(value))


def _in(col: str, values) -> str:
    return f"{col} IN ({', '.join(_lit(v) for v in values)})"


def _any(col: str, values) -> str:
    return f"array_has_any({col}, [{', '.join(_lit(v) for v in values)}])"


@dataclass
class Filters:
    """Restrictions on the ads searched; list fields match any of their values. Requirement tags are
    "dimension:value" as in ad_clean ("sprachkenntnisse:Böhmisch"). `where` adds raw SQL (LanceDB dialect)."""
    newspapers: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    labels: list[str] | None = None            # job_offer, job_search, service_offer, agency
    langs: list[str] | None = None
    countable_only: bool = False               # canonical printings without invented text
    position_categories: list[str] | None = None
    position_lemmas: list[str] | None = None
    position_gender: list[str] | None = None   # f, m, mixed, neutral
    requirement_dimensions: list[str] | None = None
    requirement_tags: list[str] | None = None
    has_pay: bool | None = None
    benefits: list[str] = field(default_factory=list)  # all of them: wohnung, kost, …
    where: str | None = None

    def sql(self) -> str | None:
        c = []
        if self.newspapers:
            c.append(_in("newspaper", self.newspapers))
        if self.year_from is not None:
            c.append(f"year >= {int(self.year_from)}")
        if self.year_to is not None:
            c.append(f"year <= {int(self.year_to)}")
        if self.labels:
            c.append(_in("label", self.labels))
        if self.langs:
            c.append(_in("lang", self.langs))
        if self.countable_only:
            c.append("countable")
        if self.position_gender:
            c.append(_in("position_gender", self.position_gender))
        for col in ("position_categories", "position_lemmas", "requirement_dimensions", "requirement_tags"):
            if values := getattr(self, col):
                c.append(_any(col, values))
        if self.has_pay is not None:
            c.append("pay_min IS NOT NULL" if self.has_pay else "pay_min IS NULL")
        for b in self.benefits:
            if b not in BENEFITS:
                raise ValueError(f"unknown benefit {b!r}; known: {', '.join(BENEFITS)}")
            c.append(f"benefit_{b}")
        if self.where:
            c.append(f"({self.where})")
        return " AND ".join(c) or None


# ------------------------------------------------------------------ keyword queries


@dataclass
class Part:
    occur: str            # MUST or MUST_NOT
    kind: str             # word, phrase, prefix
    words: list[str]      # folded


_PART = re.compile(r'(-?)"([^"]*)"|(-?)(\S+)')


def parse_keywords(q: str) -> list[Part]:
    parts = []
    for m in _PART.finditer(q):
        neg, text = (m.group(1), m.group(2)) if m.group(2) is not None else (m.group(3), m.group(4))
        words = WORD.findall(fold_spelling(text))
        if not words:
            continue
        occur = "MUST_NOT" if neg else "MUST"
        if m.group(2) is None and text.endswith("*") and len(words) == 1:
            parts.append(Part(occur, "prefix", words))
        else:
            parts.append(Part(occur, "phrase" if len(words) > 1 else "word", words))
    return parts


# ------------------------------------------------------------------ the index


class AdIndex:
    def __init__(self, cfg: Config | None = None, client=None):
        self.cfg = cfg or load_config()
        self._client = client
        self.table = connect(self.cfg).open_table(TABLE)

    @cached_property
    def client(self):
        if self._client is None:
            from hisrag.llm import DHClient
            self._client = DHClient(self.cfg)
        return self._client

    @cached_property
    def vocab(self) -> pd.DataFrame:
        files = sorted((self.cfg.path("index_dir") / "vocab").glob("*.parquet"))
        v = pd.concat([pd.read_parquet(f) for f in files]) if files else pd.DataFrame({"term": [], "df": []})
        return v.groupby("term", as_index=False)["df"].sum().sort_values("term", ignore_index=True)

    def embed_query(self, question: str) -> np.ndarray:
        ix = self.cfg["index"]
        return truncate(self.client.embed([question], ix["model"], kind="query"), ix["dims"])[0]

    def semantic(self, question: str | None = None, filters: Filters | None = None, k: int = 10, *,
                 vector: np.ndarray | None = None) -> pd.DataFrame:
        """Ads closest in meaning to the question; `score` is the cosine similarity."""
        v = self.embed_query(question) if vector is None else vector

        def fetch():
            s = self.table.search(v, vector_column_name="vector").metric("cosine")
            return s.where(filters.sql(), prefilter=True) if filters and filters.sql() else s

        out = self._distinct(fetch, k)
        out.insert(1, "score", 1 - out.pop("_distance"))
        return out

    def expand(self, prefix: str) -> list[str]:
        terms = self.vocab["term"]
        lo, hi = terms.searchsorted(prefix), terms.searchsorted(prefix + "￿")
        hits = self.vocab.iloc[lo:hi]
        return hits.nlargest(MAX_EXPANSIONS, "df")["term"].tolist()

    def keyword(self, q: str, filters: Filters | None = None, k: int | None = 10) -> pd.DataFrame:
        """Ads containing the words, BM25 ranked; k=None returns every match. The frame's attrs say what
        each prefix was expanded to."""
        from lancedb.query import BooleanQuery, MatchQuery, Occur, PhraseQuery

        col = "text_folded"
        clauses, expanded = [], {}
        for p in parse_keywords(q):
            if p.kind == "prefix":
                terms = expanded[p.words[0] + "*"] = self.expand(p.words[0])
                if not terms:
                    if p.occur == "MUST":
                        return self._empty(expanded)
                    continue
                sub = BooleanQuery([(Occur.SHOULD, MatchQuery(t, col)) for t in terms])
            elif p.kind == "phrase":
                sub = PhraseQuery(" ".join(p.words), col)
            else:
                sub = MatchQuery(p.words[0], col)
            clauses.append((Occur[p.occur], sub))
        if not any(o == Occur.MUST for o, _ in clauses):
            raise ValueError(f"no word that must occur in {q!r}")
        fts = BooleanQuery(clauses)

        def fetch():
            s = self.table.search(fts)
            return s.where(filters.sql(), prefilter=True) if filters and filters.sql() else s

        out = self._distinct(fetch, k)
        out.insert(1, "score", out.pop("_score"))
        out.attrs["expanded"] = expanded
        return out

    def _empty(self, expanded: dict) -> pd.DataFrame:
        out = pd.DataFrame(columns=["ad_id", "score"] + DISPLAY[1:])
        out.attrs["expanded"] = expanded
        return out

    def _distinct(self, fetch, k: int | None) -> pd.DataFrame:
        """Top k with one ad per printing cluster: fetch more than k and widen until k clusters are found."""
        n_rows = self.table.count_rows()
        limit = n_rows if k is None else max(4 * k, 40)
        while True:
            hits = fetch().limit(limit).select(DISPLAY).to_pandas()
            distinct = hits.drop_duplicates("dup_cluster_id")
            if k is None or len(distinct) >= k or len(hits) < limit or limit >= n_rows:
                return distinct.head(k).reset_index(drop=True) if k else distinct.reset_index(drop=True)
            limit = min(limit * 4, n_rows)
