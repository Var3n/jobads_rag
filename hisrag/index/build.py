"""Step 9: the search index, one LanceDB table `ads` with one row per searchable ad.

Each row holds the ad's vector (the step-8 winner: qwen3-embedding-8b on the enriched text, cut to
`index.dims` dimensions and renormalized), the text as printed for display, the same text with historical
spellings folded for the keyword search, and the filter columns of `ad_clean`.

Built per newspaper: a newspaper's rows are replaced as a whole, so newspapers can be added one at a time.
Vectors come from the step-8 store (`data/embeddings/<variant>/<model>/`), looked up by ad_id; ads that are
not there yet (new newspapers) are embedded and stored next to them in `newspaper=<name>/`.

Indexes, rebuilt after every build over the whole table:
  * full-text (BM25) on the folded text, unstemmed and with word positions: the keyword mode matches word
    forms exactly ("Wirthschafterin" = "Wirtschafterin", but not "Wirtschaft"), phrases in quotes, and
    prefixes ("krakau*") expanded from a word list stored per newspaper (`vocab/<newspaper>.parquet`);
  * scalar indexes on the common filter columns;
  * an ANN vector index only from `index.ann_min_rows` rows on; below that the search is exact.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hisrag.config import Config
from hisrag.data import connect as data_connect
from hisrag.data import query
from hisrag.eval import retrieval as E
from hisrag.normalize.clean import BENEFITS
from hisrag.normalize.text import fold_spelling

# Read by lancedb's Rust logger when it is first imported. Full-text search warns on every query that `_score`
# is not selected, but selecting it is an error (lancedb 0.39) and the switch it names is not in Python.
os.environ.setdefault("LANCEDB_LOG", "warn,lance::dataset::scanner=error")

TABLE = "ads"
WORD =re.compile(r"[^\W_]+")  # words as the full-text index's simple tokenizer splits them

REQUIREMENT_TAGS_SQL = "list_filter(list_transform(requirements, r -> r.dimension || ':' || r.value), x -> x IS NOT NULL)"

COLUMNS_SQL = f"""
SELECT ad_id, newspaper, year, date, decade, label, lang, countable, is_canonical,
       dup_cluster_id, dup_cluster_size, run_first_date, run_last_date, quality_warning, iiif_link,
       position_terms, position_lemmas, position_modern, position_categories, position_gender,
       requirement_dimensions,
       {REQUIREMENT_TAGS_SQL} AS requirement_tags,
       pay_min, pay_max, pay_currency, pay_standard, pay_period,
       {", ".join(f"benefit_{b}" for b in BENEFITS)}
FROM ad_clean WHERE searchable AND newspaper = ? ORDER BY ad_id
"""

SCALAR_INDEXES = {"newspaper": "Bitmap", "label": "Bitmap", "decade": "Bitmap", "year": "BTree",
                  "countable": "Bitmap", "position_categories": "LabelList",
                  "requirement_dimensions": "LabelList", "requirement_tags": "LabelList"}


def connect(cfg: Config):
    import lancedb

    return lancedb.connect(cfg.path("index_dir"))


def newspapers(cfg: Config | None = None) -> list[str]:
    return query("SELECT DISTINCT newspaper FROM ad_clean ORDER BY 1", cfg=cfg)["newspaper"].tolist()


# ------------------------------------------------------------------ vectors


def _vector_dir(cfg: Config) -> Path:
    ix = cfg["index"]
    return cfg.path("embeddings_dir") / ix["variant"] / ix["model"]


def stored_vectors(cfg: Config, ad_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(found mask, vectors of the found ads in the order of ad_ids), cut to index.dims while reading so
    the full 4096-dim vectors never sit in memory all at once."""
    want = pd.Index(ad_ids)
    found = np.zeros(len(want), dtype=bool)
    out: np.ndarray | None = None
    d = _vector_dir(cfg)
    for f in sorted(d.rglob("chunk-*.parquet")) if d.exists() else []:
        t = pq.read_table(f)
        pos = want.get_indexer(t["ad_id"].to_pylist())
        keep = pos >= 0
        if not keep.any():
            continue
        flat = t["vector"].combine_chunks()
        v = E.truncate(flat.values.to_numpy().reshape(len(flat), -1)[keep].astype(np.float32), cfg["index"]["dims"])
        if out is None:
            out = np.zeros((len(want), v.shape[1]), dtype=np.float32)
        out[pos[keep]] = v
        found[pos[keep]] = True
    return found, out[found] if out is not None else np.zeros((0, cfg["index"]["dims"]), dtype=np.float32)


def embed_missing(client, cfg: Config, newspaper: str, docs: pd.DataFrame, *, progress: bool = True) -> dict:
    """Embed the ads of `docs` (ad_id + text variant) into newspaper=<name>/ chunks; resumable like step 8."""
    ix = cfg["index"]
    d = _vector_dir(cfg) / f"newspaper={newspaper}"
    seconds = 0.0
    n_chunks = math.ceil(len(docs) / E.CHUNK)
    for c in range(n_chunks):
        part = docs.iloc[c * E.CHUNK:(c + 1) * E.CHUNK]
        path = d / f"chunk-{c:05d}.parquet"
        if path.exists() and pq.read_table(path, columns=["ad_id"])["ad_id"].to_pylist() == part["ad_id"].tolist():
            continue
        t0 = time.monotonic()
        vectors = client.embed(part[ix["variant"]].tolist(), ix["model"], kind="passage")
        seconds += time.monotonic() - t0
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"ad_id": part["ad_id"].tolist(), "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.ravel()), vectors.shape[1])}), path)
        if progress:
            print(f"  embedded {newspaper} chunk {c + 1}/{n_chunks}", flush=True)
    return {"embedded": len(docs), "seconds": round(seconds, 1)}


# ------------------------------------------------------------------ rows


def rows(cfg: Config, newspaper: str, client=None, *, progress: bool = True) -> tuple[pa.Table, dict]:
    """The index rows of one newspaper: filter columns of ad_clean, display and folded text, vector."""
    with data_connect(cfg) as con:  # Arrow, not pandas: types must not depend on one newspaper's values
        cols = con.execute(COLUMNS_SQL, [newspaper]).arrow()
        cols = cols.read_all() if isinstance(cols, pa.RecordBatchReader) else cols
    docs = E.documents(cfg)
    docs = docs[docs["ad_id"].isin(set(cols["ad_id"].to_pylist()))].reset_index(drop=True)  # both by ad_id
    report: dict = {"ads": cols.num_rows}
    found, vectors = stored_vectors(cfg, docs["ad_id"].tolist())
    if not found.all():
        missing = docs[~found]
        if client is None:
            raise RuntimeError(f"{newspaper}: {len(missing)} of {len(docs)} ads have no stored "
                               f"{cfg['index']['model']} vector; build with a client to embed them")
        report["embedding"] = embed_missing(client, cfg, newspaper, missing, progress=progress)
        found, vectors = stored_vectors(cfg, docs["ad_id"].tolist())
        assert found.all()
    vec = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), vectors.shape[1])
    return (cols.append_column("text", pa.array(docs["raw"].tolist(), pa.string()))
                .append_column("text_folded", pa.array([fold_spelling(t) for t in docs["raw"]], pa.string()))
                .append_column("vector", vec)), report


def vocabulary(texts_folded: list[str]) -> pd.DataFrame:
    """Words of the folded texts with the number of ads they occur in, for prefix expansion."""
    df = Counter(w for t in texts_folded for w in set(WORD.findall(t)))
    return pd.DataFrame({"term": list(df), "df": list(df.values())}).sort_values("term", ignore_index=True)


# ------------------------------------------------------------------ build


def build(cfg: Config, newspaper_names: list[str] | None = None, client=None, *, progress: bool = True) -> dict:
    db = connect(cfg)
    report: dict = {}
    t_start = time.monotonic()
    for name in newspaper_names or newspapers(cfg):
        t0 = time.monotonic()
        table, r = rows(cfg, name, client, progress=progress)
        r["rows_seconds"] = round(time.monotonic() - t0, 1)
        t0 = time.monotonic()
        if TABLE in db.list_tables().tables:
            tb = db.open_table(TABLE)
            tb.delete(f"newspaper = '{name.replace(chr(39), chr(39) * 2)}'")
            tb.add(table)
        else:
            db.create_table(TABLE, table)
        vocab_dir = cfg.path("index_dir") / "vocab"
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocabulary(table["text_folded"].to_pylist()).to_parquet(vocab_dir / f"{name}.parquet", index=False)
        r["write_seconds"] = round(time.monotonic() - t0, 1)
        report[name] = r
        if progress:
            print(f"  {name}: {r['ads']} ads", flush=True)
    tb = db.open_table(TABLE)
    tb.optimize(cleanup_older_than=timedelta(0))  # compact and drop the replaced rows' old versions
    report["indexes"] = create_indexes(cfg, tb)
    size = sum(f.stat().st_size for f in cfg.path("index_dir").rglob("*") if f.is_file())
    n = tb.count_rows()
    report |= {"total_rows": n, "total_seconds": round(time.monotonic() - t_start, 1),
               "size_mb": round(size / 2**20, 1), "bytes_per_ad": round(size / max(n, 1))}
    return report


def create_indexes(cfg: Config, tb) -> dict:
    from lancedb import index as I

    seconds = {}
    configs = {"text_folded": I.FTS(with_position=True, stem=False, remove_stop_words=False, ascii_folding=True)}
    configs |= {col: getattr(I, kind)() for col, kind in SCALAR_INDEXES.items()}
    if tb.count_rows() >= cfg["index"]["ann_min_rows"]:
        configs["vector"] = getattr(I, cfg["index"]["ann_index_type"])(distance_type="cosine")
    for col, config in configs.items():
        t0 = time.monotonic()
        tb.create_index(col, config=config, replace=True)
        seconds[col] = round(time.monotonic() - t0, 1)
    return seconds
