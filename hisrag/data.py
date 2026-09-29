"""Storage for the ad table and the tables derived from it.

Every table is a Hive-partitioned Parquet dataset (newspaper=…/year=…) keyed by `ad_id`:
`data/ads` from step 1, and one directory per later step under `data/derived/` (e.g.
`ad_text` from step 2). `connect()` exposes each as a DuckDB view of the same name.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from hisrag.config import Config, load_config

PARTITIONING = ds.partitioning(pa.schema([("newspaper", pa.string()), ("year", pa.int16())]), flavor="hive")


def _parquet_glob(directory: Path) -> str:
    return (directory / "**" / "*.parquet").as_posix().replace("'", "''")  # views cannot take parameters


def connect(cfg: Config | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with a view `ads` plus one view per derived table."""
    cfg = cfg or load_config()
    ads_dir = cfg.path("ads_dir")
    if not ads_dir.exists():
        raise FileNotFoundError(f"{ads_dir} does not exist yet; run `python -m hisrag.ingest` first")
    con = duckdb.connect()
    tables = {"ads": ads_dir}
    derived = cfg.path("derived_dir")
    if derived.exists():
        tables |= {d.name: d for d in sorted(derived.iterdir()) if d.is_dir() and any(d.rglob("*.parquet"))}
    for name, directory in tables.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{_parquet_glob(directory)}', "
                    "hive_partitioning = true, union_by_name = true)")
    return con


def query(sql: str, params: list | None = None, cfg: Config | None = None) -> pd.DataFrame:
    """Run SQL against the views from `connect()` and return a DataFrame."""
    with connect(cfg) as con:
        return con.execute(sql, params or []).df()


def write_partitioned(df: pd.DataFrame, out_dir: Path | str, schema: pa.Schema | None = None) -> None:
    """Write a frame with `newspaper` and `year` columns as a partitioned dataset.

    Re-running replaces only the partitions present in `df`, so data can be (re)processed one
    newspaper at a time.
    """
    table = pa.Table.from_pandas(df[schema.names] if schema else df, schema=schema, preserve_index=False)
    if table.schema.field("year").type != pa.int16():
        table = table.set_column(table.schema.get_field_index("year"), "year", table["year"].cast(pa.int16()))
    ds.write_dataset(table, out_dir, format="parquet", partitioning=PARTITIONING,
                     existing_data_behavior="delete_matching", basename_template="part-{i}.parquet")


def derived_dir(name: str, cfg: Config | None = None) -> Path:
    return (cfg or load_config()).path("derived_dir") / name
