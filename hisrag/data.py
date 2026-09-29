"""Read access to the ad table. Everything downstream queries it through DuckDB."""

from __future__ import annotations

import duckdb
import pandas as pd

from hisrag.config import Config, load_config


def connect(cfg: Config | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with a view `ads` over the Parquet dataset."""
    cfg = cfg or load_config()
    ads_dir = cfg.path("ads_dir")
    if not ads_dir.exists():
        raise FileNotFoundError(f"{ads_dir} does not exist yet; run `python -m hisrag.ingest` first")
    con = duckdb.connect()
    pattern = (ads_dir / "**" / "*.parquet").as_posix().replace("'", "''")  # views cannot take parameters
    con.execute(f"CREATE VIEW ads AS SELECT * FROM read_parquet('{pattern}', hive_partitioning = true, "
                "union_by_name = true)")
    return con


def query(sql: str, params: list | None = None, cfg: Config | None = None) -> pd.DataFrame:
    """Run SQL against the `ads` view and return a DataFrame."""
    with connect(cfg) as con:
        return con.execute(sql, params or []).df()
