"""python -m hisrag.ingest [CSV ...]   (default: paths.raw_csv from the config)"""

import argparse
import json

from hisrag.config import load_config
from hisrag.ingest.extractions import read_extractions_csv, write_ads


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction CSV → partitioned Parquet (step 1)")
    parser.add_argument("csv", nargs="*", help="one CSV per newspaper")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = cfg.path("ads_dir")
    for csv in args.csv or [cfg.path("raw_csv")]:
        df, report = read_extractions_csv(csv)
        write_ads(df, out_dir)
        report["written_to"] = str(out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
