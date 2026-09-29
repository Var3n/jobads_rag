"""python -m hisrag.normalize text   (step 2: normalized text, quality metrics and flags → derived/ad_text)"""

import argparse
import json

from hisrag.config import load_config
from hisrag.data import derived_dir, query, write_partitioned


def run_text(cfg) -> dict:
    from hisrag.normalize.quality import SCHEMA, assess, summarize

    ads = query("SELECT ad_id, newspaper, year, label, text, text_ocr, heading_text FROM ads", cfg=cfg)
    q = assess(ads)
    out = derived_dir("ad_text", cfg)
    write_partitioned(q, out, SCHEMA)
    return {**summarize(q, ads["label"]), "written_to": str(out)}


STEPS = {"text": run_text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalization steps")
    parser.add_argument("step", choices=STEPS)
    args = parser.parse_args()
    report = STEPS[args.step](load_config())
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
