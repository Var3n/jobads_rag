import datetime as dt

import pandas as pd
import pytest

from hisrag.config import load_config
from hisrag.data import derived_dir, write_partitioned
from hisrag.normalize import clean as C
from hisrag.normalize import positions as P
from hisrag.normalize import requirements as R
from hisrag.normalize import salary as S

ADS = ["a", "b", "c", "d", "h"]


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(local=tmp_path / "none.yaml")
    cfg["paths"]["ads_dir"] = str(tmp_path / "ads")
    cfg["paths"]["derived_dir"] = str(tmp_path / "derived")
    return cfg


def base(ids):
    return pd.DataFrame({"ad_id": ids, "newspaper": "wrz", "year": 1870})


def write_inputs(cfg):
    """a: offer with position, tags, pay; b: second printing of a; c: invented text; d: death register; h: heading."""
    ads = base(ADS).assign(date=dt.date(1870, 5, 1), decade=1870, page=1, iiif_link="http://iiif/x",
                           label=["job_offer", "job_offer", "job_search", "job_offer", "heading"],
                           heading_text=None, text=["Köchin gesucht, 200 fl."] * 2 + ["x", "y", "Köchin"])
    write_partitioned(ads, cfg.path("ads_dir"))
    flags = dict(flag_pc_repetition=False, flag_pc_expanded=False, flag_too_short=False)
    write_partitioned(base(ADS).assign(text_norm="t", lang="de", n_flags=[0, 0, 1, 1, 0], **flags,
                                       flag_pc_unsupported=[False, False, True, False, False],
                                       flag_death_register=[False, False, False, True, False]),
                      derived_dir("ad_text", cfg))
    write_partitioned(base(ADS).assign(dup_cluster_id=["a", "a", "c", "d", "h"], dup_cluster_size=[2, 2, 1, 1, 1],
                                       is_canonical=[True, False, True, True, True],
                                       run_first_date=dt.date(1870, 5, 1), run_last_date=dt.date(1870, 5, 8)),
                      derived_dir("ad_dups", cfg))
    pos = base(["a", "a"]).assign(source="span", span_start=0, span_end=6, surface=["Köchin", "Köchinn"], key="k",
                                  extracted_gender="f", term=["Köchin", "Köchin"], lemma="Koch", modern="Köchin",
                                  gender_form="f", category="Hauswirtschaft", confidence="high")
    write_partitioned(pos, derived_dir("ad_positions", cfg), P.AD_POSITIONS_SCHEMA)
    req = base(["a", "a", "c"]).assign(column="background", span_start=0, span_end=5, phrase="ledig",
                                       dimension=["familienstand", "familienstand", "alter"], group="Person",
                                       value=["ledig", "ledig", "jung"], detail=None, in_vocab=True)
    write_partitioned(req, derived_dir("ad_requirements", cfg), R.AD_REQUIREMENTS_SCHEMA)
    sal = base(["a"]).assign(span_start=17, span_end=24, phrase="200 fl.", amount_min=200.0, amount_max=200.0,
                             currency="fl", standard="öW", standard_source="date", component="lohn", period="jahr",
                             period_source="stated", parsed_by="rules")
    write_partitioned(sal, derived_dir("ad_salary", cfg), S.AD_SALARY_SCHEMA)
    pay = S.ad_pay(sal, pd.DataFrame({"ad_id": ["a"], "newspaper": "wrz", "year": [1870], "phrase": ["Kost"]}))
    write_partitioned(pay, derived_dir("ad_pay", cfg), S.AD_PAY_SCHEMA)


def test_clean_table_joins_steps_and_marks_usable_ads(cfg):
    write_inputs(cfg)
    table = C.build(cfg)
    df = table.to_pandas().set_index("ad_id").loc[ADS]
    assert df["searchable"].tolist() == [True, True, True, False, False]  # death register, heading
    assert df["countable"].tolist() == [True, False, False, False, False]  # 2nd printing, invented text
    assert df.loc["c", "quality_warning"].startswith("Text weicht")

    a = df.loc["a"]
    assert list(a["position_terms"]) == ["Köchin"] and a["position_gender"] == "f"
    assert [r["value"] for r in a["requirements"]] == ["ledig"]  # duplicates collapsed
    assert (a["pay_min"], a["pay_currency"], a["pay_period"], a["benefit_kost"]) == (200, "fl", "jahr", True)
    assert a["salary_amounts"][0]["component"] == "lohn"
    assert df.loc["d", "pay_amounts"] == 0 and not df.loc["d", "benefit_kost"]

    C.write(table, derived_dir("ad_clean", cfg))
    report = C.summarize(table)
    assert report["countable"] == 1 and report["pct_countable_with_pay"] == 100.0


def test_missing_step_is_reported(cfg):
    write_inputs(cfg)
    import shutil
    shutil.rmtree(derived_dir("ad_pay", cfg))
    with pytest.raises(RuntimeError, match="ad_pay"):
        C.build(cfg)
