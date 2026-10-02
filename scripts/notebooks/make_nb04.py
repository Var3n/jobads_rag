import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 04 · Salary review (step 6)

Checks the amounts read from the `salary` spans (`ad_salary`: one row per amount) and the per-ad summary (`ad_pay`: main pay
and benefits). Amounts are **nominal**. Gulden amounts carry a standard: Conventionsmünze (CM) or österreichische Währung
(öW), stated in the text or assigned by date (CM until October 1858). `component` says what an amount pays for; the main
pay of an ad is its Gehalt/Lohn/Remuneration/Taggeld, never Kaution or Pension.

Counts use **distinct ads** (canonical printing, step 3); the death register is already excluded in this step."""),
    code("""import pandas as pd
from IPython.display import display
from hisrag.data import query

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_rows", 60)

CANON = "JOIN ad_dups d USING (ad_id) WHERE d.is_canonical"

query(f\"\"\"SELECT (s.year // 10) * 10 AS decade, count(*) AS amounts,
                  round(100 * avg((parsed_by = 'rules')::int), 1) AS pct_rules,
                  round(100 * avg((parsed_by = 'llm')::int), 1) AS pct_llm,
                  round(100 * avg((parsed_by IN ('not_money', 'fragment'))::int), 1) AS pct_no_amount,
                  round(100 * avg((parsed_by IS NULL)::int), 1) AS pct_unreadable
           FROM ad_salary s {CANON} GROUP BY 1 ORDER BY 1\"\"\")"""),
    md("## What the amounts pay for, and their period"),
    code("""query(f\"\"\"SELECT component, count(*) AS amounts, count(DISTINCT ad_id) AS ads,
                  round(100 * avg((period_source = 'assumed')::int), 1) AS pct_period_assumed
           FROM ad_salary s {CANON} AND amount_min IS NOT NULL GROUP BY 1 ORDER BY amounts DESC\"\"\")"""),
    code("""query(f\"\"\"SELECT component, period, count(*) AS amounts FROM ad_salary s {CANON} AND amount_min IS NOT NULL
           GROUP BY ALL ORDER BY component, amounts DESC\"\"\")"""),
    md("""## Samples: is the reading right?
`sample(component=…)` shows amounts with the text around them. Look at the amount, the currency and whether the component
fits the words next to it. `component=None` shows amounts no keyword was found for."""),
    code("""def sample(component: str | None = "gehalt", n: int = 20, parsed_by: str | None = None, seed: int = 0):
    cond = "s.component IS NULL" if component is None else f"s.component = '{component}'"
    if parsed_by:
        cond += f" AND s.parsed_by = '{parsed_by}'"
    return query(f\"\"\"SELECT s.year, s.amount_min, s.amount_max, s.currency, s.standard, s.component, s.period,
                              '…' || substr(a.text, greatest(s.span_start - 59, 1), least(s.span_start, 60))
                                  || ' [' || s.phrase || '] ' || substr(a.text, s.span_end + 1, 40) || '…' AS context
                       FROM ad_salary s JOIN ads a USING (ad_id)
                       WHERE {cond} ORDER BY hash(s.ad_id || s.span_start || '{seed}') LIMIT {n}\"\"\")
sample("gehalt")"""),
    code("""sample("zulage")"""),
    code("""sample("quartiergeld")"""),
    code("""sample("kaution")"""),
    code("""sample(None)"""),
    code("""# Amounts read by the LLM (spelled-out numbers, OCR garbles)
query(\"\"\"SELECT s.year, s.amount_min, s.amount_max, s.currency, s.component, s.phrase
           FROM ad_salary s WHERE parsed_by = 'llm' ORDER BY hash(s.ad_id || s.span_start) LIMIT 25\"\"\")"""),
    code("""# Spans without a readable amount
query(\"\"\"SELECT s.year, s.phrase, count(*) AS n FROM ad_salary s WHERE parsed_by IS NULL
           GROUP BY ALL ORDER BY n DESC LIMIT 25\"\"\")"""),
    md("""## Currency standard
Most Gulden amounts do not say whether they are in Conventionsmünze (CM) or österreichische Währung (öW); they get the
standard by date (CM before November 1858). The amounts that *do* state it test that rule: if the stated standard
almost always matches what the date would give, the date rule is safe for the rest."""),
    code("""query(f\"\"\"SELECT standard_source,
                  count(*) AS amounts,
                  count(*) FILTER (standard = CASE WHEN a.date < DATE '1858-11-01' THEN 'CM' ELSE 'öW' END) AS matches_date_rule,
                  count(*) FILTER (standard <> CASE WHEN a.date < DATE '1858-11-01' THEN 'CM' ELSE 'öW' END) AS contradicts_date_rule
           FROM ad_salary s JOIN ads a USING (ad_id) {CANON} AND s.currency = 'fl' GROUP BY 1 ORDER BY 1\"\"\")"""),
    code("""# The contradicting amounts with their text
query(f\"\"\"SELECT a.date, s.standard, s.phrase,
                  substr(a.text, greatest(s.span_start - 59, 1), least(s.span_start, 60)) || ' [' || s.phrase || '] '
                      || substr(a.text, s.span_end + 1, 40) AS context
           FROM ad_salary s JOIN ads a USING (ad_id) {CANON} AND s.currency = 'fl' AND s.standard_source = 'stated'
             AND s.standard <> CASE WHEN a.date < DATE '1858-11-01' THEN 'CM' ELSE 'öW' END\"\"\")"""),
    md("""## Main pay per decade
Yearly pay only (`pay_period = 'jahr'`, stated or assumed). An ad offering alternatives ("840 fl. oder 735 fl.") gives
`pay_min` and `pay_max`. Gulden and Kronen are not comparable without conversion (roughly 1 fl. ö. W. = 2 K)."""),
    code("""query(f\"\"\"SELECT (p.year // 10) * 10 AS decade, pay_currency, pay_standard, count(*) AS ads,
                  median(pay_min) AS median_min, median(pay_max) AS median_max,
                  quantile_cont(pay_min, 0.1) AS p10, quantile_cont(pay_max, 0.9) AS p90
           FROM ad_pay p {CANON} AND pay_period = 'jahr'
           GROUP BY ALL HAVING ads >= 20 ORDER BY ALL\"\"\")"""),
    code("""# By position category (step 4), ads with one category only
query(f\"\"\"
    WITH cat AS (SELECT ad_id, any_value(category) AS category FROM ad_positions GROUP BY ad_id
                 HAVING count(DISTINCT category) = 1)
    SELECT category, pay_currency, count(*) AS ads, median(pay_min) AS median_min
    FROM ad_pay p JOIN cat USING (ad_id) {CANON} AND pay_period = 'jahr' AND pay_currency IN ('fl', 'K')
      AND (pay_currency = 'K' OR pay_standard = 'öW')
    GROUP BY ALL HAVING ads >= 20 ORDER BY pay_currency, median_min DESC\"\"\")"""),
    md("## Benefits: share of ads per decade (ads with any salary or benefit span)"),
    code("""b = query(f\"\"\"SELECT (p.year // 10) * 10 AS decade, count(*) AS ads,
                  {", ".join(f"round(100 * avg(benefit_{c}::int), 1) AS {c}"
                             for c in ["wohnung", "kost", "kleidung", "heizung", "licht", "deputat", "quartiergeld", "zulagen"])}
               FROM ad_pay p {CANON} GROUP BY 1 ORDER BY 1\"\"\")
b"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/04_salary_review.ipynb")
