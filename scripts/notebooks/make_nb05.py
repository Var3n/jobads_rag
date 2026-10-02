import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 05 · Clean table and sanity plots (step 7)

`ad_clean` has one row per ad with everything from steps 1–6. Two columns decide how an ad may be used:

* `searchable`: not a death-register entry and not a bare heading;
* `countable`: searchable, the canonical printing of its repeated printings, and not text invented by the
  post-correction. **All counts of ads use `countable`.**

The plots look for gaps before indexing: decades with too few ads, uneven coverage of positions, requirement tags or pay,
and where the quality flags cluster. Each plot has its table below it."""),
    code("""import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from hisrag.data import query

pd.set_option("display.max_columns", 30)

# Validated categorical order (dataviz reference palette, light mode) and recessive ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.figsize": (9, 3.6), "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


THIN = 200  # fewer ads than this: a point is drawn hollow and should not be read as a finding


def lines(df, x, cols, title, ylabel, labels=None, base=None):
    fig, ax = plt.subplots()
    thin = df[base] < THIN if base else pd.Series(False, index=df.index)
    for c, color in zip(cols, SERIES):
        ax.plot(df[x], df[c], marker="o", color=color, label=(labels or {}).get(c, c))
        ax.plot(df.loc[thin, x], df.loc[thin, c], "o", markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.5, markersize=7)
    if thin.any():
        title += f"  (hollow: fewer than {THIN} ads)"
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(df[x])
    if len(cols) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def stacked(df, x, cols, title, ylabel):
    fig, ax = plt.subplots()
    bottom = pd.Series(0.0, index=df.index)
    for c, color in zip(cols, SERIES):
        ax.bar(df[x].astype(str), df[c], bottom=bottom, color=color, label=c, width=0.7,
               edgecolor="white", linewidth=1.5)
        bottom += df[c]
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()"""),
    md("""## Overview
Totals of the table. `searchable` minus `countable` are mostly repeated printings."""),
    code("""query(\"\"\"SELECT count(*) AS regions, count(*) FILTER (searchable) AS searchable,
                  count(*) FILTER (countable) AS countable,
                  count(*) FILTER (searchable AND NOT is_canonical) AS repeated_printings,
                  count(*) FILTER (flag_pc_unsupported AND searchable AND is_canonical) AS searchable_with_warning,
                  count(*) FILTER (flag_death_register) AS death_register, count(*) FILTER (label = 'heading') AS headings
           FROM ad_clean\"\"\")"""),
    md("""## Ads per decade
The base of every statement per decade. Decades with fewer than 200 countable ads are too thin for comparisons;
answers about them must say how many ads they rest on."""),
    code("""per_decade = query(\"\"\"SELECT decade, count(*) AS regions, count(*) FILTER (countable) AS countable
                         FROM ad_clean GROUP BY 1 ORDER BY 1\"\"\")
fig, ax = plt.subplots()
x = per_decade["decade"].astype(str)
ax.bar(x, per_decade["regions"], color=SERIES[0], alpha=0.35, width=0.7, label="all regions")
ax.bar(x, per_decade["countable"], color=SERIES[0], width=0.7, label="countable ads")
for xi, n in zip(x, per_decade["countable"]):
    if n < 200:
        ax.annotate(f"{n}", (xi, n), textcoords="offset points", xytext=(0, 4), ha="center", color=INK_2, fontsize=9)
ax.set_title("Regions and countable ads per decade")
ax.set_ylabel("ads")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
per_decade.assign(thin=per_decade["countable"] < 200)"""),
    md("## What kind of ads (countable, share per decade)"),
    code("""labels = query(\"\"\"SELECT decade, label, count(*) AS n FROM ad_clean WHERE countable GROUP BY ALL\"\"\")
shares = labels.pivot_table(index="decade", columns="label", values="n", fill_value=0)
shares = (100 * shares.div(shares.sum(axis=1), axis=0)).round(1)
order = [c for c in ["job_offer", "job_search", "service_offer", "vermittlung"] if c in shares] + \\
        [c for c in shares if c not in ["job_offer", "job_search", "service_offer", "vermittlung"]]
stacked(shares[order].reset_index(), "decade", order, "Kind of ad, share of countable ads", "%")
shares[order]"""),
    md("""## Coverage of the derived fields
Share of countable **job offers** with a normalized position (step 4), at least one requirement tag (step 5) and a
readable main pay (step 6). A sudden drop marks a decade where an earlier step works worse, or where ads look different."""),
    code("""cov = query(\"\"\"SELECT decade, count(*) AS job_offers,
                         round(100 * avg((position_lemmas IS NOT NULL)::int), 1) AS position,
                         round(100 * avg((requirements IS NOT NULL)::int), 1) AS requirements,
                         round(100 * avg((pay_min IS NOT NULL)::int), 1) AS pay
                  FROM ad_clean WHERE countable AND label = 'job_offer' GROUP BY 1 ORDER BY 1\"\"\")
lines(cov, "decade", ["position", "requirements", "pay"], "Countable job offers with …", "% of job offers",
      {"position": "a position", "requirements": "requirement tags", "pay": "a main pay"}, base="job_offers")
cov"""),
    md("""## Quality flags per decade
Share of all regions with a flag (step 2). The death register is excluded from everything; invented text
(`pc_unsupported`) is searchable with a warning but not counted."""),
    code("""flags = query(\"\"\"SELECT decade, count(*) AS regions,
                           round(100 * avg(flag_death_register::int), 1) AS death_register,
                           round(100 * avg(flag_pc_unsupported::int), 1) AS pc_unsupported,
                           round(100 * avg(flag_too_short::int), 1) AS too_short
                    FROM ad_clean GROUP BY 1 ORDER BY 1\"\"\")
lines(flags, "decade", ["death_register", "pc_unsupported", "too_short"], "Regions with a quality flag", "% of regions",
      base="regions")
flags"""),
    md("## Language of the ads (countable)"),
    code("""lang = query(\"\"\"SELECT decade, lang, count(*) AS n FROM ad_clean WHERE countable GROUP BY ALL\"\"\")
lang = lang.pivot_table(index="decade", columns="lang", values="n", fill_value=0)
lang = (100 * lang.div(lang.sum(axis=1), axis=0)).round(1)
cols = [c for c in ["de", "it", "fr"] if c in lang] + [c for c in lang if c not in ["de", "it", "fr"]]
stacked(lang[cols].reset_index(), "decade", cols[:5], "Language, share of countable ads", "%")
lang[cols]"""),
    md("""## Position categories per decade
Countable ads per category (an ad with two categories counts in both). Darker = more ads. The Wiener Zeitung is
dominated by official vacancies, so teaching and administration outweigh everything else."""),
    code("""cat = query(\"\"\"SELECT decade, c AS category, count(*) AS n
                 FROM ad_clean, UNNEST(position_categories) AS u(c) WHERE countable GROUP BY ALL\"\"\")
grid = cat.pivot_table(index="category", columns="decade", values="n", fill_value=0)
grid = grid.loc[grid.sum(axis=1).sort_values(ascending=False).index]
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9cc2ef", "#2a78d6", "#0d3d7a"])
fig, ax = plt.subplots(figsize=(9, 0.35 * len(grid) + 1))
im = ax.imshow(grid.values, aspect="auto", cmap=blues)
ax.set_xticks(range(grid.shape[1]), grid.columns)
ax.set_yticks(range(grid.shape[0]), grid.index)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, label="countable ads", shrink=0.8)
ax.set_title("Countable ads per position category and decade")
plt.show()
grid"""),
    md("""## Requirement groups per decade
Share of countable job offers with at least one tag of a group (step 5)."""),
    code("""grp = query(\"\"\"WITH jo AS (SELECT ad_id, decade FROM ad_clean WHERE countable AND label = 'job_offer'),
                      g AS (SELECT DISTINCT ad_id, r."group" AS grp FROM ad_clean, UNNEST(requirements) AS u(r)
                            WHERE countable AND label = 'job_offer')
                 SELECT jo.decade, g.grp, count(DISTINCT g.ad_id) AS ads,
                        (SELECT count(*) FROM jo j2 WHERE j2.decade = jo.decade) AS base
                 FROM jo JOIN g USING (ad_id) GROUP BY ALL\"\"\")
grp["pct"] = (100 * grp.ads / grp.base).round(1)
grp.pivot_table(index="grp", columns="decade", values="pct")"""),
    md("""## Spot check: one random countable ad with all its fields
Re-run for another ad. Useful to see what the index and the agent will get."""),
    code("""ad = query(\"\"\"SELECT * FROM ad_clean WHERE countable AND pay_min IS NOT NULL AND requirements IS NOT NULL
                ORDER BY random() LIMIT 1\"\"\")  # not USING SAMPLE: DuckDB samples before WHERE
ad.T"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/05_clean_table.ipynb")
