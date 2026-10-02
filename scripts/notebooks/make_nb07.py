import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 07 · Search index (step 9)

One LanceDB table with one row per searchable ad (`data/index/`), built by `python -m hisrag.index build`. Two modes:

* **`semantic(question)`**: by meaning, with the step-8 winner (qwen3-embedding-8b on the enriched text, 1,024 dims).
  Questions in modern German find historical wording. `score` = cosine similarity.
* **`keyword(words)`**: exact words, BM25 ranked, for names, places and fixed terms. Historical spellings match
  (Wirthschafterin = Wirtschafterin, Clavier = Klavier), inflected forms do not: `krakau*` for Krakau, Krakauer, ….
  All words must occur; `"k. k. Statthalterei"` is a phrase, `-word` excludes. `k=None` returns every match.

Both take `Filters(...)` and return **one ad per printing cluster** (reprints count once; `dup_cluster_size` says
how often it was printed). Ads with `countable = False` are reprints or text invented by the post-correction
(`quality_warning`); counts of ads use `Filters(countable_only=True)`."""),
    code("""import pandas as pd
from IPython.display import Image, display
from hisrag.config import load_config
from hisrag.index import AdIndex, Filters

cfg = load_config()
index = AdIndex(cfg)
pd.set_option("display.max_colwidth", 140)
print(f"{index.table.count_rows()} ads in the index")
pd.DataFrame([{"column": i.columns[0], "type": i.index_type} for i in index.table.list_indices()])"""),
    code("""def show(hits: pd.DataFrame, chars: int = 300) -> None:
    \"\"\"One block per hit: year, kind, score, normalized position, printings, warning, text.\"\"\"
    if expanded := hits.attrs.get("expanded"):
        print("expanded:", {k: v[:12] for k, v in expanded.items()})
    for r in hits.itertuples():
        pos = ", ".join(r.position_modern) if r.position_modern is not None else "–"
        reprints = f", {r.dup_cluster_size}× printed" if r.dup_cluster_size > 1 else ""
        warn = f"  ⚠ {r.quality_warning}" if isinstance(r.quality_warning, str) else ""
        print(f"[{r.ad_id}] {r.date} {r.label}, score {r.score:.3f}{reprints} · {pos}{warn}")
        print("   ", r.text[:chars].replace("\\n", " / "), "\\n")


def clipping(hits: pd.DataFrame, i: int = 0, width: int = 500):
    \"\"\"The printed ad (IIIF clipping) of hit i.\"\"\"
    return Image(url=hits["iiif_link"].iloc[i], width=width)"""),
    md("## Semantic search"),
    code("""hits = index.semantic("Welche Sprachkenntnisse wurden von Gouvernanten verlangt?", k=5)
show(hits)"""),
    code("""clipping(hits, 0)"""),
    md("""With filters: job searches of the 1850s–60s, only countable ads (no reprints, no invented text)."""),
    code("""show(index.semantic("Hauslehrer, die eine Stelle bei einer adeligen Familie suchen",
                    Filters(year_from=1850, year_to=1869, labels=["job_search"], countable_only=True), k=5))"""),
    md("""Normalized fields as filters: position categories, requirement tags (`dimension:value`), pay, benefits.
The values present in the index:"""),
    code("""from hisrag.data import query

print(query(\"\"\"SELECT c, count(*) n FROM (SELECT unnest(position_categories) c FROM ad_clean WHERE searchable)
               GROUP BY 1 ORDER BY 2 DESC\"\"\").to_string(index=False))
query(\"\"\"SELECT t, count(*) n FROM (SELECT unnest(list_transform(requirements, r -> r.dimension || ':' || r.value)) t
         FROM ad_clean WHERE searchable) WHERE t LIKE 'sprachkenntnisse:%' GROUP BY 1 ORDER BY 2 DESC LIMIT 15\"\"\")"""),
    code("""show(index.semantic("Lehrer für Volksschulen mit freier Wohnung",
                    Filters(position_categories=["Unterricht"], benefits=["wohnung"], has_pay=True), k=5))"""),
    code("""show(index.semantic("Beamte, die mehrere Landessprachen beherrschen müssen",
                    Filters(requirement_tags=["sprachkenntnisse:Böhmisch"]), k=5))"""),
    md("""## Keyword search: names, places, fixed terms"""),
    code("""for q in ["Rothschild", "Wirthschafterin", '"k. k. Statthalterei" Lemberg', "krakau*", "böhmisch* Lehrer*",
          "Gouvernante französisch -englisch"]:
    all_hits = index.keyword(q, k=None)
    print(f"{q!r}: {len(all_hits)} ads ({all_hits['countable'].sum()} countable)")"""),
    code("""show(index.keyword('"k. k. Statthalterei" Lemberg', k=5))"""),
    code("""show(index.keyword("böhmisch*", Filters(year_from=1880, year_to=1899), k=5))"""),
    md("""## The step-8 questions

Random research questions of the evaluation with their relevance criterion: does the top 5 fit?"""),
    code("""questions = pd.read_parquet(cfg.path("eval_dir") / "queries.parquet")
for q in questions.sample(3, random_state=None).itertuples():
    print("=" * 100, f"\\n{q.question}\\nKriterium: {q.criterion}\\n")
    show(index.semantic(q.question, k=5), chars=200)"""),
    md("""## Semantic vs keyword on the same topic

Semantic search finds the topic in other words; keyword search finds exactly the words, in every spelling."""),
    code("""topic, words = "Stellen für Köchinnen in Gasthäusern", "Köchin* Gasthaus*"
sem = index.semantic(topic, k=10)
kw = index.keyword(words, k=10)
print(f"overlap of the top 10: {len(set(sem.dup_cluster_id) & set(kw.dup_cluster_id))}")
print("--- semantic"); show(sem.head(4), 160)
print("--- keyword"); show(kw.head(4), 160)"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/07_search_index.ipynb")
