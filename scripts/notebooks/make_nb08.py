import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 08 · Agent tools (step 10)

The four tools the agent of step 11 will call, run by hand on the real tables. The output below is exactly what the
model will read (JSON). Check: does each answer carry what a historian needs, is anything misleading, is it too long?

* `expand_concept(terms)`: dictionary lookup of one or more words → lemmas, historical spellings, categories, requirement tags with counts
* `search_ads(query, mode, filters, k)`: semantic or keyword search, short hits
* `get_ad(ad_ids)`: full records
* `aggregate(measure, group_by, filters, subset, dimension)`: count / share / pay over **countable** ads"""),
    code("""import json
from hisrag.agent.tools import Tools

tools = Tools()


def run(name: str, **args) -> dict:
    \"\"\"Call a tool like the agent will, print its JSON and its size.\"\"\"
    r = tools.call(name, args)
    s = json.dumps(r, ensure_ascii=False, indent=1)
    print(f"{name}({json.dumps(args, ensure_ascii=False)})  →  {len(s)} chars, ~{len(s) // 4} tokens\\n{s}")
    return r


specs = json.dumps(tools.specs(), ensure_ascii=False)
print(f"tool specs: {len(specs)} chars, ~{len(specs) // 4} tokens in every request")"""),
    md("## expand_concept"),
    code("""run("expand_concept", terms=["Köchin", "Gouvernante"]);"""),
    code("""run("expand_concept", terms=["Böhmisch", "Hauslehrer"]);"""),
    md("## search_ads"),
    code("""r = run("search_ads", query="Welche Sprachkenntnisse wurden von Gouvernanten verlangt?", k=5)"""),
    code("""run("search_ads", query='Gouvernante französisch', mode="keyword", k=3,
    filters={"labels": ["job_search"], "year_to": 1869});"""),
    md("## get_ad"),
    code("""run("get_ad", ad_ids=[h["ad_id"] for h in r["results"][:2]]);"""),
    md("## aggregate"),
    code("""run("aggregate", measure="count", group_by="decade", filters={"labels": ["job_offer"]});"""),
    code("""lemmas = [p["lemma"] for p in tools.call("expand_concept", {"terms": ["Lehrer"]})["results"][0]["positions"][:3]]
print(lemmas)
run("aggregate", measure="share", group_by="decade",
    filters={"labels": ["job_offer"], "position_lemmas": lemmas},
    subset={"requirement_tags": ["sprachkenntnisse:Böhmisch"]});"""),
    code("""run("aggregate", measure="count", group_by="requirement_value", dimension="sprachkenntnisse",
    filters={"labels": ["job_offer"]});"""),
    code("""cooks = [p["lemma"] for p in tools.call("expand_concept", {"terms": ["Köchin"]})["results"][0]["positions"][:1]]
print(cooks)
run("aggregate", measure="pay", group_by="decade", filters={"position_lemmas": cooks});"""),
    code("""run("aggregate", measure="share", group_by="decade", filters={"labels": ["job_search"]},
    subset={"keyword": "Gouvernante*"});"""),
    code("""run("aggregate", measure="count", group_by="benefit", filters={"labels": ["job_offer"], "year_to": 1879});"""),
    md("## Errors come back to the model"),
    code("""run("aggregate", measure="share", group_by="decade");
run("search_ads", query="x", k=100);"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/08_agent_tools.ipynb")
