import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 09 · Agent pilot review (step 11)

The 12 pilot questions of `python -m hisrag.agent pilot`, each answered without and with Qwen's reasoning
(`data/logs/agent_pilot.csv`; full tool traces in `data/logs/agent.jsonl`).

What to check per answer:
* **correct?** do the cited ads say what the answer claims (clippings at the end)?
* **honest about its basis?** number of ads counted or read, corpus caveats where they matter
* **sensible tool use?** expand_concept before filters, aggregate for numbers, a fitting base set for shares
* **invented IDs** (`unknown_ids`), time per answer, and whether reasoning is worth its time"""),
    code("""import json
import pandas as pd
from IPython.display import Image, Markdown, display
from hisrag.config import load_config
from hisrag.data import query

VERSION = "v5"  # prompt version of the pilot to review (v1 = first pilot, its CSV is agent_pilot.csv)

cfg = load_config()
LOGS = cfg.path("agent_log").parent
csv = {"v1": "agent_pilot.csv"}.get(VERSION, f"agent_pilot_{VERSION}.csv")
pilot = pd.read_csv(LOGS / csv).fillna({"unknown_ids": ""})
questions = list(dict.fromkeys(pilot["question"]))

# the full traces of this version: the newest log record per (question, thinking)
log = [json.loads(line) for line in cfg.path("agent_log").read_text(encoding="utf-8").splitlines()]
version = lambda r: r["settings"].get("prompt_version", "v1")
traces = {}
for r in log:
    if version(r) == VERSION:
        traces[(r["question"], r["settings"]["thinking"])] = r
print(f"prompt {VERSION}: {len(pilot)} answers to {len(questions)} questions; "
      f"{sum(version(r) == VERSION for r in log)} of {len(log)} log records")"""),
    md("""## Overview per mode

All prompt versions in the log, one column per (version, reasoning):"""),
    code("""pd.DataFrame([{"version": version(r), "thinking": r["settings"]["thinking"], "steps": r["steps"],
               "seconds": r["seconds"], "at_limit": r["stopped"] == "max_steps", "no_tools": not r["trace"],
               "cited": len(r["cited"]), "unknown_ids": len(r["unknown_ids"]) > 0,
               "reasoning_chars": r.get("reasoning_chars")} for r in log]
 ).groupby(["version", "thinking"]).mean().round(2).T"""),
    md("This version:"),
    code("""pilot.groupby("thinking").agg(
    answers=("answer", "size"), stopped_at_limit=("stopped", lambda s: (s == "max_steps").sum()),
    errors=("stopped", lambda s: (s == "error").sum()), mean_steps=("steps", "mean"),
    mean_seconds=("seconds", "mean"), max_seconds=("seconds", "max"), mean_cited=("n_cited", "mean"),
    with_unknown_ids=("unknown_ids", lambda s: (s != "").sum()),
    mean_completion_tokens=("completion_tokens", "mean"),
    mean_reasoning_chars=("reasoning_chars", "mean")).round(1).T"""),
    code("""overview = pilot.assign(q=pilot["question"].map(questions.index))[
    ["q", "thinking", "steps", "seconds", "n_cited", "unknown_ids", "stopped", "tools_used"]]
overview.sort_values(["q", "thinking"]).reset_index(drop=True)"""),
    md("## Answers side by side\n\n`compare(i)` shows question *i* (0–11) in both modes with its tool calls."),
    code("""from hisrag.agent.render import trace_markdown as tool_lines


def compare(i: int) -> None:
    q = questions[i]
    display(Markdown(f"# {i}. {q}"))
    for thinking in (False, True):
        r = traces.get((q, thinking))
        if r is None:
            continue
        head = (f"## {'mit' if thinking else 'ohne'} Reasoning · {r['steps']} Anfragen, {r['seconds']:.0f} s, "
                f"{r['tokens']['completion']} Ausgabe-Tokens, {r.get('reasoning_chars', '?')} Zeichen Reasoning, "
                f"Ende: {r['stopped']}")
        warn = f"\\n\\n**Erfundene IDs:** {', '.join(r['unknown_ids'])}" if r["unknown_ids"] else ""
        display(Markdown(f"{head}{warn}\\n\\n{r['answer']}\\n\\n**Tool-Aufrufe**\\n\\n{tool_lines(r['trace'])}"))


compare(0)"""),
    code("""compare(1)"""),
    code("""compare(2)"""),
    code("""compare(3)"""),
    code("""compare(4)"""),
    code("""compare(5)"""),
    code("""compare(6)"""),
    code("""compare(7)"""),
    code("""compare(8)"""),
    code("""compare(9)"""),
    code("""compare(10)"""),
    code("""compare(11)"""),
    md("""## A tool result in full

`result(i, thinking, n)`: the JSON the model got from its n-th tool call (from 0)."""),
    code("""def result(i: int, thinking: bool, n: int) -> None:
    t = traces[(questions[i], thinking)]["trace"][n]
    print(t["tool"], json.dumps(t["args"], ensure_ascii=False))
    print(json.dumps(t["result"], ensure_ascii=False, indent=1))


result(0, False, 0)"""),
    md("""## Cited ads: do they say what the answer claims?

`cited(i, thinking)` shows the text and the printed clipping of every ad the answer cites."""),
    code("""def cited(i: int, thinking: bool, images: bool = True) -> None:
    r = traces[(questions[i], thinking)]
    from hisrag.agent.loop import AD_ID
    ids = list(dict.fromkeys(AD_ID.findall(r["answer"])))
    if not ids:
        print("keine zitierten Anzeigen")
        return
    ads = query("SELECT ad_id, date, label, text_norm, iiif_link FROM ad_clean WHERE ad_id IN (SELECT unnest(?))",
                [ids]).set_index("ad_id")
    for ad_id in ids:
        if ad_id not in ads.index:
            print(f"[{ad_id}] gibt es nicht\\n")
            continue
        a = ads.loc[ad_id]
        print(f"[{ad_id}] {str(a['date'])[:10]} {a['label']}\\n{a['text_norm']}\\n")
        if images:
            display(Image(url=a["iiif_link"], width=450))


cited(0, False)"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/09_agent_pilot_review.ipynb")
