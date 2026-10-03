"""Step 12: how an answer is shown to a researcher (playground) and to a reviewer (notebook 09)."""

from __future__ import annotations

import json
import re

from hisrag.agent.loop import AD_ID


def tool_summary(tool: str, result: dict) -> str:
    """One line on what a tool call returned."""
    r = result
    if "error" in r:
        return f"**Fehler:** {r['error'][:200]}"
    if tool == "search_ads":
        got = f"{len(r['results'])} Treffer" + (f" von {r['total_matches']}" if "total_matches" in r else "")
        return got + (f"; erweitert: {r['expanded']}" if r.get("expanded") else "")
    if tool == "aggregate":
        return f"n_ads = {r['n_ads']}, {r['groups_total']} Gruppen"
    if tool == "expand_concept":
        return " · ".join(
            f"{x['term']}: Lemmata " + ", ".join(f"{p['lemma']} ({p['n_ads']})" for p in x.get("positions", [])[:6])
            + "; Tags " + ", ".join(f"{q['tag']} ({q['n_ads']})" for q in x.get("requirements", [])[:6])
            for x in r.get("results", [r]))  # logs before prompt v3: one term per call
    if tool == "get_ad":
        return f"{len(r.get('ads', []))} Anzeigen"
    return ""


def trace_markdown(trace: list[dict]) -> str:
    """The tool calls of an answer, one bullet per call: round, tool, arguments, what came back."""
    return "\n".join(f"- Runde {t['step']}: `{t['tool']}` {json.dumps(t['args'], ensure_ascii=False)}  \n"
                     f"  → {tool_summary(t['tool'], t['result'])}" for t in trace)


def basis(trace: list[dict]) -> dict:
    """What an answer rests on: the counts aggregate returned and the ads the model saw (hits and full records)."""
    counts = [t["result"]["n_ads"] for t in trace if t["tool"] == "aggregate" and "n_ads" in t["result"]]
    seen = set()
    for t in trace:
        if t["tool"] in ("search_ads", "get_ad"):
            seen |= {h["ad_id"] for h in t["result"].get("results", []) + t["result"].get("ads", []) if "ad_id" in h}
    read = {a["ad_id"] for t in trace if t["tool"] == "get_ad" for a in t["result"].get("ads", []) if "text" in a}
    return {"tool_calls": len(trace), "counts": counts, "ads_seen": len(seen), "ads_read_in_full": len(read)}


def basis_line(answer) -> str:
    b = basis(answer.trace)
    parts = [f"{b['tool_calls']} Tool-Aufrufe in {answer.steps} Anfragen, {answer.seconds:.0f} s"]
    if b["counts"]:
        parts.append("gezählt: " + ", ".join(f"n = {n}" for n in dict.fromkeys(b["counts"])))
    parts.append(f"{b['ads_seen']} Anzeigen gesehen, {b['ads_read_in_full']} vollständig gelesen, "
                 f"{len(answer.cited)} zitiert")
    if answer.stopped != "answer":
        parts.append({"max_steps": "Höchstzahl an Schritten erreicht", "error": "Fehler"}.get(answer.stopped,
                                                                                          answer.stopped))
    return " · ".join(parts)


def link_citations(text: str, links: dict[str, str]) -> str:
    """Every cited ad ID becomes a Markdown link to its clipping; IDs without a link are marked."""
    def repl(m: re.Match) -> str:
        ad_id = m.group(0)
        return f"[{ad_id}]({links[ad_id]})" if ad_id in links else f"~~{ad_id}~~ (unbekannt)"
    return AD_ID.sub(repl, text)
