"""python -m hisrag.agent STEP   (step 11: agent loop)

  ask "Frage" [--thinking]
          answer one question; prints the answer, the tool calls and the citation check.
  pilot [--thinking off|on|both]
          the pilot questions below, in parallel → data/logs/agent_pilot_<prompt version>.csv for review (answer, tool calls,
          citations, time, tokens; full traces in data/logs/agent.jsonl). Default: the configured mode
          (agent.thinking); --thinking both compares with and without reasoning.
"""

import argparse
import json

from hisrag.config import load_config

PILOT_QUESTIONS = [
    "Welche Sprachkenntnisse wurden von Gouvernanten verlangt?",
    "Wie veränderte sich der Anteil der Stellengesuche an allen Anzeigen zwischen 1850 und 1910?",
    "Wie viel verdienten Volksschullehrer in den 1870er Jahren laut den Ausschreibungen?",
    "Für welche Stellen wurde die Kenntnis der böhmischen Sprache verlangt, und wie häufig war das in den einzelnen "
    "Jahrzehnten?",
    "In welchen Anzeigen kommt das Haus Rothschild vor?",
    "Welche Naturalleistungen wie Kost oder Wohnung wurden Dienstboten in den 1860er Jahren angeboten?",
    "Wie beschrieben sich Frauen, die in den 1850er Jahren eine Stelle als Wirtschafterin suchten?",
    "Wie entwickelte sich die Zahl der Stellenangebote für Hausangestellte in der Zwischenkriegszeit?",
    "Welche Anforderungen an Familienstand und Religion stellten Arbeitgeber an Erzieherinnen?",
    "Wurden Frauen Stellen in Büros oder im Handel angeboten, und seit wann?",
    "Welche Stellen in Galizien verlangten die Kenntnis des Ruthenischen?",
    "Was erwarteten adelige Familien von Hauslehrern?",
]


def run_ask(cfg, question: str, thinking: bool) -> dict:
    from hisrag.agent.loop import Agent

    a = Agent(cfg).ask(question, thinking=thinking)
    print(a.answer, "\n")
    return {"tools": [{"tool": t["tool"], "args": t["args"], "ms": t["ms"]} for t in a.trace],
            "steps": a.steps, "stopped": a.stopped, "cited": len(a.cited), "unknown_ids": a.unknown_ids,
            "seconds": a.seconds, "tokens": a.tokens}


def run_pilot(cfg, modes: list[bool]) -> dict:
    import pandas as pd

    from hisrag.agent.loop import PROMPT_VERSION, Agent

    agent = Agent(cfg)
    jobs = [(q, t) for t in modes for q in PILOT_QUESTIONS]
    answers = agent.client.map(lambda job: agent.ask(job[0], thinking=job[1]), jobs, desc="agent pilot")
    rows = pd.DataFrame([{"question": a.question, "thinking": a.settings["thinking"], "answer": a.answer,
                          "tools_used": a.tools_used(),
                          "tool_args": json.dumps([{"tool": t["tool"], "args": t["args"]} for t in a.trace],
                                                  ensure_ascii=False),
                          "steps": a.steps, "stopped": a.stopped, "reasoning_chars": a.reasoning_chars,
                          "n_cited": len(a.cited),
                          "unknown_ids": ", ".join(a.unknown_ids), "seconds": a.seconds,
                          "completion_tokens": a.tokens["completion"], "prompt_tokens": a.tokens["prompt"]}
                         for a in answers])
    path = cfg.path("agent_log").parent / f"agent_pilot_{PROMPT_VERSION}.csv"
    rows.to_csv(path, index=False, encoding="utf-8-sig")
    per_mode = rows.groupby("thinking").agg(
        answers=("answer", "size"), stopped_max_steps=("stopped", lambda s: int((s == "max_steps").sum())),
        errors=("stopped", lambda s: int((s == "error").sum())), mean_steps=("steps", "mean"),
        mean_seconds=("seconds", "mean"), max_seconds=("seconds", "max"), mean_cited=("n_cited", "mean"),
        answers_with_unknown_ids=("unknown_ids", lambda s: int((s != "").sum())),
        mean_completion_tokens=("completion_tokens", "mean"), mean_reasoning_chars=("reasoning_chars", "mean"))
    return {"questions": len(PILOT_QUESTIONS), "prompt_version": PROMPT_VERSION, "review_csv": str(path),
            "per_thinking": {str(k): {c: round(float(v), 1) for c, v in r.items()} for k, r in per_mode.iterrows()},
            "usage": agent.client.usage.summary()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11: agent loop")
    parser.add_argument("step", choices=["ask", "pilot"])
    parser.add_argument("question", nargs="?", help="ask: the question")
    parser.add_argument("--thinking", nargs="?", const="on", default=None, choices=["off", "on", "both"])
    args = parser.parse_args()
    cfg = load_config()
    if args.step == "ask":
        if not args.question:
            parser.error("ask needs a question")
        report = run_ask(cfg, args.question, thinking=args.thinking == "on")
    else:
        modes = {"off": [False], "on": [True], "both": [False, True], None: [cfg["agent"]["thinking"]]}[args.thinking]
        report = run_pilot(cfg, modes)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
