"""Step 12: ratings of agent answers, and the interaction log they become (step 13).

Every answer is logged by `Agent.ask` to `paths.agent_log` (JSONL, one record per question with its trace and an
`id`); a rating from the playground is appended to `paths.ratings_log` with that id. `interactions()` joins both
into one table: the first evaluation set. Several researchers may write at the same time in one project folder:
see `hisrag.files` (group-writable files, locked appends).
"""

from __future__ import annotations

import getpass
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from hisrag.config import Config, load_config
from hisrag.files import append_line

Verdict = Literal["richtig", "teilweise", "falsch"]
VERDICTS = ("richtig", "teilweise", "falsch")


def current_user() -> str:
    return os.environ.get("JUPYTERHUB_USER") or getpass.getuser()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, record: dict) -> None:
    """One line per record, in a file the other researchers of the project folder can append to as well."""
    append_line(path, json.dumps(record, ensure_ascii=False, default=str))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut by a crash must not hide the rest
            continue
    return out


def rate(answer_id: str, verdict: Verdict, citations_fit: bool | None = None, comment: str = "", *,
         cfg: Config | None = None) -> dict:
    """Store a rating of one answer. citations_fit: do the cited ads say what the answer claims (None = not checked).
    A later rating of the same answer by the same user replaces the earlier one in `interactions()`."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    cfg = cfg or load_config()
    record = {"time": now(), "answer_id": answer_id, "user": current_user(), "verdict": verdict,
              "citations_fit": citations_fit, "comment": comment.strip()}
    append_jsonl(cfg.path("ratings_log"), record)
    return record


def interactions(cfg: Config | None = None) -> pd.DataFrame:
    """One row per logged answer (pilots included), with its latest rating per user if there is one."""
    cfg = cfg or load_config()
    answers = pd.DataFrame([{
        "answer_id": r.get("id"), "time": r["time"], "user": r.get("user"), "question": r["question"],
        "answer": r["answer"], "prompt_version": r["settings"].get("prompt_version", "v1"),
        "thinking": r["settings"].get("thinking"), "steps": r["steps"], "stopped": r["stopped"],
        "seconds": r["seconds"], "tools": " → ".join(t["tool"] for t in r["trace"]), "n_cited": len(r["cited"]),
        "unknown_ids": ", ".join(r["unknown_ids"])} for r in read_jsonl(cfg.path("agent_log"))])
    ratings = pd.DataFrame(read_jsonl(cfg.path("ratings_log")))
    if answers.empty:
        return answers
    if ratings.empty:
        return answers.assign(rated_by=None, verdict=None, citations_fit=None, comment=None)
    latest = (ratings.sort_values("time").drop_duplicates(["answer_id", "user"], keep="last")
                     .rename(columns={"user": "rated_by", "time": "rated_at"}))
    return answers.merge(latest, on="answer_id", how="left")
