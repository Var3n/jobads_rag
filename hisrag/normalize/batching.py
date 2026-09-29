"""Shared machinery for LLM dictionary steps (positions, requirements).

A dictionary step sends every distinct form once, in batches. Each batch is a frame with a
unique `key` column; the model answers {"items": [{"i": <1-based row>, ...}, ...]}. This module
handles what goes wrong at scale:

  * items the model skipped are asked again one by one;
  * an answer that cannot be parsed even after the repair attempt (e.g. it broke off mid-JSON)
    is split in half and re-sent: the prompts differ, so the failure is not simply repeated
    (temperature 0 and the response cache would return the same answer);
  * reasoning requests get a long timeout and are never restarted automatically.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import pandas as pd
from pydantic import BaseModel

from hisrag.llm.client import DHClient, StructuredOutputError

Item = TypeVar("Item", bound=BaseModel)

# With reasoning, one request can take far longer than the default read timeout (the API
# generates ~10 tokens/s per request); a timed-out request must not be restarted in a loop.
THINKING_REQUEST = {"timeout_s": 1800, "max_retries": 0}


def normalize_batch(client: DHClient, batch: pd.DataFrame, *,
                    build_messages: Callable[[pd.DataFrame], list[dict]],
                    result_model: type[BaseModel],
                    empty_item: Callable[[], Item],
                    thinking: bool = False,
                    max_tokens: int = 6000) -> dict[str, Item]:
    """Normalize one batch; returns key → item for every row of the batch."""
    opts = THINKING_REQUEST if thinking else {}
    budget = max_tokens * (3 if thinking else 1)
    try:
        result = client.chat_json(build_messages(batch), result_model, thinking=thinking,
                                  max_tokens=budget, **opts)
    except StructuredOutputError:
        if len(batch) == 1:
            return {batch["key"].iloc[0]: empty_item()}
        half = len(batch) // 2
        kw = dict(build_messages=build_messages, result_model=result_model, empty_item=empty_item,
                  thinking=thinking, max_tokens=max_tokens)
        return {**normalize_batch(client, batch.iloc[:half], **kw),
                **normalize_batch(client, batch.iloc[half:], **kw)}

    by_i = {item.i: item for item in result.items if 1 <= item.i <= len(batch)}
    out = {}
    for n, key in enumerate(batch["key"], 1):
        if n in by_i:
            out[key] = by_i[n]
            continue
        single = client.chat_json(build_messages(batch.iloc[[n - 1]]), result_model, thinking=thinking,
                                  max_tokens=max(budget // 2, 4000), **opts)
        out[key] = single.items[0] if single.items else empty_item()
    return out


def run_batches(client: DHClient, rows: pd.DataFrame, *, job: str, batch_size: int,
                progress: bool = True, **batch_kwargs) -> tuple[dict[str, Item], dict]:
    """Run normalize_batch over all rows in parallel; failures are reported, not raised."""
    batches = [rows.iloc[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    t0 = time.monotonic()
    with client.job_scope(job):
        parts = client.map(lambda b: normalize_batch(client, b, **batch_kwargs), batches,
                           desc=job if progress else None, return_exceptions=True)
    results, failed = {}, []
    for batch, part in zip(batches, parts):
        if isinstance(part, Exception):
            failed.append(f"{batch['key'].iloc[0]}…: {type(part).__name__}: {str(part)[:200]}")
        else:
            results.update(part)
    stats = {"forms": len(rows), "batches": len(batches), "failed_batches": failed,
             "seconds": round(time.monotonic() - t0, 1)}
    return results, stats
