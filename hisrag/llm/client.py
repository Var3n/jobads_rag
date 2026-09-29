"""Thin wrapper around the OpenAI-compatible DHinfra API.

Everything that talks to the cluster goes through `DHClient`, which adds:
  * a client-side requests-per-minute limiter (the openai client retries 429/5xx itself)
  * a SQLite response cache, so re-running a bulk job does not spend tokens twice
  * a token-usage log per job (data/logs/usage.jsonl)
  * `chat_json` for pydantic-validated structured output
  * `embed` with per-model query/passage prefixes from config.yaml
  * `map` for parallel jobs over a thread pool
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from hisrag.config import Config, env_file, load_config

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class RateLimiter:
    """Sliding-window limiter: at most `rpm` calls in any 60 s window."""

    def __init__(self, rpm: int | None):
        self.rpm = rpm
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self.rpm:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return
                wait = 60 - (now - self._calls[0])
            time.sleep(max(wait, 0.05))


class ResponseCache:
    """Request-hash → response JSON, in one SQLite file shared by all threads."""

    def __init__(self, path: Path | None):
        self._lock = threading.Lock()
        self._db = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
            self._db.commit()

    @staticmethod
    def key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if self._db is None:
            return None
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        if self._db is None:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
            self._db.commit()


class UsageTracker:
    """Per-job token counters, appended request by request to a JSONL log."""

    FIELDS = ("requests", "cache_hits", "errors", "prompt_tokens", "completion_tokens", "cached_tokens")

    def __init__(self, log_path: Path | None):
        self.log_path = log_path
        self.totals: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(self.FIELDS, 0))
        self._lock = threading.Lock()
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, job: str, endpoint: str, model: str, usage: dict | None, *,
               cache_hit: bool = False, error: str | None = None, latency_s: float = 0.0) -> None:
        usage = usage or {}
        details = usage.get("prompt_tokens_details") or {}
        entry = {
            "ts": time.time(), "job": job, "endpoint": endpoint, "model": model,
            "prompt_tokens": 0 if cache_hit else usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": 0 if cache_hit else usage.get("completion_tokens", 0) or 0,
            "cached_tokens": 0 if cache_hit else details.get("cached_tokens", 0) or 0,
            "cache_hit": cache_hit, "error": error, "latency_s": round(latency_s, 3),
        }
        with self._lock:
            t = self.totals[job]
            t["requests"] += 0 if cache_hit else 1
            t["cache_hits"] += int(cache_hit)
            t["errors"] += int(error is not None)
            for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                t[k] += entry[k]
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

    def summary(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {job: dict(t) for job, t in self.totals.items()}


@dataclass
class ChatResult:
    content: str | None
    reasoning: str | None
    tool_calls: list[dict]
    finish_reason: str | None
    usage: dict
    from_cache: bool
    raw: dict = field(repr=False)

    @property
    def message(self) -> dict:
        """The assistant message, ready to append to a conversation (e.g. in a tool loop)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class StructuredOutputError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    """Pull the JSON part out of a model answer (handles ```json fences and leading prose)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        return fence.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text.strip()
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1]


class DHClient:
    def __init__(self, cfg: Config | None = None, *, openai_client: Any = None, use_cache: bool = True,
                 timeout_s: float | None = None, max_retries: int | None = None):
        self.cfg = cfg or load_config()
        api = self.cfg["api"]
        if openai_client is None:
            from openai import OpenAI, Timeout

            key = self.cfg.api_key
            if not key:
                raise RuntimeError(f"{api['api_key_env']} not found (looked in the environment and {env_file()}). "
                                   "Run hisrag.set_api_key() to enter it.")
            timeout = Timeout(timeout_s or api["timeout_s"], connect=api.get("connect_timeout_s", 15))
            openai_client = OpenAI(base_url=api["base_url"], api_key=key, timeout=timeout,
                                   max_retries=api["max_retries"] if max_retries is None else max_retries)
        self._oa = openai_client
        self.model = self.cfg["llm"]["model"]
        self.max_workers = api["max_workers"]
        self.limiter = RateLimiter(api.get("requests_per_minute"))
        self.cache = ResponseCache(self.cfg.path("cache_db") if use_cache else None)
        self.usage = UsageTracker(self.cfg.path("usage_log"))
        self.structured_mode: str = self.cfg["llm"].get("structured_output", "auto")
        self.job = "default"

    @contextmanager
    def job_scope(self, name: str):
        """Attribute all requests inside the block to `name` in the usage log."""
        previous, self.job = self.job, name
        try:
            yield
        finally:
            self.job = previous

    # ------------------------------------------------------------------ chat

    def chat(self, messages: list[dict], *, model: str | None = None, thinking: bool = False,
             max_tokens: int | None = None, temperature: float | None = 0.0,
             tools: list[dict] | None = None, tool_choice: str | dict | None = None,
             response_format: dict | None = None, use_cache: bool = True,
             timeout_s: float | None = None, max_retries: int | None = None,
             **extra: Any) -> ChatResult:
        """One chat completion. timeout_s / max_retries override the client defaults for this
        request only (long reasoning answers need more than the default read timeout, and a
        timed-out request should not be restarted over and over); they are not part of the cache key."""
        payload: dict[str, Any] = {"model": model or self.model, "messages": messages}
        for k, v in (("max_tokens", max_tokens), ("temperature", temperature), ("tools", tools),
                     ("tool_choice", tool_choice), ("response_format", response_format)):
            if v is not None:
                payload[k] = v
        extra_body = dict(extra.pop("extra_body", {}) or {})
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = thinking
        payload["extra_body"] = extra_body
        payload.update(extra)

        key = ResponseCache.key(payload)
        if use_cache and (hit := self.cache.get(key)) is not None:
            self.usage.record(self.job, "chat", payload["model"], hit.get("usage"), cache_hit=True)
            return self._to_result(hit, from_cache=True)

        oa = self._oa
        if timeout_s is not None or max_retries is not None:
            opts: dict[str, Any] = {}
            if timeout_s is not None:
                from openai import Timeout

                opts["timeout"] = Timeout(timeout_s, connect=self.cfg["api"].get("connect_timeout_s", 15))
            if max_retries is not None:
                opts["max_retries"] = max_retries
            oa = oa.with_options(**opts)
        self.limiter.acquire()
        t0 = time.monotonic()
        try:
            response = oa.chat.completions.create(**payload)
        except Exception as exc:
            self.usage.record(self.job, "chat", payload["model"], None, error=type(exc).__name__,
                              latency_s=time.monotonic() - t0)
            raise
        raw = response.model_dump()
        self.usage.record(self.job, "chat", payload["model"], raw.get("usage"), latency_s=time.monotonic() - t0)
        # Only cache complete answers; a cut-off answer should be retried with more max_tokens.
        if use_cache and raw["choices"][0].get("finish_reason") != "length":
            self.cache.set(key, raw)
        return self._to_result(raw, from_cache=False)

    @staticmethod
    def _to_result(raw: dict, from_cache: bool) -> ChatResult:
        choice = raw["choices"][0]
        msg = choice.get("message") or {}
        tool_calls = [
            {"id": tc["id"], "type": tc.get("type", "function"),
             "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
            for tc in (msg.get("tool_calls") or [])
        ]
        content = msg.get("content")
        if isinstance(content, str):
            content = content.strip()  # with reasoning on, answers start with blank lines
        return ChatResult(
            content=content,
            reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=raw.get("usage") or {},
            from_cache=from_cache,
            raw=raw,
        )

    # ------------------------------------------------------- structured output

    def chat_json(self, messages: list[dict], schema: type[M], *, mode: str | None = None,
                  repair_attempts: int = 1, **kwargs: Any) -> M:
        """Chat call whose answer is validated against a pydantic model.

        mode "json_schema" uses the server's guided decoding; "prompt" puts the schema in the
        system message and parses the answer. "auto" tries json_schema and falls back to prompt
        for the rest of the session if the server rejects it.
        """
        mode = mode or self.structured_mode
        if mode == "auto":
            try:
                result = self.chat_json(messages, schema, mode="json_schema",
                                        repair_attempts=repair_attempts, **kwargs)
                self.structured_mode = "json_schema"
                return result
            except StructuredOutputError:
                raise
            except Exception as exc:  # BadRequestError etc: server does not support json_schema
                if getattr(exc, "status_code", None) != 400:
                    raise
                self.structured_mode = "prompt"
                mode = "prompt"

        schema_json = schema.model_json_schema()
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_json, "strict": True},
            }
            msgs = list(messages)
        else:
            instruction = ("Antworte ausschließlich mit JSON, das diesem JSON-Schema entspricht, "
                           "ohne weiteren Text:\n" + json.dumps(schema_json, ensure_ascii=False))
            msgs = _with_system_suffix(messages, instruction)

        result = self.chat(msgs, **kwargs)
        for attempt in range(repair_attempts + 1):
            try:
                return schema.model_validate_json(_extract_json(result.content or ""))
            except ValidationError as err:
                if attempt == repair_attempts:
                    raise StructuredOutputError(f"{schema.__name__}: {err}\n--- answer ---\n{result.content}") from err
                msgs = msgs + [result.message, {"role": "user", "content":
                    f"Das JSON ist ungültig:\n{err}\nGib nur das korrigierte JSON zurück."}]
                result = self.chat(msgs, **{**kwargs, "use_cache": False})
        raise AssertionError("unreachable")

    # -------------------------------------------------------------- embeddings

    def embedding_spec(self, name: str, kind: Literal["query", "passage"]) -> tuple[str, str]:
        """(model slug, prefix) for a configured embedding model and text kind."""
        spec = self.cfg["embeddings"]["models"][name]
        slug = spec.get(f"{kind}_model", name)
        return slug, spec.get(f"{kind}_prefix", "")

    def embed(self, texts: Sequence[str], model: str | None = None, *,
              kind: Literal["query", "passage"] = "passage", batch_size: int | None = None,
              normalize: bool = True, dimensions: int | None = None, progress: bool = False) -> np.ndarray:
        """Embed texts in parallel batches; returns float32 array of shape (len(texts), dim).

        `dimensions` truncates Matryoshka-capable models (e.g. qwen3-embedding-8b) client-side.
        """
        name = model or self.cfg["embeddings"]["default"]
        slug, prefix = self.embedding_spec(name, kind)
        batch_size = batch_size or self.cfg["embeddings"]["batch_size"]
        batches = [[prefix + t for t in texts[i:i + batch_size]] for i in range(0, len(texts), batch_size)]

        def run(batch: list[str]) -> list[list[float]]:
            self.limiter.acquire()
            t0 = time.monotonic()
            try:
                response = self._oa.embeddings.create(model=slug, input=batch)
            except Exception as exc:
                self.usage.record(self.job, "embeddings", slug, None, error=type(exc).__name__,
                                  latency_s=time.monotonic() - t0)
                raise
            raw = response.model_dump()
            self.usage.record(self.job, "embeddings", slug, raw.get("usage"), latency_s=time.monotonic() - t0)
            rows = sorted(raw["data"], key=lambda d: d["index"])  # match by index, per the API docs
            return [r["embedding"] for r in rows]

        parts = self.map(run, batches, desc=f"embed {name}" if progress else None)
        vectors = np.asarray([v for part in parts for v in part], dtype=np.float32)
        if dimensions:
            vectors = vectors[:, :dimensions]
        if normalize and len(vectors):
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        return vectors

    # --------------------------------------------------------------- parallel

    def map(self, fn: Callable[[T], Any], items: Iterable[T], *, desc: str | None = None,
            max_workers: int | None = None, return_exceptions: bool = False) -> list:
        """Run `fn` over items on a thread pool, preserving order.

        With return_exceptions=True a failing item yields its exception instead of aborting
        the whole job, so a bulk run can be inspected and the failures re-run.
        """
        items = list(items)

        def safe(item: T):
            try:
                return fn(item)
            except Exception as exc:
                if return_exceptions:
                    return exc
                raise

        with ThreadPoolExecutor(max_workers=max_workers or self.max_workers) as pool:
            results = pool.map(safe, items)
            if desc:
                from tqdm.auto import tqdm

                results = tqdm(results, total=len(items), desc=desc)
            return list(results)


def show_retries() -> None:
    """Print a line whenever the openai client retries (429, 503, timeouts), instead of waiting silently."""
    import logging

    logger = logging.getLogger("openai")
    if not any(getattr(h, "_hisrag", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s openai: %(message)s", "%H:%M:%S"))
        handler._hisrag = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _with_system_suffix(messages: list[dict], text: str) -> list[dict]:
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{text}"
    else:
        msgs.insert(0, {"role": "system", "content": text})
    return msgs
