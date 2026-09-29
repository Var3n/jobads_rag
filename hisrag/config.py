"""Configuration: config.yaml, overridden by config.local.yaml, plus secrets from .env."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(os.environ.get("HISRAG_ROOT", Path(__file__).resolve().parent.parent))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines, never overrides variables already set."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class Config(dict):
    """Nested dict with helpers for resolving paths against the repo root."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self["api"]["api_key_env"])


def load_config(path: Path | str | None = None, local: Path | str | None = None) -> Config:
    load_dotenv()
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = Path(local or os.environ.get("HISRAG_CONFIG_LOCAL", REPO_ROOT / "config.local.yaml"))
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return Config(cfg)
