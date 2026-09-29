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


def env_file() -> Path:
    """The secrets file: $HISRAG_ENV_FILE if set, else .env in the repo root."""
    return Path(os.environ.get("HISRAG_ENV_FILE", REPO_ROOT / ".env"))


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines, never overrides variables that are already non-empty."""
    path = path or env_file()
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if value and not os.environ.get(key):
            os.environ[key] = value


def set_api_key(name: str = "DHINFRA_API_KEY", *, save: bool = True, path: Path | None = None) -> None:
    """Ask for the API key without echoing it, set it for this session and optionally store it in .env.

    Meant for JupyterHub, where .env is hidden in the file browser and no editor may be available.
    The file is written with owner-only permissions.
    """
    from getpass import getpass

    key = getpass(f"{name}: ").strip()
    if not key:
        raise ValueError("No key entered")
    os.environ[name] = key
    if not save:
        return
    path = path or env_file()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [line for line in lines if line.split("=", 1)[0].strip() != name] + [f"{name}={key}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    print(f"Saved {name} to {path}")


class Config(dict):
    """Nested dict with helpers for resolving paths against the repo root."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self["api"]["api_key_env"]) or None


def load_config(path: Path | str | None = None, local: Path | str | None = None) -> Config:
    load_dotenv()
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = Path(local or os.environ.get("HISRAG_CONFIG_LOCAL", REPO_ROOT / "config.local.yaml"))
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return Config(cfg)
