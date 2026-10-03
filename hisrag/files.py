"""Files that several users write in a shared project folder (logs on the cluster).

A file or folder created here is made group-writable (folders also set-group-ID, so new files keep the group),
so the next researcher can append to it. Appends are locked with fcntl where it exists (not on Windows), because a
long line is not written atomically.
"""

from __future__ import annotations

import os
from pathlib import Path


def _share(path: Path, mode: int) -> None:
    try:
        if path.stat().st_uid == os.getuid():  # only the owner may change the mode
            os.chmod(path, mode)
    except (AttributeError, OSError):  # no os.getuid on Windows; a foreign file stays as it is
        pass


def shared_dir(path: Path) -> None:
    missing = [p for p in [path, *path.parents] if not p.exists()]
    path.mkdir(parents=True, exist_ok=True)
    for p in missing:
        _share(p, 0o2775)


def append_line(path: Path, line: str) -> None:
    """Append one line (newline added) to a file other group members can append to as well."""
    shared_dir(path.parent)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            _share(path, 0o664)
        try:
            import fcntl
        except ImportError:  # Windows
            f.write(line + "\n")
            return
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
