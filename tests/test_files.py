import os
import sys

import pytest

from hisrag.files import append_line


def test_append_line_creates_folders_and_appends(tmp_path):
    path = tmp_path / "a" / "b" / "log.jsonl"
    append_line(path, '{"x": 1}')
    append_line(path, '{"x": 2}')
    assert path.read_text(encoding="utf-8").splitlines() == ['{"x": 1}', '{"x": 2}']


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_new_files_and_folders_are_group_writable(tmp_path):
    old = os.umask(0o022)
    try:
        path = tmp_path / "logs" / "log.jsonl"
        append_line(path, "x")
    finally:
        os.umask(old)
    assert path.stat().st_mode & 0o777 == 0o664
    assert path.parent.stat().st_mode & 0o2777 == 0o2775
