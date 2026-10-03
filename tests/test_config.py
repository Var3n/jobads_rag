import os
import stat
import sys

import pytest

from hisrag import config


def test_dotenv_fills_empty_but_keeps_set_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA_KEY='from-file'\nB_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("A_KEY", "")            # empty, e.g. from `cp .env.example .env` + export
    monkeypatch.setenv("B_KEY", "from-shell")

    config.load_dotenv(env)

    assert os.environ["A_KEY"] == "from-file"
    assert os.environ["B_KEY"] == "from-shell"


def test_set_api_key_replaces_line_in_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nDHINFRA_API_KEY=\n", encoding="utf-8")
    monkeypatch.setattr("getpass.getpass", lambda prompt: "  sk-test  ")
    monkeypatch.delenv("DHINFRA_API_KEY", raising=False)

    config.set_api_key(path=env)

    assert env.read_text(encoding="utf-8") == "OTHER=1\nDHINFRA_API_KEY=sk-test\n"
    assert os.environ["DHINFRA_API_KEY"] == "sk-test"
    if sys.platform != "win32":
        assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_set_api_key_rejects_empty_input(tmp_path, monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda prompt: "")
    with pytest.raises(ValueError):
        config.set_api_key(path=tmp_path / ".env")
    assert not (tmp_path / ".env").exists()


def test_env_file_can_live_outside_the_repo(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.env"
    monkeypatch.setenv("HISRAG_ENV_FILE", str(secrets))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-outside")
    monkeypatch.delenv("DHINFRA_API_KEY", raising=False)

    config.set_api_key()
    monkeypatch.delenv("DHINFRA_API_KEY")
    config.load_dotenv()

    assert config.env_file() == secrets
    assert os.environ["DHINFRA_API_KEY"] == "sk-outside"


def test_a_researcher_in_the_owners_folder_gets_a_key_file_of_their_own(tmp_path, monkeypatch):
    monkeypatch.delenv("HISRAG_ENV_FILE", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path / "home")
    assert config.env_file() == tmp_path / "repo" / ".env"  # no .env yet: the owner's first setup
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("DHINFRA_API_KEY=owner\n", encoding="utf-8")
    assert config.env_file() == tmp_path / "repo" / ".env"
    monkeypatch.setattr(config.os, "access", lambda p, mode: False)  # the owner's file, mode 600
    assert config.env_file() == tmp_path / "home" / ".hisrag.env"
    config.load_dotenv(tmp_path / "missing.env")  # unreadable or missing: no error
