"""Check launcher synchronization and failure behavior with command stand-ins."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    shutil.copyfile(ROOT / "start.sh", tmp_path / "start.sh")
    commands = tmp_path / "commands"
    commands.mkdir()
    venv = tmp_path / ".venv/bin"
    venv.mkdir(parents=True)
    log = tmp_path / "commands.log"

    def executable(path, content):
        path.write_text("#!/bin/sh\nset -eu\n" + content)
        path.chmod(0o755)

    executable(commands / "uname", "echo aarch64\n")
    executable(
        commands / "uv",
        'echo "uv $*" >> "$LAUNCH_LOG"\n'
        'if [ "${FAIL_SYNC:-0}" = 1 ]; then exit 7; fi\n',
    )
    executable(venv / "python", "exit 0\n")
    executable(venv / "qwen-tts-api", 'echo "api $*" >> "$LAUNCH_LOG"\n')
    env = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "LAUNCH_LOG": str(log),
        "QWEN_TTS_AUTH": "none",
        "QWEN_TTS_MCP": "0",
        "QWEN_TTS_ATTENTION": "flash_attention_2",
    }
    return tmp_path, log, env


def test_launcher_syncs_existing_environment_on_every_launch(launcher):
    root, log, env = launcher
    for _ in range(2):
        subprocess.run(["sh", str(root / "start.sh")], env=env, check=True, capture_output=True)
    lines = log.read_text().splitlines()
    assert len(lines) == 4
    for offset in (0, 2):
        assert lines[offset] == "uv sync --frozen --extra api --inexact"
        assert lines[offset + 1].startswith("api ")


def test_launcher_stops_if_dependency_sync_fails(launcher):
    root, log, env = launcher
    result = subprocess.run(
        ["sh", str(root / "start.sh")],
        env={**env, "FAIL_SYNC": "1"},
        capture_output=True,
    )
    assert result.returncode == 7
    assert not any(line.startswith("api ") for line in log.read_text().splitlines())
