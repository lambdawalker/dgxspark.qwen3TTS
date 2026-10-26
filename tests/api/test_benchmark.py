"""Exercise the benchmark CLI against the pinned server without GPU inference."""

import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from tts_api_server.fake import FakeAdapter
from tts_api_server.http import create_app
from tts_api_server.models import Voice
from tts_api_server.service import Settings

ROOT = Path(__file__).resolve().parents[2]


class BenchmarkAdapter(FakeAdapter):
    def profiles(self):
        profile = super().profiles()[0]
        profile.voices = [Voice(id="ryan", model=profile.id, alias="ryan", kind="builtin")]
        return [profile]


@pytest.mark.parametrize("mode", ["anonymous", "token", "none"])
def test_benchmark_cli_auth_and_audio_download(mode, tmp_path, live_server):
    settings = Settings(
        data_dir=tmp_path / "state",
        anonymous_sessions=mode == "anonymous",
        tokens={"secret": "benchmark"} if mode == "token" else {},
    )
    app = create_app([BenchmarkAdapter()], settings)
    output = tmp_path / "results.json"
    audio = tmp_path / "audio"
    with live_server(app) as url:
        command = [
            sys.executable,
            str(ROOT / "tools/benchmark_qwen_tts.py"),
            "--url",
            url,
            "--model",
            "fake-v1",
            "--warmup",
            "0",
            "--runs",
            "1",
            "--poll-interval",
            "0.01",
            "--timeout",
            "5",
            "--save-dir",
            str(audio),
            "--output",
            str(output),
        ]
        if mode == "token":
            command += ["--api-key", "secret"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert len(report["cases"]) == 3
    for case in report["cases"]:
        sample = case["samples"][0]
        assert sample["status"] == "succeeded"
        assert sample["audio_s"] == pytest.approx(0.1)
        path = Path(sample["path"])
        assert path.stat().st_size == sample["bytes"]
        with wave.open(str(path)) as wav:
            assert wav.getframerate() == 24000
            assert wav.getnframes() == 2400
    assert "secret" not in output.read_text()


def test_benchmark_requires_key_on_static_token_server(tmp_path, live_server):
    app = create_app(
        [BenchmarkAdapter()],
        Settings(data_dir=tmp_path, tokens={"secret": "benchmark"}),
    )
    with live_server(app) as url:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/benchmark_qwen_tts.py"),
                "--url",
                url,
                "--warmup",
                "0",
                "--runs",
                "1",
                "--timeout",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    assert result.returncode != 0
    assert "--api-key" in result.stderr
