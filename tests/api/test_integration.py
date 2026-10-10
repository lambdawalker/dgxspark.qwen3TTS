"""Real HTTP/MCP transports with a native-call test double; no GPU claims."""

import asyncio
import time

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_adapter import NativeModel
from tts_api_client import TTSClient
from tts_api_server.http import create_app
from tts_api_server.service import Settings

from qwen_tts_api.adapter import QwenAdapter
from qwen_tts_api.profiles import SPECS


@pytest.fixture
def app(tmp_path):
    adapter = QwenAdapter(
        {key: "a" * 40 for key in SPECS},
        loader=lambda s, r: NativeModel(s),
        releaser=lambda: None,
    )
    return create_app(
        [adapter],
        Settings(
            data_dir=tmp_path / "state",
            default_model="qwen3-tts-1.7b-customvoice",
            tokens={"secret": "alice"},
        ),
        enable_mcp=True,
    )


def test_client_speech_design_promote_clone(app, live_server, tmp_path):
    with live_server(app) as url, TTSClient(url, api_key="secret") as client:
        assert len(client.list_models()) == 5
        assert client.guidance("tts").feature == "tts"
        assert client.capabilities().controls["instructions"]["supported"]
        job = client.generate_speech(
            text="  Hello!\n",
            voice={"alias": "ryan"},
            instructions="Calmly.",
            idempotency_key="example",
        )
        done = job.wait(timeout=5, poll_interval=0.01)
        assert done.status == "succeeded"
        assert list(client.events(job.id))[-1].data["status"] == "succeeded"
        assert (
            client.generate_speech(
                text="  Hello!\n",
                voice={"alias": "ryan"},
                instructions="Calmly.",
                idempotency_key="example",
            ).id
            == job.id
        )
        output = client.download_asset(done.results[0].asset_id, tmp_path / "speech.wav")
        assert sf.info(output).samplerate == 24000
        small = client.generate_speech(
            model="qwen3-tts-0.6b-customvoice", text="Hello.", voice={"alias": "ryan"}
        )
        assert small.wait(timeout=5, poll_interval=0.01).status == "succeeded"
        design = client.design_voice(
            model="qwen3-tts-1.7b-voicedesign",
            description="A warm voice.",
            preview_text="A sample sentence.",
        )
        candidate = design.wait(timeout=5, poll_interval=0.01).results[0]
        voice = client.register_voice(
            model="qwen3-tts-1.7b-base",
            alias="designed",
            references=[{"asset_id": candidate.asset_id, "transcript": "A sample sentence."}],
        )
        clone = client.generate_speech(
            model="qwen3-tts-1.7b-base",
            text="New words.",
            voice={"id": voice.id},
            output={"format": "flac"},
        )
        assert clone.wait(timeout=5, poll_interval=0.01).results[0].mime_type == "audio/flac"
        # Exercise upload and transcript-free speaker embedding mode as well.
        asset = client.upload_asset(output)
        ref = client.register_voice(
            model="qwen3-tts-0.6b-base", references=[{"asset_id": asset.id}]
        )
        xvector = client.generate_speech(
            model="qwen3-tts-0.6b-base",
            text="Hello.",
            voice={"id": ref.id},
            extensions={"qwen": {"x_vector_only_mode": True}},
        )
        assert xvector.wait(timeout=5, poll_interval=0.01).status == "succeeded"


@pytest.mark.parametrize(
    "payload, code",
    [
        (
            {
                "model": "qwen3-tts-0.6b-customvoice",
                "voice": {"alias": "ryan"},
                "instructions": "Happy",
            },
            "unsupported_parameter",
        ),
        ({"voice": {"alias": "missing"}}, "voice_unavailable"),
        ({"language": "xx", "voice": {"alias": "ryan"}}, "unsupported_language"),
        (
            {"voice": {"alias": "ryan"}, "extensions": {"qwen": {"seed": 1}}},
            "unsupported_parameter",
        ),
        (
            {"voice": {"alias": "ryan"}, "extensions": {"qwen": {"top_k": -1}}},
            "unsupported_parameter",
        ),
        (
            {
                "voice": {"alias": "ryan"},
                "extensions": {"qwen": {"x_vector_only_mode": True}},
            },
            "unsupported_parameter",
        ),
        ({"model": "qwen3-tts-1.7b-voicedesign"}, "unsupported_feature"),
        (
            {"voice": {"alias": "ryan"}, "output": {"sample_rate": 16000}},
            "unsupported_parameter",
        ),
    ],
)
def test_invalid_combinations_rejected_before_loading(app, payload, code):
    with TestClient(app) as client:
        response = client.post(
            "/v1/speech",
            headers={"Authorization": "Bearer secret"},
            json={"text": "hi", **payload},
        )
        assert response.status_code == (404 if code == "voice_unavailable" else 422)
        assert response.json()["error"]["code"] == code
        assert app.state.service.loaded is None


def test_mcp_uses_same_profile_validation(app, live_server):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def check(url):
        async with (
            httpx.AsyncClient(headers={"Authorization": "Bearer secret"}, trust_env=False) as http,
            streamable_http_client(url + "/mcp", http_client=http) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "generate_speech", {"text": "Hello", "voice": {"alias": "ryan"}}
            )
            assert not result.isError
            invalid = await session.call_tool(
                "validate_speech",
                {
                    "model": "qwen3-tts-0.6b-customvoice",
                    "text": "Hello",
                    "voice": {"alias": "ryan"},
                    "instructions": "Happy",
                },
            )
            assert invalid.isError
            assert invalid.structuredContent["error"]["code"] == "unsupported_parameter"

    with live_server(app) as url:
        asyncio.run(check(url))


def test_cancel_during_native_call_discards_audio(tmp_path, live_server):
    import threading

    entered, release = threading.Event(), threading.Event()

    class Blocking(NativeModel):
        def generate_custom_voice(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().generate_custom_voice(**kwargs)

    profile = "qwen3-tts-1.7b-customvoice"
    adapter = QwenAdapter(
        {profile: "a" * 40}, loader=lambda s, r: Blocking(s), releaser=lambda: None
    )
    app = create_app([adapter], Settings(data_dir=tmp_path))
    with live_server(app) as url, TTSClient(url) as client:
        try:
            job = client.generate_speech(text="hi", voice={"alias": "ryan"})
            assert entered.wait(3)
            client.cancel_job(job.id)
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while (state := client.get_job(job.id)).status not in {
            "cancelled",
            "failed",
            "succeeded",
        }:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert state.status == "cancelled" and not state.results
