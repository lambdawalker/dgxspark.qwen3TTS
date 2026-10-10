import sys

import pytest

from qwen_tts_api.factory import create_adapter, resolve_revision


def test_factory_resolves_revisions_without_loading_torch():
    seen = []

    def resolve(checkpoint, revision):
        seen.append((checkpoint, revision))
        return "b" * 40

    adapter = create_adapter(
        {
            "QWEN_TTS_PROFILES": "qwen3-tts-1.7b-base",
            "QWEN_TTS_REVISIONS": '{"qwen3-tts-1.7b-base":"v1"}',
        },
        resolver=resolve,
    )
    assert seen == [("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "v1")]
    assert adapter.profiles()[0].checkpoint_revision == "b" * 40
    assert adapter.profiles()[0].inference_mode == "native-12hz/cuda:0/bfloat16/sdpa"
    assert "torch" not in sys.modules and "qwen_tts" not in sys.modules
    assert resolve_revision("unused", "c" * 40) == "c" * 40


@pytest.mark.parametrize(
    "env",
    [
        {"QWEN_TTS_PROFILES": ""},
        {"QWEN_TTS_PROFILES": "unknown"},
        {"QWEN_TTS_REVISIONS": "[]"},
        {"QWEN_TTS_REVISIONS": '{"wrong":"main"}'},
        {"QWEN_TTS_DTYPE": "int4"},
        {"QWEN_TTS_ATTENTION": "wrong"},
    ],
)
def test_factory_rejects_invalid_configuration_before_network(env):
    with pytest.raises(ValueError):
        create_adapter(env, resolver=lambda *a: pytest.fail("unexpected network resolution"))


def test_cli_delegates_shared_auth_and_options(monkeypatch):
    import tts_api_server.cli

    from qwen_tts_api.cli import main

    calls = []
    monkeypatch.setattr(tts_api_server.cli, "main", calls.append)
    main(["--port", "8123", "--mcp"])
    assert calls == [
        [
            "--host",
            "0.0.0.0",
            "--adapter",
            "qwen_tts_api.factory:create_adapter",
            "--port",
            "8123",
            "--mcp",
        ]
    ]


@pytest.mark.parametrize(
    "host_args, expected", [([], "0.0.0.0"), (["--host", "127.0.0.1"], "127.0.0.1")]
)
def test_cli_default_host_and_override(monkeypatch, host_args, expected):
    import tts_api_server.cli

    from qwen_tts_api.cli import main

    parsed = []
    monkeypatch.setattr(
        tts_api_server.cli,
        "main",
        lambda argv: parsed.append(tts_api_server.cli.build_parser().parse_args(argv)),
    )
    main(host_args)
    assert parsed[0].host == expected
    if expected == "0.0.0.0":
        with pytest.raises(ValueError, match="TTS_API_TOKENS"):
            tts_api_server.cli.settings_from_args(parsed[0], {})
    settings = tts_api_server.cli.settings_from_args(
        parsed[0], {"TTS_API_TOKENS": '{"test-token":"caller"}'}
    )
    assert settings.tokens == {"test-token": "caller"}
