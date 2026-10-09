"""Qwen launcher using the shared server's authentication and process settings."""

import sys


def main(argv=None):
    from tts_api_server.cli import main as serve

    serve(
        [
            "--adapter",
            "qwen_tts_api.factory:create_adapter",
            *(sys.argv[1:] if argv is None else argv),
        ]
    )


if __name__ == "__main__":
    main()
