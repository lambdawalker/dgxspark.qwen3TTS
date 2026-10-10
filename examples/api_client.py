"""Run on a client machine after starting qwen-tts-api on the DGX Spark."""

import argparse
import os
from pathlib import Path

from tts_api_client import TTSClient


def main():
    parser = argparse.ArgumentParser(
        description="Exercise Qwen speech, design and reference cloning"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("api-output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TTSClient(args.url, api_key=os.environ.get("TTS_API_KEY")) as client:

        def save(job, filename):
            result = job.wait(timeout=1800, poll_interval=1)
            client.download_asset(result.results[0].asset_id, args.output_dir / filename)
            print(f"{job.id}: {result.status} -> {filename}")
            return result.results[0].asset_id

        speech = client.generate_speech(
            model="qwen3-tts-1.7b-customvoice",
            text="Hello from the shared Qwen API.",
            voice={"alias": "ryan"},
            language="en",
            instructions="Speak warmly and clearly.",
        )
        save(speech, "custom.wav")
        text = "This sample introduces a warm, calm narrator."
        candidate = save(
            client.design_voice(
                model="qwen3-tts-1.7b-voicedesign",
                description="A warm, calm adult narrator.",
                preview_text=text,
                language="en",
            ),
            "design.wav",
        )
        voice = client.register_voice(
            model="qwen3-tts-1.7b-base",
            references=[{"asset_id": candidate, "transcript": text}],
        )
        save(
            client.generate_speech(
                model="qwen3-tts-1.7b-base",
                text="Now the same reference voice says something new.",
                voice={"id": voice.id},
                language="en",
            ),
            "clone.wav",
        )


if __name__ == "__main__":
    main()
