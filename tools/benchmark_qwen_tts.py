#!/usr/bin/env python3
"""Benchmark the shared Qwen TTS HTTP API without importing Torch."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TEXTS = {
    "short": "Hello from the DGX Spark.",
    "medium": (
        "This is a warm benchmark run for Qwen three TTS. The model is already "
        "resident, so this measures generation rather than loading."
    ),
    "long": (
        "Good morning. I wanted to let you know that the furniture will arrive on "
        "Friday between nine and eleven. Please make sure the front gate is open, "
        "and call me if the delivery team needs help finding the building."
    ),
}


def request_json(url: str, payload: dict | None, api_key: str | None, timeout: float):
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as wav:
        return wav.getnframes() / wav.getframerate()


def run_once(base: str, text: str, args) -> dict:
    payload = {
        "model": args.model,
        "text": text,
        "voice": {"alias": args.voice},
        "language": args.language,
    }
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.greedy:
        payload["extensions"] = {
            "qwen": {"do_sample": False, "subtalker_dosample": False}
        }
    started = time.perf_counter()
    job = request_json(base + "/v1/speech", payload, args.api_key, args.timeout)
    while True:
        state = request_json(base + job["status_url"], None, args.api_key, args.timeout)
        if state["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            break
        if time.perf_counter() - started > args.timeout:
            raise TimeoutError(f"job {job['id']} did not finish")
        time.sleep(args.poll_interval)
    elapsed = time.perf_counter() - started
    if state["status"] != "succeeded":
        raise RuntimeError(f"job {job['id']} ended {state['status']}: {state.get('error')}")
    result = state["results"][0]
    audio = b""
    if args.save_dir:
        directory = Path(args.save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{args.label}-{args.case_name}-{args.run_index}.wav"
        with urllib.request.urlopen(
            base + "/v1/assets/" + result["asset_id"], timeout=args.timeout
        ) as response:
            audio = response.read()
        path.write_bytes(audio)
    duration = result["duration"]
    return {
        "wall_s": round(elapsed, 3),
        "audio_s": round(duration, 3),
        "rtf": round(elapsed / duration, 3) if duration else None,
        "bytes": result["size"],
        "path": str(path) if args.save_dir else None,
        "status": state["status"],
    }


def summarize(samples: list[dict]) -> dict:
    out = {}
    for key in ("wall_s", "audio_s", "rtf"):
        values = [s[key] for s in samples if s.get(key) is not None]
        if values:
            out[key] = {
                "median": round(statistics.median(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-tts-1.7b-customvoice")
    parser.add_argument("--voice", default="ryan")
    parser.add_argument("--language", default="en")
    parser.add_argument("--instructions", default="Speak warmly and clearly.")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling for a deterministic attention comparison.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--save-dir")
    parser.add_argument("--label", default="qwen")
    parser.add_argument("--output", default="qwen-benchmark.json")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    for _ in range(args.warmup):
        run_once(base, "Warm up.", args.__class__(
            **{**vars(args), "case_name": "warmup", "run_index": 0}
        ))

    cases = []
    for case_name, text in DEFAULT_TEXTS.items():
        samples = []
        for index in range(args.runs):
            args.case_name = case_name
            args.run_index = index
            samples.append(run_once(base, text, args))
        cases.append(
            {
                "case": case_name,
                "chars": len(text),
                "samples": samples,
                "summary": summarize(samples),
            }
        )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "model": args.model,
        "voice": args.voice,
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("| case | chars | audio s | wall s | RTF |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for case in cases:
        summary = case["summary"]
        print(
            f"| {case['case']} | {case['chars']} | "
            f"{summary.get('audio_s', {}).get('median', '-')} | "
            f"{summary.get('wall_s', {}).get('median', '-')} | "
            f"{summary.get('rtf', {}).get('median', '-')} |"
        )
    print(f"\nWrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
