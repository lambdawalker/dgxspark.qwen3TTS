# DGX Spark examples

Run from the repository root after `uv sync` and downloading the models:

```bash
uv run python examples/test_model_12hz_custom_voice.py
uv run python examples/test_model_12hz_voice_design.py
uv run python examples/test_model_12hz_base.py
uv run python examples/test_tokenizer_12hz.py
```

Each script loads its corresponding `Qwen/...` Hugging Face model ID. This
reuses models downloaded with `hf download` without `--local-dir`, normally
under `~/.cache/huggingface/hub`. If you used `HF_HOME` or `HF_HUB_CACHE` when
downloading, use the same environment variables when running the examples.
Missing files can be downloaded automatically during loading.

The TTS examples use the 1.7B models, CUDA device 0, bfloat16, and PyTorch SDPA;
they do not require the optional `flash-attn` package. The tokenizer example
uses `Qwen/Qwen3-TTS-Tokenizer-12Hz`. To test the smaller models, change
`1.7B` to `0.6B` in `MODEL_PATH` in the CustomVoice or Base example.
There is no 0.6B VoiceDesign model in this example set.

To require cached Hugging Face files and skip Hub network checks:

```bash
HF_HUB_OFFLINE=1 uv run python examples/test_model_12hz_custom_voice.py
```

All required model files must already be cached for this mode. The Base and
tokenizer examples still fetch reference audio from external URLs; that audio
is separate from the Hugging Face model cache. For fully offline use, replace
those URLs with local audio paths. The tokenizer example also calls
`requests.get(audio_2)` in its final NumPy-input case; replace that download
and byte-buffer read with `sf.read(audio_2)` when using a local file.

The scripts write WAV outputs into the current directory or the Base example's
`qwen3_tts_test_voice_clone_output_wav` subdirectory. These are GPU inference
examples, not lightweight unit tests; model-loading configuration can be
checked without a GPU, but actual synthesis must be tested on the Spark.
