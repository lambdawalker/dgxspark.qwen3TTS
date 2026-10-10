# Qwen TTS integration guidance

- Architecture: https://github.com/lambdawalker/design.ai/tree/main/tts.
- Shared application server: https://github.com/lambdawalker/python.tts.api.server.
- Client: https://github.com/lambdawalker/python.tts.api.client.
- `qwen_tts_api` owns the native adapter and deployment factory. Do not duplicate HTTP,
  MCP, queues, job persistence or authorization inside this repository.
- Keep discovery imports free of Torch, CUDA, Gradio and model-weight loading.
- Preserve caller content exactly. Reject unsupported size/family combinations.
- Use immutable checkpoint snapshots; native component loaders do not consistently
  forward `revision`. Do not remove the snapshot boundary.
- CPU fixtures are not GPU or acoustic validation. Follow `docs/api.md` for checks.
- Keep existing Gradio and native Python interfaces working. Avoid unrelated core edits.
- This fork targets Linux ARM64/Python 3.13/CUDA 13. CPU CI installs only
  `tests/api/requirements.txt`; do not resolve a CPU Torch build into the deployment lock.
