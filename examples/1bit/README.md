# 1bit PA Local-First Pipecat Example

This example keeps Pipecat focused on voice I/O and leaves authority to the 1bit PA control plane.

Default local path:

- STT: local Faster Whisper through `WhisperSTTService`
- LLM: local OpenAI-compatible endpoint, for example llama.cpp server
- TTS: local Piper through `PiperTTSService`
- Transport: local microphone/speaker through `LocalAudioTransport`

Cloud services are intentionally not required for this example.

## Start A Local LLM

Example llama.cpp server:

```bash
/home/bcloud/prism-llama.cpp/build-rocm/bin/llama-server \
  --host 127.0.0.1 \
  --port 18080 \
  -m /home/bcloud/models/deepseek-r1-distill-gguf/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  -ngl 999 \
  -fa 1 \
  --no-mmap
```

## Install Local Extras

```bash
uv sync --extra local --extra whisper --extra piper
```

Piper may download the configured voice model on first run.

## Run

```bash
ONEBIT_LLM_BASE_URL=http://127.0.0.1:18080/v1 \
ONEBIT_LLM_MODEL=deepseek-r1-distill-qwen-7b-q4 \
ONEBIT_PIPER_VOICE=en_US-ryan-high \
uv run python examples/1bit/voice-agent-local-first.py
```

## Architecture Rule

Do not grant tool authority directly in the Pipecat pipeline. Route any action requests to the 1bit PA control plane so work primitives, memory zones, approvals, and audit hooks decide what can happen.
