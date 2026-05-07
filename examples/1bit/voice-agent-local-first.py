#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local-first 1bit PA voice pipeline.

Pipecat owns the realtime voice pipeline here. The 1bit PA control plane should
own tools, memory writes, approvals, and other durable authority.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level=os.getenv("ONEBIT_LOG_LEVEL", "INFO"))


SYSTEM_INSTRUCTION = """You are 1bit PA in a live voice conversation.
Keep spoken replies brief and natural.
Do not use markdown, bullets, emoji, or visual formatting.
You may explain what you can do, but do not claim to have completed actions unless
the 1bit control plane has actually confirmed them.
For actions that change files, spend money, send messages, deploy software, touch
secrets, or modify memory, say that you need approval from the control plane.
"""


def env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


async def main() -> None:
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = WhisperSTTService(
        model=env("ONEBIT_WHISPER_MODEL", "distil-medium.en"),
        device=env("ONEBIT_WHISPER_DEVICE", "cpu"),
        compute_type=env("ONEBIT_WHISPER_COMPUTE_TYPE", "int8"),
    )

    voice_dir = Path(env("ONEBIT_PIPER_VOICE_DIR", str(Path.home() / ".local/share/piper")))
    voice_dir.mkdir(parents=True, exist_ok=True)
    tts = PiperTTSService(
        download_dir=voice_dir,
        settings=PiperTTSService.Settings(
            voice=env("ONEBIT_PIPER_VOICE", "en_US-ryan-high"),
        ),
    )

    llm = OpenAILLMService(
        api_key=env("ONEBIT_LLM_API_KEY", "local"),
        base_url=env("ONEBIT_LLM_BASE_URL", "http://127.0.0.1:18080/v1"),
        settings=OpenAILLMService.Settings(
            model=env("ONEBIT_LLM_MODEL", "local-model"),
            temperature=float(env("ONEBIT_LLM_TEMPERATURE", "0.4")),
            max_tokens=int(env("ONEBIT_LLM_MAX_TOKENS", "160")),
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    context.add_message(
        {
            "role": "developer",
            "content": "Briefly introduce yourself as the local 1bit PA voice prototype.",
        }
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
