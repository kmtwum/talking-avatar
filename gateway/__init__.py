"""
Gateway Module - LLM-to-Video Streaming Pipeline

This module provides the bridge between LLM text streams and the 
Talking Avatar video generation service.

Architecture:
    Client (React) <-- WebSocket --> Gateway (Python) <-- WebSocket --> Avatar Service

Components:
    - AvatarSession: Manages WebSocket connection to avatar service
    - LLMStreamProcessor: Buffers LLM tokens into sentences
    - GPT: OpenAI integration for chat-to-video
    - stream_llm_to_video: Main entry point for custom LLM streams

Quick Start:
    from gateway import stream_llm_to_video, GatewayConfig
    
    async def my_llm():
        for token in llm_response:
            yield token
    
    await stream_llm_to_video(
        client_ws=websocket,
        llm_stream=my_llm(),
        config=GatewayConfig(avatar="sunny")
    )
"""

from gateway.avatar_session import AvatarSession, AvatarConfig
from gateway.llm_to_video import (
    GatewayConfig,
    SentenceBuffer,
    LLMStreamProcessor,
    stream_llm_to_video,
    openai_stream_to_video,
    create_gateway_app,
)
from gateway.gpt import GPT, chat_to_video

__all__ = [
    # Core classes
    "AvatarSession",
    "AvatarConfig",
    "GatewayConfig",
    "SentenceBuffer",
    "LLMStreamProcessor",
    # Main functions
    "stream_llm_to_video",
    "openai_stream_to_video",
    "chat_to_video",
    # GPT integration
    "GPT",
    # FastAPI app factory
    "create_gateway_app",
]
