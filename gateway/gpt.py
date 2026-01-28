"""
GPT Integration with LLM-to-Video Pipeline

Connects OpenAI streaming responses to the Talking Avatar service
for real-time video generation.
"""

from typing import Optional, Callable, Awaitable
import re
import time

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from gateway.llm_to_video import (
    GatewayConfig,
    AvatarSession,
    LLMStreamProcessor,
    stream_llm_to_video,
)


# Sentence end pattern for legacy compatibility
SENTENCE_END_RE = re.compile(r"[.!?]\s$")


class CloneResponse(BaseModel):
    """Response model for clone actions."""
    text: str = Field(description="Action text to display")
    slug: Optional[str] = Field(description="Action slug")
    link: Optional[str] = Field(description="Action link URL")


class GPT:
    """
    GPT integration for chat-to-video streaming.
    
    Supports both legacy sentence-based callbacks and the new
    streaming pipeline for real-time video generation.
    """
    
    @staticmethod
    async def ask_with_video(
        client_ws,
        question: str,
        avatar: str = "sunny",
        voice_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = "gpt-4",
        avatar_ws_url: str = "ws://77.68.21.101:8002/ws/generate",
        on_token: Optional[Callable[[str, int], Awaitable[None]]] = None,
        on_sentence: Optional[Callable[[str, int], Awaitable[None]]] = None,
    ):
        """
        Stream GPT response to video via avatar service.
        
        This is the new unified API that handles the full pipeline:
        1. Streams GPT response
        2. Buffers into sentences
        3. Sends to avatar service
        4. Relays video to client
        
        Args:
            client_ws: WebSocket connection to the React client
            question: User's question/prompt
            avatar: Avatar identifier
            voice_id: Voice ID for TTS (ElevenLabs)
            system_prompt: System prompt for GPT
            model: GPT model to use
            avatar_ws_url: URL of the avatar WebSocket endpoint
            on_token: Optional callback for each token
            on_sentence: Optional callback for each sentence
        """
        config = GatewayConfig(
            avatar_ws_url=avatar_ws_url,
            avatar=avatar,
            voice_id=voice_id,
            gateway_aggregate=True,
            gateway_min_chars=40,
            gateway_max_chars=200,
            gateway_timeout=0.8,
        )
        
        client = AsyncOpenAI()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        async def llm_stream():
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
            
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
        await stream_llm_to_video(
            client_ws=client_ws,
            llm_stream=llm_stream(),
            config=config,
            on_token=on_token,
            on_sentence=on_sentence,
        )
    
    @staticmethod
    async def ask_legacy(
        question: str,
        context: Optional[str] = None,
        socket=None,
        on_token: Optional[Callable] = None,
        on_sentence: Optional[Callable] = None,
        session: Optional[AvatarSession] = None
    ):
        """
        Legacy API: Stream GPT response with callbacks.
        
        Use ask_with_video() for the new unified pipeline.
        """
        client = OpenAI()
        
        messages = [{"role": "user", "content": question}]
        if context:
            messages.insert(0, {"role": "system", "content": context})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        
        buffer = ""
        last_emit = time.monotonic()
        seq_token = 0
        seq_sentence = 0
        
        for chunk in response:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            
            token = content
            buffer += token
            seq_token += 1
            
            if on_token:
                await on_token(token, seq_token)
            
            now = time.monotonic()
            
            # Check if should emit sentence
            should_emit = (
                SENTENCE_END_RE.search(buffer)
                or len(buffer) >= 120
                or (now - last_emit) >= 0.6
            )
            
            if should_emit:
                text = buffer.strip()
                buffer = ""
                last_emit = now
                seq_sentence += 1
                
                if on_sentence and text:
                    await on_sentence(text, seq_sentence)
                
                # Send to avatar session if available
                if session and text:
                    await session.send_sentence(text, seq_sentence)
        
        # Flush remaining buffer
        if buffer.strip():
            seq_sentence += 1
            if on_sentence:
                await on_sentence(buffer.strip(), seq_sentence)
            if session:
                await session.send_sentence(buffer.strip(), seq_sentence)


# Convenience function for direct usage
async def chat_to_video(
    websocket,
    question: str,
    avatar: str = "sunny",
    voice_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """
    Simplified API for streaming chat to video.
    
    Usage:
        @app.websocket("/ws/chat")
        async def chat(websocket: WebSocket):
            await websocket.accept()
            msg = await websocket.receive_json()
            await chat_to_video(
                websocket,
                question=msg["prompt"],
                avatar=msg.get("avatar", "sunny")
            )
    """
    await GPT.ask_with_video(
        client_ws=websocket,
        question=question,
        avatar=avatar,
        voice_id=voice_id,
        system_prompt=system_prompt,
    )