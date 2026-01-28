"""
LLM Stream to Video Gateway

This module handles the pipeline:
1. Client (React) connects via WebSocket
2. LLM generates text stream
3. Text chunks are sent to Talking Avatar service
4. Video chunks are streamed back to client

The gateway integrates with the new socket-based avatar API that supports:
- Chunk aggregation (buffering small chunks into sentences)
- Audio pre-buffering (smooth video playback)
"""

import asyncio
import json
import uuid
import time
import re
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field

import websockets
from websockets.exceptions import ConnectionClosed


# Sentence detection patterns
SENTENCE_END_RE = re.compile(r'[.!?。！？]\s*$')
CONTINUATION_RE = re.compile(r'[,;:—–]\s*$|\.{2,}\s*$')


@dataclass
class GatewayConfig:
    """Configuration for the gateway."""
    avatar_ws_url: str = "ws://77.68.21.101:8002/ws/generate"
    
    # Avatar session settings
    avatar: str = "sunny"
    size: int = 256
    tts_preference: str = "coqui"  # or "elevenlabs"
    voice_id: Optional[str] = None
    
    # Aggregation (gateway-side sentence buffering)
    gateway_aggregate: bool = True  # Buffer at gateway level too
    gateway_min_chars: int = 40
    gateway_max_chars: int = 200
    gateway_timeout: float = 0.8
    
    # Avatar-side settings (sent to avatar service)
    avatar_aggregate: bool = True
    avatar_min_chars: int = 50
    avatar_max_chars: int = 500
    avatar_aggregate_timeout: float = 1.5
    prebuffer_enabled: bool = True
    prebuffer_min_chunks: int = 1
    prebuffer_min_seconds: float = 1.0


@dataclass
class SentenceBuffer:
    """Buffers tokens into sentences for better TTS quality."""
    min_chars: int = 40
    max_chars: int = 200
    timeout_seconds: float = 0.8
    
    buffer: str = ""
    last_token_time: float = field(default_factory=time.time)
    sentences_emitted: int = 0
    
    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token to the buffer.
        
        Returns:
            Complete sentence if ready, None otherwise
        """
        self.buffer += token
        self.last_token_time = time.time()
        
        # Check if we should emit
        if self._should_emit():
            return self._emit()
        return None
    
    def check_timeout(self) -> Optional[str]:
        """Check if timeout has elapsed and emit if so."""
        if self.buffer and (time.time() - self.last_token_time) >= self.timeout_seconds:
            return self._emit()
        return None
    
    def flush(self) -> Optional[str]:
        """Force emit any remaining content."""
        if self.buffer.strip():
            return self._emit()
        return None
    
    def _should_emit(self) -> bool:
        """Check if buffer should be emitted."""
        text = self.buffer
        
        # Force emit at max length
        if len(text) >= self.max_chars:
            return True
        
        # Check for sentence end (only if min length met)
        if len(text) >= self.min_chars:
            if SENTENCE_END_RE.search(text) and not CONTINUATION_RE.search(text):
                return True
        
        return False
    
    def _emit(self) -> str:
        """Emit the buffer content."""
        text = self.buffer.strip()
        self.buffer = ""
        self.sentences_emitted += 1
        return text


class AvatarSession:
    """
    Manages bidirectional WebSocket connection to the Talking Avatar service.
    
    Handles:
    - Session lifecycle (start, send chunks, end)
    - Video chunk forwarding to client
    - Error handling and reconnection
    """
    
    def __init__(
        self,
        config: GatewayConfig,
        client_ws,
        user_id: Optional[str] = None
    ):
        self.config = config
        self.client_ws = client_ws
        self.user_id = user_id
        
        self.session_id = str(uuid.uuid4())[:8]
        self.avatar_ws = None
        self.closed = False
        self._ending = False
        
        # Queues for async pipeline
        self.sentence_queue: asyncio.Queue = asyncio.Queue()
        self.video_queue: asyncio.Queue = asyncio.Queue(maxsize=120)
        
        # Event for session completion (set when avatar sends SESSION_COMPLETE)
        self.session_complete = asyncio.Event()
        
        # Statistics
        self.sentences_sent = 0
        self.video_chunks_received = 0
        self.start_time: Optional[float] = None
        
        # Tasks
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._client_sender_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the avatar session and background tasks."""
        self.start_time = time.time()
        
        print(f"[Gateway {self.session_id}] Connecting to avatar: {self.config.avatar_ws_url}")
        
        try:
            # Extract host from URL for proper headers
            import urllib.parse
            parsed = urllib.parse.urlparse(self.config.avatar_ws_url)
            host = parsed.netloc  # e.g., "77.68.21.101:8002"
            origin = f"http://{host}" if parsed.scheme == "ws" else f"https://{host}"
            
            # Add extra headers to bypass potential proxy issues
            extra_headers = {
                "Origin": origin,
                "Host": host,
            }
            
            print(f"[Gateway {self.session_id}] Using headers: {extra_headers}")
            
            self.avatar_ws = await websockets.connect(
                self.config.avatar_ws_url,
                max_size=10 * 1024 * 1024,  # 10MB max message
                ping_interval=20,
                ping_timeout=10
            )
        except Exception as e:
            print(f"[Gateway {self.session_id}] Failed to connect to avatar: {e}")
            raise
        
        # Send SESSION_START with all configuration
        await self.avatar_ws.send(json.dumps({
            "type": "SESSION_START",
            "avatar": self.config.avatar,
            "size": self.config.size,
            "tts_preference": self.config.tts_preference,
            "tts_voice_id": self.config.voice_id,
            "user_id": self.user_id,
            # Aggregation settings (avatar-side)
            "aggregate_chunks": self.config.avatar_aggregate,
            "aggregate_min_chars": self.config.avatar_min_chars,
            "aggregate_max_chars": self.config.avatar_max_chars,
            "aggregate_timeout": self.config.avatar_aggregate_timeout,
            # Pre-buffer settings
            "prebuffer_enabled": self.config.prebuffer_enabled,
            "prebuffer_min_chunks": self.config.prebuffer_min_chunks,
            "prebuffer_min_seconds": self.config.prebuffer_min_seconds,
        }))
        
        # Wait for SESSION_STARTED confirmation
        response = await asyncio.wait_for(self.avatar_ws.recv(), timeout=10.0)
        msg = json.loads(response)
        
        if msg.get("type") != "SESSION_STARTED":
            raise RuntimeError(f"Expected SESSION_STARTED, got {msg}")
        
        avatar_session_id = msg.get("session_id")
        print(f"[Gateway {self.session_id}] Avatar session started: {avatar_session_id}")
        
        # Start background tasks
        self._sender_task = asyncio.create_task(self._avatar_sender())
        self._receiver_task = asyncio.create_task(self._avatar_receiver())
        self._client_sender_task = asyncio.create_task(self._client_sender())
        
    async def _avatar_sender(self):
        """Send sentences from queue to avatar service."""
        try:
            while not self.closed:
                try:
                    msg = await asyncio.wait_for(
                        self.sentence_queue.get(),
                        timeout=1.0
                    )
                    
                    if msg is None:
                        # End signal - send SESSION_END to avatar
                        print(f"[Gateway {self.session_id}] Sender finished, sending SESSION_END")
                        await self.avatar_ws.send(json.dumps({
                            "type": "SESSION_END"
                        }))
                        break
                    
                    await self.avatar_ws.send(json.dumps(msg))
                    self.sentences_sent += 1
                    print(f"[Gateway {self.session_id}] Sent sentence {self.sentences_sent} to avatar")
                    
                except asyncio.TimeoutError:
                    continue
                    
        except ConnectionClosed:
            print(f"[Gateway {self.session_id}] Avatar connection closed in sender")
        except Exception as e:
            print(f"[Gateway {self.session_id}] Avatar sender error: {e}")
            
    async def _avatar_receiver(self):
        """Receive video chunks from avatar and queue for client."""
        try:
            async for message in self.avatar_ws:
                if isinstance(message, bytes):
                    # Binary video chunk
                    try:
                        self.video_queue.put_nowait(message)
                        self.video_chunks_received += 1
                    except asyncio.QueueFull:
                        # Drop frame - better than blocking realtime
                        print(f"[Gateway {self.session_id}] Video queue full, dropping frame")
                else:
                    # JSON message
                    msg = json.loads(message)
                    msg_type = msg.get("type")
                    
                    if msg_type == "SESSION_COMPLETE":
                        print(f"[Gateway {self.session_id}] Avatar session complete: {msg}")
                        # Forward to client
                        await self._forward_to_client_json(msg)
                        # Signal completion
                        self.session_complete.set()
                        break
                        
                    elif msg_type == "ERROR":
                        print(f"[Gateway {self.session_id}] Avatar error: {msg}")
                        await self._forward_to_client_json(msg)
                        
                    elif msg_type == "STATUS":
                        # Optionally forward status updates
                        pass
                        
        except ConnectionClosed:
            print(f"[Gateway {self.session_id}] Avatar connection closed in receiver")
            # Set complete so end() doesn't hang
            self.session_complete.set()
        except Exception as e:
            print(f"[Gateway {self.session_id}] Avatar receiver error: {e}")
            self.session_complete.set()
            
    async def _client_sender(self):
        """Send video chunks from queue to client WebSocket."""
        chunks_sent = 0
        try:
            while not self.closed:
                try:
                    chunk = await asyncio.wait_for(
                        self.video_queue.get(),
                        timeout=1.0
                    )
                    await self.client_ws.send_bytes(chunk)
                    chunks_sent += 1
                    if chunks_sent == 1:
                        print(f"[Gateway {self.session_id}] Sent first chunk to client ({len(chunk)} bytes)")
                    elif chunks_sent % 10 == 0:
                        print(f"[Gateway {self.session_id}] Sent {chunks_sent} chunks to client")
                except asyncio.TimeoutError:
                    continue
            
            # Drain any remaining chunks after closed
            while not self.video_queue.empty():
                try:
                    chunk = self.video_queue.get_nowait()
                    await self.client_ws.send_bytes(chunk)
                    chunks_sent += 1
                except:
                    break
                    
            print(f"[Gateway {self.session_id}] Client sender complete: {chunks_sent} chunks sent")
                    
        except ConnectionClosed:
            print(f"[Gateway {self.session_id}] Client connection closed")
        except Exception as e:
            print(f"[Gateway {self.session_id}] Client sender error: {e}")
            
    async def _forward_to_client_json(self, msg: dict):
        """Forward JSON message to client."""
        try:
            await self.client_ws.send_json(msg)
        except Exception as e:
            print(f"[Gateway {self.session_id}] Error forwarding to client: {e}")
            
    async def send_sentence(self, text: str, seq: int):
        """Queue a sentence for sending to avatar."""
        await self.sentence_queue.put({
            "type": "SPEECH_CHUNK",
            "seq": seq,
            "text": text
        })
        print(f"[Gateway {self.session_id}] Queued sentence {seq}: '{text[:50]}...'")
        
    async def end(self):
        """End the avatar session - wait for pending work to complete."""
        if self._ending:
            return
        self._ending = True
        
        # Signal sender to finish (it will send SESSION_END after draining queue)
        await self.sentence_queue.put(None)
        
        # Wait for sender to finish (which sends SESSION_END after draining)
        if self._sender_task and not self._sender_task.done():
            try:
                await asyncio.wait_for(self._sender_task, timeout=30.0)
            except asyncio.TimeoutError:
                print(f"[Gateway {self.session_id}] Sender task timed out")
                self._sender_task.cancel()
            except asyncio.CancelledError:
                pass
        
        # Wait for video completion (receiver sets this when SESSION_COMPLETE arrives)
        print(f"[Gateway {self.session_id}] Waiting for video completion...")
        try:
            await asyncio.wait_for(self.session_complete.wait(), timeout=120.0)
            print(f"[Gateway {self.session_id}] Video complete, received {self.video_chunks_received} chunks")
        except asyncio.TimeoutError:
            print(f"[Gateway {self.session_id}] Video completion timed out (120s)")
        
        # Now we can close
        self.closed = True
        
        # Cancel remaining tasks
        for task in [self._receiver_task, self._client_sender_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close avatar connection
        if self.avatar_ws:
            try:
                await self.avatar_ws.close()
            except:
                pass
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[Gateway {self.session_id}] Session ended: "
              f"{self.sentences_sent} sentences, "
              f"{self.video_chunks_received} video chunks, "
              f"{elapsed:.1f}s total")


class LLMStreamProcessor:
    """
    Processes LLM token stream and sends sentences to avatar.
    
    Integrates with the SentenceBuffer for gateway-side aggregation
    and AvatarSession for avatar communication.
    """
    
    def __init__(
        self,
        avatar_session: AvatarSession,
        config: GatewayConfig
    ):
        self.avatar_session = avatar_session
        self.config = config
        
        # Sentence buffer for gateway-side aggregation
        if config.gateway_aggregate:
            self.buffer = SentenceBuffer(
                min_chars=config.gateway_min_chars,
                max_chars=config.gateway_max_chars,
                timeout_seconds=config.gateway_timeout
            )
        else:
            self.buffer = None
        
        self.seq = 0
        self.tokens_processed = 0
        
    async def process_token(self, token: str):
        """
        Process a single token from the LLM stream.
        
        With gateway aggregation, buffers tokens into sentences.
        Without, sends each token directly.
        """
        self.tokens_processed += 1
        
        if self.buffer:
            # Gateway-side aggregation
            sentence = self.buffer.add_token(token)
            if sentence:
                await self._send_sentence(sentence)
        else:
            # Direct mode - send each token (avatar will aggregate)
            self.seq += 1
            await self.avatar_session.send_sentence(token, self.seq)
    
    async def check_timeout(self):
        """Check for buffer timeout and flush if needed."""
        if self.buffer:
            sentence = self.buffer.check_timeout()
            if sentence:
                await self._send_sentence(sentence)
    
    async def flush(self):
        """Flush any remaining content."""
        if self.buffer:
            sentence = self.buffer.flush()
            if sentence:
                await self._send_sentence(sentence)
    
    async def _send_sentence(self, text: str):
        """Send a sentence to the avatar session."""
        self.seq += 1
        await self.avatar_session.send_sentence(text, self.seq)


async def stream_llm_to_video(
    client_ws,
    llm_stream,  # Async generator yielding tokens
    config: GatewayConfig,
    user_id: Optional[str] = None,
    on_token: Optional[Callable[[str, int], Awaitable[None]]] = None,
    on_sentence: Optional[Callable[[str, int], Awaitable[None]]] = None,
):
    """
    Main entry point: Stream LLM tokens to video via avatar service.
    
    Args:
        client_ws: WebSocket connection to the client
        llm_stream: Async generator yielding text tokens
        config: Gateway configuration
        user_id: Optional user ID for voice cloning
        on_token: Optional callback for each token
        on_sentence: Optional callback for each sentence
        
    Usage:
        async def my_llm_stream():
            for chunk in openai_response:
                yield chunk.choices[0].delta.content
                
        await stream_llm_to_video(
            client_ws=websocket,
            llm_stream=my_llm_stream(),
            config=GatewayConfig(avatar="sunny")
        )
    """
    # Create avatar session
    avatar_session = AvatarSession(
        config=config,
        client_ws=client_ws,
        user_id=user_id
    )
    
    # Create LLM processor
    processor = LLMStreamProcessor(
        avatar_session=avatar_session,
        config=config
    )
    
    try:
        # Start avatar session
        await avatar_session.start()
        
        # Process LLM stream
        token_count = 0
        async for token in llm_stream:
            if token:
                token_count += 1
                
                # Process token
                await processor.process_token(token)
                
                # Optional callback
                if on_token:
                    await on_token(token, token_count)
        
        # Flush remaining buffer
        await processor.flush()
        
        # End session
        await avatar_session.end()
        
        # Report stats
        print(f"[Gateway] Stream complete: {token_count} tokens, "
              f"{processor.seq} sentences sent")
        
    except Exception as e:
        print(f"[Gateway] Error in stream_llm_to_video: {e}")
        import traceback
        traceback.print_exc()
        await avatar_session.end()
        raise


# ============================================================================
# Example: OpenAI Integration
# ============================================================================

async def openai_stream_to_video(
    client_ws,
    prompt: str,
    config: GatewayConfig,
    model: str = "gpt-4",
    system_prompt: Optional[str] = None,
):
    """
    Stream OpenAI response to video.
    
    Example usage:
        await openai_stream_to_video(
            client_ws=websocket,
            prompt="Explain quantum computing in simple terms",
            config=GatewayConfig(avatar="sunny"),
            model="gpt-4"
        )
    """
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    async def llm_stream():
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    await stream_llm_to_video(
        client_ws=client_ws,
        llm_stream=llm_stream(),
        config=config
    )


# ============================================================================
# FastAPI Integration Example
# ============================================================================

def create_gateway_app():
    """
    Create a FastAPI app for the gateway.
    
    Usage:
        app = create_gateway_app()
        uvicorn.run(app, host="0.0.0.0", port=8001)
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    
    app = FastAPI(title="LLM-to-Video Gateway")
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket endpoint for chat-to-video.
        
        Client sends:
            {"type": "START", "prompt": "Hello...", "avatar": "sunny"}
            
        Server streams:
            Binary video chunks
            {"type": "COMPLETE", ...}
        """
        await websocket.accept()
        
        try:
            # Wait for START message
            message = await websocket.receive_json()
            
            if message.get("type") != "START":
                await websocket.send_json({
                    "type": "ERROR",
                    "message": "Expected START message"
                })
                return
            
            prompt = message.get("prompt", "")
            avatar = message.get("avatar", "sunny")
            
            config = GatewayConfig(
                avatar=avatar,
                size=message.get("size", 256),
                voice_id=message.get("voice_id"),
            )
            
            # Stream response
            await openai_stream_to_video(
                client_ws=websocket,
                prompt=prompt,
                config=config,
                system_prompt=message.get("system_prompt")
            )
            
        except WebSocketDisconnect:
            print("[Gateway] Client disconnected")
        except Exception as e:
            print(f"[Gateway] Error: {e}")
            try:
                await websocket.send_json({
                    "type": "ERROR",
                    "message": str(e)
                })
            except:
                pass
    
    return app


# ============================================================================
# Standalone Test
# ============================================================================

async def test_gateway():
    """Test the gateway with mock LLM stream."""
    
    # Mock client WebSocket
    class MockClientWS:
        def __init__(self):
            self.messages = []
            self.bytes_received = 0
            
        async def send_bytes(self, data):
            self.bytes_received += len(data)
            
        async def send_json(self, data):
            self.messages.append(data)
            print(f"[MockClient] Received: {data}")
    
    # Mock LLM stream
    async def mock_llm_stream():
        text = (
            "Hello! I'm your AI assistant. "
            "Today I'll explain how neural networks work. "
            "They're inspired by the human brain. "
            "Each neuron processes information and passes it along."
        )
        words = text.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate token delay
    
    config = GatewayConfig(
        avatar_ws_url="ws://localhost:8000/ws/generate",
        gateway_aggregate=True,
        gateway_min_chars=30,
    )
    
    client = MockClientWS()
    
    print("Starting gateway test...")
    await stream_llm_to_video(
        client_ws=client,
        llm_stream=mock_llm_stream(),
        config=config
    )
    print(f"Test complete: {client.bytes_received} bytes received")


if __name__ == "__main__":
    asyncio.run(test_gateway())
