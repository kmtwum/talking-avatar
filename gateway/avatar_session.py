"""
Avatar Session - WebSocket connection manager for Talking Avatar service.

This is the bridge between the gateway and the avatar video generation service.
Handles bidirectional streaming of text chunks and video data.
"""

import asyncio
import json
import uuid
import time
from typing import Optional
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosed


@dataclass
class AvatarConfig:
    """Configuration for avatar session."""
    avatar_ws_url: str = "ws://77.68.21.101:8002/ws/generate"
    avatar: str = "sunny"
    size: int = 256
    tts_preference: str = "coqui"
    voice_id: Optional[str] = None
    user_id: Optional[str] = None
    voice_source: Optional[str] = None
    
    # Avatar-side aggregation settings
    aggregate_chunks: bool = True
    aggregate_min_chars: int = 50
    aggregate_max_chars: int = 500
    aggregate_timeout: float = 1.5
    
    # Pre-buffer settings
    prebuffer_enabled: bool = True
    prebuffer_min_chunks: int = 1
    prebuffer_min_seconds: float = 1.0
    prebuffer_timeout: float = 10.0


class AvatarSession:
    """
    Manages bidirectional WebSocket connection to the Talking Avatar service.
    
    Architecture:
        Client (React) <--WS--> Gateway (this) <--WS--> Avatar Service
        
    Pipeline:
        1. Receives text sentences from GPT stream
        2. Sends SPEECH_CHUNK messages to avatar
        3. Receives binary video chunks from avatar
        4. Forwards video to client WebSocket
    """
    
    def __init__(
        self,
        client_ws,
        config: Optional[AvatarConfig] = None,
        user=None  # User object with avatar/voice info
    ):
        self.session_id = str(uuid.uuid4())[:8]
        self.config = config or AvatarConfig()
        self.client_ws = client_ws
        self.user = user
        
        # Override config with user settings if available
        if user:
            if hasattr(user, 'talent') and user.talent:
                if user.talent.avatar:
                    self.config.avatar = user.talent.avatar
                if user.talent.voice_id:
                    self.config.voice_id = user.talent.voice_id
            if hasattr(user, 'id'):
                self.config.user_id = str(user.id)
        
        # Connection state
        self.avatar_ws = None
        self.closed = False
        self.started = False
        
        # Async queues
        self.sentence_queue: asyncio.Queue = asyncio.Queue()
        self.video_queue: asyncio.Queue = asyncio.Queue(maxsize=120)
        
        # Background tasks
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._client_sender_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.sentences_sent = 0
        self.video_chunks_received = 0
        self.video_bytes_received = 0
        self.start_time: Optional[float] = None
        self.first_video_time: Optional[float] = None
        
    async def start(self):
        """
        Connect to avatar service and start background tasks.
        
        Raises:
            ConnectionError: If unable to connect to avatar service
            TimeoutError: If avatar service doesn't respond
        """
        if self.started:
            return
            
        self.start_time = time.time()
        print(f"[AvatarSession {self.session_id}] Connecting to {self.config.avatar_ws_url}")
        
        try:
            # Add extra headers to bypass potential origin checks from proxies
            extra_headers = {
                "Origin": self.config.avatar_ws_url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/", 1)[0],
            }
            
            self.avatar_ws = await websockets.connect(
                self.config.avatar_ws_url,
                max_size=10 * 1024 * 1024,  # 10MB max message
                ping_interval=20,
                ping_timeout=10,
                extra_headers=extra_headers,
            )
        except Exception as e:
            print(f"[AvatarSession {self.session_id}] Connection failed: {e}")
            raise ConnectionError(f"Failed to connect to avatar service: {e}")
        
        # Send SESSION_START with full configuration
        start_msg = {
            "type": "SESSION_START",
            "avatar": self.config.avatar,
            "size": self.config.size,
            "tts_preference": self.config.tts_preference,
            "tts_voice_id": self.config.voice_id,
            "user_id": self.config.user_id,
            "voice_source": self.config.voice_source,
            # Aggregation settings
            "aggregate_chunks": self.config.aggregate_chunks,
            "aggregate_min_chars": self.config.aggregate_min_chars,
            "aggregate_max_chars": self.config.aggregate_max_chars,
            "aggregate_timeout": self.config.aggregate_timeout,
            # Pre-buffer settings
            "prebuffer_enabled": self.config.prebuffer_enabled,
            "prebuffer_min_chunks": self.config.prebuffer_min_chunks,
            "prebuffer_min_seconds": self.config.prebuffer_min_seconds,
            "prebuffer_timeout": self.config.prebuffer_timeout,
        }
        
        await self.avatar_ws.send(json.dumps(start_msg))
        
        # Wait for confirmation
        try:
            response = await asyncio.wait_for(self.avatar_ws.recv(), timeout=10.0)
            msg = json.loads(response)
            
            if msg.get("type") != "SESSION_STARTED":
                raise RuntimeError(f"Unexpected response: {msg}")
                
            avatar_session_id = msg.get("session_id")
            print(f"[AvatarSession {self.session_id}] Avatar session: {avatar_session_id}")
            
        except asyncio.TimeoutError:
            await self.avatar_ws.close()
            raise TimeoutError("Avatar service did not respond to SESSION_START")
        
        # Start background tasks
        self._sender_task = asyncio.create_task(self._avatar_sender())
        self._receiver_task = asyncio.create_task(self._avatar_receiver())
        self._client_sender_task = asyncio.create_task(self._client_sender())
        
        self.started = True
        
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
                        break
                    
                    await self.avatar_ws.send(json.dumps(msg))
                    self.sentences_sent += 1
                    
                except asyncio.TimeoutError:
                    continue
                    
        except ConnectionClosed:
            print(f"[AvatarSession {self.session_id}] Avatar connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[AvatarSession {self.session_id}] Sender error: {e}")
            
    async def _avatar_receiver(self):
        """Receive video/messages from avatar and queue for client."""
        try:
            async for message in self.avatar_ws:
                if isinstance(message, bytes):
                    # Binary video chunk
                    if self.first_video_time is None:
                        self.first_video_time = time.time()
                        latency = self.first_video_time - self.start_time
                        print(f"[AvatarSession {self.session_id}] First video frame: {latency:.2f}s")
                    
                    try:
                        self.video_queue.put_nowait(message)
                        self.video_chunks_received += 1
                        self.video_bytes_received += len(message)
                    except asyncio.QueueFull:
                        # Drop frame to avoid blocking
                        print(f"[AvatarSession {self.session_id}] Queue full, dropping frame")
                else:
                    # JSON message
                    msg = json.loads(message)
                    await self._handle_avatar_message(msg)
                    
        except ConnectionClosed:
            print(f"[AvatarSession {self.session_id}] Avatar connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[AvatarSession {self.session_id}] Receiver error: {e}")
            
    async def _handle_avatar_message(self, msg: dict):
        """Handle JSON message from avatar service."""
        msg_type = msg.get("type")
        
        if msg_type == "SESSION_COMPLETE":
            print(f"[AvatarSession {self.session_id}] Session complete: "
                  f"duration={msg.get('total_duration_ms')}ms, "
                  f"chunks={msg.get('chunks_processed')}, "
                  f"frames={msg.get('frames_generated')}")
            
            # Forward to client
            await self._forward_json(msg)
            
        elif msg_type == "ERROR":
            print(f"[AvatarSession {self.session_id}] Avatar error: {msg}")
            await self._forward_json(msg)
            
        elif msg_type == "STATUS":
            # Optionally forward status updates
            pass
            
    async def _client_sender(self):
        """Forward video chunks to client WebSocket."""
        try:
            while not self.closed:
                try:
                    chunk = await asyncio.wait_for(
                        self.video_queue.get(),
                        timeout=1.0
                    )
                    await self.client_ws.send_bytes(chunk)
                except asyncio.TimeoutError:
                    continue
                    
        except ConnectionClosed:
            print(f"[AvatarSession {self.session_id}] Client disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[AvatarSession {self.session_id}] Client sender error: {e}")
            
    async def _forward_json(self, msg: dict):
        """Forward JSON message to client."""
        try:
            if hasattr(self.client_ws, 'send_json'):
                await self.client_ws.send_json(msg)
            else:
                await self.client_ws.send(json.dumps(msg))
        except Exception as e:
            print(f"[AvatarSession {self.session_id}] Forward error: {e}")
            
    async def send_sentence(self, text: str, seq: int):
        """
        Queue a sentence for sending to avatar service.
        
        Args:
            text: The sentence text
            seq: Sequence number for ordering
        """
        if not text.strip():
            return
            
        await self.sentence_queue.put({
            "type": "SPEECH_CHUNK",
            "seq": seq,
            "text": text
        })
        
        print(f"[AvatarSession {self.session_id}] Sentence {seq}: "
              f"'{text[:50]}{'...' if len(text) > 50 else ''}'")
        
    async def end(self):
        """
        End the avatar session gracefully.
        
        Signals the avatar service that no more text is coming,
        waits for remaining video, then closes connections.
        """
        if self.closed:
            return
            
        self.closed = True
        
        # Signal sender to stop
        await self.sentence_queue.put(None)
        
        # Send SESSION_END to avatar
        if self.avatar_ws and not self.avatar_ws.closed:
            try:
                await self.avatar_ws.send(json.dumps({
                    "type": "SESSION_END"
                }))
            except Exception as e:
                print(f"[AvatarSession {self.session_id}] Error sending end: {e}")
        
        # Wait briefly for final video chunks
        await asyncio.sleep(0.5)
        
        # Cancel tasks
        for task in [self._sender_task, self._receiver_task, self._client_sender_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close avatar connection
        if self.avatar_ws and not self.avatar_ws.closed:
            await self.avatar_ws.close()
        
        # Log statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[AvatarSession {self.session_id}] Ended: "
              f"{self.sentences_sent} sentences, "
              f"{self.video_chunks_received} chunks, "
              f"{self.video_bytes_received / 1024:.1f}KB, "
              f"{elapsed:.1f}s")
        
    def get_stats(self) -> dict:
        """Get session statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        first_frame_latency = (
            self.first_video_time - self.start_time 
            if self.first_video_time and self.start_time 
            else None
        )
        
        return {
            "session_id": self.session_id,
            "sentences_sent": self.sentences_sent,
            "video_chunks_received": self.video_chunks_received,
            "video_bytes_received": self.video_bytes_received,
            "elapsed_seconds": elapsed,
            "first_frame_latency": first_frame_latency,
        }
