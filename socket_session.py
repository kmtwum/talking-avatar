"""
Socket Session - Manages state for a single WebSocket connection.

Handles:
- Session configuration and lifecycle
- Chunk queuing and ordering
- Audio/video pipeline coordination
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class SessionState(Enum):
    """Session lifecycle states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class SessionConfig:
    """Configuration for a socket session."""
    avatar: str = "sunny"
    size: int = 256
    tts_preference: str = "coqui"
    tts_voice_id: Optional[str] = None
    user_id: Optional[str] = None
    voice_source: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionConfig":
        """Create config from dictionary."""
        return cls(
            avatar=data.get("avatar", "sunny"),
            size=int(data.get("size", 256)),
            tts_preference=data.get("tts_preference", "coqui"),
            tts_voice_id=data.get("tts_voice_id"),
            user_id=data.get("user_id"),
            voice_source=data.get("voice_source"),
        )


@dataclass
class ChunkInfo:
    """Information about a received text chunk."""
    seq: int
    text: str
    received_at: float = field(default_factory=lambda: __import__('time').time())
    audio_path: Optional[str] = None
    processed: bool = False


class SocketSession:
    """
    Manages the state for a single WebSocket connection.
    
    Coordinates between:
    - Incoming text chunks from client
    - TTS audio generation
    - Video frame generation
    - Output streaming to client
    """
    
    def __init__(self, session_id: str = None, config: SessionConfig = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.config = config or SessionConfig()
        
        # State
        self.state = SessionState.INITIALIZING
        self._start_time = None
        
        # Chunk tracking
        self.chunks: Dict[int, ChunkInfo] = {}
        self.last_seq = -1
        self.chunks_received = 0
        self.chunks_processed = 0
        
        # Async queues for pipeline coordination
        self.text_queue: asyncio.Queue = asyncio.Queue()
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.video_queue: asyncio.Queue = asyncio.Queue()
        
        # Events for synchronization
        self.session_started = asyncio.Event()
        self.session_complete = asyncio.Event()
        self.first_audio_ready = asyncio.Event()
        self.generation_started = asyncio.Event()
        
        # Error tracking
        self.error: Optional[Exception] = None
        
        # Video state
        self.init_segment: Optional[bytes] = None
        self.frames_generated = 0
        self.total_duration_ms = 0
        
    def start(self):
        """Mark session as started."""
        import time
        self._start_time = time.time()
        self.state = SessionState.ACTIVE
        self.session_started.set()
        print(f"[Session {self.session_id}] Started with avatar={self.config.avatar}, size={self.config.size}")
        
    async def handle_chunk(self, chunk_data: dict) -> ChunkInfo:
        """
        Process incoming SPEECH_CHUNK message.
        
        Args:
            chunk_data: Dict with 'seq' and 'text' keys
            
        Returns:
            ChunkInfo for the processed chunk
            
        Raises:
            ValueError: If chunk sequence is invalid
        """
        seq = chunk_data.get("seq")
        text = chunk_data.get("text", "")
        
        if seq is None:
            raise ValueError("Chunk missing 'seq' field")
            
        if not text:
            raise ValueError("Chunk has empty 'text' field")
        
        # Create chunk info
        chunk = ChunkInfo(seq=seq, text=text)
        self.chunks[seq] = chunk
        self.chunks_received += 1
        self.last_seq = max(self.last_seq, seq)
        
        # Queue for TTS processing
        await self.text_queue.put(chunk)
        
        print(f"[Session {self.session_id}] Received chunk {seq}: '{text[:50]}...' ({len(text)} chars)")
        
        return chunk
    
    async def queue_audio(self, seq: int, audio_path: str):
        """Queue generated audio for video processing."""
        if seq in self.chunks:
            self.chunks[seq].audio_path = audio_path
            
        await self.audio_queue.put((seq, audio_path))
        
        # Signal that first audio is ready
        if not self.first_audio_ready.is_set():
            self.first_audio_ready.set()
            print(f"[Session {self.session_id}] First audio ready")
    
    async def queue_video_segment(self, segment: bytes):
        """Queue video segment for output streaming."""
        await self.video_queue.put(segment)
        
    def mark_chunk_processed(self, seq: int):
        """Mark a chunk as fully processed."""
        if seq in self.chunks:
            self.chunks[seq].processed = True
            self.chunks_processed += 1
            
    def is_all_processed(self) -> bool:
        """Check if all received chunks have been processed."""
        return self.chunks_processed >= self.chunks_received
    
    def end(self):
        """Mark session as ending - no more chunks will be received."""
        self.state = SessionState.CLOSING
        print(f"[Session {self.session_id}] Session ending, {self.chunks_received} chunks received")
        
    def complete(self):
        """Mark session as complete."""
        import time
        if self._start_time:
            self.total_duration_ms = int((time.time() - self._start_time) * 1000)
            
        self.state = SessionState.CLOSED
        self.session_complete.set()
        
        print(f"[Session {self.session_id}] Complete: {self.chunks_processed}/{self.chunks_received} chunks, "
              f"{self.frames_generated} frames, {self.total_duration_ms}ms total")
        
    def set_error(self, error: Exception):
        """Record an error and mark session as failed."""
        self.error = error
        self.state = SessionState.CLOSED
        self.session_complete.set()
        print(f"[Session {self.session_id}] Error: {error}")
        
    def get_status(self) -> dict:
        """Get current session status."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "chunks_received": self.chunks_received,
            "chunks_processed": self.chunks_processed,
            "frames_generated": self.frames_generated,
            "total_duration_ms": self.total_duration_ms,
            "has_error": self.error is not None,
        }
    
    def get_image_path(self) -> str:
        """Get the source image path for this session's avatar."""
        return f"/app/user_img/{self.config.avatar}.jpg"


class SessionManager:
    """
    Manages multiple active socket sessions.
    
    Provides session lookup and cleanup utilities.
    """
    
    _instance: Optional["SessionManager"] = None
    
    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
        return cls._instance
    
    def create_session(self, config: SessionConfig = None) -> SocketSession:
        """Create and register a new session."""
        session = SocketSession(config=config)
        self._sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SocketSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session from management."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            
    def get_active_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self._sessions.values() 
                   if s.state in (SessionState.ACTIVE, SessionState.PROCESSING))
    
    def cleanup_stale(self, max_age_seconds: float = 300):
        """Remove stale/closed sessions."""
        import time
        now = time.time()
        stale = [
            sid for sid, session in self._sessions.items()
            if session.state == SessionState.CLOSED and 
               session._start_time and (now - session._start_time) > max_age_seconds
        ]
        for sid in stale:
            del self._sessions[sid]
        
        if stale:
            print(f"[SessionManager] Cleaned up {len(stale)} stale sessions")
