"""
TTS Streamer - Generates TTS audio for each text chunk incrementally.

Processes text chunks as they arrive and queues audio for video generation.
"""

import asyncio
import uuid
import os
import httpx
from typing import Optional

from socket_session import SocketSession


class TTSStreamer:
    """
    Generates TTS audio for each text chunk in the session.
    
    Runs as an async task, continuously processing the text queue
    and generating audio files for video synthesis.
    """
    
    def __init__(self, session: SocketSession):
        self.session = session
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the TTS processing task."""
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        print(f"[TTSStreamer {self.session.session_id}] Started")
        
    async def stop(self):
        """Stop the TTS processing task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print(f"[TTSStreamer {self.session.session_id}] Stopped")
        
    async def _process_loop(self):
        """Main processing loop - generates TTS for each text chunk."""
        try:
            while self._running:
                try:
                    # Wait for text chunk with timeout
                    chunk = await asyncio.wait_for(
                        self.session.text_queue.get(),
                        timeout=1.0
                    )
                    
                    if chunk is None:
                        # Sentinel value - end of stream
                        await self.session.audio_queue.put(None)
                        break
                        
                    # Generate TTS for this chunk
                    print(f"[TTSStreamer {self.session.session_id}] Generating TTS for chunk {chunk.seq}")
                    
                    try:
                        audio_path = await self._generate_tts(chunk.text)
                        
                        # Queue audio for video generation
                        await self.session.queue_audio(chunk.seq, audio_path)
                        
                        print(f"[TTSStreamer {self.session.session_id}] Chunk {chunk.seq} TTS complete: {audio_path}")
                        
                    except Exception as e:
                        print(f"[TTSStreamer {self.session.session_id}] TTS error for chunk {chunk.seq}: {e}")
                        self.session.set_error(e)
                        break
                        
                except asyncio.TimeoutError:
                    # Check if session is ending
                    if self.session.state.value == "closing":
                        # Signal end of audio stream
                        await self.session.audio_queue.put(None)
                        break
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TTSStreamer {self.session.session_id}] Fatal error: {e}")
            self.session.set_error(e)
            
    async def _generate_tts(self, text: str) -> str:
        """
        Generate TTS audio for a single text chunk.
        
        Uses async HTTP to avoid blocking the event loop.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to generated audio file
        """
        config = self.session.config
        
        if config.tts_preference == "coqui":
            return await self._generate_coqui(text)
        else:
            return await self._generate_elevenlabs(text)
            
    async def _generate_coqui(self, text: str) -> str:
        """Generate TTS using Coqui TTS server."""
        tts_url = "http://tts:8000/generate"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(tts_url, json={
                "text": text,
                "split_sentences": False,
                "source_aud": self.session.config.voice_source,
                "clone": self.session.config.user_id,
                "streaming": False,
                "model": "tts_models/multilingual/multi-dataset/xtts_v2"
            })
            response.raise_for_status()
            
            # Save audio to temp file
            request_id = str(uuid.uuid4())[:8]
            audio_path = f"/tmp/tts_{self.session.session_id}_{request_id}.wav"
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
                
            return audio_path
            
    async def _generate_elevenlabs(self, text: str) -> str:
        """Generate TTS using ElevenLabs API."""
        # Run in thread pool as ElevenLabs SDK is not async-native
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._elevenlabs_sync, text)
        
    def _elevenlabs_sync(self, text: str) -> str:
        """Synchronous ElevenLabs generation (runs in thread pool)."""
        import tempfile
        from elevenlabs.client import ElevenLabs
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = self.session.config.tts_voice_id or os.getenv("VOICE_ID")
        
        elevenlabs = ElevenLabs(api_key=api_key)
        response = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
                    
        return f.name


class ChunkAggregator:
    """
    Aggregates small chunks into larger segments for more efficient TTS.
    
    Optional component that can buffer incoming chunks until a sentence
    boundary or timeout, reducing TTS API calls.
    """
    
    def __init__(
        self, 
        min_chars: int = 50,
        max_chars: int = 500,
        timeout_seconds: float = 2.0
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.timeout_seconds = timeout_seconds
        
        self._buffer = ""
        self._chunk_seqs = []
        self._last_chunk_time = None
        
    def add_chunk(self, seq: int, text: str) -> Optional[tuple]:
        """
        Add a chunk to the buffer.
        
        Returns:
            Tuple of (combined_text, seq_list) if ready to process,
            None if still buffering
        """
        import time
        
        self._buffer += text
        self._chunk_seqs.append(seq)
        self._last_chunk_time = time.time()
        
        # Check if we should flush
        if self._should_flush():
            return self._flush()
            
        return None
        
    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        if len(self._buffer) >= self.max_chars:
            return True
            
        if len(self._buffer) >= self.min_chars:
            # Check for sentence endings
            if self._buffer.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
                return True
                
        return False
        
    def _flush(self) -> tuple:
        """Flush the buffer and return aggregated content."""
        result = (self._buffer, list(self._chunk_seqs))
        self._buffer = ""
        self._chunk_seqs = []
        return result
        
    def flush_remaining(self) -> Optional[tuple]:
        """Force flush any remaining content."""
        if self._buffer:
            return self._flush()
        return None
