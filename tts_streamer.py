"""
TTS Streamer - Generates TTS audio for each text chunk incrementally.

Supports two modes:
1. Direct mode: Each chunk generates TTS immediately
2. Aggregated mode: Chunks are buffered until sentence boundary or timeout

Processes text chunks as they arrive and queues audio for video generation.
"""

import asyncio
import uuid
import os
import time
import httpx
from typing import Optional, List, Tuple
from dataclasses import dataclass

from socket_session import SocketSession, ChunkInfo
from dotenv import load_dotenv
load_dotenv()


@dataclass
class AggregatedChunk:
    """Represents multiple chunks aggregated together."""
    text: str
    source_seqs: List[int]
    first_received_at: float
    last_received_at: float
    
    @property
    def seq(self) -> int:
        """Return the last sequence number for ordering."""
        return self.source_seqs[-1] if self.source_seqs else 0


class ChunkAggregator:
    """
    Aggregates small chunks into larger segments for more efficient TTS.
    
    Buffers incoming chunks until:
    - A sentence boundary is detected (., !, ?, etc.)
    - Maximum character limit is reached
    - Timeout expires since last chunk
    
    This reduces TTS API calls and improves speech prosody by ensuring
    complete sentences are synthesized together.
    """
    
    # Sentence ending patterns (multiple languages)
    SENTENCE_ENDINGS = ('.', '!', '?', '。', '！', '？', '؟', '।')
    
    # Patterns that suggest more text is coming (don't flush on these)
    CONTINUATION_PATTERNS = ('...', '..', '—', '–', ',', ':', ';')
    
    def __init__(
        self, 
        min_chars: int = 50,
        max_chars: int = 500,
        timeout_seconds: float = 1.5
    ):
        """
        Initialize the chunk aggregator.
        
        Args:
            min_chars: Minimum chars before considering sentence-based flush
            max_chars: Force flush when buffer reaches this size
            timeout_seconds: Flush after this many seconds of no new chunks
        """
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.timeout_seconds = timeout_seconds
        
        # Buffer state
        self._buffer = ""
        self._chunk_seqs: List[int] = []
        self._first_chunk_time: Optional[float] = None
        self._last_chunk_time: Optional[float] = None
        
        # Statistics
        self.total_chunks_received = 0
        self.total_aggregations_produced = 0
        
    def add_chunk(self, seq: int, text: str) -> Optional[AggregatedChunk]:
        """
        Add a chunk to the buffer.
        
        Args:
            seq: Chunk sequence number
            text: Chunk text content
            
        Returns:
            AggregatedChunk if buffer should be flushed, None if buffering
        """
        now = time.time()
        
        # Add space between chunks if needed
        if self._buffer and not self._buffer.endswith((' ', '\n')):
            if not text.startswith((' ', '\n', '.', ',', '!', '?', ';', ':')):
                self._buffer += " "
        
        self._buffer += text
        self._chunk_seqs.append(seq)
        self.total_chunks_received += 1
        
        if self._first_chunk_time is None:
            self._first_chunk_time = now
        self._last_chunk_time = now
        
        # Check if we should flush
        if self._should_flush():
            return self._flush()
            
        return None
    
    def check_timeout(self) -> Optional[AggregatedChunk]:
        """
        Check if timeout has expired and flush if needed.
        
        Call this periodically when waiting for more chunks.
        
        Returns:
            AggregatedChunk if timeout triggered flush, None otherwise
        """
        if not self._buffer:
            return None
            
        now = time.time()
        if self._last_chunk_time and (now - self._last_chunk_time) >= self.timeout_seconds:
            return self._flush()
            
        return None
        
    def _should_flush(self) -> bool:
        """Determine if buffer should be flushed."""
        buffer_len = len(self._buffer)
        
        # Always flush if we hit max chars
        if buffer_len >= self.max_chars:
            return True
            
        # Don't flush if below minimum
        if buffer_len < self.min_chars:
            return False
            
        # Check for sentence endings
        stripped = self._buffer.rstrip()
        
        # Don't flush on continuation patterns
        for pattern in self.CONTINUATION_PATTERNS:
            if stripped.endswith(pattern):
                return False
        
        # Flush on sentence endings
        for ending in self.SENTENCE_ENDINGS:
            if stripped.endswith(ending):
                return True
                
        return False
        
    def _flush(self) -> AggregatedChunk:
        """Flush the buffer and return aggregated content."""
        result = AggregatedChunk(
            text=self._buffer.strip(),
            source_seqs=list(self._chunk_seqs),
            first_received_at=self._first_chunk_time or time.time(),
            last_received_at=self._last_chunk_time or time.time()
        )
        
        # Reset buffer
        self._buffer = ""
        self._chunk_seqs = []
        self._first_chunk_time = None
        self._last_chunk_time = None
        
        self.total_aggregations_produced += 1
        
        return result
        
    def flush_remaining(self) -> Optional[AggregatedChunk]:
        """
        Force flush any remaining content.
        
        Call this when session is ending to process remaining buffer.
        """
        if self._buffer.strip():
            return self._flush()
        return None
        
    def get_buffer_status(self) -> dict:
        """Get current buffer status for debugging."""
        return {
            "buffer_length": len(self._buffer),
            "pending_chunks": len(self._chunk_seqs),
            "waiting_since": self._first_chunk_time,
            "last_activity": self._last_chunk_time,
            "total_received": self.total_chunks_received,
            "total_produced": self.total_aggregations_produced,
        }


class TTSStreamer:
    """
    Generates TTS audio for each text chunk in the session.
    
    Runs as an async task, continuously processing the text queue
    and generating audio files for video synthesis.
    
    Supports optional chunk aggregation for more efficient TTS.

    The direct (non-aggregated) path runs TTS calls in bounded parallel
    (see ``MAX_CONCURRENT_TTS``) while preserving sequence order when
    queueing audio to the video pipeline. The aggregated path remains
    serial — it is only used by external clients that send raw tokens
    (the notchup gateway disables avatar aggregation; see Bottleneck 3
    in notchup-py/VIDEO_INFERENCE.md).
    """

    # Maximum number of concurrent TTS calls per session. Set conservatively
    # to stay well under ElevenLabs paid-tier concurrency limits while still
    # giving meaningful parallelism for typical 3-6 sentence responses.
    MAX_CONCURRENT_TTS = 4

    def __init__(self, session: SocketSession):
        self.session = session
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Initialize aggregator if enabled
        config = session.config
        self._aggregator: Optional[ChunkAggregator] = None
        
        if config.aggregate_chunks:
            self._aggregator = ChunkAggregator(
                min_chars=config.aggregate_min_chars,
                max_chars=config.aggregate_max_chars,
                timeout_seconds=config.aggregate_timeout
            )
            print(f"[TTSStreamer {session.session_id}] Aggregation enabled: "
                  f"min={config.aggregate_min_chars}, max={config.aggregate_max_chars}, "
                  f"timeout={config.aggregate_timeout}s")
        
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
        """Main processing loop - generates TTS for chunks or aggregated text."""
        try:
            if self._aggregator:
                await self._process_with_aggregation()
            else:
                await self._process_direct()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TTSStreamer {self.session.session_id}] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            self.session.set_error(e)
            
    async def _process_direct(self):
        """
        Process chunks with bounded-parallel TTS, queueing audio in seq order.

        Architecture:
        - Main loop pulls chunks from ``session.text_queue`` and spawns a
          TTS task per chunk (gated by ``MAX_CONCURRENT_TTS`` semaphore).
        - The (seq, task) tuple is pushed onto an internal ordering queue.
        - A dedicated drainer task dequeues in FIFO order, awaits each
          task, and queues the resulting audio to ``session.audio_queue``.
          This preserves seq order downstream while letting TTS calls run
          concurrently.

        For a 4-sentence response with ~1.5 s/call this turns ~6 s of
        serial TTS into ~1.5-2 s of wall time (limited by the slowest call).
        """
        tts_sem = asyncio.Semaphore(self.MAX_CONCURRENT_TTS)
        task_queue: asyncio.Queue = asyncio.Queue()

        async def _run_tts(seq: int, text: str) -> str:
            async with tts_sem:
                print(f"[TTSStreamer {self.session.session_id}] Generating TTS for segment {seq}: "
                      f"'{text[:50]}...' ({len(text)} chars)")
                start = time.time()
                audio_path = await self._generate_tts(text)
                elapsed = time.time() - start
                print(f"[TTSStreamer {self.session.session_id}] Segment {seq} TTS "
                      f"complete in {elapsed:.2f}s: {audio_path}")
                return audio_path

        async def _drainer():
            """Await TTS tasks in seq order, queue audio downstream."""
            try:
                while True:
                    item = await task_queue.get()
                    if item is None:
                        break
                    seq, task = item
                    audio_path = await task
                    await self.session.queue_audio(seq, audio_path)
            finally:
                # Always unblock downstream consumers, even on error —
                # video pipeline waits on these signals.
                await self.session.audio_queue.put(None)
                self.session.all_audio_ready.set()

        drainer_task = asyncio.create_task(_drainer())

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        self.session.text_queue.get(),
                        timeout=1.0
                    )

                    if chunk is None:
                        await task_queue.put(None)
                        break

                    task = asyncio.create_task(_run_tts(chunk.seq, chunk.text))
                    await task_queue.put((chunk.seq, task))

                except asyncio.TimeoutError:
                    if self.session.state.value == "closing":
                        await task_queue.put(None)
                        break
                    continue

            # Wait for drainer to finish — re-raises any TTS error in
            # seq order so _process_loop can mark the session failed.
            await drainer_task
        except Exception:
            if not drainer_task.done():
                drainer_task.cancel()
                try:
                    await drainer_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise
                
    async def _process_with_aggregation(self):
        """Process chunks with aggregation for better TTS efficiency."""
        aggregation_seq = 0  # Sequence number for aggregated chunks
        
        while self._running:
            try:
                # Use shorter timeout to check for aggregation timeout
                chunk = await asyncio.wait_for(
                    self.session.text_queue.get(),
                    timeout=0.25  # Check frequently for timeouts
                )
                
                if chunk is None:
                    # Session ending - flush remaining buffer
                    remaining = self._aggregator.flush_remaining()
                    if remaining:
                        print(f"[TTSStreamer {self.session.session_id}] Flushing remaining buffer: "
                              f"'{remaining.text[:50]}...' ({len(remaining.text)} chars)")
                        await self._generate_and_queue(aggregation_seq, remaining.text)
                        aggregation_seq += 1
                    
                    await self.session.audio_queue.put(None)
                    
                    # Signal that all audio is ready for video generation
                    self.session.all_audio_ready.set()
                    
                    # Log aggregation stats
                    status = self._aggregator.get_buffer_status()
                    print(f"[TTSStreamer {self.session.session_id}] Aggregation complete: "
                          f"{status['total_received']} chunks -> {status['total_produced']} TTS calls")
                    break
                    
                # Add chunk to aggregator
                aggregated = self._aggregator.add_chunk(chunk.seq, chunk.text)
                
                if aggregated:
                    print(f"[TTSStreamer {self.session.session_id}] Aggregated {len(aggregated.source_seqs)} chunks: "
                          f"'{aggregated.text[:50]}...' ({len(aggregated.text)} chars)")
                    await self._generate_and_queue(aggregation_seq, aggregated.text)
                    aggregation_seq += 1
                    
            except asyncio.TimeoutError:
                # Check if session is ending
                if self.session.state.value == "closing":
                    remaining = self._aggregator.flush_remaining()
                    if remaining:
                        await self._generate_and_queue(aggregation_seq, remaining.text)
                        aggregation_seq += 1
                    await self.session.audio_queue.put(None)
                    break
                    
                # Check for aggregation timeout
                timed_out = self._aggregator.check_timeout()
                if timed_out:
                    print(f"[TTSStreamer {self.session.session_id}] Timeout flush: "
                          f"'{timed_out.text[:50]}...' ({len(timed_out.text)} chars)")
                    await self._generate_and_queue(aggregation_seq, timed_out.text)
                    aggregation_seq += 1
                    
                continue
                
    async def _generate_and_queue(self, seq: int, text: str):
        """Generate TTS for text and queue the audio."""
        print(f"[TTSStreamer {self.session.session_id}] Generating TTS for segment {seq}: "
              f"'{text[:50]}...' ({len(text)} chars)")
        
        try:
            start_time = time.time()
            audio_path = await self._generate_tts(text)
            elapsed = time.time() - start_time
            
            await self.session.queue_audio(seq, audio_path)
            
            print(f"[TTSStreamer {self.session.session_id}] Segment {seq} TTS complete in {elapsed:.2f}s: {audio_path}")
            
        except Exception as e:
            print(f"[TTSStreamer {self.session.session_id}] TTS error for segment {seq}: {e}")
            raise
            
    async def _generate_tts(self, text: str) -> str:
        """
        Generate TTS audio for text.
        
        Uses async HTTP to avoid blocking the event loop.
        """
        config = self.session.config

        print(f"[TTSStreamer {self.session.session_id}] Generating TTS for '{text[:50]} using {config.tts_preference}...' ")
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
