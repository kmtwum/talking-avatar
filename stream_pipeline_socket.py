"""
Socket Streaming Pipeline - SDK variant that accepts audio chunks progressively.

Extends StreamingSDK to support incremental audio input, allowing video
generation to start before all audio is available.
"""

import threading
import queue
import asyncio
import numpy as np
import math
import librosa
import os
import tempfile
from typing import AsyncIterator, Optional, List
from dataclasses import dataclass

from stream_pipeline_streaming import StreamingSDK
from core.atomic_components.fmp4_writer import FMP4StreamWriter


@dataclass
class AudioSegment:
    """Represents a segment of audio to be processed."""
    seq: int
    audio_path: str
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 16000
    is_final: bool = False


class SocketStreamingSDK(StreamingSDK):
    """
    SDK variant that accepts audio chunks progressively via a queue.
    
    Instead of requiring all audio upfront, this SDK accepts audio
    segments as they become available, enabling progressive video
    generation alongside TTS processing.
    """
    
    def __init__(self, cfg_pkl, data_root, **kwargs):
        super().__init__(cfg_pkl, data_root, **kwargs)
        
        # Audio input queue (receives AudioSegment objects)
        self.audio_input_queue: asyncio.Queue = None
        
        # Accumulated audio for processing
        self._audio_segments: List[AudioSegment] = []
        self._total_audio: Optional[np.ndarray] = None
        self._processed_samples = 0
        self._frame_offset = 0
        
        # Synchronization
        self._audio_complete = threading.Event()
        self._processing_lock = threading.Lock()
        
    def setup_socket_streaming(
        self,
        source_path: str,
        width: int = 256,
        height: int = 256,
        **kwargs
    ):
        """
        Setup for socket-based streaming output.
        
        Args:
            source_path: Path to source image/video
            width: Output video width
            height: Output video height  
            **kwargs: Additional setup kwargs
        """
        # Initialize audio input queue
        self.audio_input_queue = asyncio.Queue()
        
        # Use parent streaming setup
        self.setup_streaming(source_path, width, height, **kwargs)
        
        print(f"[SocketSDK] Setup complete: {width}x{height}")
        
    async def append_audio(self, segment: AudioSegment):
        """
        Append a new audio segment to the processing queue.
        
        This allows video generation to continue progressively
        as more audio becomes available.
        
        Args:
            segment: AudioSegment containing path and metadata
        """
        # Load audio data if not already loaded
        if segment.audio_data is None and segment.audio_path:
            audio, sr = librosa.core.load(segment.audio_path, sr=16000)
            segment.audio_data = audio
            segment.sample_rate = sr
            
        self._audio_segments.append(segment)
        
        print(f"[SocketSDK] Appended audio segment {segment.seq}: "
              f"{len(segment.audio_data) if segment.audio_data is not None else 0} samples")
        
        if segment.is_final:
            self._audio_complete.set()
            
    async def generate_progressive(self) -> AsyncIterator[bytes]:
        """
        Async generator that yields fMP4 segments as audio arrives.
        
        Starts video generation immediately when first audio is available,
        and continues processing as more audio segments arrive.
        
        Yields:
            bytes: fMP4 segments (init segment first, then media segments)
        """
        import time
        start_time = time.time()
        print("[SocketSDK] Starting progressive generation")
        
        if not self._streaming_mode:
            raise RuntimeError("Must call setup_socket_streaming() before generate_progressive()")
        
        # Wait for first audio segment
        print("[SocketSDK] Waiting for first audio segment...")
        
        first_segment = None
        while not self._audio_segments:
            await asyncio.sleep(0.05)
            if self._audio_complete.is_set():
                break
                
        if not self._audio_segments:
            raise RuntimeError("No audio segments received")
            
        first_segment = self._audio_segments[0]
        print(f"[SocketSDK] First audio received at {time.time() - start_time:.3f}s")
        
        # Setup fMP4 writer with first audio for timing info
        # Note: We'll use a combined audio file later for proper muxing
        temp_audio = self._create_temp_combined_audio()
        self._setup_streaming_writer(temp_audio)
        
        # Calculate initial frame count
        initial_frames = self._calculate_frame_count()
        self.setup_Nd(N_d=initial_frames)
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=self._run_progressive_generation,
            args=(start_time,)
        )
        generation_thread.start()
        
        # Yield init segment
        print("[SocketSDK] Waiting for init segment...")
        try:
            init_segment = self._fmp4_writer.get_init_segment(timeout=15.0)
            print(f"[SocketSDK] Init segment ready ({len(init_segment)} bytes)")
            yield init_segment
        except TimeoutError:
            raise RuntimeError("Failed to get initialization segment")
        
        # Yield media segments as they become available
        # NOTE: iter_segments uses blocking queue.get() - we need to run it
        # in an executor to not block the asyncio event loop
        segment_count = 0
        print("[SocketSDK] Starting segment iteration...", flush=True)
        
        loop = asyncio.get_event_loop()
        segment_iter = self._fmp4_writer.iter_segments(timeout=0.5)  # Short timeout for responsiveness
        
        while True:
            try:
                # Run the blocking next() in a thread pool to not block event loop
                segment = await loop.run_in_executor(
                    None,  # Use default executor
                    lambda: next(segment_iter, None)
                )
                
                if segment is None:
                    break
                    
                segment_count += 1
                if segment_count == 1:
                    print(f"[SocketSDK] First media segment ({len(segment)} bytes)", flush=True)
                elif segment_count % 10 == 0:
                    print(f"[SocketSDK] Yielded {segment_count} segments", flush=True)
                yield segment
                
            except StopIteration:
                break
            except Exception as e:
                print(f"[SocketSDK] Segment iteration error: {e}", flush=True)
                break
            
        print(f"[SocketSDK] Segment iteration complete: {segment_count} segments", flush=True)
        
        generation_thread.join(timeout=30.0)
        if generation_thread.is_alive():
            print("[SocketSDK] Warning: Generation thread still alive after join", flush=True)
        print(f"[SocketSDK] Progressive generation complete at {time.time() - start_time:.3f}s", flush=True)
        
        # Cleanup
        self._fmp4_writer.close()
        self._cleanup_temp()
        
    def _create_temp_combined_audio(self) -> str:
        """
        Create a temporary audio file with all current segments.
        
        This is updated as more segments arrive.
        """
        if not self._audio_segments:
            return None
            
        # Combine all audio segments
        audio_arrays = [s.audio_data for s in self._audio_segments if s.audio_data is not None]
        
        if not audio_arrays:
            return None
            
        combined = np.concatenate(audio_arrays)
        self._total_audio = combined
        
        # Save to temp file
        import soundfile as sf
        temp_path = f"{self._temp_dir}/combined_audio.wav"
        sf.write(temp_path, combined, 16000)
        
        return temp_path
        
    def _calculate_frame_count(self) -> int:
        """Calculate total frame count from current audio."""
        if self._total_audio is None:
            return 0
        return math.ceil(len(self._total_audio) / 16000 * 25)
        
    def _run_progressive_generation(self, start_time: float):
        """
        Run progressive audio-to-video generation.
        
        Processes audio in chunks, updating as more segments arrive.
        """
        import time
        
        try:
            print(f"[SocketSDK] Starting generation thread at {time.time() - start_time:.3f}s")
            
            # Chunk configuration
            chunk_size = (2, 3, 1)
            
            # Process until all audio is consumed or stopped
            processed_chunks = 0
            last_segment_count = 0
            
            while not self.stop_event.is_set():
                current_segment_count = len(self._audio_segments)
                
                # Check if we have new audio to process
                if current_segment_count > last_segment_count:
                    # Update combined audio
                    self._create_temp_combined_audio()
                    last_segment_count = current_segment_count
                    
                    # Update frame count
                    new_frame_count = self._calculate_frame_count()
                    if new_frame_count > self._frame_offset:
                        self.setup_Nd(N_d=new_frame_count)
                
                # Process available audio
                if self._total_audio is not None:
                    # Pad for chunking
                    audio = np.concatenate([
                        np.zeros((chunk_size[0] * 640,), dtype=np.float32),
                        self._total_audio
                    ], 0)
                    
                    split_len = int(sum(chunk_size) * 0.04 * 16000) + 80
                    
                    # Process chunks we haven't done yet
                    start_idx = processed_chunks * chunk_size[1] * 640
                    
                    for i in range(start_idx, len(audio), chunk_size[1] * 640):
                        if self.stop_event.is_set():
                            print(f"[SocketSDK] Stop event triggered at chunk {processed_chunks}", flush=True)
                            break
                            
                        audio_chunk = audio[i:i + split_len]
                        if len(audio_chunk) < split_len:
                            audio_chunk = np.pad(
                                audio_chunk,
                                (0, split_len - len(audio_chunk)),
                                mode="constant"
                            )
                        
                        processed_chunks += 1
                        if processed_chunks == 1:
                            print(f"[SocketSDK] Processing first audio chunk", flush=True)
                        elif processed_chunks % 20 == 0:
                            print(f"[SocketSDK] Processed {processed_chunks} audio chunks", flush=True)
                        
                        self.run_chunk(audio_chunk, chunk_size)
                
                # Check if we're done
                if self._audio_complete.is_set():
                    # Process any remaining audio
                    total_samples = len(self._total_audio) if self._total_audio is not None else 0
                    processed_samples = processed_chunks * chunk_size[1] * 640
                    print(f"[SocketSDK] Audio complete check: processed={processed_samples}, total={total_samples}", flush=True)
                    if processed_samples >= total_samples:
                        print(f"[SocketSDK] All audio processed, exiting generation loop", flush=True)
                        break
                else:
                    # Wait for more audio
                    time.sleep(0.1)
            
            print(f"[SocketSDK] Finished processing {processed_chunks} chunks", flush=True)
            
            # Signal end of audio
            self.audio2motion_queue.put(None)
            print("[SocketSDK] Signaled end of audio to motion queue", flush=True)
            
        except Exception as e:
            print(f"[SocketSDK] Error in progressive generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            self.worker_exception = e
            self.stop_event.set()


class SocketVideoGenerator:
    """
    High-level coordinator for socket-based video generation.
    
    Manages the SDK lifecycle and provides a clean async interface
    for the WebSocket handler.
    """
    
    def __init__(
        self,
        source_path: str,
        width: int = 512,
        height: int = 512,
        cfg_pkl: str = None,
        data_root: str = None,
        watermark: bool = False,
    ):
        self.source_path = source_path
        self.width = width
        self.height = height
        self.watermark = watermark
        
        # Use defaults if not provided
        self.cfg_pkl = cfg_pkl or "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
        self.data_root = data_root or "/app/checkpoints/ditto_trt_Ampere_Plus"
        
        self._sdk: Optional[SocketStreamingSDK] = None
        self._generation_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the SDK for streaming."""
        print(f"[VideoGen] Initializing with source={self.source_path}")
        
        self._sdk = SocketStreamingSDK(self.cfg_pkl, self.data_root)
        self._sdk.setup_socket_streaming(
            self.source_path,
            width=self.width,
            height=self.height,
            watermark=self.watermark,
        )
        
    async def add_audio(self, seq: int, audio_path: str, is_final: bool = False):
        """Add an audio segment for processing."""
        segment = AudioSegment(
            seq=seq,
            audio_path=audio_path,
            is_final=is_final
        )
        await self._sdk.append_audio(segment)
        
    async def generate(self) -> AsyncIterator[bytes]:
        """Generate video segments progressively."""
        async for chunk in self._sdk.generate_progressive():
            yield chunk
            
    def cleanup(self):
        """Cleanup resources."""
        if self._sdk:
            try:
                # Signal stop to any running generation
                self._sdk.stop_event.set()
                self._sdk._audio_complete.set()
                
                # Close the SDK
                self._sdk.close()
                
                # Clear audio segments
                self._sdk._audio_segments = []
                self._sdk._total_audio = None
                
            except Exception as e:
                print(f"[VideoGen] Cleanup error: {e}")
            finally:
                self._sdk = None
            
        # Clean up temp audio files
        # (files are cleaned by SDK close, but ensure nothing remains)
        print("[VideoGen] Cleanup complete")
