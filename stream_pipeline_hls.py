"""
HLS Streaming Pipeline - SDK variant that outputs HLS segments instead of fMP4.

Replaces FMP4StreamWriter with HLSStreamWriter to produce .ts segments
and .m3u8 playlists for native iOS Safari playback. The architecture
mirrors stream_pipeline_socket.py but outputs HLS files served via HTTP
instead of streaming binary fMP4 chunks over WebSocket.
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
from core.atomic_components.hls_writer import HLSStreamWriter


@dataclass
class AudioSegment:
    """Represents a segment of audio to be processed."""
    seq: int
    audio_path: str
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 16000
    is_final: bool = False


class HLSStreamingSDK(StreamingSDK):
    """
    SDK variant that accepts audio chunks progressively and outputs HLS segments.
    
    Instead of producing fMP4 chunks for MSE (which iOS doesn't support),
    this SDK writes HLS .ts segments + .m3u8 playlist to a directory
    that gets served via HTTP endpoints.
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
        
        # HLS writer (replaces _fmp4_writer)
        self._hls_writer: Optional[HLSStreamWriter] = None
        
        # Synchronization
        self._audio_complete = threading.Event()
        self._processing_lock = threading.Lock()
        
    def setup_hls_streaming(
        self,
        source_path: str,
        width: int = 256,
        height: int = 256,
        hls_output_dir: str = None,
        session_id: str = None,
        **kwargs
    ):
        """
        Setup for HLS-based streaming output.
        
        Args:
            source_path: Path to source image/video
            width: Output video width
            height: Output video height
            hls_output_dir: Directory to write HLS files
            session_id: Session identifier
            **kwargs: Additional setup kwargs
        """
        # Initialize audio input queue
        self.audio_input_queue = asyncio.Queue()
        
        # Store HLS-specific settings
        self._hls_output_dir = hls_output_dir
        self._hls_session_id = session_id
        
        # Use parent streaming setup
        self.setup_streaming(source_path, width, height, **kwargs)
        
        print(f"[HLS-SDK] Setup complete: {width}x{height}, session={session_id}")
        
    def _setup_hls_writer(self, audio_path: str = None):
        """Initialize the HLS writer (replaces _setup_streaming_writer)."""
        print(f"[HLS-SDK] Creating HLSStreamWriter with audio: {audio_path is not None}")
        
        self._hls_writer = HLSStreamWriter(
            width=self._streaming_output_width,
            height=self._streaming_output_height,
            fps=25,
            segment_duration=2.0,
            audio_path=audio_path,
            output_dir=self._hls_output_dir,
            session_id=self._hls_session_id,
        )
        
        print("[HLS-SDK] Starting HLSStreamWriter")
        self._hls_writer.start()
        
        # Signal fMP4 ready (reuse the event from parent for compatibility)
        self._fmp4_ready.set()
        print("[HLS-SDK] HLSStreamWriter ready")

    def writer_worker(self):
        """
        Override writer_worker to use HLS writer instead of fMP4 writer.
        
        This is called by the base class setup() when creating worker threads.
        """
        try:
            self._hls_writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _hls_writer_worker(self):
        """
        Writer worker that outputs to HLS writer instead of fMP4.
        
        Waits for the HLS writer to be initialized, then processes
        frames from the writer queue.
        """
        # Wait for HLS writer to be ready
        while not self.stop_event.is_set():
            if self._fmp4_ready.wait(timeout=1):
                break
        
        if self.stop_event.is_set():
            return
            
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
                
            res_frame_rgb = item
            
            # Resize frame if needed
            if res_frame_rgb.shape[0] != self._streaming_output_height or \
               res_frame_rgb.shape[1] != self._streaming_output_width:
                import cv2
                res_frame_rgb = cv2.resize(
                    res_frame_rgb,
                    (self._streaming_output_width, self._streaming_output_height),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Write to HLS writer
            self._hls_writer.write_frame(res_frame_rgb)
        
        # All frames written - finalize HLS stream
        if self._hls_writer:
            print("[HLS-SDK] All frames written, finalizing HLS stream")
            self._hls_writer.finalize()
        
    async def append_audio(self, segment: AudioSegment):
        """
        Append a new audio segment to the processing queue.
        
        Args:
            segment: AudioSegment containing path and metadata
        """
        # Load audio data if not already loaded
        if segment.audio_data is None and segment.audio_path:
            audio, sr = librosa.core.load(segment.audio_path, sr=16000)
            segment.audio_data = audio
            segment.sample_rate = sr
            
        self._audio_segments.append(segment)
        
        print(f"[HLS-SDK] Appended audio segment {segment.seq}: "
              f"{len(segment.audio_data) if segment.audio_data is not None else 0} samples")
        
        if segment.is_final:
            self._audio_complete.set()
            
    async def generate_hls(self) -> dict:
        """
        Generate HLS stream from progressive audio input.
        
        Instead of yielding fMP4 chunks, this starts the generation and returns
        the HLS session info (playlist URL, etc). The segments are written to
        disk and served via HTTP.
        
        Returns:
            dict with session info: {session_id, playlist_path, output_dir}
        """
        import time
        start_time = time.time()
        print("[HLS-SDK] Starting HLS generation")
        
        if not self._streaming_mode:
            raise RuntimeError("Must call setup_hls_streaming() before generate_hls()")
        
        # Wait for first audio segment
        print("[HLS-SDK] Waiting for first audio segment...")
        
        while not self._audio_segments:
            await asyncio.sleep(0.05)
            if self._audio_complete.is_set():
                break
                
        if not self._audio_segments:
            raise RuntimeError("No audio segments received")
            
        print(f"[HLS-SDK] First audio received at {time.time() - start_time:.3f}s")
        
        # Create combined audio file
        temp_audio = self._create_temp_combined_audio()
        
        # Setup HLS writer (instead of fMP4 writer)
        self._setup_hls_writer(temp_audio)
        
        # Calculate initial frame count
        initial_frames = self._calculate_frame_count()
        self.setup_Nd(N_d=initial_frames)
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=self._run_progressive_generation,
            args=(start_time,)
        )
        generation_thread.start()
        
        # Wait for playlist to be ready
        print("[HLS-SDK] Waiting for playlist...")
        playlist_ready = self._hls_writer.wait_for_playlist(timeout=30.0)
        if playlist_ready:
            print(f"[HLS-SDK] Playlist ready at {time.time() - start_time:.3f}s")
        else:
            print("[HLS-SDK] Warning: Playlist not yet created, continuing anyway")
        
        return {
            "session_id": self._hls_session_id,
            "playlist_path": self._hls_writer.playlist_path,
            "output_dir": self._hls_writer.output_dir,
            "generation_thread": generation_thread,
        }
    
    async def wait_for_completion(self, generation_thread: threading.Thread, timeout: float = 300.0):
        """Wait for generation to complete."""
        import time
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        
        # Wait for thread in executor to avoid blocking event loop
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: generation_thread.join(timeout=timeout)),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print(f"[HLS-SDK] Generation timed out after {timeout}s")
        
        if generation_thread.is_alive():
            print("[HLS-SDK] Warning: Generation thread still alive")
        
        print(f"[HLS-SDK] Generation complete at {time.time() - start_time:.3f}s")
        
    def _create_temp_combined_audio(self) -> str:
        """Create a temporary audio file with all current segments."""
        if not self._audio_segments:
            return None
            
        audio_arrays = [s.audio_data for s in self._audio_segments if s.audio_data is not None]
        
        if not audio_arrays:
            return None
            
        combined = np.concatenate(audio_arrays)
        self._total_audio = combined
        
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
            print(f"[HLS-SDK] Starting generation thread at {time.time() - start_time:.3f}s")
            
            chunk_size = (2, 3, 1)
            processed_chunks = 0
            last_segment_count = 0
            
            while not self.stop_event.is_set():
                current_segment_count = len(self._audio_segments)
                
                # Check if we have new audio to process
                if current_segment_count > last_segment_count:
                    self._create_temp_combined_audio()
                    last_segment_count = current_segment_count
                    
                    new_frame_count = self._calculate_frame_count()
                    if new_frame_count > self._frame_offset:
                        self.setup_Nd(N_d=new_frame_count)
                
                # Process available audio
                if self._total_audio is not None:
                    audio = np.concatenate([
                        np.zeros((chunk_size[0] * 640,), dtype=np.float32),
                        self._total_audio
                    ], 0)
                    
                    split_len = int(sum(chunk_size) * 0.04 * 16000) + 80
                    
                    start_idx = processed_chunks * chunk_size[1] * 640
                    
                    for i in range(start_idx, len(audio), chunk_size[1] * 640):
                        if self.stop_event.is_set():
                            print(f"[HLS-SDK] Stop event triggered at chunk {processed_chunks}", flush=True)
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
                            print(f"[HLS-SDK] Processing first audio chunk", flush=True)
                        elif processed_chunks % 20 == 0:
                            print(f"[HLS-SDK] Processed {processed_chunks} audio chunks", flush=True)
                        
                        self.run_chunk(audio_chunk, chunk_size)
                
                # Check if we're done
                if self._audio_complete.is_set():
                    total_samples = len(self._total_audio) if self._total_audio is not None else 0
                    processed_samples = processed_chunks * chunk_size[1] * 640
                    print(f"[HLS-SDK] Audio complete check: processed={processed_samples}, total={total_samples}", flush=True)
                    if processed_samples >= total_samples:
                        print(f"[HLS-SDK] All audio processed, exiting generation loop", flush=True)
                        break
                else:
                    time.sleep(0.1)
            
            print(f"[HLS-SDK] Finished processing {processed_chunks} chunks", flush=True)
            
            # Signal end of audio
            self.audio2motion_queue.put(None)
            print("[HLS-SDK] Signaled end of audio to motion queue", flush=True)
            
        except Exception as e:
            print(f"[HLS-SDK] Error in progressive generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            self.worker_exception = e
            self.stop_event.set()
    
    def close(self):
        """Close the SDK and cleanup resources."""
        if self._hls_writer:
            self._hls_writer.close()
        super().close()
        self._cleanup_temp()


class HLSVideoGenerator:
    """
    High-level coordinator for HLS-based video generation.
    
    Manages the SDK lifecycle and provides a clean async interface
    for the WebSocket handler. Outputs HLS files to a session directory
    instead of streaming binary chunks.
    """
    
    def __init__(
        self,
        source_path: str,
        width: int = 512,
        height: int = 512,
        cfg_pkl: str = None,
        data_root: str = None,
        watermark: bool = False,
        watermark_position: str = "bottom-right",
        hls_base_dir: str = "/tmp/hls_streams",
        session_id: str = None,
    ):
        self.source_path = source_path
        self.width = width
        self.height = height
        self.watermark = watermark
        self.watermark_position = watermark_position
        import time as _time
        self.session_id = session_id or f"hls_{int(_time.time() * 1000)}"
        
        # HLS output directory for this session
        self.hls_output_dir = os.path.join(hls_base_dir, self.session_id)
        
        # Use defaults if not provided
        self.cfg_pkl = cfg_pkl or "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
        self.data_root = data_root or "/app/checkpoints/ditto_trt_Ampere_Plus"
        
        self._sdk: Optional[HLSStreamingSDK] = None
        self._generation_thread: Optional[threading.Thread] = None
        
    async def initialize(self):
        """Initialize the SDK for HLS streaming."""
        print(f"[HLS-VideoGen] Initializing with source={self.source_path}")
        
        self._sdk = HLSStreamingSDK(self.cfg_pkl, self.data_root)
        self._sdk.setup_hls_streaming(
            self.source_path,
            width=self.width,
            height=self.height,
            hls_output_dir=self.hls_output_dir,
            session_id=self.session_id,
            watermark=self.watermark,
            watermark_position=self.watermark_position,
        )
        
    async def add_audio(self, seq: int, audio_path: str, is_final: bool = False):
        """Add an audio segment for processing."""
        segment = AudioSegment(
            seq=seq,
            audio_path=audio_path,
            is_final=is_final
        )
        await self._sdk.append_audio(segment)
        
    async def start_generation(self) -> dict:
        """
        Start HLS generation.
        
        Returns:
            dict with {session_id, playlist_path, output_dir, playlist_url}
        """
        result = await self._sdk.generate_hls()
        self._generation_thread = result.get("generation_thread")
        
        return {
            "session_id": self.session_id,
            "playlist_path": result["playlist_path"],
            "output_dir": result["output_dir"],
        }
    
    async def wait_for_completion(self, timeout: float = 300.0):
        """Wait for generation to complete."""
        if self._generation_thread:
            await self._sdk.wait_for_completion(self._generation_thread, timeout)
    
    def get_playlist_url(self, base_url: str = "") -> str:
        """Get the URL for the HLS playlist."""
        return f"{base_url}/hls/{self.session_id}/stream.m3u8"
    
    def cleanup(self):
        """Cleanup resources."""
        if self._sdk:
            try:
                self._sdk.stop_event.set()
                self._sdk._audio_complete.set()
                self._sdk.close()
                self._sdk._audio_segments = []
                self._sdk._total_audio = None
            except Exception as e:
                print(f"[HLS-VideoGen] Cleanup error: {e}")
            finally:
                self._sdk = None
        
        # Clean up HLS output files
        if os.path.exists(self.hls_output_dir):
            import shutil
            try:
                shutil.rmtree(self.hls_output_dir)
            except Exception as e:
                print(f"[HLS-VideoGen] Output cleanup error: {e}")
        
        print("[HLS-VideoGen] Cleanup complete")
