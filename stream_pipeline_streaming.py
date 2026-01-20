"""
Streaming Pipeline - SDK variant that yields fMP4 chunks instead of writing to file.

This extends the base StreamSDK to support real-time streaming output
for MediaSource Extensions playback in browsers.
"""

import threading
import queue
import asyncio
import numpy as np
import math
import librosa
from typing import AsyncIterator, Optional
from tqdm import tqdm

from stream_pipeline_offline import StreamSDK
from core.atomic_components.fmp4_writer import FMP4StreamWriter
from core.atomic_components.condition_handler import _mirror_index


class StreamingSDK(StreamSDK):
    """
    SDK variant that yields fMP4 chunks as frames are generated.
    
    Instead of writing to a file, this SDK streams video segments
    to an async queue for real-time consumption.
    """
    
    def __init__(self, cfg_pkl, data_root, **kwargs):
        super().__init__(cfg_pkl, data_root, **kwargs)
        
        self._streaming_mode = False
        self._fmp4_writer: Optional[FMP4StreamWriter] = None
        self._fmp4_ready = threading.Event()  # Signal when fMP4 writer is ready
        self._chunk_queue: asyncio.Queue = None
        self._loop: asyncio.AbstractEventLoop = None
        
    def setup_streaming(
        self,
        source_path: str,
        width: int = 256,
        height: int = 256,
        **kwargs
    ):
        """
        Setup for streaming output.
        
        Args:
            source_path: Path to source image/video
            width: Output video width
            height: Output video height  
            **kwargs: Additional setup kwargs
        """
        self._streaming_mode = True
        
        # Merge defaults with streaming-optimized settings
        streaming_defaults = {
            "sampling_timesteps": 15,  # Faster inference
            "max_size": max(width, height),
            "smo_k_s": 5,
            "smo_k_d": 1,
        }
        streaming_defaults.update(kwargs)
        
        # Create output path for temporary processing
        # We won't actually write to this - the fmp4_writer handles output
        self._streaming_output_width = width
        self._streaming_output_height = height
        
        # Setup base SDK with a dummy output path
        # The actual streaming happens through the fMP4 writer
        import tempfile
        self._temp_dir = tempfile.mkdtemp()
        dummy_output = f"{self._temp_dir}/stream.mp4"
        
        # Call parent setup
        super().setup(source_path, dummy_output, **streaming_defaults)
        
    def _setup_streaming_writer(self):
        """Initialize the fMP4 writer for streaming output."""
        self._fmp4_writer = FMP4StreamWriter(
            width=self._streaming_output_width,
            height=self._streaming_output_height,
            fps=25,
            fragment_duration_frames=5
        )
        self._fmp4_writer.start()
        # Signal that the fMP4 writer is ready
        self._fmp4_ready.set()
        
    def writer_worker(self):
        """
        Override base class writer_worker to use fMP4 streaming output.
        
        This is called by the base class setup() when creating worker threads.
        """
        try:
            self._streaming_writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
            
    def _streaming_writer_worker(self):
        """
        Modified writer worker that outputs to fMP4 writer instead of file.
        
        Redirects frames to the fMP4 streaming writer instead of the file writer.
        Waits for the fMP4 writer to be initialized before processing.
        """
        # Wait for fMP4 writer to be ready (blocks until generate_chunks starts)
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
            
            # Write to fMP4 writer
            self._fmp4_writer.write_frame(res_frame_rgb)
            
    async def generate_chunks(self, audio_path: str) -> AsyncIterator[bytes]:
        """
        Async generator that yields fMP4 segments as frames are generated.
        
        This is the main entry point for streaming video generation.
        
        Args:
            audio_path: Path to the audio file
            
        Yields:
            bytes: fMP4 segments (init segment first, then media segments)
        """
        if not self._streaming_mode:
            raise RuntimeError("Must call setup_streaming() before generate_chunks()")
        
        # Store event loop reference for cross-thread communication
        self._loop = asyncio.get_event_loop()
        self._chunk_queue = asyncio.Queue()
        
        # Setup the fMP4 writer
        self._setup_streaming_writer()
        
        # Load audio and compute frame count
        audio, sr = librosa.core.load(audio_path, sr=16000)
        num_frames = math.ceil(len(audio) / 16000 * 25)
        
        # Setup frame count
        self.setup_Nd(N_d=num_frames)
        
        # Process audio to features
        aud_feat = self.wav2feat.wav2feat(audio)
        
        # Start frame generation in background thread
        generation_thread = threading.Thread(
            target=self._run_generation_background,
            args=(aud_feat,)
        )
        generation_thread.start()
        
        # Yield init segment first
        try:
            init_segment = self._fmp4_writer.get_init_segment(timeout=10.0)
            yield init_segment
        except TimeoutError:
            raise RuntimeError("Failed to get initialization segment")
        
        # Yield media segments as they become available
        for segment in self._fmp4_writer.iter_segments(timeout=2.0):
            yield segment
            
        # Wait for generation to complete
        generation_thread.join()
        
        # Cleanup
        self._fmp4_writer.close()
        self._cleanup_temp()
        
    def _run_generation_background(self, aud_feat):
        """
        Run the video generation pipeline in background thread.
        
        This feeds audio features to the audio2motion queue and
        waits for all workers to complete.
        """
        try:
            # Feed audio features to the pipeline
            self.audio2motion_queue.put(aud_feat)
            self.audio2motion_queue.put(None)  # Signal end
            
            # Wait for all workers to complete
            for thread in self.thread_list:
                thread.join()
                
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
            
    def _cleanup_temp(self):
        """Clean up temporary files."""
        import shutil
        try:
            if hasattr(self, '_temp_dir') and self._temp_dir:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except:
            pass
            
    def get_init_segment(self) -> bytes:
        """
        Get the initialization segment for the fMP4 stream.
        
        This contains the ftyp and moov boxes needed for MSE playback.
        Must be called after some frames have been processed.
        
        Returns:
            bytes: ftyp + moov boxes
        """
        if self._fmp4_writer is None:
            raise RuntimeError("Streaming not started")
        return self._fmp4_writer.get_init_segment()
        
    def close(self):
        """Close the SDK and cleanup resources."""
        if self._fmp4_writer:
            self._fmp4_writer.close()
            
        super().close()
        self._cleanup_temp()


class StreamingSDKWithAudio(StreamingSDK):
    """
    Extended streaming SDK that handles audio muxing.
    
    For full fMP4 streaming with audio, the audio track needs to be
    muxed into the stream. This class handles that complexity.
    """
    
    def __init__(self, cfg_pkl, data_root, **kwargs):
        super().__init__(cfg_pkl, data_root, **kwargs)
        self._audio_path: Optional[str] = None
        
    async def generate_chunks_with_audio(
        self,
        audio_path: str
    ) -> AsyncIterator[bytes]:
        """
        Generate fMP4 chunks with video and audio muxed together.
        
        For now, this generates video-only fMP4 which works with MSE.
        Audio can be played separately for initial implementation.
        
        Args:
            audio_path: Path to audio file
            
        Yields:
            bytes: fMP4 segments
        """
        self._audio_path = audio_path
        
        async for chunk in self.generate_chunks(audio_path):
            yield chunk
