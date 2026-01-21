"""
fMP4 Streaming Writer - Outputs fragmented MP4 chunks for real-time streaming.

Uses FFmpeg to encode frames into fMP4 format with proper box structure
for MediaSource Extensions (MSE) playback in browsers.

MIME Type: video/mp4; codecs="avc1.42E01E, mp4a.40.2"
"""

import subprocess
import struct
import threading
import queue
from typing import Optional, Iterator
import numpy as np


# MP4 Box type identifiers
BOX_FTYP = b'ftyp'
BOX_MOOV = b'moov'
BOX_MOOF = b'moof'
BOX_MDAT = b'mdat'


def parse_box_header(data: bytes, offset: int = 0) -> tuple[int, bytes, int]:
    """
    Parse an MP4 box header.
    
    Args:
        data: Raw bytes containing the box
        offset: Starting offset in the data
        
    Returns:
        Tuple of (box_size, box_type, header_size)
        header_size is 8 for normal boxes, 16 for extended size boxes
    """
    if len(data) < offset + 8:
        raise ValueError("Not enough data for box header")
    
    size = struct.unpack('>I', data[offset:offset+4])[0]
    box_type = data[offset+4:offset+8]
    header_size = 8
    
    if size == 1:
        # Extended size (64-bit)
        if len(data) < offset + 16:
            raise ValueError("Not enough data for extended box header")
        size = struct.unpack('>Q', data[offset+8:offset+16])[0]
        header_size = 16
    elif size == 0:
        # Box extends to end of file
        size = len(data) - offset
    
    return size, box_type, header_size


def extract_init_segment(data: bytes) -> bytes:
    """
    Extract the initialization segment (ftyp + moov) from fMP4 data.
    
    Uses proper MP4 box parsing to find and extract the ftyp and moov boxes.
    These boxes contain codec configuration needed before media segments.
    
    Args:
        data: Raw fMP4 data from FFmpeg output
        
    Returns:
        Bytes containing ftyp + moov boxes
    """
    init_segment = b''
    offset = 0
    
    while offset < len(data):
        try:
            size, box_type, header_size = parse_box_header(data, offset)
        except ValueError:
            break
        
        if box_type == BOX_FTYP:
            init_segment += data[offset:offset + size]
        elif box_type == BOX_MOOV:
            init_segment += data[offset:offset + size]
            break  # moov is the last box we need for init
        
        offset += size
    
    return init_segment


def extract_media_segments(data: bytes, start_offset: int = 0) -> Iterator[bytes]:
    """
    Extract media segments (moof + mdat pairs) from fMP4 data.
    
    Each media segment contains one or more video frames and is
    independently decodable after the initialization segment.
    
    Args:
        data: Raw fMP4 data from FFmpeg output
        start_offset: Offset to start parsing from (skip init segment)
        
    Yields:
        Bytes containing moof + mdat pairs
    """
    offset = start_offset
    current_segment = b''
    
    while offset < len(data):
        try:
            size, box_type, header_size = parse_box_header(data, offset)
        except ValueError:
            break
        
        box_data = data[offset:offset + size]
        
        if box_type == BOX_MOOF:
            # Start of a new segment - yield previous if exists
            if current_segment:
                yield current_segment
            current_segment = box_data
        elif box_type == BOX_MDAT:
            # Media data - append to current segment
            current_segment += box_data
        
        offset += size
    
    # Yield any remaining segment
    if current_segment:
        yield current_segment


class FMP4StreamWriter:
    """
    Streaming fMP4 writer that yields media segments as frames are generated.
    
    Uses FFmpeg subprocess with fragmented MP4 output for browser-compatible
    streaming via MediaSource Extensions.
    """
    
    MIME_TYPE = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"'
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 25,
        fragment_duration_frames: int = 5,
        audio_path: Optional[str] = None
    ):
        """
        Initialize the fMP4 writer.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels  
            fps: Frames per second (default 25)
            fragment_duration_frames: Frames per fragment (default 5)
            audio_path: Optional path to audio file for muxing
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.fragment_duration_frames = fragment_duration_frames
        self.audio_path = audio_path
        
        self._process: Optional[subprocess.Popen] = None
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._init_segment: Optional[bytes] = None
        self._init_segment_size: int = 0
        self._started = False
        self._closed = False
        
    def start(self):
        """Start the FFmpeg encoding process."""
        if self._started:
            raise RuntimeError("Writer already started")
        
        print(f"[FMP4] Starting FFmpeg process")
        
        # FFmpeg command for fMP4 with browser-compatible codecs
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0'
        ]
        
        # Add audio input if provided
        if self.audio_path:
            cmd.extend(['-i', self.audio_path])
            print(f"[FMP4] Adding audio input: {self.audio_path}")
            
        # Output format and codecs
        cmd.extend([
            '-f', 'mp4',
            '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(self.fragment_duration_frames),  # GOP size = fragment size
            '-keyint_min', str(self.fragment_duration_frames),
            '-pix_fmt', 'yuv420p'
        ])
        
        # Add audio codec if audio is present
        if self.audio_path:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            
        cmd.append('pipe:1')
        
        print(f"[FMP4] FFmpeg command: {' '.join(cmd)}")
        
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        print(f"[FMP4] FFmpeg process started (PID: {self._process.pid})")
        
        # Start reader thread to consume stdout
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        
        self._started = True
        print(f"[FMP4] Writer started successfully")
        
    def _read_output(self):
        """Background thread that reads FFmpeg output and queues chunks."""
        buffer = b''
        init_segment_complete = False
        
        while True:
            chunk = self._process.stdout.read(4096)
            if not chunk:
                break
            
            buffer += chunk
            
            # Parse boxes and queue complete segments
            offset = 0
            while offset < len(buffer):
                if len(buffer) - offset < 8:
                    break  # Need more data for header
                
                try:
                    size, box_type, header_size = parse_box_header(buffer, offset)
                except ValueError:
                    break
                
                if len(buffer) - offset < size:
                    break  # Incomplete box
                
                box_data = buffer[offset:offset + size]
                
                if box_type == BOX_FTYP:
                    if self._init_segment is None:
                        self._init_segment = box_data
                    else:
                        self._init_segment += box_data
                elif box_type == BOX_MOOV:
                    if self._init_segment is None:
                        self._init_segment = box_data
                    else:
                        self._init_segment += box_data
                    self._init_segment_size = len(self._init_segment)
                    init_segment_complete = True
                    self._output_queue.put(('init', self._init_segment))
                elif box_type == BOX_MOOF:
                    # Start collecting media segment
                    pass
                elif box_type == BOX_MDAT:
                    # Complete media segment (moof came before)
                    pass
                
                # For media segments, we need to handle moof+mdat pairs
                if init_segment_complete and box_type in (BOX_MOOF, BOX_MDAT):
                    self._output_queue.put(('media', box_data))
                
                offset += size
            
            # Keep unconsumed data
            buffer = buffer[offset:]
        
        # Signal end of stream
        self._output_queue.put(None)
        
    def write_frame(self, frame_rgb: np.ndarray):
        """
        Write a frame to the encoder.
        
        Args:
            frame_rgb: RGB frame as numpy array (H, W, 3), dtype uint8
        """
        if not self._started:
            raise RuntimeError("Writer not started")
        if self._closed:
            raise RuntimeError("Writer is closed")
        
        # Ensure correct format
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        
        # Write raw frame data
        try:
            self._process.stdin.write(frame_rgb.tobytes())
            self._process.stdin.flush()
        except BrokenPipeError:
            print(f"[FMP4] Broken pipe - FFmpeg process may have died")
            raise
        
    def get_init_segment(self, timeout: float = 5.0) -> bytes:
        """
        Get the initialization segment (ftyp + moov).
        
        Blocks until the init segment is available.
        Must be called after start() and after at least one frame is written.
        
        Args:
            timeout: Maximum seconds to wait for init segment
            
        Returns:
            Bytes containing ftyp + moov boxes
        """
        if self._init_segment is not None:
            return self._init_segment
        
        # Wait for init segment from queue
        import time
        start = time.time()
        while time.time() - start < timeout:
            try:
                item = self._output_queue.get(timeout=0.1)
                if item is None:
                    raise RuntimeError("Stream ended before init segment")
                segment_type, data = item
                if segment_type == 'init':
                    return data
            except queue.Empty:
                continue
        
        raise TimeoutError("Timeout waiting for init segment")
        
    def iter_segments(self, timeout: float = 1.0) -> Iterator[bytes]:
        """
        Iterate over media segments as they become available.
        
        Yields moof+mdat segment pairs for each fragment.
        
        Args:
            timeout: Seconds to wait for each segment
            
        Yields:
            Bytes containing media segments
        """
        moof_data = None
        
        while True:
            try:
                item = self._output_queue.get(timeout=timeout)
            except queue.Empty:
                continue
            
            if item is None:
                break
            
            segment_type, data = item
            
            if segment_type == 'init':
                # Skip init segment (already handled)
                continue
            
            # Determine box type
            if len(data) >= 8:
                box_type = data[4:8]
                if box_type == BOX_MOOF:
                    moof_data = data
                elif box_type == BOX_MDAT and moof_data is not None:
                    # Complete segment
                    yield moof_data + data
                    moof_data = None
                    
    def close(self):
        """Close the writer and FFmpeg process."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._process:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except:
                self._process.kill()
                
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.close()
