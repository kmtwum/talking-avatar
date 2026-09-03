"""
HLS Streaming Writer - Outputs HTTP Live Streaming segments for iOS/cross-platform playback.

Uses FFmpeg to produce MPEG-TS segments and an m3u8 playlist suitable for
native playback on iOS Safari and other browsers via hls.js.

Unlike fMP4/MSE, HLS is natively supported by iOS Safari and doesn't require
the MediaSource API.
"""

import subprocess
import threading
import queue
import os
import time
import shutil
import tempfile
from typing import Optional, Iterator
import numpy as np


class HLSStreamWriter:
    """
    Streaming HLS writer that produces .ts segments and .m3u8 playlist as frames arrive.
    
    Uses FFmpeg subprocess with HLS muxer to produce browser-compatible
    HLS streams. Segments and playlists are written to a session-specific
    directory and served via HTTP endpoints.
    """
    
    MIME_TYPE = 'application/vnd.apple.mpegurl'
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 25,
        segment_duration: float = 2.0,
        audio_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        live_mode: bool = False,
    ):
        """
        Initialize the HLS writer.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second (default 25)
            segment_duration: Target duration per HLS segment in seconds (default 2.0)
            audio_path: Optional path to audio file for muxing
            output_dir: Directory to write HLS files to. Created if not exists.
            session_id: Unique session identifier for this stream
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.segment_duration = segment_duration
        self.audio_path = audio_path
        self.session_id = session_id or f"hls_{int(time.time() * 1000)}"
        self.live_mode = live_mode
        
        # Output directory for HLS files
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = tempfile.mkdtemp(prefix=f"hls_{self.session_id}_")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._process: Optional[subprocess.Popen] = None
        self._started = False
        self._closed = False
        self._finalized = False
        self._frames_written = 0
        
        # Segment tracking
        self._segment_queue: queue.Queue = queue.Queue()
        self._watcher_thread: Optional[threading.Thread] = None
        self._known_segments: set = set()
        self._playlist_ready = threading.Event()
        self._stream_complete = threading.Event()
        
    @property
    def playlist_path(self) -> str:
        """Path to the m3u8 playlist file."""
        return os.path.join(self.output_dir, "stream.m3u8")
    
    @property
    def segment_pattern(self) -> str:
        """FFmpeg segment filename pattern."""
        return os.path.join(self.output_dir, "segment_%05d.ts")
        
    def start(self):
        """Start the FFmpeg HLS encoding process."""
        if self._started:
            raise RuntimeError("Writer already started")
        
        print(f"[HLS] Starting FFmpeg process for session {self.session_id}")
        print(f"[HLS] Output dir: {self.output_dir}")
        
        # Calculate keyframe interval based on segment duration
        gop_size = int(self.fps * self.segment_duration)
        
        # FFmpeg command for HLS output
        cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
        ]
        
        # Add audio input if provided
        if self.audio_path:
            cmd.extend(['-i', self.audio_path])
            print(f"[HLS] Adding audio input: {self.audio_path}")
        
        # Video codec settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(gop_size),
            '-keyint_min', str(gop_size),
            '-sc_threshold', '0',
            '-pix_fmt', 'yuv420p',
        ])
        
        # Audio codec if audio is present
        if self.audio_path:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-ar', '44100'])
        
        # HLS muxer settings
        hls_flags = (
            "append_list+omit_endlist+independent_segments"
            if self.live_mode
            else "independent_segments"
        )
        cmd.extend([
            '-f', 'hls',
            '-hls_time', str(self.segment_duration),
            '-hls_list_size', '0',  # Keep all segments in playlist
            '-hls_flags', hls_flags,
            '-hls_segment_type', 'mpegts',
            '-hls_segment_filename', self.segment_pattern,
            self.playlist_path,
        ])
        
        print(f"[HLS] FFmpeg command: {' '.join(cmd)}")
        
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        
        print(f"[HLS] FFmpeg process started (PID: {self._process.pid})")
        
        # Start directory watcher to detect new segments
        self._watcher_thread = threading.Thread(
            target=self._watch_segments, daemon=True
        )
        self._watcher_thread.start()
        
        self._started = True
        print(f"[HLS] Writer started successfully")
        
    def _watch_segments(self):
        """
        Background thread that watches for new HLS segments.
        
        Polls the output directory for new .ts files and queues them
        as they appear.
        """
        while not self._stream_complete.is_set():
            try:
                # List current segment files
                if os.path.exists(self.output_dir):
                    files = set(
                        f for f in os.listdir(self.output_dir) 
                        if f.endswith('.ts')
                    )
                    
                    # Find new segments
                    new_segments = files - self._known_segments
                    
                    for seg_name in sorted(new_segments):
                        seg_path = os.path.join(self.output_dir, seg_name)
                        # Wait briefly to ensure file is fully written
                        time.sleep(0.1)
                        
                        seg_size = os.path.getsize(seg_path)
                        self._segment_queue.put(seg_name)
                        self._known_segments.add(seg_name)
                        
                        if len(self._known_segments) == 1:
                            print(f"[HLS] First segment ready: {seg_name} ({seg_size} bytes)")
                    
                    # Check if playlist exists
                    if os.path.exists(self.playlist_path):
                        self._playlist_ready.set()
                
                time.sleep(0.2)  # Poll interval
                
            except Exception as e:
                print(f"[HLS] Watcher error: {e}")
                time.sleep(0.5)
        
        # Final check for any remaining segments
        if os.path.exists(self.output_dir):
            files = set(
                f for f in os.listdir(self.output_dir)
                if f.endswith('.ts')
            )
            new_segments = files - self._known_segments
            for seg_name in sorted(new_segments):
                self._segment_queue.put(seg_name)
                self._known_segments.add(seg_name)
        
        # Signal end
        self._segment_queue.put(None)
        
    def write_frame(self, frame_rgb: np.ndarray):
        """
        Write a frame to the HLS encoder.
        
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
            self._frames_written += 1
        except BrokenPipeError:
            print(f"[HLS] Broken pipe - FFmpeg process may have died")
            stderr = self._process.stderr.read().decode() if self._process.stderr else ""
            print(f"[HLS] FFmpeg stderr: {stderr}")
            raise
        
    def wait_for_playlist(self, timeout: float = 15.0) -> bool:
        """
        Wait for the m3u8 playlist to be created.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if playlist is ready, False if timed out
        """
        return self._playlist_ready.wait(timeout=timeout)
        
    def iter_new_segments(self, timeout: float = 1.0) -> Iterator[str]:
        """
        Iterate over new segment filenames as they become available.
        
        Args:
            timeout: Seconds to wait for each segment
            
        Yields:
            str: Segment filename (e.g. "segment_00000.ts")
        """
        empty_count = 0
        max_empty = 5
        
        while True:
            try:
                item = self._segment_queue.get(timeout=timeout)
                empty_count = 0
            except queue.Empty:
                empty_count += 1
                if empty_count >= max_empty:
                    print(f"[HLS] iter_segments: no data for {max_empty * timeout}s, exiting")
                    break
                continue
            
            if item is None:
                break
            
            yield item
    
    def get_segment_count(self) -> int:
        """Get the number of segments produced so far."""
        return len(self._known_segments)

    def list_segments_since(self, seen: set) -> list[str]:
        """Return sorted segment filenames not yet in *seen*."""
        return sorted(n for n in self._known_segments if n not in seen)

    @property
    def stream_complete(self) -> bool:
        return self._stream_complete.is_set()
    
    def get_segment_path(self, segment_name: str) -> str:
        """Get the full path to a segment file."""
        return os.path.join(self.output_dir, segment_name)
    
    def read_segment(self, segment_name: str) -> bytes:
        """
        Read a segment file's contents.
        
        Args:
            segment_name: The segment filename
            
        Returns:
            Raw bytes of the .ts segment
        """
        path = self.get_segment_path(segment_name)
        with open(path, 'rb') as f:
            return f.read()
    
    def read_playlist(self) -> str:
        """
        Read the current m3u8 playlist content.
        
        Returns:
            The playlist text
        """
        if not os.path.exists(self.playlist_path):
            return ""
        with open(self.playlist_path, 'r') as f:
            return f.read()
            
    def finalize(self):
        """
        Finalize the stream — close FFmpeg stdin, wait for it to exit,
        then post-process the playlist into a clean VOD.
        """
        if self._finalized:
            return
        self._finalized = True
            
        print(f"[HLS] Finalizing stream ({self._frames_written} frames written)")
        
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception as e:
                print(f"[HLS] Error closing stdin: {e}")
        
        # Wait for FFmpeg to finish
        if self._process:
            try:
                self._process.wait(timeout=30)
                returncode = self._process.returncode
                if returncode != 0:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    print(f"[HLS] FFmpeg exited with code {returncode}: {stderr}")
                else:
                    print(f"[HLS] FFmpeg process completed successfully")
            except subprocess.TimeoutExpired:
                print(f"[HLS] FFmpeg process timed out, killing")
                self._process.kill()
        
        # Post-process playlist into a clean VOD (skip mid-stream rewrite in live mode)
        if not self.live_mode:
            self._postprocess_playlist()
        
        # Signal watcher that stream is complete
        self._stream_complete.set()
        
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
    
    def _postprocess_playlist(self):
        """
        Rewrite the m3u8 playlist as a clean VOD:
        - Remove #EXT-X-DISCONTINUITY lines
        - Add #EXT-X-PLAYLIST-TYPE:VOD
        - Ensure #EXT-X-ENDLIST is present at the end
        """
        if not os.path.exists(self.playlist_path):
            print("[HLS] No playlist to post-process")
            return
        
        try:
            with open(self.playlist_path, 'r') as f:
                lines = f.readlines()
            
            out = []
            has_playlist_type = False
            has_endlist = False
            
            for line in lines:
                stripped = line.strip()
                
                # Skip discontinuity tags
                if stripped == '#EXT-X-DISCONTINUITY':
                    continue
                
                # Track existing tags
                if stripped.startswith('#EXT-X-PLAYLIST-TYPE'):
                    has_playlist_type = True
                    out.append('#EXT-X-PLAYLIST-TYPE:VOD\n')
                    continue
                if stripped == '#EXT-X-ENDLIST':
                    has_endlist = True
                    # Don't append yet — we'll add it at the end
                    continue
                
                out.append(line)
            
            # Insert PLAYLIST-TYPE:VOD after the EXTM3U header if not present
            if not has_playlist_type:
                insert_idx = 1  # after #EXTM3U
                for i, line in enumerate(out):
                    if line.strip().startswith('#EXT-X-VERSION') or \
                       line.strip().startswith('#EXT-X-TARGETDURATION'):
                        insert_idx = i
                        break
                out.insert(insert_idx, '#EXT-X-PLAYLIST-TYPE:VOD\n')
            
            # Ensure ENDLIST at end
            out.append('#EXT-X-ENDLIST\n')
            
            with open(self.playlist_path, 'w') as f:
                f.writelines(out)
            
            seg_count = sum(1 for l in out if l.strip().endswith('.ts'))
            print(f"[HLS] Post-processed playlist as VOD ({seg_count} segments)")
            
        except Exception as e:
            print(f"[HLS] Error post-processing playlist: {e}")
    
    def close(self):
        """Close the writer, finalize, and cleanup."""
        if self._closed:
            return
        self._closed = True
        self.finalize()
        
    def cleanup(self):
        """Remove all HLS output files."""
        self.close()
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
                print(f"[HLS] Cleaned up output dir: {self.output_dir}")
            except Exception as e:
                print(f"[HLS] Cleanup error: {e}")
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.close()
