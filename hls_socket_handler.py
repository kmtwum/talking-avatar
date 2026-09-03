"""
HLS Socket Handler - WebSocket message routing for HLS-based video generation.

Handles the WebSocket protocol for streaming speech-to-video generation
using HLS output instead of fMP4 binary streaming. The key difference:
instead of sending binary video chunks over WebSocket, this handler starts
HLS generation and sends the client a playlist URL for HTTP-based playback.

This enables iOS Safari compatibility since HLS is natively supported.
"""

import asyncio
import json
import time
import uuid
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect

from socket_session import SocketSession, SessionConfig, SessionManager, SessionState
from tts_streamer import TTSStreamer
from stream_pipeline_hls import HLSVideoGenerator, AudioSegment
from gpu_concurrency import GPUConcurrencyManager


class HLSMessageType:
    """WebSocket message types for HLS protocol."""
    # Client -> Server
    SESSION_START = "SESSION_START"
    SPEECH_CHUNK = "SPEECH_CHUNK"
    SESSION_END = "SESSION_END"
    
    # Server -> Client
    SESSION_STARTED = "SESSION_STARTED"
    HLS_READY = "HLS_READY"
    HLS_SEGMENT = "HLS_SEGMENT"
    STATUS = "STATUS"
    SESSION_COMPLETE = "SESSION_COMPLETE"
    ERROR = "ERROR"


class HLSSocketHandler:
    """
    Handles WebSocket connections for HLS-based video generation.
    
    Coordinates:
    - Message parsing and routing
    - TTS generation pipeline
    - Video generation pipeline (HLS output)
    - HLS playlist URL delivery to client
    
    Key difference from SocketHandler:
    - Instead of streaming binary fMP4 chunks, sends HLS_READY with playlist URL
    - Client fetches .m3u8 and .ts segments via HTTP endpoints
    - Compatible with iOS Safari native HLS support
    """
    
    def __init__(self, websocket: WebSocket, hls_base_url: str = ""):
        self.websocket = websocket
        self.hls_base_url = hls_base_url
        self.session: Optional[SocketSession] = None
        self.tts_streamer: Optional[TTSStreamer] = None
        self.video_generator: Optional[HLSVideoGenerator] = None
        
        # GPU concurrency
        self._gpu_acquired = False
        
        # Pipeline tasks
        self._tts_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._audio_bridge_task: Optional[asyncio.Task] = None
        self._segment_task: Optional[asyncio.Task] = None
        self._hls_ready_sent = False
        
    async def handle_connection(self):
        """
        Main entry point for handling a WebSocket connection.
        
        Runs until the connection is closed or an error occurs.
        """
        await self.websocket.accept()
        print("[HLS-Handler] Connection accepted")
        
        try:
            # Wait for session start message
            await self._wait_for_session_start()
            
            # Start processing tasks
            await self._start_pipeline()
            
            # Main message loop
            await self._message_loop()
            
        except WebSocketDisconnect:
            print("[HLS-Handler] Client disconnected")
        except Exception as e:
            print(f"[HLS-Handler] Error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_error(str(e), "HANDLER_ERROR")
        finally:
            await self._cleanup()
            
    async def _wait_for_session_start(self):
        """Wait for and process SESSION_START message."""
        timeout = 30.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive_json(),
                    timeout=5.0
                )
                
                msg_type = message.get("type")
                
                if msg_type == HLSMessageType.SESSION_START:
                    config = SessionConfig.from_dict(message)
                    self.session = SessionManager().create_session(config)
                    self.session.start()
                    
                    # Send session started confirmation
                    await self._send_json({
                        "type": HLSMessageType.SESSION_STARTED,
                        "session_id": self.session.session_id,
                        "protocol": "hls",
                    })
                    
                    print(f"[HLS-Handler] Session {self.session.session_id} started (HLS mode)")
                    return
                else:
                    await self._send_error(
                        f"Expected SESSION_START, got {msg_type}",
                        "PROTOCOL_ERROR"
                    )
                    
            except asyncio.TimeoutError:
                continue
                
        raise TimeoutError("No SESSION_START received within timeout")
        
    async def _start_pipeline(self):
        """Start the TTS and HLS video generation pipeline."""
        if not self.session:
            raise RuntimeError("No session started")
        
        # Initialize TTS streamer
        self.tts_streamer = TTSStreamer(self.session)
        await self.tts_streamer.start()
        
        # Initialize HLS video generator (loads model weights, no GPU compute yet)
        self.video_generator = HLSVideoGenerator(
            source_path=self.session.get_image_path(),
            width=self.session.config.size,
            height=self.session.config.size,
            watermark=self.session.config.watermark,
            watermark_position=self.session.config.watermark_position,
            session_id=self.session.session_id,
            live_mode=self.session.config.live_mode,
            hls_segment_duration=self.session.config.hls_segment_duration,
        )
        await self.video_generator.initialize()
        
        # Start audio bridge (moves audio from session queue to video generator)
        self._audio_bridge_task = asyncio.create_task(self._audio_bridge())
        
        # Start HLS generation task (GPU is acquired/released inside this task)
        self._video_task = asyncio.create_task(self._run_hls_generation())
        
        print(f"[HLS-Handler] Pipeline started for session {self.session.session_id}")
        
    async def _message_loop(self):
        """Process incoming messages until session ends."""
        session_end_received = False
        message_count = 0
        timeout_count = 0
        
        print("[HLS-Handler] Message loop started", flush=True)
        
        while self.session and self.session.state != SessionState.CLOSED:
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive_json(),
                    timeout=1.0
                )
                message_count += 1
                timeout_count = 0
                
                await self._handle_message(message)
                
                if message.get("type") == HLSMessageType.SESSION_END:
                    session_end_received = True
                    print(f"[HLS-Handler] SESSION_END received after {message_count} messages", flush=True)
                    break
                
            except asyncio.TimeoutError:
                timeout_count += 1
                if timeout_count <= 3 or timeout_count % 10 == 0:
                    print(f"[HLS-Handler] Message timeout #{timeout_count} (received {message_count} msgs so far)", flush=True)
                if session_end_received and self._video_task and self._video_task.done():
                    break
                continue
                
            except WebSocketDisconnect:
                print(f"[HLS-Handler] WebSocket disconnected after {message_count} messages", flush=True)
                raise
        
        print(f"[HLS-Handler] Message loop ended: {message_count} messages, session_end={session_end_received}", flush=True)
        
        # After SESSION_END, wait for video generation to complete
        if session_end_received and self._video_task and not self._video_task.done():
            print("[HLS-Handler] SESSION_END received, waiting for HLS generation to complete...", flush=True)
            try:
                await asyncio.wait_for(self._video_task, timeout=180.0)
            except asyncio.TimeoutError:
                print("[HLS-Handler] HLS generation timed out after 180s", flush=True)
            except Exception as e:
                print(f"[HLS-Handler] Error waiting for HLS generation: {e}", flush=True)
                
    async def _handle_message(self, message: dict):
        """Route incoming message to appropriate handler."""
        msg_type = message.get("type")
        print(f"[HLS-Handler] Received message type: {msg_type}")
        
        if msg_type == HLSMessageType.SPEECH_CHUNK:
            print(f"[HLS-Handler] Processing SPEECH_CHUNK: seq={message.get('seq')}")
            await self._handle_speech_chunk(message)
            
        elif msg_type == HLSMessageType.SESSION_END:
            await self._handle_session_end()
            
        else:
            print(f"[HLS-Handler] Unknown message type: {msg_type}")
            
    async def _handle_speech_chunk(self, message: dict):
        """Handle incoming SPEECH_CHUNK message."""
        if not self.session:
            await self._send_error("No active session", "NO_SESSION")
            return
            
        try:
            chunk = await self.session.handle_chunk(message)
            if self.session.chunks_received % 5 == 0:
                await self._send_status()
                
        except ValueError as e:
            await self._send_error(str(e), "CHUNK_ERROR")
            
    async def _handle_session_end(self):
        """Handle SESSION_END message."""
        if not self.session:
            return
            
        print("[HLS-Handler] Session end received")
        self.session.end()
        await self.session.text_queue.put(None)
        
    async def _audio_bridge(self):
        """
        Bridge audio from session queue to HLS video generator.
        """
        try:
            seq = 0
            while True:
                try:
                    item = await asyncio.wait_for(
                        self.session.audio_queue.get(),
                        timeout=1.0
                    )
                    
                    if item is None:
                        if self.video_generator and self.video_generator._sdk:
                            self.video_generator._sdk._audio_complete.set()
                        break
                        
                    audio_seq, audio_path = item
                    await self.video_generator.add_audio(
                        seq=audio_seq,
                        audio_path=audio_path,
                        is_final=False
                    )
                    seq = audio_seq
                    
                except asyncio.TimeoutError:
                    if self.session.state == SessionState.CLOSED:
                        break
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[HLS-Handler] Audio bridge error: {e}")
            
    async def _run_hls_generation(self):
        """
        Start HLS generation and notify the client.

        VOD mode (default): waits for all TTS audio, generates fully, then
        sends a single HLS_READY with the complete playlist.

        Live mode (``live_mode: true`` in SESSION_START): starts after the
        pre-buffer threshold, sends HLS_READY when the first segment lands,
        pushes HLS_SEGMENT events as segments appear, then SESSION_COMPLETE.
        """
        if self.session and self.session.config.live_mode:
            await self._run_hls_generation_live()
        else:
            await self._run_hls_generation_vod()

    async def _run_hls_generation_vod(self):
        """Original VOD flow — HLS_READY after full generation."""
        try:
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                print("[HLS-Handler] Waiting for all audio to be ready...", flush=True)
                try:
                    await asyncio.wait_for(self.session.all_audio_ready.wait(), timeout=120.0)
                    print(f"[HLS-Handler] All audio ready "
                          f"({self.session.audio_segments_buffered} segments, "
                          f"{self.session.audio_duration_buffered:.2f}s)", flush=True)
                except asyncio.TimeoutError:
                    print("[HLS-Handler] Timeout waiting for audio, starting anyway", flush=True)

                await self._acquire_gpu_and_generate()
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            playlist_url = self.video_generator.get_playlist_url(self.hls_base_url)
            segment_count = self._segment_count()

            print(f"[HLS-Handler] HLS generation complete: {segment_count} segments", flush=True)

            hls_sent = await self._send_json({
                "type": HLSMessageType.HLS_READY,
                "session_id": self.session.session_id,
                "playlist_url": playlist_url,
                "segments": segment_count,
                "mode": "vod",
            })
            if hls_sent:
                print(f"[HLS-Handler] Sent HLS_READY (VOD): {playlist_url}", flush=True)

            await self._send_session_complete(playlist_url, segment_count)

        except asyncio.CancelledError:
            print("[HLS-Handler] HLS generation cancelled", flush=True)
        except Exception as e:
            await self._handle_generation_error(e)

    async def _run_hls_generation_live(self):
        """Progressive HLS for realtime consumers (Teams, etc.)."""
        try:
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                print("[HLS-Handler] Live mode: waiting for pre-buffer...", flush=True)
                try:
                    await asyncio.wait_for(
                        self.session.prebuffer_ready.wait(), timeout=120.0
                    )
                    print(
                        f"[HLS-Handler] Pre-buffer ready "
                        f"({self.session.audio_segments_buffered} segments, "
                        f"{self.session.audio_duration_buffered:.2f}s)",
                        flush=True,
                    )
                except asyncio.TimeoutError:
                    print("[HLS-Handler] Pre-buffer timeout, starting anyway", flush=True)

                await self._acquire_gpu_and_generate(
                    start_segment_notifier=True,
                    wait_for_pipeline=False,
                )

                try:
                    await asyncio.wait_for(self.session.all_audio_ready.wait(), timeout=300.0)
                except asyncio.TimeoutError:
                    print("[HLS-Handler] Live mode: audio wait timed out", flush=True)

                await self.video_generator.wait_for_completion()

            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            if self._segment_task and not self._segment_task.done():
                await asyncio.wait_for(self._segment_task, timeout=30.0)

            playlist_url = self.video_generator.get_playlist_url(self.hls_base_url)
            segment_count = self._segment_count()
            await self._send_session_complete(playlist_url, segment_count)

        except asyncio.CancelledError:
            print("[HLS-Handler] Live HLS generation cancelled", flush=True)
        except Exception as e:
            await self._handle_generation_error(e)

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(10.0)
            try:
                await self._send_json({
                    "type": HLSMessageType.STATUS,
                    "message": "Generating video...",
                    **(self.session.get_status() if self.session else {}),
                })
            except Exception:
                break

    async def _acquire_gpu_and_generate(
        self,
        *,
        start_segment_notifier: bool = False,
        wait_for_pipeline: bool = True,
    ):
        gpu_mgr = GPUConcurrencyManager()
        if gpu_mgr.is_at_capacity:
            await self._send_json({
                "type": HLSMessageType.STATUS,
                "queue_position": gpu_mgr.queue_position + 1,
                "message": "Waiting for GPU availability...",
            })

        acquired = await gpu_mgr.acquire()
        if not acquired:
            await self._send_error(
                "Server is at capacity. Please try again shortly.",
                "SERVER_BUSY",
            )
            raise RuntimeError("GPU acquisition timed out")
        self._gpu_acquired = True
        print("[HLS-Handler] GPU slot acquired for generation", flush=True)

        try:
            await self.video_generator.start_generation()

            if start_segment_notifier:
                self._segment_task = asyncio.create_task(self._notify_hls_segments())

            await self._wait_for_playlist_and_notify_live()

            if wait_for_pipeline:
                await self.video_generator.wait_for_completion()
        finally:
            if self._gpu_acquired:
                GPUConcurrencyManager().release()
                self._gpu_acquired = False
                print("[HLS-Handler] GPU slot released after generation", flush=True)

    async def _wait_for_playlist_and_notify_live(self):
        """Send HLS_READY once playlist + first segment exist (live mode only)."""
        if not self.session or not self.session.config.live_mode:
            return
        if self._hls_ready_sent:
            return

        writer = self._hls_writer()
        if not writer:
            return

        deadline = time.time() + 60.0
        while time.time() < deadline:
            playlist_ok = writer.wait_for_playlist(timeout=0.5)
            if playlist_ok and writer.get_segment_count() >= 1:
                playlist_url = self.video_generator.get_playlist_url(self.hls_base_url)
                sent = await self._send_json({
                    "type": HLSMessageType.HLS_READY,
                    "session_id": self.session.session_id,
                    "playlist_url": playlist_url,
                    "segments": writer.get_segment_count(),
                    "mode": "live",
                })
                if sent:
                    self._hls_ready_sent = True
                    print(f"[HLS-Handler] Sent HLS_READY (live): {playlist_url}", flush=True)
                return
            await asyncio.sleep(0.15)

        print("[HLS-Handler] Live HLS_READY timed out waiting for first segment", flush=True)

    async def _notify_hls_segments(self):
        """Push HLS_SEGMENT JSON events as .ts files land on disk."""
        seen: set = set()
        seq = 0
        try:
            while True:
                writer = self._hls_writer()
                if not writer:
                    await asyncio.sleep(0.2)
                    continue

                for name in writer.list_segments_since(seen):
                    seq += 1
                    url = f"{self.hls_base_url}/hls/{self.session.session_id}/{name}"
                    await self._send_json({
                        "type": HLSMessageType.HLS_SEGMENT,
                        "session_id": self.session.session_id,
                        "segment": name,
                        "seq": seq,
                        "url": url,
                    })
                    seen.add(name)

                if writer.stream_complete:
                    # Final drain
                    for name in writer.list_segments_since(seen):
                        seq += 1
                        url = f"{self.hls_base_url}/hls/{self.session.session_id}/{name}"
                        await self._send_json({
                            "type": HLSMessageType.HLS_SEGMENT,
                            "session_id": self.session.session_id,
                            "segment": name,
                            "seq": seq,
                            "url": url,
                        })
                        seen.add(name)
                    break

                await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[HLS-Handler] Segment notifier error: {e}", flush=True)

    def _hls_writer(self):
        if (
            self.video_generator
            and self.video_generator._sdk
            and self.video_generator._sdk._hls_writer
        ):
            return self.video_generator._sdk._hls_writer
        return None

    def _segment_count(self) -> int:
        writer = self._hls_writer()
        return writer.get_segment_count() if writer else 0

    async def _send_session_complete(self, playlist_url: str, segment_count: int):
        self.session.complete()
        complete_sent = await self._send_json({
            "type": HLSMessageType.SESSION_COMPLETE,
            "total_duration_ms": self.session.total_duration_ms,
            "chunks_processed": self.session.chunks_processed,
            "frames_generated": self.session.frames_generated,
            "audio_buffered_seconds": round(self.session.audio_duration_buffered, 2),
            "hls_segments": segment_count,
            "playlist_url": playlist_url,
        })
        if complete_sent:
            print("[HLS-Handler] SESSION_COMPLETE sent successfully", flush=True)

    async def _handle_generation_error(self, e: Exception):
        print(f"[HLS-Handler] HLS generation error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if self.session:
            self.session.set_error(e)
        await self._send_error(str(e), "VIDEO_ERROR")
            
    async def _send_json(self, data: dict) -> bool:
        """Send JSON message to client. Returns True if sent, False if failed."""
        try:
            await self.websocket.send_json(data)
            return True
        except Exception as e:
            print(f"[HLS-Handler] Failed to send {data.get('type', '?')}: {e}")
            return False
            
    async def _send_status(self):
        """Send status update to client."""
        if not self.session:
            return
            
        await self._send_json({
            "type": HLSMessageType.STATUS,
            **self.session.get_status()
        })
        
    async def _send_error(self, message: str, code: str):
        """Send error message to client."""
        await self._send_json({
            "type": HLSMessageType.ERROR,
            "message": message,
            "code": code
        })
        
    async def _cleanup(self):
        """Cleanup resources when connection closes.
        
        In VOD mode, video generation writes to disk, so it's useful even
        if the WebSocket dies. We let _video_task finish (with a timeout)
        instead of cancelling it, so the playlist is always complete.
        """
        print("[HLS-Handler] Cleaning up")
        
        # Cancel TTS and audio bridge tasks (they depend on the WebSocket)
        for task in [self._tts_task, self._audio_bridge_task, self._segment_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Let _video_task finish — it writes to disk and the playlist
        # is still useful for HTTP fetches even if the WS is dead.
        if self._video_task and not self._video_task.done():
            print("[HLS-Handler] Waiting for video generation to finish...", flush=True)
            try:
                await asyncio.wait_for(self._video_task, timeout=180.0)
            except asyncio.TimeoutError:
                print("[HLS-Handler] Video generation timed out, cancelling", flush=True)
                self._video_task.cancel()
                try:
                    await self._video_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
                    
        # Stop TTS streamer
        if self.tts_streamer:
            await self.tts_streamer.stop()
            
        # Cleanup SDK resources but KEEP HLS files on disk
        if self.video_generator:
            self.video_generator.cleanup_sdk()
            
            # Schedule deferred file deletion (2 minutes)
            generator = self.video_generator
            async def _deferred_file_cleanup():
                await asyncio.sleep(120)  # 2 minutes
                generator.cleanup_files()
            asyncio.create_task(_deferred_file_cleanup())
        
        # Release GPU slot (safety net — should already be released
        # by _run_hls_generation, but guard against error paths)
        if self._gpu_acquired:
            GPUConcurrencyManager().release()
            self._gpu_acquired = False
            print("[HLS-Handler] GPU slot released in cleanup (was still held)", flush=True)
            
        # Remove session from manager
        if self.session:
            SessionManager().remove_session(self.session.session_id)
            
        print("[HLS-Handler] Cleanup complete (HLS files will be removed in 5 minutes)")
