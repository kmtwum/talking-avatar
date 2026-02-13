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
        
        # Pipeline tasks
        self._tts_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._audio_bridge_task: Optional[asyncio.Task] = None
        
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
        
        # Initialize HLS video generator
        self.video_generator = HLSVideoGenerator(
            source_path=self.session.get_image_path(),
            width=self.session.config.size,
            height=self.session.config.size,
            watermark=self.session.config.watermark,
            watermark_position=self.session.config.watermark_position,
            session_id=self.session.session_id,
        )
        await self.video_generator.initialize()
        
        # Start audio bridge (moves audio from session queue to video generator)
        self._audio_bridge_task = asyncio.create_task(self._audio_bridge())
        
        # Start HLS generation task
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
        Start HLS generation and notify client when playlist is ready.
        
        Instead of streaming binary chunks, this:
        1. Waits for all audio to be ready
        2. Starts HLS generation
        3. Sends HLS_READY message with playlist URL
        4. Monitors segment production and notifies client
        5. Sends SESSION_COMPLETE when done
        """
        try:
            # Wait for all audio to be ready
            print("[HLS-Handler] Waiting for all audio to be ready...", flush=True)
            try:
                await asyncio.wait_for(self.session.all_audio_ready.wait(), timeout=120.0)
                print(f"[HLS-Handler] All audio ready "
                      f"({self.session.audio_segments_buffered} segments, "
                      f"{self.session.audio_duration_buffered:.2f}s)", flush=True)
            except asyncio.TimeoutError:
                print("[HLS-Handler] Timeout waiting for audio, starting anyway", flush=True)
            
            # Start HLS generation
            result = await self.video_generator.start_generation()
            
            # Build playlist URL
            playlist_url = self.video_generator.get_playlist_url(self.hls_base_url)
            
            # Notify client that HLS stream is ready
            await self._send_json({
                "type": HLSMessageType.HLS_READY,
                "session_id": self.session.session_id,
                "playlist_url": playlist_url,
            })
            print(f"[HLS-Handler] Sent HLS_READY: {playlist_url}", flush=True)
            
            # Wait for generation to complete
            await self.video_generator.wait_for_completion()
            
            # Notify about individual segments
            segment_count = 0
            if self.video_generator._sdk and self.video_generator._sdk._hls_writer:
                segment_count = self.video_generator._sdk._hls_writer.get_segment_count()
            
            print(f"[HLS-Handler] HLS generation complete: {segment_count} segments", flush=True)
            
            # Send completion message
            print("[HLS-Handler] Sending SESSION_COMPLETE...", flush=True)
            self.session.complete()
            await self._send_json({
                "type": HLSMessageType.SESSION_COMPLETE,
                "total_duration_ms": self.session.total_duration_ms,
                "chunks_processed": self.session.chunks_processed,
                "frames_generated": self.session.frames_generated,
                "audio_buffered_seconds": round(self.session.audio_duration_buffered, 2),
                "hls_segments": segment_count,
                "playlist_url": playlist_url,
            })
            print("[HLS-Handler] SESSION_COMPLETE sent successfully", flush=True)
            
        except asyncio.CancelledError:
            print("[HLS-Handler] HLS generation cancelled", flush=True)
        except Exception as e:
            print(f"[HLS-Handler] HLS generation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            self.session.set_error(e)
            await self._send_error(str(e), "VIDEO_ERROR")
            
    async def _send_json(self, data: dict):
        """Send JSON message to client."""
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            print(f"[HLS-Handler] Failed to send JSON: {e}")
            
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
        """Cleanup resources when connection closes."""
        print("[HLS-Handler] Cleaning up")
        
        # Cancel tasks
        for task in [self._tts_task, self._audio_bridge_task, self._video_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Stop TTS streamer
        if self.tts_streamer:
            await self.tts_streamer.stop()
            
        # Cleanup video generator (removes HLS files)
        if self.video_generator:
            self.video_generator.cleanup()
            
        # Remove session from manager
        if self.session:
            SessionManager().remove_session(self.session.session_id)
            
        print("[HLS-Handler] Cleanup complete")
