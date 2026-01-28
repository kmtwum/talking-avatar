"""
Socket Handler - WebSocket message routing and connection management.

Handles the WebSocket protocol for streaming speech-to-video generation.
"""

import asyncio
import json
import time
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect

from socket_session import SocketSession, SessionConfig, SessionManager, SessionState
from tts_streamer import TTSStreamer
from stream_pipeline_socket import SocketVideoGenerator, AudioSegment


class MessageType:
    """WebSocket message types."""
    # Client -> Server
    SESSION_START = "SESSION_START"
    SPEECH_CHUNK = "SPEECH_CHUNK"
    SESSION_END = "SESSION_END"
    
    # Server -> Client
    SESSION_STARTED = "SESSION_STARTED"
    STATUS = "STATUS"
    SESSION_COMPLETE = "SESSION_COMPLETE"
    ERROR = "ERROR"


class SocketHandler:
    """
    Handles WebSocket connections for streaming video generation.
    
    Coordinates:
    - Message parsing and routing
    - TTS generation pipeline
    - Video generation pipeline
    - Binary output streaming
    """
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session: Optional[SocketSession] = None
        self.tts_streamer: Optional[TTSStreamer] = None
        self.video_generator: Optional[SocketVideoGenerator] = None
        
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
        print("[SocketHandler] Connection accepted")
        
        try:
            # Wait for session start message
            await self._wait_for_session_start()
            
            # Start processing tasks
            await self._start_pipeline()
            
            # Main message loop
            await self._message_loop()
            
        except WebSocketDisconnect:
            print("[SocketHandler] Client disconnected")
        except Exception as e:
            print(f"[SocketHandler] Error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_error(str(e), "HANDLER_ERROR")
        finally:
            await self._cleanup()
            
    async def _wait_for_session_start(self):
        """Wait for and process SESSION_START message."""
        timeout = 30.0  # 30 second timeout for session start
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive_json(),
                    timeout=5.0
                )
                
                msg_type = message.get("type")
                
                if msg_type == MessageType.SESSION_START:
                    config = SessionConfig.from_dict(message)
                    self.session = SessionManager().create_session(config)
                    self.session.start()
                    
                    # Send session started confirmation
                    await self._send_json({
                        "type": MessageType.SESSION_STARTED,
                        "session_id": self.session.session_id
                    })
                    
                    print(f"[SocketHandler] Session {self.session.session_id} started")
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
        """Start the TTS and video generation pipeline."""
        if not self.session:
            raise RuntimeError("No session started")
            
        # Initialize TTS streamer
        self.tts_streamer = TTSStreamer(self.session)
        await self.tts_streamer.start()
        
        # Initialize video generator
        self.video_generator = SocketVideoGenerator(
            source_path=self.session.get_image_path(),
            width=self.session.config.size,
            height=self.session.config.size
        )
        await self.video_generator.initialize()
        
        # Start audio bridge (moves audio from session queue to video generator)
        self._audio_bridge_task = asyncio.create_task(self._audio_bridge())
        
        # Start video output streaming
        self._video_task = asyncio.create_task(self._stream_video_output())
        
        print(f"[SocketHandler] Pipeline started for session {self.session.session_id}")
        
    async def _message_loop(self):
        """Process incoming messages until session ends."""
        session_end_received = False
        message_count = 0
        timeout_count = 0
        
        print("[SocketHandler] Message loop started", flush=True)
        
        while self.session and self.session.state != SessionState.CLOSED:
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive_json(),
                    timeout=1.0
                )
                message_count += 1
                timeout_count = 0  # Reset on successful receive
                
                await self._handle_message(message)
                
                # Check if we received SESSION_END
                if message.get("type") == MessageType.SESSION_END:
                    session_end_received = True
                    print(f"[SocketHandler] SESSION_END received after {message_count} messages", flush=True)
                    break  # Exit message loop, but will wait for video below
                
            except asyncio.TimeoutError:
                timeout_count += 1
                if timeout_count <= 3 or timeout_count % 10 == 0:
                    print(f"[SocketHandler] Message timeout #{timeout_count} (received {message_count} msgs so far)", flush=True)
                # If video task is done and we got session_end, exit
                if session_end_received and self._video_task and self._video_task.done():
                    break
                continue
                
            except WebSocketDisconnect:
                print(f"[SocketHandler] WebSocket disconnected after {message_count} messages", flush=True)
                raise
        
        print(f"[SocketHandler] Message loop ended: {message_count} messages, session_end={session_end_received}", flush=True)
        
        # After SESSION_END, wait for video generation to complete
        if session_end_received and self._video_task and not self._video_task.done():
            print("[SocketHandler] SESSION_END received, waiting for video generation to complete...", flush=True)
            try:
                # Wait for video task to complete (with timeout)
                await asyncio.wait_for(self._video_task, timeout=180.0)
            except asyncio.TimeoutError:
                print("[SocketHandler] Video generation timed out after 180s", flush=True)
            except Exception as e:
                print(f"[SocketHandler] Error waiting for video: {e}", flush=True)
                
    async def _handle_message(self, message: dict):
        """Route incoming message to appropriate handler."""
        msg_type = message.get("type")
        print(f"[SocketHandler] Received message type: {msg_type}")
        
        if msg_type == MessageType.SPEECH_CHUNK:
            print(f"[SocketHandler] Processing SPEECH_CHUNK: seq={message.get('seq')}")
            await self._handle_speech_chunk(message)
            
        elif msg_type == MessageType.SESSION_END:
            await self._handle_session_end()
            
        else:
            print(f"[SocketHandler] Unknown message type: {msg_type}")
            
    async def _handle_speech_chunk(self, message: dict):
        """Handle incoming SPEECH_CHUNK message."""
        if not self.session:
            await self._send_error("No active session", "NO_SESSION")
            return
            
        try:
            chunk = await self.session.handle_chunk(message)
            
            # Send status update periodically
            if self.session.chunks_received % 5 == 0:
                await self._send_status()
                
        except ValueError as e:
            await self._send_error(str(e), "CHUNK_ERROR")
            
    async def _handle_session_end(self):
        """Handle SESSION_END message."""
        if not self.session:
            return
            
        print("[SocketHandler] Session end received")
        self.session.end()
        
        # Signal end of text queue
        await self.session.text_queue.put(None)
        
    async def _audio_bridge(self):
        """
        Bridge audio from session queue to video generator.
        
        Transfers generated audio segments to the video pipeline
        as they become available.
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
                        # End of audio - signal video generator
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
            print(f"[SocketHandler] Audio bridge error: {e}")
            
    async def _stream_video_output(self):
        """
        Stream video segments to the client as they're generated.
        
        Waits for audio pre-buffer before starting video generation
        to ensure smooth playback.
        
        Sends binary fMP4 segments via WebSocket.
        """
        try:
            # Wait for ALL audio to be ready before starting video
            # This ensures FFmpeg has complete audio and won't cut off early
            print("[SocketHandler] Waiting for all audio to be ready...", flush=True)
            try:
                await asyncio.wait_for(self.session.all_audio_ready.wait(), timeout=120.0)
                print(f"[SocketHandler] All audio ready, starting video stream "
                      f"({self.session.audio_segments_buffered} segments, "
                      f"{self.session.audio_duration_buffered:.2f}s)", flush=True)
            except asyncio.TimeoutError:
                print("[SocketHandler] Timeout waiting for audio, starting anyway", flush=True)
            
            # Stream video segments
            segment_count = 0
            async for segment in self.video_generator.generate():
                # Send binary segment
                await self.websocket.send_bytes(segment)
                segment_count += 1
                
                if segment_count == 1:
                    print(f"[SocketHandler] Sent init segment ({len(segment)} bytes)", flush=True)
                elif segment_count % 10 == 0:
                    print(f"[SocketHandler] Sent {segment_count} segments", flush=True)
                    
            print(f"[SocketHandler] Video streaming complete: {segment_count} segments", flush=True)
            
            # Send completion message
            print("[SocketHandler] Sending SESSION_COMPLETE...", flush=True)
            self.session.complete()
            await self._send_json({
                "type": MessageType.SESSION_COMPLETE,
                "total_duration_ms": self.session.total_duration_ms,
                "chunks_processed": self.session.chunks_processed,
                "frames_generated": self.session.frames_generated,
                "audio_buffered_seconds": round(self.session.audio_duration_buffered, 2)
            })
            print("[SocketHandler] SESSION_COMPLETE sent successfully", flush=True)
            
        except asyncio.CancelledError:
            print("[SocketHandler] Video streaming cancelled", flush=True)
        except Exception as e:
            print(f"[SocketHandler] Video stream error: {e}", flush=True)
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
            print(f"[SocketHandler] Failed to send JSON: {e}")
            
    async def _send_status(self):
        """Send status update to client."""
        if not self.session:
            return
            
        await self._send_json({
            "type": MessageType.STATUS,
            **self.session.get_status()
        })
        
    async def _send_error(self, message: str, code: str):
        """Send error message to client."""
        await self._send_json({
            "type": MessageType.ERROR,
            "message": message,
            "code": code
        })
        
    async def _cleanup(self):
        """Cleanup resources when connection closes."""
        print("[SocketHandler] Cleaning up")
        
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
            
        # Cleanup video generator
        if self.video_generator:
            self.video_generator.cleanup()
            
        # Remove session from manager
        if self.session:
            SessionManager().remove_session(self.session.session_id)
            
        print("[SocketHandler] Cleanup complete")