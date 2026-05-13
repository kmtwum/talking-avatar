import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import requests
import httpx
import subprocess
import tempfile
import os
from PIL import Image
from io import BytesIO

app = FastAPI(
    root_path="/video"
)

# Add CORS middleware for WebSocket and HTTP requests
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv

load_dotenv()


# SDK warmup on startup
@app.on_event("startup")
async def startup():
    """Pre-load SDK on startup to avoid cold start latency."""
    try:
        from sdk_manager import SDKManager
        # Configure paths for online mode
        cfg_pkl = os.getenv("SDK_CFG_PKL", "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
        data_root = os.getenv("SDK_DATA_ROOT", "/app/checkpoints/ditto_trt_Ampere_Plus")
        SDKManager.configure(cfg_pkl=cfg_pkl, data_root=data_root)
        SDKManager.warmup()
    except Exception as e:
        print(f"[STARTUP] SDK warmup failed (will load on first request): {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup SDK on shutdown."""
    try:
        from sdk_manager import SDKManager
        SDKManager.cleanup()
    except Exception as e:
        print(f"[SHUTDOWN] SDK cleanup error: {e}")


def generate_tts(text: str, tts_preference: str = "coqui", tts_voice_id: str = None, user_id: str = None,
                 voice_source: str = None) -> str:
    """Generate TTS audio"""
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        tts_response = requests.post(tts_url, json={
            "text": text,
            "split_sentences": True,
            "source_aud": voice_source,
            "clone": user_id,
            "speed": 1.2,
            "streaming": False,
            "model": "tts_models/multilingual/multi-dataset/xtts_v2"
        })
        tts_response.raise_for_status()
        request_id = str(uuid.uuid4())[:8]
        audio_path = f"/tmp/tts_{request_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_response.content)
        return audio_path
    else:
        from elevenlabs.client import ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("VOICE_ID")
        if tts_voice_id:
            print(f"Using voice ID from request: {tts_voice_id}")
            voice_id = tts_voice_id
        else:
            print("Using default voice ID from environment variable")

        elevenlabs = ElevenLabs(api_key=api_key)
        response = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings={
                "stability": 0.7
            }
        )

        print("Saving 11 audio file...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"Audio file saved to {f.name}")

        return f.name


async def generate_tts_async(
    text: str,
    tts_preference: str = "coqui",
    tts_voice_id: str = None,
    user_id: str = None,
    voice_source: str = None
) -> str:
    """
    Generate TTS audio asynchronously.
    
    Uses httpx for non-blocking HTTP requests, freeing the event loop
    for other requests during the TTS API call.
    
    Note: Video generation depends on audio (needs duration for frame count),
    so TTS must complete before video generation starts.
    """
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(tts_url, json={
                "text": text,
                "split_sentences": True,
                "source_aud": voice_source,
                "clone": user_id,
                "streaming": False,
                "model": "tts_models/multilingual/multi-dataset/xtts_v2"
            })
            response.raise_for_status()

            request_id = str(uuid.uuid4())[:8]
            audio_path = f"/tmp/tts_{request_id}.wav"
            with open(audio_path, "wb") as f:
                f.write(response.content)
            return audio_path
    else:
        # ElevenLabs - run in thread pool as it's not async-native
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_tts(text, tts_preference, tts_voice_id, user_id, voice_source)
        )


def get_secret_key(secret):
    key_file = os.getenv(secret)
    if key_file:
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
            return api_key
    return None


@app.post("/generate")
async def quick_generate(
        text: Optional[str] = Form(...),
        inference: str = Form("inference_minimal"),
        size: str = Form("256"),
        tts_preference: str = Form("coqui"),
        tts_voice_id: str = Form(None),
        user_id: Optional[str] = Form(None),
        avatar: Optional[str] = Form("sunny"),
        source_img: str = Form(None),
        source_aud: str = Form(None),
        audio: UploadFile = File(None),
        watermark: bool = Form(False),
        watermark_position: str = Form("bottom-right")
):
    """Optimized endpoint for fast generation"""
    from gpu_concurrency import GPUConcurrencyManager
    gpu_mgr = GPUConcurrencyManager()

    img_path = f"/app/user_img/{avatar}.jpg"
    if not os.path.exists(img_path):
        img_path = "/app/user_img/jamal.jpg"

    print(f"Using img_path: {img_path}")

    request_id = str(uuid.uuid4())[:8]

    # Use provided audio or generate TTS
    if audio:
        audio_path = f"/app/aud/{user_id}_{request_id}_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        print(f"Using uploaded audio: {audio_path}")
    else:
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        print("Generating TTS...")
        audio_path = generate_tts(text, tts_preference, tts_voice_id, user_id=user_id, voice_source=source_aud)

    # Generate video with streaming optimizations
    output_path = f"/tmp/quick_{hash(request_id)}.mp4"

    cmd = [
        "python", f"/app/{inference}.py",
        "--audio_path", audio_path,
        "--source_path", img_path,
        "--output_path", output_path,
        "--size", size,
        "--steps", "10",
        "--fast",
    ]
    if watermark:
        cmd.append("--watermark")
        cmd.extend(["--watermark_position", watermark_position])

    # Acquire GPU slot before running the subprocess
    acquired = await gpu_mgr.acquire()
    if not acquired:
        return JSONResponse(
            {"error": "Server is at capacity. Please try again shortly."},
            status_code=503
        )
    try:
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()
    finally:
        gpu_mgr.release()

    # Clean up temp audio
    if os.path.exists(audio_path):
        os.unlink(audio_path)

    # Stream response
    def video_stream():
        with open(output_path, "rb") as _f:
            while chunk := _f.read(8192):
                yield chunk

    return StreamingResponse(video_stream(), media_type="video/mp4")


@app.post("/presave-photo")
async def upload_photo(user_id: str = Form(...), image: UploadFile = File(...)):
    os.makedirs("/app/user_img", exist_ok=True)
    img_path = f"/app/user_img/{user_id}.jpg"

    # Read upload once, resize to 512x512, and save as JPEG
    content = await image.read()
    img = Image.open(BytesIO(content))
    img = img.convert("RGB")
    img = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
    img.save(img_path, format="JPEG", quality=90, optimize=True)

    return {"message": "Photo uploaded successfully", "user_id": user_id}


@app.delete("/presave-photo")
async def delete_photo(user_id: str):
    img_path = f"/app/user_img/{user_id}.jpg"
    if os.path.exists(img_path):
        os.remove(img_path)
        return {"message": "Photo deleted successfully", "user_id": user_id}
    else:
        return {"message": "Photo not found", "user_id": user_id}


@app.get("/health")
def health():
    return {"status": "ready"}


@app.get("/gpu/status")
async def gpu_status():
    """Return current GPU concurrency status for monitoring."""
    from gpu_concurrency import GPUConcurrencyManager
    return GPUConcurrencyManager().get_status()


# Simple WebSocket test endpoint
@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple test endpoint to verify WebSocket connectivity."""
    # Log connection attempt details
    print(f"[WS Test] Connection attempt from: {websocket.client}")
    print(f"[WS Test] Headers: {dict(websocket.headers)}")
    
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "WebSocket test successful"})
    await websocket.close()


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for streaming speech-to-video generation.
    
    Protocol:
    1. Client connects and sends SESSION_START with config
    2. Client sends SPEECH_CHUNK messages progressively
    3. Server streams back binary fMP4 video chunks
    4. Client sends SESSION_END when done sending text
    5. Server sends SESSION_COMPLETE and closes connection
    
    Message Types (Client -> Server):
        SESSION_START: {"type": "SESSION_START", "avatar": "sunny", "size": 256, ...}
        SPEECH_CHUNK: {"type": "SPEECH_CHUNK", "seq": 0, "text": "Hello..."}
        SESSION_END: {"type": "SESSION_END"}
    
    Message Types (Server -> Client):
        SESSION_STARTED: {"type": "SESSION_STARTED", "session_id": "abc123"}
        <binary fMP4 video data>
        STATUS: {"type": "STATUS", "chunks_processed": 3, ...}
        SESSION_COMPLETE: {"type": "SESSION_COMPLETE", ...}
        ERROR: {"type": "ERROR", "message": "...", "code": "..."}
    """
    # Log connection details for debugging
    print(f"[SocketHandler] Connection attempt from: {websocket.client}")
    print(f"[SocketHandler] Origin: {websocket.headers.get('origin', 'Not provided')}")
    
    from socket_handler import SocketHandler
    
    handler = SocketHandler(websocket)
    await handler.handle_connection()


# ============================================================================
# HLS Endpoints - iOS-compatible video streaming
# ============================================================================

HLS_BASE_DIR = os.getenv("HLS_BASE_DIR", "/tmp/hls_streams")
os.makedirs(HLS_BASE_DIR, exist_ok=True)


@app.websocket("/ws/generate/hls")
async def websocket_generate_hls(websocket: WebSocket):
    """
    WebSocket endpoint for HLS-based speech-to-video generation.
    
    Same protocol as /ws/generate but outputs HLS instead of fMP4:
    
    Protocol:
    1. Client connects and sends SESSION_START with config
    2. Client sends SPEECH_CHUNK messages progressively
    3. Server sends HLS_READY with playlist URL when stream is available
    4. Client fetches .m3u8 and .ts segments via HTTP
    5. Client sends SESSION_END when done sending text
    6. Server sends SESSION_COMPLETE and closes connection
    
    Message Types (Client -> Server):
        SESSION_START: {"type": "SESSION_START", "avatar": "sunny", "size": 256, ...}
        SPEECH_CHUNK: {"type": "SPEECH_CHUNK", "seq": 0, "text": "Hello..."}
        SESSION_END: {"type": "SESSION_END"}
    
    Message Types (Server -> Client):
        SESSION_STARTED: {"type": "SESSION_STARTED", "session_id": "abc123", "protocol": "hls"}
        HLS_READY: {"type": "HLS_READY", "session_id": "abc123", "playlist_url": "/hls/abc123/stream.m3u8"}
        STATUS: {"type": "STATUS", "chunks_processed": 3, ...}
        SESSION_COMPLETE: {"type": "SESSION_COMPLETE", ..., "playlist_url": "..."}
        ERROR: {"type": "ERROR", "message": "...", "code": "..."}
    """
    print(f"[HLS-WS] Connection attempt from: {websocket.client}")
    print(f"[HLS-WS] Origin: {websocket.headers.get('origin', 'Not provided')}")
    
    from hls_socket_handler import HLSSocketHandler
    
    # Determine base URL for HLS from request headers
    host = websocket.headers.get("host", "localhost:8000")
    scheme = "https" if websocket.headers.get("x-forwarded-proto") == "https" else "http"
    hls_base_url = f"{scheme}://{host}"
    
    handler = HLSSocketHandler(websocket, hls_base_url=hls_base_url)
    await handler.handle_connection()


@app.get("/hls/{session_id}/stream.m3u8")
async def hls_playlist(session_id: str):
    """
    Serve the HLS playlist for a session.
    
    Returns the m3u8 playlist file with proper CORS and caching headers.
    """
    from fastapi.responses import Response
    
    playlist_path = os.path.join(HLS_BASE_DIR, session_id, "stream.m3u8")
    
    if not os.path.exists(playlist_path):
        return JSONResponse(
            {"error": "Playlist not found", "session_id": session_id},
            status_code=404
        )
    
    with open(playlist_path, 'r') as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.get("/hls/{session_id}/{segment_name}")
async def hls_segment(session_id: str, segment_name: str):
    """
    Serve an HLS .ts segment for a session.
    
    Returns the MPEG-TS segment file with proper headers.
    """
    from fastapi.responses import Response
    
    # Validate segment name to prevent path traversal
    if ".." in segment_name or "/" in segment_name:
        return JSONResponse({"error": "Invalid segment name"}, status_code=400)
    
    if not segment_name.endswith('.ts'):
        return JSONResponse({"error": "Invalid segment type"}, status_code=400)
    
    segment_path = os.path.join(HLS_BASE_DIR, session_id, segment_name)
    
    if not os.path.exists(segment_path):
        return JSONResponse(
            {"error": "Segment not found", "segment": segment_name},
            status_code=404
        )
    
    with open(segment_path, 'rb') as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="video/mp2t",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/generate/stream")
async def generate_stream(
    text: str = Form(...),
    size: str = Form("256"),
    avatar: str = Form("sunny"),
    tts_preference: str = Form("coqui"),
    tts_voice_id: str = Form(None),
    user_id: Optional[str] = Form(None),
    source_img: str = Form(None),
    source_aud: str = Form(None),
    audio: UploadFile = File(None),
    watermark: bool = Form(False),
    watermark_position: str = Form("bottom-right")
):
    """
    Stream fMP4 chunks for real-time playback via MediaSource Extensions.
    
    Returns a chunked streaming response:
    1. First chunk: Initialization segment (ftyp + moov boxes)
    2. Subsequent chunks: Media segments (moof + mdat pairs)
    
    Client should use MSE to append chunks to a SourceBuffer.
    MIME type: video/mp4; codecs="avc1.42E01E, mp4a.40.2"
    """
    import time
    start_time = time.time()
    print(f"[ENDPOINT] Stream request started at {start_time:.3f}")
    
    # Resolve image path
    img_path = f"/app/user_img/{avatar}.jpg"
    print(f"[ENDPOINT] Using image: {img_path} at {time.time() - start_time:.3f}s")

    request_id = str(uuid.uuid4())[:8]
    audio_path = f"/app/aud/{user_id}_{request_id}_audio.wav"
    
    # Use provided audio or generate TTS
    if audio:
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        print(f"[ENDPOINT] Using uploaded audio: {audio_path} at {time.time() - start_time:.3f}s")
    else:
        print(f"[ENDPOINT] Generating TTS at {time.time() - start_time:.3f}s")
        audio_path = await generate_tts_async(
            text, tts_preference, tts_voice_id,
            user_id=user_id, voice_source=source_aud
        )
        print(f"[ENDPOINT] TTS complete: {audio_path} at {time.time() - start_time:.3f}s")
        print(f"[ENDPOINT] Text: '{text}' -> Audio: {audio_path}")
    
    async def stream_chunks():
        """Async generator yielding fMP4 segments."""
        from gpu_concurrency import GPUConcurrencyManager
        gpu_mgr = GPUConcurrencyManager()
        gpu_acquired = False
        try:
            print(f"[ENDPOINT] Starting stream_chunks at {time.time() - start_time:.3f}s")
            
            from stream_pipeline_streaming import StreamingSDK
            import os
            
            print(f"[ENDPOINT] Imported StreamingSDK at {time.time() - start_time:.3f}s")
            
            # Acquire GPU slot before creating SDK
            acquired = await gpu_mgr.acquire()
            if not acquired:
                raise RuntimeError("Server is at capacity. Please try again shortly.")
            gpu_acquired = True
            
            # Get SDK configuration
            cfg_pkl = "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
            data_root = "/app/checkpoints/ditto_trt_Ampere_Plus"

            # Create streaming SDK instance
            print(f"[ENDPOINT] Creating StreamingSDK at {time.time() - start_time:.3f}s")
            streaming_sdk = StreamingSDK(cfg_pkl, data_root)
            
            print(f"[ENDPOINT] Setting up streaming at {time.time() - start_time:.3f}s")
            streaming_sdk.setup_streaming(
                source_path=img_path,
                width=int(size),
                height=int(size),
                watermark=watermark,
                watermark_position=watermark_position,
            )
            
            print(f"[ENDPOINT] Starting chunk generation at {time.time() - start_time:.3f}s")
            # Yield chunks as they're generated, with pre-buffering
            segment_count = 0
            media_buffer = []
            prebuffer_size = 3
            is_buffering = True

            async for chunk in streaming_sdk.generate_chunks(audio_path):
                segment_count += 1
                if segment_count == 1:
                    yield chunk
                    continue
                
                if is_buffering:
                    media_buffer.append(chunk)
                    if len(media_buffer) >= prebuffer_size:
                        is_buffering = False
                        for buf_seg in media_buffer:
                            yield buf_seg
                        media_buffer.clear()
                else:
                    yield chunk

            for buf_seg in media_buffer:
                yield buf_seg
                
        except Exception as e:
            print(f"[ENDPOINT] Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Release GPU slot
            if gpu_acquired:
                gpu_mgr.release()
            # Cleanup audio file
            try:
                os.remove(audio_path)
            except:
                pass
    
    print(f"[ENDPOINT] Returning StreamingResponse at {time.time() - start_time:.3f}s")
    return StreamingResponse(
        stream_chunks(),
        media_type='video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Transfer-Encoding": "chunked",
        }
    )