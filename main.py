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

app = FastAPI()

from dotenv import load_dotenv

load_dotenv()


# SDK warmup on startup
@app.on_event("startup")
async def startup():
    """Pre-load SDK on startup to avoid cold start latency."""
    try:
        from sdk_manager import SDKManager
        # Configure paths (these can be overridden by environment variables)
        cfg_pkl = os.getenv("SDK_CFG_PKL", "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
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


class VideoCallManager:
    def __init__(self):
        self.active_calls = {}

    async def start_generation(self, call_id: str, audio_path: str, image_path: str):
        """Start video generation and yield chunks as they're ready"""

        output_path = f"/tmp/call_{call_id}.mp4"

        # Use streaming inference
        process = subprocess.Popen([
            "python", "/app/inference_streaming.py",
            "--audio_path", audio_path,
            "--source_path", image_path,
            "--output_path", output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for completion (in real streaming, you'd yield chunks)
        await asyncio.create_subprocess_exec(*process.args)

        # Stream the completed video
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk


manager = VideoCallManager()


@app.websocket("/video_call/{call_id}")
async def video_call_websocket(websocket: WebSocket, call_id: str):
    """WebSocket for real-time video call simulation"""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data["type"] == "generate_response":
                text = data["text"]

                # 3. Wait for TTS
                audio_path = generate_tts(text)

                # 4. Start video generation
                await websocket.send_json({"status": "generating_video"})

                # 5. Stream video chunks as ready
                async for chunk in manager.start_generation(call_id, audio_path, "/app/avatar.jpg"):
                    await websocket.send_bytes(chunk)

                await websocket.send_json({"status": "complete"})

    except Exception as e:
        await websocket.send_json({"error": str(e)})


def generate_tts(text: str, tts_preference: str = "coqui", tts_voice_id: str = None, user_id: str = None,
                 voice_source: str = None) -> str:
    """Generate TTS audio"""
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        tts_response = requests.post(tts_url, json={
            "text": text,
            "split_sentences": False,
            "source_aud": voice_source,
            "clone": user_id,
            "streaming": False,
            "model": "tts_models/multilingual/multi-dataset/xtts_v2"
        })
        tts_response.raise_for_status()
        audio_path = f"/tmp/tts_{hash(text)}.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_response.content)
        return audio_path
    else:
        from elevenlabs.client import ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("VOICE_ID")
        if tts_voice_id:
            voice_id = tts_voice_id

        elevenlabs = ElevenLabs(api_key=api_key)
        response = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
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
                "split_sentences": False,
                "source_aud": voice_source,
                "clone": user_id,
                "streaming": False,
                "model": "tts_models/multilingual/multi-dataset/xtts_v2"
            })
            response.raise_for_status()
            
            audio_path = f"/tmp/tts_{hash(text)}.wav"
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
        text: str = Form(...),
        inference: str = Form("inference_minimal"),
        size: str = Form("256"),
        tts_preference: str = Form("coqui"),
        tts_voice_id: str = Form(None),
        user_id: Optional[str] = Form(None),
        avatar: Optional[str] = Form("sunny"),
        source_img: str = Form(None),
        source_aud: str = Form(None),
        audio: UploadFile = File(None)
):
    """Optimized endpoint for fast generation"""

    img_path = f"/app/user_img/{avatar}.jpg"
    request_id = str(uuid.uuid4())[:8]

    # Use provided audio or generate TTS
    if audio:
        audio_path = f"/app/aud/{user_id}_{request_id}_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        print(f"Using uploaded audio: {audio_path}")
    else:
        print("Generating TTS...")
        audio_path = generate_tts(text, tts_preference, tts_voice_id, user_id=user_id, voice_source=source_aud)

    # Generate video with streaming optimizations
    output_path = f"/tmp/quick_{hash(text)}.mp4"

    process = await asyncio.create_subprocess_exec(
        "python", f"/app/{inference}.py",
        "--audio_path", audio_path,
        "--source_path", img_path,
        "--output_path", output_path,
        "--size", size,
        "--steps", "10",
        "--fast",
    )
    await process.wait()

    # Clean up temp audio
    if os.path.exists(audio_path):
        os.unlink(audio_path)

    # Stream response
    def video_stream():
        with open(output_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(video_stream(), media_type="video/mp4")


@app.post("/presave-photo")
async def upload_photo(user_id: str = Form(...), image: UploadFile = File(...)):
    os.makedirs("/app/user_img", exist_ok=True)
    img_path = f"/app/user_img/{user_id}.jpg"

    with open(img_path, "wb") as f:
        f.write(await image.read())

    return {"message": "Photo uploaded successfully", "user_id": user_id}


@app.get("/health")
def health():
    return {"status": "ready"}


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
    audio: UploadFile = File(None)
):
    """
    Stream fMP4 chunks for real-time playback via MediaSource Extensions.
    
    Returns a chunked streaming response:
    1. First chunk: Initialization segment (ftyp + moov boxes)
    2. Subsequent chunks: Media segments (moof + mdat pairs)
    
    Client should use MSE to append chunks to a SourceBuffer.
    MIME type: video/mp4; codecs="avc1.42E01E, mp4a.40.2"
    """
    
    # Resolve image path
    img_path = f"/app/user_img/{avatar}.jpg"
    
    # Use provided audio or generate TTS
    if audio:
        audio_path = f"/tmp/stream_uploaded_{hash(text)}.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        print(f"[STREAM] Using uploaded audio: {audio_path}")
    else:
        print("[STREAM] Generating TTS...")
        audio_path = await generate_tts_async(
            text, tts_preference, tts_voice_id,
            user_id=user_id, voice_source=source_aud
        )
        print(f"[STREAM] TTS complete: {audio_path}")
    
    async def stream_chunks():
        """Async generator yielding fMP4 segments."""
        try:
            from stream_pipeline_streaming import StreamingSDK
            import os
            
            # Get SDK configuration
            cfg_pkl = os.getenv(
                "SDK_ONLINE_CFG_PKL",
                "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
            )
            data_root = os.getenv(
                "SDK_DATA_ROOT",
                "/app/checkpoints/ditto_trt_Ampere_Plus"
            )
            
            # Create streaming SDK instance
            streaming_sdk = StreamingSDK(cfg_pkl, data_root)
            streaming_sdk.setup_streaming(
                source_path=img_path,
                width=int(size),
                height=int(size)
            )
            
            # Yield chunks as they're generated
            async for chunk in streaming_sdk.generate_chunks(audio_path):
                yield chunk
                
        except Exception as e:
            print(f"[STREAM] Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Cleanup audio file
            try:
                os.remove(audio_path)
            except:
                pass
    
    return StreamingResponse(
        stream_chunks(),
        media_type='video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Transfer-Encoding": "chunked",
        }
    )