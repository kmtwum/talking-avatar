from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import asyncio
import json
import requests
import subprocess
import tempfile
import os

app = FastAPI()

from dotenv import load_dotenv

load_dotenv()


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
            # Receive message from gateway
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


def generate_tts(text: str, tts_preference: str = "coqui") -> str:
    """Generate TTS audio"""
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        tts_response = requests.post(tts_url, json={"text": text})
        tts_response.raise_for_status()
        audio_path = f"/tmp/tts_{hash(text)}.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_response.content)
        return audio_path
    else:
        from elevenlabs.client import ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("VOICE_ID")

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


@app.post("/generate")
async def quick_generate(
        text: str = Form(...),
        inference: str = Form("inference_minimal"),
        size: str = Form("512"),
        tts_preference: str = Form("coqui"),
        audio: UploadFile = File(None),
        image: UploadFile = File(None),
):
    """Optimized endpoint for fast generation"""

    # Save image
    if image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as im:
            im.write(await image.read())
            img_path = im.name
    else:
        img_path = "/app/img/avatar.png"

    # Save audio
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as au:
            au.write(await audio.read())
            audio_path = au.name
    else:
        # Generate TTS
        audio_path = generate_tts(text, tts_preference)

    # Generate video with streaming optimizations
    output_path = f"/tmp/quick_{hash(text)}.mp4"

    process = await asyncio.create_subprocess_exec(
        "python", f"/app/{inference}.py",
        "--audio_path", audio_path,
        "--source_path", img_path,
        "--output_path", output_path,
        "--size", size,
        "--steps", "15",
        "--fast",
    )
    await process.wait()

    # Stream response
    def video_stream():
        with open(output_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(video_stream(), media_type="video/mp4")


@app.get("/health")
def health():
    return {"status": "ready"}
