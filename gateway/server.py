"""
Gateway Server Example

Complete FastAPI application that:
1. Accepts WebSocket connections from React clients
2. Streams LLM responses (OpenAI) to video
3. Forwards video to client in real-time

Usage:
    # Start the avatar service first (port 8000)
    uvicorn main:app --port 8000
    
    # Start this gateway (port 8001)  
    python gateway/server.py
    # or
    uvicorn gateway.server:app --port 8001

Client Protocol:
    1. Connect to ws://localhost:8001/ws/chat
    2. Send: {"type": "START", "prompt": "Hello!", "avatar": "sunny"}
    3. Receive: Binary video chunks + {"type": "COMPLETE", ...}
"""

import os
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

from gateway.llm_to_video import GatewayConfig, stream_llm_to_video


# Configuration from environment
AVATAR_SERVICE_URL = os.getenv("AVATAR_WS_URL", "ws://77.68.21.101:8002/ws/generate")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    type: str = "START"
    prompt: str
    avatar: str = "sunny"
    size: int = 512
    voice_id: Optional[str] = None
    system_prompt: Optional[str] = None
    model: str = "gpt-4.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print(f"[Gateway] Starting, avatar service: {AVATAR_SERVICE_URL}")
    yield
    print("[Gateway] Shutting down")


# Create FastAPI app
app = FastAPI(
    title="LLM-to-Video Gateway",
    description="Streams LLM responses to video via Talking Avatar service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for React client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "avatar_service": AVATAR_SERVICE_URL}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat-to-video streaming.
    
    Protocol:
        Client -> Server:
            {
                "type": "START",
                "prompt": "Hello, tell me about AI",
                "avatar": "sunny",
                "size": 256,
                "voice_id": "optional_voice_id",
                "system_prompt": "You are a helpful assistant",
                "model": "gpt-4"
            }
            
        Server -> Client:
            Binary video chunks (fMP4 format)
            {"type": "STATUS", "chunks_processed": 5}
            {"type": "COMPLETE", "total_duration_ms": 12500, ...}
            {"type": "ERROR", "message": "...", "code": "..."}
    """
    await websocket.accept()
    print("[Gateway] Client connected")
    
    try:
        # Loop to handle multiple requests on same connection
        while True:
            try:
                # Wait for START message
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=300.0  # 5 minute idle timeout
                )
                
                request = ChatRequest(**message)
                
                if request.type != "START":
                    await websocket.send_json({
                        "type": "ERROR",
                        "message": "Expected START message",
                        "code": "PROTOCOL_ERROR"
                    })
                    continue  # Wait for next message instead of closing
                
                print(f"[Gateway] Processing: avatar={request.avatar}, "
                      f"prompt='{request.prompt[:50]}...'")
                
                # Configure the pipeline
                config = GatewayConfig(
                    avatar_ws_url=AVATAR_SERVICE_URL,
                    avatar=request.avatar,
                    size=request.size,
                    voice_id=request.voice_id,
                    # Gateway-side aggregation
                    gateway_aggregate=True,
                    gateway_min_chars=40,
                    gateway_max_chars=200,
                    gateway_timeout=0.8,
                    # Avatar-side settings
                    avatar_aggregate=True,
                    prebuffer_enabled=True,
                    prebuffer_min_seconds=1.0,
                )
                
                # Stream LLM to video
                await stream_openai_to_video(
                    websocket=websocket,
                    prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    model=request.model,
                    config=config,
                )
                
                # Send ready for next request
                await websocket.send_json({
                    "type": "READY",
                    "message": "Ready for next prompt"
                })
                
            except asyncio.TimeoutError:
                # Idle timeout - close connection
                print("[Gateway] Idle timeout, closing connection")
                break
                
    except WebSocketDisconnect:
        print("[Gateway] Client disconnected")
    except Exception as e:
        print(f"[Gateway] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "ERROR",
                "message": str(e),
                "code": "INTERNAL_ERROR"
            })
        except:
            pass


async def stream_openai_to_video(
    websocket: WebSocket,
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    config: GatewayConfig,
):
    """Stream OpenAI response to video."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    async def llm_stream():
        """Async generator for OpenAI tokens."""
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=5000,  # Limit for demo
        )
        
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    
    # Stream to video
    await stream_llm_to_video(
        client_ws=websocket,
        llm_stream=llm_stream(),
        config=config,
    )


@app.websocket("/ws/text-to-video")
async def websocket_text_to_video(websocket: WebSocket):
    """
    WebSocket endpoint for direct text-to-video (no LLM).
    
    Useful for pre-generated scripts or testing.
    
    Protocol:
        Client -> Server:
            {"type": "START", "avatar": "sunny", "size": 256}
            {"type": "TEXT", "seq": 0, "text": "Hello!"}
            {"type": "TEXT", "seq": 1, "text": "How are you?"}
            {"type": "END"}
            
        Server -> Client:
            Binary video chunks
            {"type": "COMPLETE", ...}
    """
    await websocket.accept()
    print("[Gateway] Text-to-video client connected")
    
    from gateway.avatar_session import AvatarSession, AvatarConfig
    
    session = None
    
    try:
        # Wait for START
        message = await websocket.receive_json()
        
        if message.get("type") != "START":
            await websocket.send_json({
                "type": "ERROR",
                "message": "Expected START message"
            })
            return
        
        # Create avatar session
        config = AvatarConfig(
            avatar_ws_url=AVATAR_SERVICE_URL,
            avatar=message.get("avatar", "sunny"),
            size=message.get("size", 256),
            voice_id=message.get("voice_id"),
        )
        
        session = AvatarSession(
            client_ws=websocket,
            config=config
        )
        
        await session.start()
        
        # Process text messages
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            
            if msg_type == "TEXT":
                await session.send_sentence(
                    text=msg.get("text", ""),
                    seq=msg.get("seq", 0)
                )
                
            elif msg_type == "END":
                await session.end()
                break
                
            else:
                print(f"[Gateway] Unknown message type: {msg_type}")
        
    except WebSocketDisconnect:
        print("[Gateway] Client disconnected")
    except Exception as e:
        print(f"[Gateway] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session and not session.closed:
            await session.end()


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("GATEWAY_PORT", "8001"))
    
    print(f"Starting gateway server on port {port}")
    print(f"Avatar service: {AVATAR_SERVICE_URL}")
    print(f"WebSocket endpoints:")
    print(f"  - ws://localhost:{port}/ws/chat (LLM chat)")
    print(f"  - ws://localhost:{port}/ws/text-to-video (direct text)")
    
    uvicorn.run(
        "gateway.server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
