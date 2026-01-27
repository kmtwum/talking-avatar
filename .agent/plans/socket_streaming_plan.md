# WebSocket Streaming Architecture Plan

## Overview

This plan outlines a WebSocket-based approach for the talking avatar API that receives progressive speech text chunks and triggers TTS + video generation in real-time while waiting for more input.

## Input Protocol

The WebSocket endpoint will receive JSON messages in the format:

```json
{
    "type": "SPEECH_CHUNK",
    "seq": 0,
    "text": "Hello, this is the first chunk"
}
```

### Message Types

| Type | Description |
|------|-------------|
| `SPEECH_CHUNK` | Text chunk to be processed for TTS + video |
| `SESSION_START` | Initialize session with avatar/config |
| `SESSION_END` | Signal end of speech stream |

---

## Architecture Design

### High-Level Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   WebSocket     │───▶│   Chunk Queue   │───▶│   TTS Worker    │
│   Connection    │     │   (asyncio)     │     │   (async)       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   fMP4 Output   │◀───│   Video Gen     │◀───│   Audio Queue   │
│   (WebSocket)   │     │   Pipeline      │     │   (per chunk)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Core Components

#### 1. `SocketSession` Class
Manages the state for a single WebSocket connection:
- Session ID and configuration
- Chunk queue for incoming text
- Audio queue for generated TTS segments
- Video generation state
- Output streaming to client

#### 2. `ChunkProcessor` Class
Processes incoming speech chunks in order:
- Buffers and orders chunks by sequence number
- Triggers TTS generation per chunk
- Manages sentence boundary detection (optional)

#### 3. `StreamingTTSGenerator`
Generates TTS audio for each chunk:
- Async TTS generation (Coqui/ElevenLabs)
- Outputs audio segments to be muxed
- Handles voice consistency across chunks

#### 4. `IncrementalVideoGenerator`
Modified StreamingSDK for incremental audio:
- Accepts audio in progressive chunks
- Continues rendering as more audio arrives
- Outputs fMP4 segments immediately

---

## Detailed Implementation Plan

### Phase 1: WebSocket Endpoint Setup

**File: `main.py`**

Add new WebSocket endpoint:

```python
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for streaming speech-to-video.
    
    Protocol:
    1. Client sends SESSION_START with config
    2. Client sends SPEECH_CHUNK messages progressively
    3. Server streams back fMP4 video chunks
    4. Client sends SESSION_END or closes connection
    """
```

### Phase 2: Session Management

**New File: `socket_session.py`**

```python
class SocketSession:
    def __init__(self, session_id: str, config: dict):
        self.session_id = session_id
        self.avatar = config.get("avatar", "sunny")
        self.size = config.get("size", 256)
        self.tts_preference = config.get("tts_preference", "coqui")
        self.voice_source = config.get("voice_source")
        
        # Queues
        self.text_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
        
        # State
        self.is_active = True
        self.chunks_received = 0
        self.last_seq = -1
        
    async def handle_chunk(self, chunk: dict):
        """Process incoming SPEECH_CHUNK message."""
        seq = chunk["seq"]
        text = chunk["text"]
        
        # Validate sequence
        if seq <= self.last_seq:
            raise ValueError(f"Out of order chunk: {seq} <= {self.last_seq}")
        
        self.last_seq = seq
        self.chunks_received += 1
        
        await self.text_queue.put((seq, text))
```

### Phase 3: Incremental TTS Pipeline

**New File: `tts_streamer.py`**

```python
class TTSStreamer:
    """Generates TTS audio for each text chunk in parallel."""
    
    def __init__(self, session: SocketSession):
        self.session = session
        self.tts_tasks = {}
        
    async def process_text_queue(self):
        """Continuously process text chunks into audio."""
        while self.session.is_active:
            try:
                seq, text = await asyncio.wait_for(
                    self.session.text_queue.get(),
                    timeout=1.0
                )
                
                # Generate TTS for this chunk
                audio_path = await self._generate_tts(text)
                
                # Queue audio for video generation
                await self.session.audio_queue.put((seq, audio_path))
                
            except asyncio.TimeoutError:
                continue
                
    async def _generate_tts(self, text: str) -> str:
        """Generate TTS audio for a single chunk."""
        # Use existing async TTS generation
        return await generate_tts_async(
            text,
            self.session.tts_preference,
            voice_source=self.session.voice_source
        )
```

### Phase 4: Incremental Video Generation

**Modified File: `stream_pipeline_socket.py`**

Key modifications to support incremental audio input:

```python
class SocketStreamingSDK(StreamingSDK):
    """SDK variant that accepts audio chunks progressively."""
    
    def __init__(self, cfg_pkl, data_root, **kwargs):
        super().__init__(cfg_pkl, data_root, **kwargs)
        self.audio_buffer = []
        self.frame_offset = 0
        
    async def append_audio(self, audio_path: str, is_final: bool = False):
        """
        Append new audio chunk to the processing queue.
        
        This allows video generation to continue while more
        audio is being generated.
        """
        audio, sr = librosa.core.load(audio_path, sr=16000)
        self.audio_buffer.append(audio)
        
        # Process available audio immediately
        await self._process_buffered_audio(is_final)
        
    async def _process_buffered_audio(self, is_final: bool):
        """Process buffered audio and yield frames."""
        # Concatenate all buffered audio
        full_audio = np.concatenate(self.audio_buffer) if self.audio_buffer else np.array([])
        
        # Generate frames for new audio portion
        # ... (detailed implementation)
```

### Phase 5: Output Streaming

**File: `main.py`** (WebSocket output)

```python
async def stream_output(session: SocketSession, websocket: WebSocket):
    """Stream fMP4 chunks back to client via WebSocket."""
    
    # Send init segment first
    if session.init_segment:
        await websocket.send_bytes(session.init_segment)
    
    # Stream media segments as they become available
    while session.is_active:
        try:
            segment = await asyncio.wait_for(
                session.video_queue.get(),
                timeout=1.0
            )
            
            if segment is None:
                break
                
            await websocket.send_bytes(segment)
            
        except asyncio.TimeoutError:
            continue
```

---

## Message Protocol (Client ↔ Server)

### Client → Server

```typescript
// Start session
{
    "type": "SESSION_START",
    "avatar": "sunny",
    "size": 256,
    "tts_preference": "coqui",
    "voice_source": null
}

// Send text chunks
{
    "type": "SPEECH_CHUNK",
    "seq": 0,
    "text": "Hello there!"
}

// End session
{
    "type": "SESSION_END"
}
```

### Server → Client

```typescript
// Session started
{
    "type": "SESSION_STARTED",
    "session_id": "abc123"
}

// Binary fMP4 chunks (sent as binary frames)
<init_segment_bytes>
<media_segment_bytes>
...

// Status updates (optional)
{
    "type": "STATUS",
    "chunks_processed": 3,
    "frames_generated": 75
}

// Session complete
{
    "type": "SESSION_COMPLETE",
    "total_duration_ms": 5240
}

// Error
{
    "type": "ERROR",
    "message": "TTS generation failed",
    "code": "TTS_ERROR"
}
```

---

## File Structure

```
talking-avatar/
├── main.py                      # Add WebSocket endpoint
├── socket_session.py            # NEW: Session management
├── socket_handler.py            # NEW: WebSocket message handling
├── tts_streamer.py              # NEW: Incremental TTS pipeline
├── stream_pipeline_socket.py    # NEW: Incremental video SDK
└── sdk_manager.py               # Update for socket SDK
```

---

## Implementation Order

### Step 1: Core Infrastructure ✓ (this document)
- [ ] Create implementation plan

### Step 2: Session Management
- [ ] Create `socket_session.py`
- [ ] Implement `SocketSession` class
- [ ] Add session lifecycle management

### Step 3: WebSocket Endpoint
- [ ] Add `/ws/generate` endpoint to `main.py`
- [ ] Implement message parsing and routing
- [ ] Add connection state handling

### Step 4: Incremental TTS
- [ ] Create `tts_streamer.py`
- [ ] Implement async TTS generation per chunk
- [ ] Add audio queuing logic

### Step 5: Incremental Video Generation
- [ ] Create `stream_pipeline_socket.py`
- [ ] Modify SDK for progressive audio input
- [ ] Implement frame generation for partial audio

### Step 6: Output Streaming
- [ ] Implement binary WebSocket output
- [ ] Add init segment handling
- [ ] Stream media segments progressively

### Step 7: Testing & Integration
- [ ] Create test client
- [ ] End-to-end testing
- [ ] Performance optimization

---

## Key Technical Considerations

### 1. Audio Consistency
- Voice cloning requires consistent speaker reference
- Solution: Cache speaker embedding from first chunk

### 2. Chunk Boundaries
- Speech chunks may break mid-sentence
- Solution: Buffer and use sentence detection, or accept natural pauses

### 3. Latency Optimization
- TTS should run in parallel with video generation
- Pre-generate TTS for next chunk while current is rendering

### 4. Memory Management
- Audio buffers should be cleared after processing
- Limit maximum buffered audio to prevent OOM

### 5. Error Recovery
- Handle TTS failures gracefully
- Allow reconnection without losing state

---

## Performance Targets

| Metric | Target |
|--------|--------|
| First frame latency | < 1 second after first chunk |
| Chunk processing time | < 500ms per chunk |
| Video output rate | > 25 fps realtime |
| Memory overhead | < 2GB per session |

---

## Dependencies

- FastAPI WebSocket support (already imported)
- asyncio queues for concurrent processing
- Existing StreamingSDK for video generation
- Existing TTS integration (Coqui/ElevenLabs)

---

## Notes

This architecture enables:
1. **Progressive rendering**: Video starts before all text is received
2. **Low latency**: Each chunk processed immediately
3. **Scalability**: Async design allows concurrent connections
4. **Flexibility**: Works with any TTS provider
