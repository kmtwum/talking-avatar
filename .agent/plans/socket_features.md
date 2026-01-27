# Socket Streaming - Feature Enhancements

## Recommended Features (Priority Order)

---

## 🔴 High Priority

### 1. **Chunk Buffering & Sentence Aggregation** ✅ IMPLEMENTED
**Problem:** Small text chunks (e.g., word-by-word streaming from LLM) generate many tiny TTS calls, increasing latency and overhead.

**Solution:** Buffer incoming chunks and aggregate them at sentence boundaries before TTS.

```python
# SessionConfig options
{
    "aggregate_chunks": True,      # Enable/disable
    "aggregate_min_chars": 50,     # Buffer at least 50 chars
    "aggregate_max_chars": 500,    # Force flush at 500 chars
    "aggregate_timeout": 1.5       # Flush after 1.5s silence
}
```

**Implementation:**
- `ChunkAggregator` class in `tts_streamer.py`
- Detects sentence endings (., !, ?, etc.) across multiple languages
- Handles continuation patterns (..., —, etc.)
- Timeout-based flush for partial sentences
- Full statistics tracking

**Testing:**
```bash
# Word-by-word streaming (demonstrates aggregation)
python test_socket_client.py --word-stream "Hello! This is a test. How are you?"

# Compare with aggregation disabled
python test_socket_client.py --word-stream "Hello there!" --no-aggregate
```

### 2. **Audio Pre-buffering Before Video Starts**
**Problem:** Starting video immediately on first audio chunk means video may stutter if subsequent TTS is slow.

**Solution:** Wait for N audio chunks (or N seconds of audio) before starting video generation.

```python
class SocketSession:
    min_audio_buffer_chunks = 2  # Wait for 2 chunks before starting video
    min_audio_buffer_seconds = 1.5  # Or 1.5 seconds of audio
```

**Benefits:**
- Smoother video playback
- Graceful handling of TTS latency spikes

---

### 3. **Connection Heartbeat/Keep-Alive**
**Problem:** Long pauses between chunks may cause connection timeouts or appear as disconnects.

**Solution:** Implement heartbeat messages both directions.

```python
# Server sends HEARTBEAT every 10 seconds
{"type": "HEARTBEAT", "timestamp": 1706364647}

# Client should respond with HEARTBEAT_ACK
{"type": "HEARTBEAT_ACK"}
```

**Benefits:**
- Reliable connection state detection
- Works with proxies/load balancers
- Early detection of dead connections

---

### 4. **Graceful Error Recovery**
**Problem:** If TTS fails for one chunk, the entire session fails.

**Solution:** Skip failed chunks with fallback behavior.

```python
async def _generate_tts(self, text: str) -> str:
    for attempt in range(3):
        try:
            return await self._generate_coqui(text)
        except Exception as e:
            if attempt == 2:
                # Generate silence placeholder instead of failing
                return self._generate_silence(duration_ms=500)
            await asyncio.sleep(0.5 * (attempt + 1))
```

**Benefits:**
- More resilient streaming
- Partial output is better than no output

---

## 🟡 Medium Priority

### 5. **Voice Embedding Cache**
**Problem:** Voice cloning recomputes speaker embeddings for each TTS call.

**Solution:** Cache embeddings at session start.

```python
class TTSStreamer:
    async def start(self):
        # Pre-compute speaker embedding from voice source
        if self.session.config.voice_source:
            self._cached_embedding = await self._compute_embedding(
                self.session.config.voice_source
            )
```

**Benefits:**
- ~200-500ms faster per TTS call
- More consistent voice across chunks

---

### 6. **Progressive Quality Upgrade**
**Problem:** Users want fast first response, but high quality final output.

**Solution:** Start with fast low-quality settings, upgrade mid-stream.

```python
# First chunks: Fast settings
{
    "sampling_timesteps": 5,
    "size": 192
}

# After buffer fills: Quality settings  
{
    "sampling_timesteps": 10,
    "size": 256
}
```

**Benefits:**
- Sub-second first frame latency
- High quality sustained playback

---

### 7. **Bandwidth Adaptation**
**Problem:** Clients on slow networks may buffer if video bitrate is too high.

**Solution:** Monitor client ACK timing, reduce quality if lagging.

```python
# Client sends playback status
{"type": "PLAYBACK_STATUS", "buffered_ms": 250, "played_ms": 5000}

# Server adapts quality
if buffered_ms < 500:
    reduce_bitrate()
```

**Benefits:**
- Adaptive streaming like YouTube/Netflix
- Better experience on poor networks

---

### 8. **Multi-Language Support**
**Problem:** Text chunks may be in different languages.

**Solution:** Auto-detect language per chunk.

```python
from langdetect import detect

async def _generate_tts(self, text: str) -> str:
    language = detect(text)  # Returns 'en', 'es', 'fr', etc.
    return await self._generate_coqui(text, language=language)
```

**Benefits:**
- Seamless multilingual conversations
- Correct pronunciation per language

---

## 🟢 Nice to Have

### 9. **Interruption Handling**
**Problem:** User may want to interrupt current speech (e.g., "Stop!").

**Solution:** Add INTERRUPT message type.

```python
# Client sends
{"type": "INTERRUPT"}

# Server: 
# 1. Stops current TTS/video generation
# 2. Clears queues
# 3. Sends INTERRUPTED acknowledgment
```

**Benefits:**
- More conversational feel
- Faster response to user corrections

---

### 10. **Chunk Priority/Urgency**
**Problem:** Some text (like alerts) should be processed faster.

**Solution:** Add priority field to SPEECH_CHUNK.

```python
{
    "type": "SPEECH_CHUNK",
    "seq": 5,
    "text": "Warning! Please evacuate.",
    "priority": "high"  # Skip queue, process immediately
}
```

---

### 11. **Audio-Only Mode**
**Problem:** Some clients only need audio, not video.

**Solution:** Add mode to SESSION_START.

```python
{
    "type": "SESSION_START",
    "mode": "audio_only",  # or "video", "both"
    ...
}
```

**Benefits:**
- Much faster for audio-only use cases
- Lower server GPU usage

---

### 12. **Session Resume**
**Problem:** If connection drops, all progress is lost.

**Solution:** Allow reconnection with session ID.

```python
{
    "type": "SESSION_RESUME",
    "session_id": "abc123",
    "last_received_seq": 42
}
```

**Benefits:**
- Resilient to network blips
- Better mobile experience

---

### 13. **Metrics & Observability**
**Problem:** Hard to debug latency issues in production.

**Solution:** Emit detailed timing metrics.

```python
# Include in SESSION_COMPLETE
{
    "type": "SESSION_COMPLETE",
    "metrics": {
        "tts_total_ms": 2340,
        "tts_per_chunk_ms": [450, 520, 680, 690],
        "video_total_ms": 3200,
        "first_frame_latency_ms": 890,
        "total_frames": 125
    }
}
```

---

## Implementation Roadmap

| Phase | Features | Effort |
|-------|----------|--------|
| **Phase 1** | Chunk Aggregation, Audio Pre-buffer | 1-2 days |
| **Phase 2** | Heartbeat, Error Recovery | 1 day |
| **Phase 3** | Voice Cache, Progressive Quality | 2-3 days |
| **Phase 4** | Interruption, Bandwidth Adapt | 2-3 days |
| **Phase 5** | Metrics, Session Resume | 2 days |

---

## Quick Wins (< 1 hour each)

1. ✅ Add `websockets` to environment.yaml (DONE)
2. [ ] Add configurable timeouts to session config
3. [ ] Add `chunk_received_at` timestamps for latency tracking
4. [ ] Add connection close reason codes
5. [ ] Add request ID to all log messages for tracing
