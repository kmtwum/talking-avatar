# Progressive Video Streaming Implementation Plan

## Overview

Enable video generation to start streaming **as soon as the first TTS audio segment completes**, rather than waiting for all audio. This provides a dramatically better user experience with the avatar appearing to respond in real-time as the LLM generates text.

## Current State

```
LLM Stream → [All sentences] → [All TTS] → [Wait] → [Generate Video] → Stream
            └─────────────────────────────────────────────────────────────────┘
                                    Total Latency
```

## Target State

```
LLM Token → Sentence 1 → TTS 1 ─┬→ Video starts → Stream chunk 1
LLM Token → Sentence 2 → TTS 2 ─┤→ Extend video → Stream chunk 2
LLM Token → Sentence 3 → TTS 3 ─┴→ ...          → Stream chunk N
```

---

## Implementation Steps

### Phase 1: Socket Handler Changes
**File: `socket_handler.py`**

#### 1.1 Remove "Wait for All Audio" Block
**Lines 289-298** - Replace the `all_audio_ready` wait with `prebuffer_ready`:

```python
# BEFORE:
await asyncio.wait_for(self.session.all_audio_ready.wait(), timeout=120.0)

# AFTER:
await asyncio.wait_for(self.session.prebuffer_ready.wait(), timeout=30.0)
```

**Rationale:** The prebuffer system already exists and provides a small buffer (1-2 seconds of audio) before starting video. This ensures smooth playback without waiting for everything.

#### 1.2 Update Logging
Update logging messages to reflect the new behavior.

---

### Phase 2: Audio Bridge Enhancement
**File: `socket_handler.py`**

#### 2.1 Continue Forwarding Audio During Generation
The `_audio_bridge()` method already forwards audio segments to the video generator. Ensure it:
- Continues running even after video generation starts
- Properly signals when all audio is complete (for final segment handling)

**Current code is mostly correct** - just verify it doesn't stop prematurely.

---

### Phase 3: SocketStreamingSDK Progressive Generation
**File: `stream_pipeline_socket.py`**

#### 3.1 Dynamic Audio Extension
Modify `_run_progressive_generation()` to:
1. Start processing with initial audio
2. Check for new audio segments periodically
3. Extend the combined audio file when new segments arrive
4. Continue generating frames for the extended audio

**Key changes in `_run_progressive_generation()`:**

```python
def _run_progressive_generation(self, start_time: float):
    """Run progressive audio-to-video generation with dynamic extension."""
    
    # Track what we've processed
    last_known_segments = 0
    processed_frames = 0
    
    while not self.stop_event.is_set():
        # Check for new audio segments
        current_segments = len(self._audio_segments)
        
        if current_segments > last_known_segments:
            # New audio arrived - update combined audio
            self._create_temp_combined_audio()
            new_frame_count = self._calculate_frame_count()
            
            # Update SDK with new frame target
            if new_frame_count > self._frame_offset:
                self.setup_Nd(N_d=new_frame_count)
                
            last_known_segments = current_segments
        
        # Process available frames
        # ... (existing chunk processing logic)
        
        # If audio complete and all frames processed, exit
        if self._audio_complete.is_set() and processed_frames >= total_frames:
            break
```

#### 3.2 Thread-Safe Audio Updates
Ensure `_audio_segments` list is accessed safely:
- Use `threading.Lock()` for list modifications
- Or use `queue.Queue` for thread-safe append operations

---

### Phase 4: FMP4 Writer Timestamp Continuity
**File: `core/atomic_components/fmp4_writer.py`**

#### 4.1 Verify Segment Timestamps
When extending video with new audio, ensure:
- Media segment timestamps are continuous
- No gaps or overlaps in the fMP4 timeline
- Sequence numbers increment correctly

**The current implementation should handle this** since it uses a running frame counter, but verify:
- `_segment_sequence` increments correctly
- `_current_pts` continues from previous segment

---

### Phase 5: TTS Streamer Optimization (Optional)
**File: `tts_streamer.py`**

#### 5.1 Parallel TTS Requests
For even lower latency, consider:
- Starting next TTS request before current one completes
- Using a small pool of concurrent TTS workers
- This is a future optimization, not required for MVP

---

## Testing Plan

### Unit Tests
1. **Prebuffer triggers correctly** - Video starts after min buffer reached
2. **Audio extension works** - New segments are incorporated
3. **Timestamps are continuous** - fMP4 plays without gaps
4. **Session completes properly** - All data sent before close

### Integration Tests
1. **Short prompt** (1-2 sentences) - Video starts quickly
2. **Long prompt** (10+ sentences) - Video extends smoothly
3. **Interrupted prompt** (client disconnects) - Cleanup works

### Manual Testing
1. Connect via WebSocket tester
2. Send multi-sentence prompt
3. Verify first video chunk arrives < 5 seconds after first TTS completes
4. Verify smooth playback (no stutters or gaps)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Audio/video desync | Medium | High | Verify timestamp math, add logging |
| Playback stutter | Medium | Medium | Increase prebuffer threshold if needed |
| Memory leak (temp files) | Low | Medium | Ensure cleanup in all paths |
| Thread race conditions | Medium | High | Use proper locking, test thoroughly |

---

## Files to Modify

| File | Changes |
|------|---------|
| `socket_handler.py` | Use `prebuffer_ready`, update logging |
| `stream_pipeline_socket.py` | Dynamic audio extension, thread safety |
| `socket_session.py` | Possibly adjust prebuffer defaults |
| `fmp4_writer.py` | Verify timestamp handling (likely no changes) |

---

## Rollback Plan

If issues arise:
1. Revert `socket_handler.py` to use `all_audio_ready.wait()`
2. This returns to current "wait for all audio" behavior
3. No other changes should break existing functionality

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to first video chunk | 10-30s (varies by length) | < 5s after first TTS |
| Playback smoothness | N/A (all-or-nothing) | No visible stutter |
| Total generation time | Same | Same (no regression) |

---

## Implementation Order

1. ✅ Create feature branch: `feature/progressive-video-streaming`
2. ✅ Phase 1: Socket handler prebuffer switch (`socket_handler.py`)
3. [ ] Phase 1 Test: Verify video starts early
4. ✅ Phase 3: SDK progressive generation fixes (`stream_pipeline_socket.py`)
   - Added thread-safe locking (`_audio_lock`)
   - Added `_new_audio_available` event for efficient waiting
   - Improved `append_audio` with lock protection
   - Enhanced `_create_temp_combined_audio` with lock
   - Upgraded `_run_progressive_generation` with event-based waiting
5. ✅ Phase 5: Updated session config defaults (`socket_session.py`)
   - Increased `prebuffer_min_seconds` to 2.0s
   - Increased `prebuffer_timeout` to 15.0s
   - Added progressive streaming documentation
6. [ ] Phase 4: Verify fMP4 timestamps (likely no changes)
7. [ ] Integration testing
8. [ ] Code review and merge
