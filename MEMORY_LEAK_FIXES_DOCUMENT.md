# VideoV8 Memory Leak Investigation & Solutions

**Date**: 2026-06-19  
**Problem**: Multithreaded video processing causing 4GB+ memory spikes  
**Video Specs**: 4096×2160 @ 24fps = ~2356 frames, ~35MB per frame (RGBA ImageData)

---

## 1. Initial Problem Analysis

### Symptoms
- Memory usage: 100MB → 4GB+ during multithreaded export
- Single-threaded mode worked fine (~100MB stable)
- FaceMesh AI effects rendered correctly in single-threaded only

### Root Cause Identification
Frame storage pattern:
```javascript
// Original code - MEMORY KILLER
processFrames[] = [
  {imageData: Uint8ClampedArray(35MB)},  // Frame 0
  {imageData: Uint8ClampedArray(35MB)},  // Frame 1
  ...
  {imageData: Uint8ClampedArray(35MB)}   // Frame 2356
]
// Total: 2356 frames × 35MB = 82GB potential!
```

---

## 2. Attempted Solutions (Chronological Order)

### **Attempt 1: Remove Unnecessary Frame Data**
**Status**: ❌ Partial success  
**Approach**: Remove imageData field from processFrames array, store only metadata  
**Result**: Still had memory spike (canvas objects still referenced)  
**Lesson**: Storing any frame data accumulates memory

---

### **Attempt 2: Streaming Architecture (Single-threaded)**
**Status**: ✅ Success for single-threaded  
**Approach**: Process frame-by-frame without storage
```javascript
for frame in video:
  decode() → process() → encode() → release()  // Immediate cleanup
```
**Result**: Memory stayed stable at ~100MB  
**Memory Pattern**: Only 1 frame in memory at a time  
**Lesson**: Frame-by-frame streaming eliminates accumulation

---

### **Attempt 3: Multithreading with ImageData Transfers**
**Status**: ❌ Failed - introduced new spike  
**Approach**: Divide video into segments, process each with Web Workers
```javascript
// Attempted flow
Main: Decode frames → Extract ImageData → Queue in frameQueue[]
Workers: Fetch from queue → Process → Return ImageData
Main: Store in pendingFrames Map → Wait for all → Encode
```
**Memory Spike**: 
- ImageData extraction creates copies (duplicates memory temporarily)
- Queuing caused unbounded buffer (decode outpaced processing)
- Results accumulated in pendingFrames Map

**Issues**:
- frameQueue could grow to 2356 frames (82GB)
- pendingFrames Map held all results until ready to encode
- Workers couldn't access React state (FaceMesh model, UI updates)

**Lesson**: Queuing + accumulation = exponential memory growth

---

### **Attempt 4: Promise.all() with Parallel Workers**
**Status**: ❌ Failed - worst memory pattern  
**Approach**: Launch all 3 workers in parallel, wait for all to complete
```javascript
Promise.all([
  worker0.process(frames 0-785),
  worker1.process(frames 786-1571),
  worker2.process(frames 1572-2356)
])
```
**Memory Pattern**:
- All 3 workers' results held simultaneously
- allProcessedFrames array accumulated all results before encoding
- 3 × 785 frames = 2355 frames × 35MB = 82GB in memory

**Why it failed**:
```javascript
const allProcessedFrames = [];
for (let workerIndex = 0; workerIndex < 3; workerIndex++) {
  const results = await processSegmentInWorker(...);
  allProcessedFrames.push(...results);  // ACCUMULATION!
}
// At this point: allProcessedFrames has 2356 frames = 82GB
```

**Lesson**: Promise.all() = waiting for everything = memory accumulation

---

### **Attempt 5: Sequential Workers + Batch Processing**
**Status**: ❌ Still spiking  
**Approach**: Process workers sequentially, but accumulate each worker's results
```javascript
for worker in [0, 1, 2]:
  results = await processSegmentInWorker(worker)
  allProcessedFrames.push(...results)  // STILL ACCUMULATING!
```
**Memory Pattern**:
- Worker 0 finishes: ~785 × 35MB = 27.5GB in memory
- Worker 1 finishes: ~1570 × 35MB = 55GB in memory
- Worker 2 finishes: ~2355 × 35MB = 82GB in memory
- Then encoding starts

**Why it still failed**:
- Results accumulated before encoding began
- Encoding loop took time, holding full results in memory
- No interleaving of decode/process/encode

**Lesson**: Can't accumulate results at all, period

---

### **Attempt 6: Canvas Warning & Timer Issues**
**Status**: ⚠️ Addressed, not root cause  
**Issues Fixed**:
1. `Canvas2D: Multiple readback operations...` warning
   - Solution: Added `willReadFrequently: true` to all getContext() calls
2. Timer stuck at 0s
   - Cause: Captured `elapsedTime` state in closure
   - Solution: Calculate fresh in onProgress callback

**Result**: Fixed UI issues, but memory spike remained

---

## 3. Final Solution: Immediate Frame Encoding

### **Attempt 7: Encode as Frames Complete (Current)**
**Status**: ✅ WORKING  
**Approach**: Don't accumulate results, encode immediately

#### Architecture
```
Worker 0 (Frames 0-785):
├─ Batch 1 (2 frames)
│  ├─ Decode → Process → [Store in map]
│  ├─ Encode frames 0-1 → Release ✓
│  └─ Memory freed immediately
├─ Batch 2 (2 frames)
│  ├─ Decode → Process → [Store in map]
│  ├─ Encode frames 2-3 → Release ✓
│  └─ Memory freed immediately
└─ ... (repeat)

Worker 1 (Frames 786-1571): [Same pattern]
Worker 2 (Frames 1572-2356): [Same pattern]
```

#### Key Implementation Details

1. **Frame Ordering with Map**
```javascript
let nextFrameToEncode = 0;
const encodedFrameMap = new Map();  // frameIndex → imageData

// As each batch completes:
encodedFrameMap.set(frameIndex, imageData);
await encodeBufferedFrames();  // Encode immediately
```

2. **Encode Only Available Frames**
```javascript
const encodeBufferedFrames = async () => {
  while (encodedFrameMap.has(nextFrameToEncode)) {
    imageData = encodedFrameMap.get(nextFrameToEncode);
    encodedFrameMap.delete(nextFrameToEncode);  // Release!
    
    // Encode to video
    await videoSource.add(timestamp, frameDuration);
    nextFrameToEncode++;
  }
}
```

3. **Memory Pattern**
- At any time: Only 1-4 frames in memory (current batch + buffered awaiting encoding)
- After each batch: Frames encoded and deleted from map
- Total max memory: ~4 frames × 35MB = 140MB (not 82GB!)

#### Configuration
```javascript
BATCH_SIZE = 2;  // Ultra-conservative: 2 frames = 70MB per batch
Workers = 3;      // Sequential processing, encode immediately
```

#### Flow Diagram
```
Main Thread:
├─ Decode batch (2 frames) → 70MB
├─ Send to worker (transfer buffers, main copy cleared)
├─ Wait for worker result
├─ Receive batch results → Store in map
├─ Encode buffered frames in order → Release
└─ Repeat

Result: Max memory at any point = 70MB (current batch) + encoded frame
```

---

## 4. Memory Comparisons

| Approach | Max Memory | Status | Issues |
|----------|-----------|--------|--------|
| Original (accumulated) | 82GB | ❌ Failed | Stored all frames |
| Single-threaded stream | 100MB | ✅ Works | Slow (1 core) |
| Promise.all() + accum | 82GB | ❌ Failed | All workers + all results |
| Sequential + accum | 82GB | ❌ Failed | Wait for all before encode |
| **Immediate encode** | **~150MB** | **✅ Works** | **Encodes as available** |

---

## 5. Key Learnings

### What Works
✅ **Frame-by-frame streaming**: Never hold multiple frames  
✅ **Immediate encoding**: Encode as frames complete, don't accumulate  
✅ **Transferable buffers**: Zero-copy transfer to workers  
✅ **Sequential worker processing**: Easier to manage memory order  
✅ **Small batch size**: 2 frames = 70MB per batch (manageable)  
✅ **willReadFrequently: true**: Optimizes canvas getImageData()

### What Doesn't Work
❌ **Array accumulation**: `results.push()` into one unbounded array = memory leak  
❌ **Unbounded look-ahead**: decoding/processing far ahead of the encoder = holding everything  
❌ **Deferred encoding**: encode only after *all* processing = holding all results

> **Note (superseded):** an earlier version claimed "worker-based rendering
> doesn't work because workers can't access React state / FaceMesh." That was
> true for the old `FaceMesh` (which required `window`/DOM). The current build
> uses `@mediapipe/tasks-vision` `FaceLandmarker`, which runs entirely in a
> worker via WASM, so **rendering now happens in the workers** (see Section 6).
> `Promise.all()` is likewise fine now — it awaits a bounded set of worker
> *pumps*, not all frame results at once.

### Memory Management Principles
1. **One direction flow**: Decode → Process → Encode → Release
2. **No intermediate storage**: Don't cache, stream immediately
3. **Encode as available**: Don't wait for all processing to finish
4. **Explicit cleanup**: `sample.close()`, `clearRect()`, `delete`
5. **Order via map not array**: Use Map with sequential lookup

---

## 6. Current Implementation

> **Updated 2026-08-25.** The core memory principle (encode-as-available, no
> unbounded accumulation) still holds, but the pipeline has evolved
> significantly. The historical attempts above remain for context; this section
> reflects the code as it stands now. See `src/lib/MULTITHREADED_ARCHITECTURE.md`
> for the full architecture write-up.

### Files

- **`src/lib/videoV8_multithreaded.js`** — orchestrator (main thread).
- **`src/lib/faceWorker.js`** — real ES-module Web Worker (replaced the old
  inline `WORKER_CODE` string). Runs `@mediapipe/tasks-vision` `FaceLandmarker`
  and composites the effect.

### Method: `processAndEncodeFramesWithWorkers()`

- Spins up a pool of `WORKER_COUNT = min(navigator.hardwareConcurrency, 8)`
  workers (no longer hard-coded to 3).
- Workers run **in parallel** via a **work-stealing pump**: the decoded chunk is
  split into batches and every worker keeps pulling the next batch off a shared
  queue until the chunk is drained (`Promise.all(workerPumps)`). This is *not*
  the old "wait for all results then encode" — encoding is triggered as frames
  become available.

### Two pipelines (selected by `renderStride`)

- **`renderStride === 1` (default): draw-in-worker.** Each worker receives the
  full-res frame as a transferred `ImageBitmap`, downscales it to 512px for
  inference, runs `FaceLandmarker`, composites sunglasses + overlay on a full-res
  `OffscreenCanvas`, and transfers a **finished `ImageBitmap`** back. The main
  thread only decodes, blits the finished frame, and encodes.
- **`renderStride > 1`: legacy main-thread compositing.** Workers return
  landmark coordinates only; the main thread interpolates between keyframes and
  draws. Kept as a quality/speed fallback.

### Decode: sequential iterator (major change)

Frames are now pulled from MediaBunny's `sink.samples(startTime, endTime)`
iterator, which decodes **each packet at most once**. The previous
`sink.getSample(timestamp)`-per-frame approach re-seeked to the nearest keyframe
and re-decoded whole GOPs every call — catastrophic on 4K inter-frame video.

### Memory model: budgeted chunks + encode back-pressure

The "1–4 frames in memory" model has been replaced by a deliberate,
RAM-budgeted buffering scheme (the goal is to keep workers saturated):

1. **Chunk sizing** — `CHUNK_FRAME_COUNT` is derived from
   `APPROX_DEVICE_MEMORY_GB × MEMORY_BUDGET_FRACTION` (minus worker runtime and a
   safety reserve), capped at 320 frames.
2. **Decode / process overlap** — chunk N+1 is decoded on the main thread while
   the workers process chunk N.
3. **Encode back-pressure** — `MAX_PENDING_ENCODE` bounds how many
   finished-but-not-yet-encoded frames may sit in `encodedFrameMap`. If the
   encoder falls behind, decoding pauses. This is what keeps peak memory bounded
   now (roughly one chunk of frames in flight, by design).

### Ordering & cleanup (unchanged in spirit)

- `encodedFrameMap` + `nextFrameToEncode` still guarantee in-order encoding,
  draining only contiguous available frames.
- Explicit cleanup remains essential: `sample.close()`, `bitmap.close()`,
  `clearRect()`, and `Map.delete()` after encode.
- Frame handoffs use **transferable `ImageBitmap`s** (zero-copy), not copied
  `ImageData`/`ArrayBuffer`s.

### UI Integration: `src/components/home.js`
- Timer updates live; elapsed seconds computed fresh in `onProgress` (avoids the
  stale-closure bug from Attempt 6).
- Progress is a two-phase bar: decode fills 0→50%, encode fills 50→100%.

---

## 7. Remaining Optimization Opportunities

### Already implemented (were "future" in the original doc)
- ✅ **Parallel workers, encode-as-available** — work-stealing pump + `encodeChain`.
- ✅ **Progressive/overlapped encoding** — decode(N+1) overlaps process(N).
- ✅ **Memory-aware buffering** — chunk size derived from device RAM budget.
- ✅ **Single-decode pipeline** — sequential `samples()` iterator.

### Still open
1. **Encode throughput** — encode is now the bottleneck (main-thread, serial,
   one 4K frame at a time). Options: decoder/encoder hardware-accel hints, or
   segment-based parallel encoding muxed together.
2. **GPU compositing** — WebGL/WebGPU effects instead of 2D canvas for heavy
   future effects.
3. **`SharedArrayBuffer`** — direct memory sharing (needs COOP/COEP headers).

### Current trade-offs
- **`renderStride = 1`** (every frame, highest quality, draw-in-worker) vs.
  **`renderStride > 1`** (skip + interpolate, faster, main-thread compositing).
- **Larger chunks** (better worker saturation) vs. **memory headroom** — tuned
  via `MEMORY_BUDGET_FRACTION`.

---

## 8. Validation Checklist

- [x] Peak memory bounded by the RAM budget (chunk + back-pressure), not by total frame count
- [x] No unbounded accumulation of decoded or finished frames
- [x] Decoder and worker pool stay busy simultaneously (decode/process overlap)
- [x] Each source packet decoded at most once (`samples()` iterator)
- [x] Output frames encoded strictly in order
- [x] Timer updates correctly (no stale closure)
- [x] Progress reflects both decode and encode phases
- [x] `ImageBitmap`s and samples explicitly closed; workers terminate cleanly

---

## 9. Document Summary

### TL;DR
The original memory leak was caused by **accumulating frame data** before
encoding. The fix was **encode-as-available** ordering via `encodedFrameMap` +
`nextFrameToEncode`, so results are never held all at once. The current build
keeps that principle but replaces the ultra-conservative "2–4 frames in memory"
model with a **RAM-budgeted chunk pipeline plus encode back-pressure**
(`MAX_PENDING_ENCODE`): it deliberately buffers a bounded number of frames to
keep the parallel workers saturated, while peak memory stays capped by the
device-memory budget rather than by total frame count. (See Section 6.)

### Key Code Pattern
```javascript
// Instead of this (WRONG):
allFrames.push(...processedFrames);  // Accumulate
for frame of allFrames:
  encode(frame);

// Do this (CORRECT):
encodedFrameMap.set(frameIndex, imageData);
encodeBufferedFrames();  // Encode what's available
encodedFrameMap.delete(frameIndex);  // Release immediately
```
