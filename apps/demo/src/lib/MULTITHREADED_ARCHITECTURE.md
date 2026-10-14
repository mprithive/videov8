# VideoV8 Multithreaded Architecture

## Overview

`VideoV8Multithreaded` (`src/lib/videoV8_multithreaded.js`) processes a video
entirely in the browser: it decodes each frame, runs one selected AI effect,
composites the result, and re-encodes to MP4.

The design is shaped by one hard constraint:

- **MediaBunny decode/encode must run on the main thread** (its WASM/WebCodecs
  pipeline is not used from workers here).
- **MediaPipe tasks (`FaceLandmarker` or `ImageSegmenter`) run inside Web
  Workers** (pure WASM, no DOM), and that — plus the per-frame drawing — is the
  parallelizable work.

So the pipeline is **decode on main → infer + draw in parallel workers → encode
on main, in order**. The main thread owns ordering and I/O; workers are the
compute pool.

### Files

| File | Role |
|------|------|
| `videoV8_multithreaded.js` | Orchestrator (main thread): decode, dispatch, encode, memory/pipeline events |
| `faceWorker.js` | Web Worker: loads **one** MediaPipe task + composites the chosen effect |
| `segmentationEffect.js` | Shared pattern-fill compositor + pattern image loader (worker + main thread) |
| `videoV8.js` | Single-thread streaming path (one frame at a time; no workers) |
| `MULTITHREADED_ARCHITECTURE.md` | This document |

Configuration lives in `src/components/home.js`:

```javascript
const WORKER_COUNT = Math.min(navigator.hardwareConcurrency || 4, 8);
const TARGET_AI_INFERENCE_FRAMES = 160;   // budget of "real" AI inferences
const APPROX_DEVICE_MEMORY_GB = navigator.deviceMemory || 8;
const MEMORY_BUDGET_FRACTION = 0.4;        // fraction of RAM used for buffering
const RENDER_FRAME_STRIDE = 1;             // 1 = render every frame (see below)
```

`hardwareConcurrency` is the logical CPU count (capped at 8; fallback 4 if
missing). `deviceMemory` is a **bucketed hint** (0.25–8 GB, Chrome/Edge only;
fallback 8). Neither is exact hardware — the chunk cap (≤320) exists because
these APIs are approximate.

### AI effects (UI dropdown)

The home page **AI effect** dropdown selects which model each worker loads.
Only one effect runs per export — FaceMesh is unchanged when the other is
selected.

| `aiEffect` | Model | What is drawn |
|------------|--------|----------------|
| `facemesh` (default) | `FaceLandmarker` | Sunglasses + overlay on landmarks |
| `segmentation` | Selfie `ImageSegmenter` | Pattern image clipped to the character mask |

Passed through `INIT` / `PROCESS_BATCH` so workers never load both models.

**Pattern fill** (`src/lib/segmentationEffect.js` + `src/Patterns/1.png`–`3.png`):

- Segmentation at 256px; compositing at full resolution.
- A cover-fitted pattern fills the masked character region.
- Image changes every **0.5s** (`PATTERN_HOLD_MS`). The pick is
  **timestamp-seeded** so every worker shows the same pattern in the same slot
  (no flicker).
- Mask polarity is left as-is to match the current segmenter output.

## Two pipelines

The orchestrator chooses a pipeline based on `renderStride`:

```javascript
const DRAW_IN_WORKER = renderStride === 1 || this.aiEffect === 'segmentation';
```

Segmentation always uses draw-in-worker (masks must be applied where inference
runs; transferring full-res masks back would be expensive).

### 1. Draw-in-worker pipeline — `renderStride === 1` (current default)

Every frame is rendered, so each output frame is **fully self-contained** (no
cross-frame blending). This lets the workers own the entire per-frame pipeline:
AI inference **and** effect compositing. The main thread is reduced to
**decode → transfer → encode**.

```
MAIN THREAD                                WORKER POOL (N workers)
───────────────────────────────────       ─────────────────────────────────────
sink.samples() iterator (sequential,       ┌─ Worker k
  each packet decoded once)                │  - receive full-res ImageBitmap
      │  full-res ImageBitmap  ───────────►│  - downscale (512px face / 256px seg)
      │  (transferred, zero-copy)          │  - FaceLandmarker **or** ImageSegmenter
      │                                     │  - sunglasses **or** pattern fill
encode in frame order (MediaBunny) ◄───────┤    on a full-res OffscreenCanvas
  drawImage(finishedBitmap) → add()        │  - transferToImageBitmap() back
      │                                     └─
MP4 Blob
```

Key property: workers return a **finished full-resolution `ImageBitmap`**; no
pixel data is drawn on the main thread except the final blit into the encoder
canvas.

### 2. Legacy main-thread compositing — `renderStride > 1`

To go faster at the cost of quality, only every Nth frame is decoded
("keyframes"); the in-between frames reuse a neighbour. This requires
**interpolation between two neighbouring keyframes** (landmark lerp + bitmap
cross-fade), which spans batch/chunk boundaries and therefore cannot be done by
a single worker in isolation. In this mode:

- Workers return **landmark coordinates only**.
- The main thread interpolates landmarks and composites every output frame.

This path is preserved as a fallback and is unchanged by the worker-drawing
work. The rest of this document focuses on the default (`renderStride === 1`)
pipeline.

## Stage 1 — Sequential decode (main thread)

Frames are pulled from MediaBunny's **sequential iterator**:

```javascript
const sampleIterator = this.sink.samples(startTime, iterEnd);
```

This is critical for performance. The iterator decodes **each packet at most
once** and pre-decodes a few frames ahead. The earlier implementation called
`sink.getSample(timestamp)` once per frame, which re-seeks to the nearest
keyframe and re-decodes the whole GOP on every call — on 4K inter-frame video
that meant decoding each frame ~15–30× and was the dominant cost.

Each decoded sample is drawn to a reused `decodeCanvas` and converted to a
single full-resolution `ImageBitmap`. The worker does its own 512px downscale
for inference, so the main thread creates **one** bitmap per frame.

## Stage 2 — Parallel inference + drawing (workers)

### Worker pool

`initializeWorkerPool()` spins up `WORKER_COUNT` module workers. Each loads
`@mediapipe/tasks-vision` and creates **either** a `FaceLandmarker` **or** an
`ImageSegmenter`, according to `aiEffect`:

- **WASM served locally** from `public/mediapipe-wasm/` (no CDN).
- **Delegate**: `CPU` when `WORKER_COUNT > 1`, `GPU` when single-worker. Multiple
  workers contending for one GPU queue serialize; CPU gives each worker its own
  core.
- Models are downloaded once and browser-cached (FaceLandmarker ~4 MB; selfie
  landscape segmenter is smaller). Init has a 90 s timeout; a failed worker is
  skipped rather than blocking the pool.
- `runningMode: 'VIDEO'`. Face mode: `numFaces: 1`. Seg mode: category mask
  only (`outputCategoryMask: true`).

### Work-stealing batch pump

A decoded **chunk** is split into small **batches** (`WORKER_BATCH_SIZE`). Every
worker runs a loop that keeps pulling the next batch off a shared queue until the
chunk is drained — a fast worker simply grabs more batches (auto load-balancing):

```javascript
const workerPumps = this.workers.map((worker) => (async () => {
  while (batches.length > 0) {
    const batch = batches.shift();
    if (!batch) return;
    const results = await sendToWorker(worker, batch);
    for (const frame of results) sourceFrameResults.set(frame.frameIndex, frame);
  }
})());
await Promise.all(workerPumps);
```

Batches are sent with the full-res `ImageBitmap`s as **transferables** (zero-copy
ownership handoff). Workers return finished `ImageBitmap`s, also transferred.

### Inside the worker (per frame)

1. Downscale the received full-res bitmap onto the detect canvas (512px for
   FaceMesh, 256px for segmentation).
2. Run inference only every `inferenceStride` frames; otherwise reuse
   `cachedLandmarks` or `cachedCategoryMask`.
3. Composite on a full-res `OffscreenCanvas`:
   - **facemesh:** original frame + `drawSunglasses()` (or "No Face").
   - **segmentation:** `applyPatternFill()` — original frame, mask cutout,
     cover-fitted pattern from `src/Patterns/` for the current 0.5s slot.
4. `transferToImageBitmap()` and hand it back to the main thread.

`WORKER_BATCH_SIZE` is kept `>= inferenceStride` so each batch usually contains
one true inference and the rest reuse cached landmarks.

## Stage 3 — In-order encode (main thread)

Workers finish out of order, but video must be encoded strictly in order. This is
handled by:

- `encodedFrameMap` — finished frames keyed by output index.
- `encodeChain` — a promise chain that serialises encode calls (never
  concurrent, never dropped).
- `scheduleEncode()` drains only while the **next expected** frame
  (`nextFrameToEncode`) is present, guaranteeing order:

```javascript
while (encodedFrameMap.has(nextFrameToEncode) && (DRAW_IN_WORKER || nextFrameToEncode < totalFrames)) {
  const frameData = encodedFrameMap.get(nextFrameToEncode);
  encodedFrameMap.delete(nextFrameToEncode);
  if (frameData.rendered) {                 // worker-drawn finished frame
    encodeCtx.drawImage(frameData.rendered, 0, 0, this.width, this.height);
    frameData.rendered.close();
    await videoSource.add(nextFrameToEncode * frameDuration, frameDuration);
    nextFrameToEncode++;
  }
  // ... (legacy compositing branch for renderStride > 1) ...
}
```

Output is re-timed to a uniform `frameDuration`, independent of the source
timestamps.

## Pipelining & memory management

### Chunked, memory-budgeted decoding

The orchestrator sizes a decode **chunk** from approximate device RAM
(`APPROX_DEVICE_MEMORY_GB × MEMORY_BUDGET_FRACTION`, minus worker runtime and a
safety reserve), so it buffers as many frames as fit while keeping workers
saturated. `CHUNK_FRAME_COUNT` is capped (≤320) to avoid browser instability.

### Decode / process overlap

Instead of alternating (decode a chunk → then process it, leaving one side idle),
the default pipeline **decodes chunk N+1 while the workers process chunk N**:

```javascript
let decodePromise = decodeNextChunk();
while (true) {
  const decodedChunk = await decodePromise;
  if (decodedChunk.decodedFrames.length === 0) break;
  decodePromise = iteratorDone ? Promise.resolve({ decodedFrames: [] }) : decodeNextChunk();
  await processChunk(decodedChunk);        // workers run while next chunk decodes
  if (encodedFrameMap.size > MAX_PENDING_ENCODE) await encodeChain; // back-pressure
}
```

### Encode back-pressure

Finished 4K frames are large (~33 MB each). `MAX_PENDING_ENCODE` bounds how many
finished-but-not-yet-encoded frames are held in memory (~one chunk). If encoding
falls behind, decoding pauses until the encoder drains, keeping peak memory
bounded.

## Data transfer summary

| Direction | Payload | Mechanism |
|-----------|---------|-----------|
| main → worker | full-res `ImageBitmap` (per frame) | transferable (zero-copy) |
| worker → main | finished full-res `ImageBitmap` | transferable (zero-copy) |
| main → worker (init) | width, height, wasmBasePath, delegate, **aiEffect** | structured clone |

No `ImageData`/`ArrayBuffer` pixel buffers are copied between threads; all frame
handoffs are `ImageBitmap` transfers.

## Progress reporting & BTS dashboard

Progress counts **two units per frame** — decode fills 0→50%, encode fills
50→100% — so the bar advances monotonically through both phases and never goes
backward:

```
progress = (totalDecoded + nextFrameToEncode) / (totalFrames * 2)
```

During export, **View BTS** on the home page hides that bar and shows a live
workflow. The orchestrator emits `onPipelineEvent` (throttled in the UI with
`requestAnimationFrame`):

| Event | When |
|-------|------|
| `pipeline-ready` | Worker pool finished init |
| `decoder-start` / `decoder-idle` | A decode chunk begins / ends |
| `worker-receive` | A worker takes a batch |
| `worker-processed` | That batch comes back |
| `encoder-start` / `encoder-frame` / `encoder-idle` | Encode drain |

- **Multithreaded:** Decoder → Worker 1…N → Encoder. A worker block **glows**
  from receive until its frames are encoded (`pending` hits 0).
- **Single-threaded:** Decoder → static **AI Inference** block → Encoder.
  No worker tiles (nothing is spawned). Decoder/encoder still glow from
  synthetic progress events. FaceMesh or pattern fill still run on the main
  thread via `videoV8.js`.

Toggle BTS off to restore the progress bar. Events keep flowing either way.

## Frame-rate reduction levers

Three independent knobs reduce AI/compute cost and compound:

1. **`renderStride`** — decode/render fewer frames (interpolate the rest).
   `1` = every frame (current default); `>1` switches to the legacy interpolation
   pipeline.
2. **`inferenceStride`** — derived from `TARGET_AI_INFERENCE_FRAMES`; run true
   inference on a subset of processed frames and reuse cached landmarks between.
3. **Low-res detection** — FaceMesh infers at 512px; segmentation at 256px.
   Landmarks/masks upscale; cost drops vs. full 4K inference.

## Error handling

- Worker init failure is non-fatal: that worker is skipped; AI-less frames pass
  through with the overlay only.
- Per-batch worker timeout (3 min) rejects the batch.
- On any fatal error the pool is terminated and the error propagated.

## Performance notes & current bottleneck

With sequential decoding and decode/process overlap in place, **encode is the
remaining floor**: MediaBunny encodes on the main thread, one frame at a time, in
order. Once decode is cheap and inference/drawing are parallel, total time
approaches encode time.

Further speedups would require attacking encode throughput itself (e.g. decoder
hardware-acceleration hints, or segment-based parallel encoding muxed together) —
not more worker parallelism.

## Browser support

Chromium-first. Current Firefox and Safari 16.4+ should work; older Safari
does not (no `OffscreenCanvas` / weak WebCodecs).

- Web Workers (module workers): Chrome 80+, Firefox 114+, Safari 15.2+.
- `OffscreenCanvas`: Chrome 69+, Firefox 105+, Safari 16.4+.
- Transferable `ImageBitmap`: all modern browsers.
- WebCodecs (MediaBunny decode/encode): Chrome/Edge strong; Firefox recent;
  Safari 16.4+.
- `navigator.deviceMemory`: Chrome/Edge only (fallback 8 GB).
- MediaPipe WASM: best on Chromium; 4K + workers can be heavier on Safari.

IE / old Android WebViews are unsupported. Single-thread is slightly more
compatible (no worker `OffscreenCanvas`) but still needs MediaBunny/WebCodecs
to encode.
