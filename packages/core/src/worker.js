/* eslint-disable no-restricted-globals, no-new-func */
/**
 * VideoV8 worker entry.
 *
 * Loaded only via `new Worker(new URL('./worker.js', import.meta.url))`.
 * One file, two roles, chosen by the first INIT message:
 *
 *   • role: 'coordinator' — decode + encode (MediaBunny / OffscreenCanvas) and
 *     orchestration. Spawned by the main thread. Spawns the inference pool.
 *   • role: 'inference'   — AI model + per-frame compositing. Spawned by the
 *     coordinator (nested workers).
 *
 * Webpack 5 bundles this as its own chunk, including the nested workers the
 * coordinator creates from the same URL.
 */

import { createInferenceAdapters, emptyInferenceResult } from './shared.js';

/** Rebuild a function from its source string. */
function rehydrateFunction(src) {
	if (!src) return null;
	// Wrap in parens so both `(a) => {}` and `function (a) {}` parse as expressions.
	return new Function(`return (${src});`)();
}

function installWorkerHandler() {
	let role = null;
	let inferenceApi = null;

	self.onmessage = async (event) => {
		const { type, payload } = event.data || {};
		try {
			if (type === 'INIT') {
				role = payload && payload.role === 'coordinator' ? 'coordinator' : 'inference';
				if (role === 'coordinator') {
					await runCoordinatorWorker(payload);
				} else {
					inferenceApi = createInferenceWorker();
					await inferenceApi.init(payload);
				}
			} else if (role === 'inference' && inferenceApi) {
				if (type === 'PROCESS_BATCH') await inferenceApi.processBatch(payload);
				else if (type === 'TERMINATE') inferenceApi.terminate();
			}
		} catch (error) {
			const message = String((error && error.message) || error);
			self.postMessage({ type: role === 'coordinator' ? 'FATAL' : 'WORKER_ERROR', error: message });
		}
	};
}

// ── Inference role ────────────────────────────────────────────────────────────
// Loads one AI model and, for each frame, runs (or reuses cached) inference,
// composites the full-res frame + user effect, and transfers a finished
// ImageBitmap back to the coordinator for in-order encoding.
function createInferenceWorker() {
	let adapter = null;
	let model = null;
	let detectCtx = null; // small OffscreenCanvas for inference
	let drawCtx = null; // full-res OffscreenCanvas for compositing
	let userDraw = null;
	let userParams = {};
	let inferenceStride = 1;
	let cachedInference = null;

	async function init(payload) {
		const { inference, wasmBasePath, width, height, inferenceStride: stride, userEffect } = payload;
		inferenceStride = Math.max(1, stride || 1);
		drawCtx = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: false });

		// inference is optional. Without it the worker only composites the user effect.
		if (inference) {
			const vision = await import('@mediapipe/tasks-vision');
			const adapters = createInferenceAdapters(vision);
			adapter = adapters[inference.task];

			const detectWidth = adapter.detectWidth;
			const detectHeight = Math.max(1, Math.round(detectWidth * height / width));
			detectCtx = new OffscreenCanvas(detectWidth, detectHeight).getContext('2d', { willReadFrequently: true });

			try {
				// This file is a module worker, so importScripts cannot install the
				// classic wasm loader. The module build assigns globalThis.ModuleFactory.
				const resolver = await vision.FilesetResolver.forVisionTasks(wasmBasePath, true);
				model = await adapter.create(resolver, inference);
			} catch (err) {
				console.warn('[VideoV8 worker] model init failed, running without AI:', err && err.message);
				model = null;
			}
		}

		// Rebuild the user's effect (if any) and run its one-time setup.
		if (userEffect) {
			userDraw = rehydrateFunction(userEffect.drawSrc);
			userParams = userEffect.params || {};
			const setup = rehydrateFunction(userEffect.setupSrc);
			if (setup) {
				const extra = await setup(userParams);
				if (extra && typeof extra === 'object') userParams = { ...userParams, ...extra };
			}
		}

		self.postMessage({ type: 'INIT_COMPLETE' });
	}

	async function processBatch(payload) {
		const { frames, width, height } = payload;
		const processedFrames = [];
		const transferables = [];

		for (const frame of frames) {
			const { frameIndex, timestamp, fullBitmap } = frame;

			// Run (or reuse cached) inference.
			if (model && fullBitmap) {
				const shouldInfer = !cachedInference || (frameIndex % inferenceStride === 0);
				if (shouldInfer) {
					detectCtx.clearRect(0, 0, detectCtx.canvas.width, detectCtx.canvas.height);
					detectCtx.drawImage(fullBitmap, 0, 0, detectCtx.canvas.width, detectCtx.canvas.height);
					try {
						cachedInference = adapter.infer(model, detectCtx.canvas, timestamp);
					} catch (_) {
						// Keep last-known result on transient inference errors.
					}
				}
			}

			// Composite the full-res frame + user effect.
			drawCtx.clearRect(0, 0, width, height);
			if (fullBitmap) drawCtx.drawImage(fullBitmap, 0, 0, width, height);

			if (userDraw) {
				try {
					// Awaited so users can run async image manipulation on the
					// inferred frame (createImageBitmap, async filters, etc.).
					await userDraw(
						drawCtx,
						{ frameIndex, timestamp, width, height },
						cachedInference || emptyInferenceResult(),
						userParams
					);
				} catch (err) {
					self.postMessage({ type: 'BATCH_ERROR', error: `workerEffect.draw frame ${frameIndex}: ${err.message}` });
					if (fullBitmap) fullBitmap.close();
					return;
				}
			}

			if (fullBitmap) fullBitmap.close();
			const rendered = drawCtx.canvas.transferToImageBitmap();
			processedFrames.push({ frameIndex, timestamp, rendered });
			transferables.push(rendered);
		}

		self.postMessage({ type: 'PROCESS_BATCH_COMPLETE', payload: { processedFrames } }, transferables);
	}

	function terminate() {
		if (model) { try { model.close(); } catch (_) {} model = null; }
		cachedInference = null;
		self.postMessage({ type: 'TERMINATED' });
	}

	return { init, processBatch, terminate };
}

// ── Coordinator role ──────────────────────────────────────────────────────────
// Runs the entire decode → dispatch → encode pipeline inside a worker, so the
// main thread does nothing but relay progress and receive the final buffer.
async function runCoordinatorWorker(payload) {
	const {
		file, inference, inferenceStride, userEffect, workerCount,
		wasmBasePath, width, height, frameRate, duration,
		startTime = 0, endTime = duration, maxFrames = null,
		codec = 'avc', bitrate = null,
		approxDeviceMemoryGB = 8, memoryBudgetFraction = 0.4,
	} = payload;

	const postProgress = (data) => self.postMessage({ type: 'PROGRESS', data });
	const emit = (event) => self.postMessage({ type: 'PIPELINE_EVENT', event });

	const {
		Input, ALL_FORMATS, BlobSource, VideoSampleSink,
		Output, BufferTarget, Mp4OutputFormat, CanvasSource, QUALITY_HIGH,
	} = await import('mediabunny');

	// Decode source — the coordinator reads the file itself (File is cloneable).
	const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
	const track = await input.getPrimaryVideoTrack();
	if (!track) throw new Error('No video track found in the file');
	const sink = new VideoSampleSink(track);

	// Spawn the nested inference worker pool.
	const workers = await spawnInferenceWorkers({
		workerCount, inference, inferenceStride, userEffect, wasmBasePath, width, height,
	});
	emit({ type: 'pipeline-ready', workerCount: workers.length });

	// Encoder (OffscreenCanvas — we're inside a worker, no DOM).
	const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() });
	const encodeCanvas = new OffscreenCanvas(width, height);
	const encodeCtx = encodeCanvas.getContext('2d', { willReadFrequently: true });
	const videoSource = new CanvasSource(encodeCanvas, { codec, bitrate: bitrate || QUALITY_HIGH });
	output.addVideoTrack(videoSource);
	await output.start();

	const frameInterval = 1 / frameRate;
	const frameDuration = 1 / frameRate;
	let totalFrames = Math.ceil((endTime - startTime) * frameRate);
	if (maxFrames) totalFrames = Math.min(totalFrames, maxFrames);

	// ── Chunk sizing from a rough memory budget ────────────────────────────────
	const bytesPerFrame = width * height * 4;
	const budgetBytes = approxDeviceMemoryGB * 1024 * 1024 * 1024 * memoryBudgetFraction;
	const workerRuntimeBytes = workers.length * 140 * 1024 * 1024;
	const frameBudget = Math.max(128 * 1024 * 1024, budgetBytes - workerRuntimeBytes - 256 * 1024 * 1024);
	const chunkFromBudget = Math.floor(frameBudget / (bytesPerFrame * 1.35));
	const CHUNK_FRAME_COUNT = Math.max(workers.length * 24, Math.min(320, chunkFromBudget || 192));
	const WORKER_BATCH_SIZE = Math.max(16, Math.min(32, Math.ceil(CHUNK_FRAME_COUNT / (workers.length * 4))));
	const MAX_PENDING_ENCODE = Math.max(CHUNK_FRAME_COUNT, workers.length * WORKER_BATCH_SIZE * 2);

	// ── Ordered encode (serialised promise chain) ──────────────────────────────
	let nextFrameToEncode = 0;
	let totalDecoded = 0;
	const encodedFrameMap = new Map();
	let encodeChain = Promise.resolve();

	const scheduleEncode = () => {
		encodeChain = encodeChain.then(async () => {
			while (encodedFrameMap.has(nextFrameToEncode)) {
				const { rendered } = encodedFrameMap.get(nextFrameToEncode);
				encodedFrameMap.delete(nextFrameToEncode);
				encodeCtx.clearRect(0, 0, width, height);
				encodeCtx.drawImage(rendered, 0, 0, width, height);
				rendered.close();
				await videoSource.add(nextFrameToEncode * frameDuration, frameDuration);
				nextFrameToEncode++;
				postProgress({
					current: nextFrameToEncode,
					total: totalFrames,
					progress: Math.min(1, (totalDecoded + nextFrameToEncode) / (totalFrames * 2)),
				});
			}
		});
		return encodeChain;
	};

	// ── Dispatch a batch to an inference worker ────────────────────────────────
	const sendToWorker = (worker, frames) => new Promise((resolve, reject) => {
		const timeout = setTimeout(() => reject(new Error('Worker timeout after 3 minutes')), 180000);
		const handler = (event) => {
			const { type, payload: p, error } = event.data || {};
			if (type === 'PROCESS_BATCH_COMPLETE') {
				clearTimeout(timeout);
				worker.removeEventListener('message', handler);
				resolve(p.processedFrames);
			} else if (type === 'BATCH_ERROR' || type === 'WORKER_ERROR') {
				clearTimeout(timeout);
				worker.removeEventListener('message', handler);
				reject(new Error(error));
			}
		};
		worker.addEventListener('message', handler);
		const transferables = frames.map((f) => f.fullBitmap).filter(Boolean);
		worker.postMessage({ type: 'PROCESS_BATCH', payload: { frames, width, height } }, transferables);
	});

	// ── Sequential decode via the sink iterator (each packet decoded once) ──────
	const iterEnd = (endTime && endTime < duration) ? endTime : undefined;
	const iterator = sink.samples(startTime, iterEnd);
	let iteratorDone = false;
	let decodedOutputIndex = 0;

	const decodeCanvas = new OffscreenCanvas(width, height);
	const decodeCtx = decodeCanvas.getContext('2d', { willReadFrequently: true });

	const decodeNextChunk = async () => {
		const decodedFrames = [];
		emit({ type: 'decoder-start' });
		for (let n = 0; n < CHUNK_FRAME_COUNT; n++) {
			if (maxFrames && decodedOutputIndex >= maxFrames) { iteratorDone = true; break; }
			const { value: sample, done } = await iterator.next();
			if (done || !sample) { iteratorDone = true; break; }
			decodeCtx.clearRect(0, 0, width, height);
			sample.draw(decodeCtx, 0, 0);
			sample.close();
			const fi = decodedOutputIndex++;
			decodedFrames.push({
				frameIndex: fi,
				timestamp: Math.round((startTime + fi * frameInterval) * 1000),
				fullBitmap: await createImageBitmap(decodeCanvas),
			});
			totalDecoded++;
			postProgress({
				current: totalDecoded,
				total: totalFrames,
				progress: Math.min(1, (totalDecoded + nextFrameToEncode) / (totalFrames * 2)),
			});
		}
		emit({ type: 'decoder-idle', decoded: decodedOutputIndex });
		return decodedFrames;
	};

	// ── Workers pull sub-batches from the chunk until it's drained ──────────────
	const processChunk = async (decodedFrames) => {
		const batches = [];
		for (let s = 0; s < decodedFrames.length; s += WORKER_BATCH_SIZE) {
			batches.push(decodedFrames.slice(s, Math.min(s + WORKER_BATCH_SIZE, decodedFrames.length)));
		}
		const pumps = workers.map((worker, workerIndex) => (async () => {
			while (batches.length > 0) {
				const batch = batches.shift();
				if (!batch) return;
				emit({ type: 'worker-receive', workerIndex, frameCount: batch.length });
				const results = await sendToWorker(worker, batch);
				emit({ type: 'worker-processed', workerIndex, frameCount: results.length });
				for (const r of results) encodedFrameMap.set(r.frameIndex, { rendered: r.rendered });
			}
		})());
		await Promise.all(pumps);
		scheduleEncode();
	};

	try {
		// Pipeline: decode chunk N+1 while workers process chunk N.
		let decodePromise = decodeNextChunk();
		while (true) {
			const chunk = await decodePromise;
			if (chunk.length === 0) break;
			decodePromise = iteratorDone ? Promise.resolve([]) : decodeNextChunk();
			await processChunk(chunk);
			if (encodedFrameMap.size > MAX_PENDING_ENCODE) await encodeChain;
		}

		await encodeChain;
		await output.finalize();
		terminateWorkerPool(workers);

		// Transfer the encoded buffer back to the main thread (zero-copy).
		const buffer = output.target.buffer;
		self.postMessage({ type: 'DONE', buffer }, [buffer]);
	} catch (error) {
		terminateWorkerPool(workers);
		throw error;
	}
}

/** Spawn + INIT the nested inference worker pool (called from the coordinator). */
async function spawnInferenceWorkers({ workerCount, inference, inferenceStride, userEffect, wasmBasePath, width, height }) {
	const workers = [];
	const ready = [];
	for (let i = 0; i < workerCount; i++) {
		const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
		workers.push(worker);
		ready.push(new Promise((resolve) => {
			let timeout;
			const onInit = (event) => {
				if (event.data && event.data.type === 'INIT_COMPLETE') {
					clearTimeout(timeout);
					worker.removeEventListener('message', onInit);
					resolve();
				}
			};
			worker.addEventListener('message', onInit);
			worker.onerror = () => { clearTimeout(timeout); resolve(); }; // don't block the pool
			worker.postMessage({
				type: 'INIT',
				payload: { role: 'inference', inference, inferenceStride, userEffect, wasmBasePath, width, height },
			});
			// 90s allows first-run model download.
			timeout = setTimeout(resolve, 90000);
		}));
	}
	await Promise.all(ready);
	return workers;
}

/** Terminate a pool of inference workers. */
function terminateWorkerPool(workers) {
	for (const worker of workers) {
		try { worker.postMessage({ type: 'TERMINATE' }); worker.terminate(); } catch (_) { /* ignore */ }
	}
}

installWorkerHandler();
