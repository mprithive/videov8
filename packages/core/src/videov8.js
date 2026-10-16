/**
 * VideoV8 — video AI processing library (main thread).
 * ============================================================================
 *
 * Two execution modes:
 *
 *   • Single-threaded  (workerCount <= 1) — decode + inference + effect + encode
 *                        all run on the calling thread. Simplest, lowest memory.
 *   • Multi-threaded   (workerCount  > 1) — a COORDINATOR Web Worker
 *                        (worker.js) owns decode + encode AND spawns N
 *                        nested inference workers. The main thread only sends the
 *                        File and receives progress + the final Blob.
 *
 * Worker code lives in worker.js. The model registry shared by both
 * sides lives in shared.js. Webpack 5 bundles the worker from
 * `new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })`.
 *
 * ── Public API ──────────────────────────────────────────────────────────────
 *
 *   const player = new VideoV8({ workerCount: 3 });
 *   await player.loadVideo(file);
 *
 *   const blob = await player.process({
 *     inference: {                          // optional — omit to skip AI
 *       engine: 'mediapipe',
 *       task: 'face-landmarker',          // or 'image-segmenter'
 *       modelUrl: '/models/face_landmarker.task',   // optional override
 *       delegate: 'CPU',                  // 'CPU' | 'GPU'
 *       options: { numFaces: 1 },         // forwarded to the MediaPipe task
 *     },
 *     workerEffect: {
 *       params: { tint: 'rgba(255,0,0,0.2)' },      // structured-cloneable only
 *       setup: async (params) => ({ ... }),         // optional, runs once/worker
 *       draw: (ctx, frame, inference, params) => {  // runs per output frame
 *         ctx.fillStyle = params.tint;
 *         ctx.fillRect(0, 0, frame.width, frame.height);
 *         if (inference.type === 'face') {
 *           const landmarks = inference.faceLandmarks?.[0];
 *           if (landmarks) { ... }
 *         } else if (inference.type === 'image') {
 *           const mask = inference.categoryMask;
 *         }
 *       },
 *     },
 *     onProgress: ({ current, total, progress }) => {},
 *   });
 *
 * `processAndEncodeFramesWithWorkers()` is kept as an alias of `process()`.
 *
 * ── Notes / limitations (v1) ─────────────────────────────────────────────────
 *   • `workerEffect.draw`/`setup` are serialized with Function.prototype.toString()
 *     for the worker path, so they must be self-contained: no closures, no imports,
 *     no DOM. Pass everything via `params` (build heavy resources in `setup`).
 *   • Each inference worker loads its own model (memory scales with workers).
 *   • Every output frame is rendered (no cross-frame interpolation); inference
 *     can still be throttled via `targetInferenceFrames` / `inferenceStride`.
 *   • The coordinator spawns NESTED workers. Supported in Chrome/Firefox and
 *     Safari 16.4+. On older Safari, use workerCount <= 1 (single-threaded).
 */

import { createInferenceAdapters, emptyInferenceResult, resolveInferenceConfig } from './shared.js';

/**
 * Serialise a `workerEffect` so it can be posted to a worker. Functions can't
 * be cloned, so we send their source and rebuild them inside the worker.
 */
function serializeWorkerEffect(workerEffect) {
	if (!workerEffect || typeof workerEffect.draw !== 'function') return null;
	return {
		drawSrc: workerEffect.draw.toString(),
		setupSrc: typeof workerEffect.setup === 'function' ? workerEffect.setup.toString() : null,
		params: workerEffect.params || {},
	};
}

// ─────────────────────────────────────────────────────────────────────────────
// Main-thread class
// ─────────────────────────────────────────────────────────────────────────────

class VideoV8 {
	/**
	 * @param {Object} [options]
	 * @param {number} [options.workerCount=1]  1 = single-threaded, >1 = workers.
	 * @param {number} [options.approxDeviceMemoryGB]
	 * @param {number} [options.memoryBudgetFraction=0.4]
	 * @param {string} [options.wasmBasePath]  Where MediaPipe wasm files live.
	 */
	constructor(options = {}) {
		this.input = null;
		this.videoTrack = null;
		this.sink = null;
		this.file = null;
		this.duration = 0;
		this.width = 0;
		this.height = 0;
		this.frameRate = 24;
		this.onError = null;

		this.workerCount = Math.max(1, options.workerCount || 1);
		this.approxDeviceMemoryGB = options.approxDeviceMemoryGB
			|| (typeof navigator !== 'undefined' && navigator.deviceMemory) || 8;
		this.memoryBudgetFraction = options.memoryBudgetFraction || 0.4;
		this.wasmBasePath = options.wasmBasePath || this._defaultWasmBasePath();

		this.coordinator = null; // the coordinator worker, when multithreaded
	}

	_defaultWasmBasePath() {
		if (typeof window === 'undefined') return '/mediapipe-wasm';
		const publicUrl = (typeof process !== 'undefined' && process.env && process.env.PUBLIC_URL) || '';
		return `${window.location.origin}${publicUrl}/mediapipe-wasm`;
	}

	/**
	 * Load a video file.
	 * @param {File|Blob} file
	 * @returns {Promise<boolean>}
	 */
	async loadVideo(file) {
		try {
			const { Input, ALL_FORMATS, BlobSource, VideoSampleSink } = await import('mediabunny');

			// Keep the original file so the coordinator worker can decode it itself.
			this.file = file;
			this.input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
			this.videoTrack = await this.input.getPrimaryVideoTrack();
			if (!this.videoTrack) throw new Error('No video track found in the file');

			const canDecode = await this.videoTrack.canDecode();
			if (!canDecode) throw new Error('Video codec is not supported');

			this.sink = new VideoSampleSink(this.videoTrack);
			this.duration = await this.videoTrack.computeDuration();
			this.width = await this.videoTrack.getDisplayWidth();
			this.height = await this.videoTrack.getDisplayHeight();
			const packetStats = await this.videoTrack.computePacketStats(100);
			this.frameRate = packetStats.averagePacketRate || 24;

			return true;
		} catch (error) {
			if (this.onError) this.onError(error);
			else console.error('[VideoV8] Failed to load video:', error);
			return false;
		}
	}

	/** @returns {Object} video metadata */
	getMetadata() {
		return {
			duration: this.duration,
			width: this.width,
			height: this.height,
			frameRate: this.frameRate,
		};
	}

	/**
	 * Extract a thumbnail canvas at `time` seconds.
	 * @param {number} [time=0]
	 * @returns {Promise<HTMLCanvasElement>}
	 */
	async getThumbnail(time = 0) {
		if (!this.sink) throw new Error('Video not loaded. Call loadVideo first.');
		time = Math.max(0, Math.min(time, this.duration));
		const sample = await this.sink.getSample(time);
		const canvas = document.createElement('canvas');
		canvas.width = this.width;
		canvas.height = this.height;
		sample.draw(canvas.getContext('2d'), 0, 0);
		sample.close();
		return canvas;
	}

	/**
	 * Process + encode the whole video. Runs single-threaded (main thread) or via
	 * a coordinator worker based on `workerCount` passed to the constructor.
	 * @param {Object} options  see file header for the full shape.
	 * @returns {Promise<Blob>} encoded MP4
	 */
	async process(options = {}) {
		if (!this.file || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}
		return this.workerCount > 1
			? this._processWithCoordinator(options)
			: this._processSingleThreaded(options);
	}

	/**
	 * Back-compat alias. The first argument (legacy renderFunction) is ignored;
	 * use `options.workerEffect.draw` instead.
	 */
	async processAndEncodeFramesWithWorkers(_legacyRenderFn, options = {}) {
		return this.process(options);
	}

	// ── Shared planning helpers ────────────────────────────────────────────────

	_plan(options) {
		const {
			maxFrames = null,
			startTime = 0,
			endTime = this.duration,
			targetInferenceFrames = 160,
			inferenceStride: explicitStride = null,
		} = options;

		let totalFrames = Math.ceil((endTime - startTime) * this.frameRate);
		if (maxFrames) totalFrames = Math.min(totalFrames, maxFrames);

		const inferenceStride = explicitStride
			? Math.max(1, explicitStride)
			: (targetInferenceFrames > 0 ? Math.max(1, Math.ceil(totalFrames / targetInferenceFrames)) : 1);

		return { totalFrames, startTime, endTime, inferenceStride };
	}

	async _createEncoder(options) {
		const { Output, BufferTarget, Mp4OutputFormat, CanvasSource, QUALITY_HIGH } = await import('mediabunny');
		const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() });

		const encodeCanvas = document.createElement('canvas');
		encodeCanvas.width = this.width;
		encodeCanvas.height = this.height;
		const encodeCtx = encodeCanvas.getContext('2d', { willReadFrequently: true });

		const videoSource = new CanvasSource(encodeCanvas, {
			codec: options.codec || 'avc',
			bitrate: options.bitrate || QUALITY_HIGH,
		});
		output.addVideoTrack(videoSource);
		await output.start();

		return { output, encodeCanvas, encodeCtx, videoSource };
	}

	// ── Single-threaded path (runs entirely on the calling thread) ──────────────

	async _processSingleThreaded(options) {
		const { onProgress = null, workerEffect = null } = options;
		const inference = resolveInferenceConfig(options.inference, { defaultDelegate: 'GPU' });
		const { totalFrames, startTime, endTime, inferenceStride } = this._plan(options);

		let adapter = null;
		let model = null;
		if (inference) {
			const vision = await import('@mediapipe/tasks-vision');
			const adapters = createInferenceAdapters(vision);
			adapter = adapters[inference.task];
			try {
				const resolver = await vision.FilesetResolver.forVisionTasks(this.wasmBasePath);
				model = await adapter.create(resolver, inference);
			} catch (err) {
				console.warn('[VideoV8] model init failed, running without AI:', err && err.message);
			}
		}

		// User effect runs directly (no serialization needed on the main thread).
		let userDraw = null;
		let userParams = (workerEffect && workerEffect.params) || {};
		if (workerEffect && typeof workerEffect.draw === 'function') {
			userDraw = workerEffect.draw;
			if (typeof workerEffect.setup === 'function') {
				const extra = await workerEffect.setup(userParams);
				if (extra && typeof extra === 'object') userParams = { ...userParams, ...extra };
			}
		}

		const { output, encodeCtx, videoSource } = await this._createEncoder(options);

		// Detect canvas for downscaled inference. Unused when no model is loaded.
		let detectCanvas = null;
		let detectCtx = null;
		let detectWidth = 0;
		let detectHeight = 0;
		if (adapter) {
			detectWidth = adapter.detectWidth;
			detectHeight = Math.max(1, Math.round(detectWidth * this.height / this.width));
			detectCanvas = document.createElement('canvas');
			detectCanvas.width = detectWidth;
			detectCanvas.height = detectHeight;
			detectCtx = detectCanvas.getContext('2d', { willReadFrequently: true });
		}

		const frameInterval = 1 / this.frameRate;
		const frameDuration = 1 / this.frameRate;
		let cachedInference = null;
		let frameIndex = 0;

		const iterEnd = (endTime && endTime < this.duration) ? endTime : undefined;
		const iterator = this.sink.samples(startTime, iterEnd);

		try {
			while (true) {
				if (options.maxFrames && frameIndex >= options.maxFrames) break;
				const { value: sample, done } = await iterator.next();
				if (done || !sample) break;

				encodeCtx.clearRect(0, 0, this.width, this.height);
				sample.draw(encodeCtx, 0, 0);
				sample.close();

				const timestamp = Math.round((startTime + frameIndex * frameInterval) * 1000);

				if (model) {
					const shouldInfer = !cachedInference || (frameIndex % inferenceStride === 0);
					if (shouldInfer) {
						detectCtx.clearRect(0, 0, detectWidth, detectHeight);
						detectCtx.drawImage(encodeCtx.canvas, 0, 0, detectWidth, detectHeight);
						try {
							cachedInference = adapter.infer(model, detectCanvas, timestamp);
						} catch (_) { /* keep last-known */ }
					}
				}

				if (userDraw) {
					// Awaited so an async user function is fully supported.
					await userDraw(
						encodeCtx,
						{ frameIndex, timestamp, width: this.width, height: this.height },
						cachedInference || emptyInferenceResult(),
						userParams
					);
				}

				await videoSource.add(frameIndex * frameDuration, frameDuration);
				frameIndex++;

				if (onProgress) {
					onProgress({ current: frameIndex, total: totalFrames, progress: Math.min(1, frameIndex / totalFrames) });
				}
			}

			await output.finalize();
			if (model) { try { model.close(); } catch (_) {} }
			return new Blob([output.target.buffer], { type: 'video/mp4' });
		} catch (error) {
			if (model) { try { model.close(); } catch (_) {} }
			throw error;
		}
	}

	// ── Multi-threaded path (coordinator worker owns decode + encode) ───────────

	async _processWithCoordinator(options) {
		const { onProgress = null, onPipelineEvent = null, workerEffect = null } = options;
		// Validate early on the main thread so callers get clean errors.
		const inference = resolveInferenceConfig(options.inference, {
			// Multiple workers contend for one GPU queue, so default to CPU.
			defaultDelegate: 'CPU',
		});
		const { inferenceStride } = this._plan(options);
		const userEffect = serializeWorkerEffect(workerEffect);

		return new Promise((resolve, reject) => {
			let coordinator;
			try {
				coordinator = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
			} catch (err) {
				reject(err);
				return;
			}
			this.coordinator = coordinator;

			const cleanup = () => {
				try { coordinator.terminate(); } catch (_) { /* ignore */ }
				if (this.coordinator === coordinator) this.coordinator = null;
			};

			coordinator.onmessage = (event) => {
				const data = event.data || {};
				switch (data.type) {
					case 'PROGRESS':
						if (onProgress) onProgress(data.data);
						break;
					case 'PIPELINE_EVENT':
						if (onPipelineEvent) onPipelineEvent(data.event);
						break;
					case 'DONE':
						cleanup();
						resolve(new Blob([data.buffer], { type: 'video/mp4' }));
						break;
					case 'FATAL':
						cleanup();
						reject(new Error(data.error));
						break;
					default:
						break;
				}
			};
			coordinator.onerror = (err) => {
				cleanup();
				reject(new Error(err.message || 'Coordinator worker error'));
			};

			coordinator.postMessage({
				type: 'INIT',
				payload: {
					role: 'coordinator',
					file: this.file,
					inference,
					inferenceStride,
					userEffect,
					workerCount: this.workerCount,
					wasmBasePath: this.wasmBasePath,
					width: this.width,
					height: this.height,
					frameRate: this.frameRate,
					duration: this.duration,
					startTime: options.startTime || 0,
					endTime: options.endTime || this.duration,
					maxFrames: options.maxFrames || null,
					codec: options.codec || 'avc',
					bitrate: options.bitrate || null,
					approxDeviceMemoryGB: this.approxDeviceMemoryGB,
					memoryBudgetFraction: this.memoryBudgetFraction,
				},
			});
		});
	}

	/** Terminate the coordinator worker (which terminates its inference pool). */
	terminateWorkers() {
		if (this.coordinator) {
			try { this.coordinator.terminate(); } catch (_) { /* ignore */ }
			this.coordinator = null;
		}
	}
}

export default VideoV8;
