
import { createInferenceAdapters, emptyInferenceResult, resolveInferenceConfig } from './videov8_shared';


function serializeWorkerEffect(workerEffect) {
	if (!workerEffect || typeof workerEffect.draw !== 'function') return null;
	return {
		drawSrc: workerEffect.draw.toString(),
		setupSrc: typeof workerEffect.setup === 'function' ? workerEffect.setup.toString() : null,
		params: workerEffect.params || {},
	};
}


class VideoV8 {

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

	async process(options = {}) {
		if (!this.file || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}
		return this.workerCount > 1
			? this._processWithCoordinator(options)
			: this._processSingleThreaded(options);
	}


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



	async _processWithCoordinator(options) {
		const { onProgress = null, onPipelineEvent = null, workerEffect = null } = options;

		const inference = resolveInferenceConfig(options.inference, {

			defaultDelegate: 'CPU',
		});
		const { inferenceStride } = this._plan(options);
		const userEffect = serializeWorkerEffect(workerEffect);

		return new Promise((resolve, reject) => {
			let coordinator;
			try {
				coordinator = new Worker(new URL('./videov8_worker.js', import.meta.url));
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


	terminateWorkers() {
		if (this.coordinator) {
			try { this.coordinator.terminate(); } catch (_) { }
			this.coordinator = null;
		}
	}
}

export default VideoV8;
