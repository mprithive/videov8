import { Input, ALL_FORMATS, BlobSource, VideoSampleSink, Output, BufferTarget, Mp4OutputFormat, CanvasSource, QUALITY_HIGH } from 'mediabunny';
import { drawCoolers, loadCoolersBitmap } from './coolersEffect';


class VideoV8Multithreaded {
	constructor(options = {}) {
		this.input = null;
		this.videoTrack = null;
		this.sink = null;
		this.isPlaying = false;
		this.currentTime = 0;
		this.duration = 0;
		this.frameRate = 24;
		this.playbackRate = 1.0;
		this.animationFrameId = null;
		this.onTimeUpdate = null;
		this.onEnded = null;
		this.onError = null;

		// Multithreading configuration — set via WORKER_COUNT in home.js
		this.workerCount = options.workerCount || 3;
		this.approxDeviceMemoryGB = options.approxDeviceMemoryGB || 8;
		this.memoryBudgetFraction = options.memoryBudgetFraction || 0.4;
		this.aiEffect = options.aiEffect || 'facemesh';
		this.workers = [];
		this.workerPool = [];
		this.frameQueue = [];
		this.pendingFrames = new Map();
		this.nextFrameIndex = 0;
	}


	async loadVideo(file) {
		try {
			this.input = new Input({
				formats: ALL_FORMATS,
				source: new BlobSource(file),
			});

			this.videoTrack = await this.input.getPrimaryVideoTrack();

			if (!this.videoTrack) {
				throw new Error('No video track found in the file');
			}

			const canDecode = await this.videoTrack.canDecode();
			if (!canDecode) {
				throw new Error('Video codec is not supported');
			}

			this.sink = new VideoSampleSink(this.videoTrack);

			this.duration = await this.videoTrack.computeDuration();
			this.width = await this.videoTrack.getDisplayWidth();
			this.height = await this.videoTrack.getDisplayHeight();

			const packetStats = await this.videoTrack.computePacketStats(100);
			this.frameRate = packetStats.averagePacketRate || 24;

			console.log('Video loaded successfully', {
				duration: this.duration,
				width: this.width,
				height: this.height,
				frameRate: this.frameRate,
			});

			return true;
		} catch (error) {
			console.error('Failed to load video:', error);
			if (this.onError) {
				this.onError(error);
			}
			return false;
		}
	}

	getMetadata() {
		return {
			duration: this.duration,
			width: this.width,
			height: this.height,
			frameRate: this.frameRate,
			currentTime: this.currentTime,
			isPlaying: this.isPlaying,
		};
	}

	async initializeWorkerPool() {
		return new Promise((resolve, reject) => {
			const initPromises = [];
			let readyCount = 0;

			for (let i = 0; i < this.workerCount; i++) {
				const promise = new Promise((resolveWorker) => {
					try {
						
						const worker = new Worker(new URL('./faceWorker.js', import.meta.url));
						const delegate = this.workerCount > 1 ? 'CPU' : 'GPU';

						let initTimeout;

						worker.onmessage = (event) => {
							if (event.data.type === 'INIT_COMPLETE') {
								clearTimeout(initTimeout);
								console.log(`Worker ${i} ready (${this.aiEffect} AI loaded)`);
								readyCount++;
								resolveWorker();
							}
						};

						worker.onerror = (error) => {
							clearTimeout(initTimeout);
							console.error(`Worker ${i} error:`, error);
							resolveWorker(); // don't block other workers
						};

						console.log(`Initializing worker ${i} (${delegate})...`);
				
						const wasmBasePath = `${window.location.origin}${process.env.PUBLIC_URL || ''}/mediapipe-wasm`;
						worker.postMessage({
							type: 'INIT',
							payload: { width: this.width, height: this.height, wasmBasePath, delegate, aiEffect: this.aiEffect },
						});

						this.workers.push(worker);
						this.workerPool.push(worker);

						// 90 s — allows model download (~4 MB) on first run
						initTimeout = setTimeout(() => {
							console.warn(`Worker ${i} init timeout — continuing without it`);
							resolveWorker();
						}, 90000);
					} catch (error) {
						console.error(`Failed to create worker ${i}:`, error);
						resolveWorker();
					}
				});

				initPromises.push(promise);
			}

			Promise.all(initPromises)
				.then(() => {
					console.log(`Worker pool initialization complete. Ready: ${readyCount}/${this.workerCount}`);
					resolve();
				})
				.catch(reject);
		});
	}




	async processAndEncodeFramesWithWorkers(renderFunction, options = {}) {
		if (!this.sink || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}

		const {
			maxFrames = null,
			startTime = 0,
			endTime = this.duration,
			targetInferenceFrames = 160,
			renderStride = 4,
			aiEffect = this.aiEffect || 'facemesh',
			onProgress = null,
			onPipelineEvent = null,
		} = options;

		this.aiEffect = aiEffect;
		const emitPipeline = (event) => {
			if (onPipelineEvent) onPipelineEvent(event);
		};

		try {
			console.log('Initializing worker pool for true parallel processing...');
			await this.initializeWorkerPool();
			console.log(`${this.workers.length} workers ready`);
			emitPipeline({ type: 'pipeline-ready', workerCount: this.workers.length });

			let coolersBitmap = null;
			if (this.aiEffect !== 'segmentation') {
				try {
					coolersBitmap = await loadCoolersBitmap();
				} catch (err) {
					console.warn('Failed to load coolers.png:', err.message);
				}
			}

			const frameInterval = 1 / this.frameRate;
			const frameDuration = 1 / this.frameRate;
			let totalFrames = Math.ceil((endTime - startTime) * this.frameRate);
			if (maxFrames) totalFrames = Math.min(totalFrames, maxFrames);
			const inferenceStride = targetInferenceFrames > 0
				? Math.max(1, Math.ceil(totalFrames / targetInferenceFrames))
				: 1;
			const effectiveInferenceStride = renderStride > 1 ? 1 : inferenceStride;

			console.log(
				`Processing ${totalFrames} frames with ${this.workers.length} parallel workers ` +
				`(AI every ${effectiveInferenceStride} processed frames, render stride ${renderStride}, target ${targetInferenceFrames})`
			);

			const DRAW_IN_WORKER = renderStride === 1 || this.aiEffect === 'segmentation';
			console.log(`Draw-in-worker pipeline: ${DRAW_IN_WORKER ? 'ON (parallel AI + drawing)' : 'OFF (main-thread compositing)'} | effect: ${this.aiEffect}`);

			const output = new Output({
				format: new Mp4OutputFormat(),
				target: new BufferTarget(),
			});

			const DETECT_WIDTH = 512;

			const encodeCanvas = document.createElement('canvas');
			encodeCanvas.width = this.width;
			encodeCanvas.height = this.height;
			const encodeCtx = encodeCanvas.getContext('2d', { willReadFrequently: true });

			const videoSource = new CanvasSource(encodeCanvas, {
				codec: 'avc',
				bitrate: QUALITY_HIGH,
			});
			output.addVideoTrack(videoSource);
			await output.start();

			const decodeCanvas = document.createElement('canvas');
			decodeCanvas.width = this.width;
			decodeCanvas.height = this.height;
			const decodeCtx = decodeCanvas.getContext('2d', { willReadFrequently: true });
			const detectCanvas = document.createElement('canvas');
			detectCanvas.width = DETECT_WIDTH;
			detectCanvas.height = Math.round(DETECT_WIDTH * this.height / this.width);
			const detectCtx = detectCanvas.getContext('2d');


			let totalDecoded = 0;         // incremented inside decodeRange()
			let nextFrameToEncode = 0;    // incremented inside scheduleEncode()
			const encodedFrameMap = new Map();


			let encodeChain = Promise.resolve();

			const drawSunglasses = (ctx, landmarks) => {
				drawCoolers(ctx, landmarks, this.width, this.height, coolersBitmap);
			};

			const drawOverlay = (ctx, frameIndex, suffix) => {
				ctx.font = 'bold 14px Arial';
				ctx.shadowColor = 'rgba(0,0,0,0.7)';
				ctx.shadowBlur = 4;
				ctx.fillStyle = '#ffffff';
				ctx.fillText(`Frame: ${frameIndex} ${suffix}`, 10, 30);
				ctx.shadowColor = 'transparent';
			};

			const interpolateLandmarks = (fromLandmarks, toLandmarks, alpha) => {
				if (!fromLandmarks) return toLandmarks;
				if (!toLandmarks) return fromLandmarks;
				if (alpha <= 0) return fromLandmarks;
				if (alpha >= 1) return toLandmarks;

				return fromLandmarks.map((landmark, index) => {
					const nextLandmark = toLandmarks[index] || landmark;
					return {
						x: landmark.x + ((nextLandmark.x ?? landmark.x) - landmark.x) * alpha,
						y: landmark.y + ((nextLandmark.y ?? landmark.y) - landmark.y) * alpha,
						z: (landmark.z ?? 0) + (((nextLandmark.z ?? landmark.z ?? 0) - (landmark.z ?? 0)) * alpha),
					};
				});
			};

			const scheduleEncode = () => {
				encodeChain = encodeChain.then(async () => {
					let encoding = false;
					while (encodedFrameMap.has(nextFrameToEncode) && (DRAW_IN_WORKER || nextFrameToEncode < totalFrames)) {
						const frameData = encodedFrameMap.get(nextFrameToEncode);
						encodedFrameMap.delete(nextFrameToEncode);
						if (!encoding) {
							emitPipeline({ type: 'encoder-start', frameIndex: nextFrameToEncode });
							encoding = true;
						}
						emitPipeline({
							type: 'encoder-frame',
							frameIndex: nextFrameToEncode,
							workerIndex: frameData.workerIndex,
						});


						if (frameData.rendered) {
							encodeCtx.clearRect(0, 0, this.width, this.height);
							encodeCtx.drawImage(frameData.rendered, 0, 0, this.width, this.height);
							frameData.rendered.close();
							await videoSource.add(nextFrameToEncode * frameDuration, frameDuration);
							nextFrameToEncode++;
							if (onProgress) {
								onProgress({
									current: nextFrameToEncode,
									total: totalFrames,
									progress: (totalDecoded + nextFrameToEncode) / (totalFrames * 2),
								});
							}
							continue;
						}

						encodeCtx.clearRect(0, 0, this.width, this.height);
						encodeCtx.drawImage(frameData.bitmapRef.bitmap, 0, 0, this.width, this.height);
						frameData.bitmapRef.remainingUses--;
						if (frameData.bitmapRef.remainingUses === 0) {
							frameData.bitmapRef.bitmap.close();
						}

						const interpolatedLandmarks = interpolateLandmarks(
							frameData.landmarks,
							frameData.nextLandmarks,
							frameData.blendAlpha || 0
						);

						if (frameData.nextBitmapRef && frameData.blendAlpha > 0) {
							encodeCtx.save();
							encodeCtx.globalAlpha = frameData.blendAlpha;
							encodeCtx.drawImage(frameData.nextBitmapRef.bitmap, 0, 0, this.width, this.height);
							encodeCtx.restore();
							frameData.nextBitmapRef.remainingUses--;
							if (frameData.nextBitmapRef.remainingUses === 0) {
								frameData.nextBitmapRef.bitmap.close();
							}
						}

						if (interpolatedLandmarks) {
							drawSunglasses(encodeCtx, interpolatedLandmarks);
						} else {
							drawOverlay(encodeCtx, nextFrameToEncode, '| No Face');
						}
						drawOverlay(encodeCtx, nextFrameToEncode, interpolatedLandmarks ? '| AI Worker ✓' : '| Worker');
						await videoSource.add(nextFrameToEncode * frameDuration, frameDuration);
						nextFrameToEncode++;
						if (onProgress) {
							onProgress({
								current: nextFrameToEncode,
								total: totalFrames,
								// (decoded + encoded) / 2×total  →  always 0–1, never decreases
								progress: (totalDecoded + nextFrameToEncode) / (totalFrames * 2),
							});
						}
					}
					if (encoding) {
						emitPipeline({ type: 'encoder-idle', frameIndex: nextFrameToEncode });
					}
				});
				return encodeChain;
			};

	
			const sendToWorker = (worker, frames) => new Promise((resolve, reject) => {
				const timeout = setTimeout(() => reject(new Error('Worker timeout after 3 minutes')), 180000);
				const handler = (event) => {
					if (event.data.type === 'PROCESS_BATCH_COMPLETE') {
						clearTimeout(timeout);
						worker.removeEventListener('message', handler);
						resolve(event.data.payload.processedFrames);
					} else if (event.data.type === 'BATCH_ERROR') {
						clearTimeout(timeout);
						worker.removeEventListener('message', handler);
						reject(new Error(event.data.error));
					}
				};
				worker.addEventListener('message', handler);
				const transferables = DRAW_IN_WORKER
					? frames.map(f => f.fullBitmap).filter(Boolean)
					: frames.map(f => f.detectBitmap).filter(Boolean);
				worker.postMessage(
					{ type: 'PROCESS_BATCH', payload: { frames, width: this.width, height: this.height, inferenceStride: effectiveInferenceStride, drawInWorker: DRAW_IN_WORKER, aiEffect: this.aiEffect } },
					transferables
				);
			});

			const bytesPerFrame = this.width * this.height * 4;
			const approxOperationBudgetBytes = this.approxDeviceMemoryGB * 1024 * 1024 * 1024 * this.memoryBudgetFraction;
			const approxWorkerRuntimeBytes = this.workers.length * 140 * 1024 * 1024;
			const approxSafetyReserveBytes = 256 * 1024 * 1024;
			const approxFrameBudgetBytes = Math.max(
				128 * 1024 * 1024,
				approxOperationBudgetBytes - approxWorkerRuntimeBytes - approxSafetyReserveBytes
			);

			const chunkFramesFromBudget = Math.floor(approxFrameBudgetBytes / (bytesPerFrame * 1.35));
			const CHUNK_FRAME_COUNT = Math.max(
				this.workers.length * 24,
				Math.min(320, chunkFramesFromBudget || 192)
			);

			const WORKER_BATCH_SIZE = Math.max(
				16,
				Math.min(32, Math.max(effectiveInferenceStride, Math.ceil(CHUNK_FRAME_COUNT / (this.workers.length * 4))))
			);

			console.log(
				`Memory budget: ~${this.approxDeviceMemoryGB}GB device RAM × ${Math.round(this.memoryBudgetFraction * 100)}% ` +
				`=> ${(approxOperationBudgetBytes / (1024 * 1024)).toFixed(0)}MB op budget | ` +
				`chunk ${CHUNK_FRAME_COUNT} | worker-batch ${WORKER_BATCH_SIZE}`
			);

	
			const decodeRange = async (rangeStart, rangeEnd) => {
				const decodedFrames = [];
				const coverageBySource = new Map();
				const bitmapRefs = new Map();
				let currentSourceFrameIndex = null;
				emitPipeline({ type: 'decoder-start', from: rangeStart, to: rangeEnd });

				for (let fi = rangeStart; fi < rangeEnd; fi++) {
					const shouldDecodeFrame = currentSourceFrameIndex === null || ((fi - rangeStart) % renderStride === 0);

					if (shouldDecodeFrame) {
						const sample = await this.sink.getSample(startTime + fi * frameInterval);
						decodeCtx.clearRect(0, 0, this.width, this.height);
						sample.draw(decodeCtx, 0, 0);
						sample.close();

						if (DRAW_IN_WORKER) {
						
							decodedFrames.push({
								frameIndex: fi,
								fullBitmap: await createImageBitmap(decodeCanvas),
								timestamp: Math.round((startTime + fi * frameInterval) * 1000),
							});
						} else {
							detectCtx.clearRect(0, 0, detectCanvas.width, detectCanvas.height);
							detectCtx.drawImage(decodeCanvas, 0, 0, detectCanvas.width, detectCanvas.height);

							const bitmapRef = {
								bitmap: await createImageBitmap(decodeCanvas),
								remainingUses: 0,
							};

							decodedFrames.push({
								frameIndex: fi,
								detectBitmap: await createImageBitmap(detectCanvas),
								timestamp: Math.round((startTime + fi * frameInterval) * 1000),
							});

							bitmapRefs.set(fi, bitmapRef);
						}

						coverageBySource.set(fi, []);
						currentSourceFrameIndex = fi;
						totalDecoded++;
					}

					coverageBySource.get(currentSourceFrameIndex).push(fi);

					if (onProgress) {
						onProgress({
							current: fi + 1,
							total: totalFrames,
							progress: (totalDecoded + nextFrameToEncode) / (totalFrames * 2),
						});
					}
				}

				emitPipeline({ type: 'decoder-idle', decoded: totalDecoded });
				return { decodedFrames, coverageBySource, bitmapRefs };
			};


			const buildChunkBatches = (chunkFrames) => {
				const batches = [];
				for (let start = 0; start < chunkFrames.length; start += WORKER_BATCH_SIZE) {
					batches.push(chunkFrames.slice(start, Math.min(start + WORKER_BATCH_SIZE, chunkFrames.length)));
				}
				return batches;
			};

			const processChunk = async (chunkFrames) => {
				const batches = buildChunkBatches(chunkFrames.decodedFrames);
				const sourceFrameResults = new Map();
				const workerPumps = this.workers.map((worker, workerIndex) => (async () => {
					while (batches.length > 0) {
						const batch = batches.shift();
						if (!batch) return;
						emitPipeline({
							type: 'worker-receive',
							workerIndex,
							frameCount: batch.length,
							from: batch[0].frameIndex,
							to: batch[batch.length - 1].frameIndex,
						});
						const results = await sendToWorker(worker, batch);
						emitPipeline({
							type: 'worker-processed',
							workerIndex,
							frameCount: results.length,
							from: results[0] ? results[0].frameIndex : null,
							to: results[results.length - 1] ? results[results.length - 1].frameIndex : null,
						});
						for (const frame of results) {
							sourceFrameResults.set(frame.frameIndex, { ...frame, workerIndex });
						}
					}
				})());

				await Promise.all(workerPumps);

				if (DRAW_IN_WORKER) {

					for (const [frameIndex, result] of sourceFrameResults) {
						encodedFrameMap.set(frameIndex, { rendered: result.rendered, workerIndex: result.workerIndex });
					}
					scheduleEncode();
					return;
				}

				const sourceFrameIndexes = Array.from(chunkFrames.coverageBySource.keys()).sort((a, b) => a - b);
				for (let sourceIndex = 0; sourceIndex < sourceFrameIndexes.length; sourceIndex++) {
					const sourceFrameIndex = sourceFrameIndexes[sourceIndex];
					const nextSourceFrameIndex = sourceFrameIndexes[sourceIndex + 1];
					const sourceResult = sourceFrameResults.get(sourceFrameIndex);
					const nextSourceResult = nextSourceFrameIndex !== undefined ? sourceFrameResults.get(nextSourceFrameIndex) : null;
					const bitmapRef = chunkFrames.bitmapRefs.get(sourceFrameIndex);
					const nextBitmapRef = nextSourceFrameIndex !== undefined ? chunkFrames.bitmapRefs.get(nextSourceFrameIndex) : null;
					const coveredFrames = chunkFrames.coverageBySource.get(sourceFrameIndex) || [sourceFrameIndex];

					for (const outputFrameIndex of coveredFrames) {
						const span = nextSourceFrameIndex !== undefined ? Math.max(1, nextSourceFrameIndex - sourceFrameIndex) : 1;
						const blendAlpha = nextSourceFrameIndex !== undefined
							? Math.min(1, Math.max(0, (outputFrameIndex - sourceFrameIndex) / span))
							: 0;

						bitmapRef.remainingUses++;
						if (nextBitmapRef && blendAlpha > 0) {
							nextBitmapRef.remainingUses++;
						}

						encodedFrameMap.set(outputFrameIndex, {
							bitmapRef,
							nextBitmapRef,
							landmarks: sourceResult ? sourceResult.landmarks : null,
							nextLandmarks: nextSourceResult ? nextSourceResult.landmarks : null,
							blendAlpha,
							workerIndex: sourceResult ? sourceResult.workerIndex : undefined,
						});
					}
				}

				scheduleEncode();
			};

			if (DRAW_IN_WORKER) {
		
				const iterEnd = (endTime && endTime < this.duration) ? endTime : undefined;
				const sampleIterator = this.sink.samples(startTime, iterEnd);
				let iteratorDone = false;
				let decodedOutputIndex = 0;

				const decodeNextChunk = async () => {
					const decodedFrames = [];
					emitPipeline({ type: 'decoder-start' });
					for (let n = 0; n < CHUNK_FRAME_COUNT; n++) {
						const { value: sample, done } = await sampleIterator.next();
						if (done || !sample) { iteratorDone = true; break; }
						decodeCtx.clearRect(0, 0, this.width, this.height);
						sample.draw(decodeCtx, 0, 0);
						sample.close();
						const fi = decodedOutputIndex++;
						decodedFrames.push({
							frameIndex: fi,
							fullBitmap: await createImageBitmap(decodeCanvas),
							timestamp: Math.round((startTime + fi * frameInterval) * 1000),
						});
						totalDecoded++;
						if (onProgress) {
							onProgress({
								current: totalDecoded,
								total: totalFrames,
								progress: Math.min(1, (totalDecoded + nextFrameToEncode) / (totalFrames * 2)),
							});
						}
					}
					emitPipeline({ type: 'decoder-idle', decoded: decodedOutputIndex });
					return { decodedFrames, coverageBySource: new Map(), bitmapRefs: new Map() };
				};

				// Cap on finished-but-not-yet-encoded frames held in memory (~one chunk).
				const MAX_PENDING_ENCODE = Math.max(CHUNK_FRAME_COUNT, this.workers.length * WORKER_BATCH_SIZE * 2);
				let decodePromise = decodeNextChunk();
				let chunkNo = 0;

				while (true) {
					const t0 = performance.now();
					const decodedChunk = await decodePromise;
					if (decodedChunk.decodedFrames.length === 0) break;

				
					decodePromise = iteratorDone ? Promise.resolve({ decodedFrames: [] }) : decodeNextChunk();

					await processChunk(decodedChunk);

					// Back-pressure: if encoding lags behind, let it drain before racing ahead.
					if (encodedFrameMap.size > MAX_PENDING_ENCODE) {
						await encodeChain;
					}

					const chunkMs = Math.round(performance.now() - t0);
					console.log(
						`Chunk ${++chunkNo}: ${decodedChunk.decodedFrames.length} frames | ` +
						`worker-batch ${WORKER_BATCH_SIZE} | time ${chunkMs}ms | ` +
						`pending-encode: ${encodedFrameMap.size}`
					);
				}
			} else {
				for (let chunkStart = 0; chunkStart < totalFrames; chunkStart += CHUNK_FRAME_COUNT) {
					const t0 = performance.now();
					const chunkEnd = Math.min(chunkStart + CHUNK_FRAME_COUNT, totalFrames);
					const decodedChunk = await decodeRange(chunkStart, chunkEnd);

					await processChunk(decodedChunk);

					const chunkMs = Math.round(performance.now() - t0);
					console.log(
						`Chunk ${Math.floor(chunkStart / CHUNK_FRAME_COUNT) + 1}: ` +
						`frames ${chunkStart}–${chunkEnd - 1} | ` +
						`${decodedChunk.decodedFrames.length} decoded keyframes | ` +
						`worker-batch ${WORKER_BATCH_SIZE} | ` +
						`time ${chunkMs}ms | pending-encode: ${encodedFrameMap.size}`
					);
				}
			}

			// All rounds dispatched — wait for the encode chain to fully drain.
			await encodeChain;

			await output.finalize();
			if (coolersBitmap) {
				try { coolersBitmap.close(); } catch (_) {}
			}
			const blob = new Blob([output.target.buffer], { type: 'video/mp4' });
			this.terminateWorkers();
			console.log(`✓ Parallel processing complete: ${totalFrames} frames, ${this.workers.length} workers`);
			return blob;
		} catch (error) {
			console.error('Parallel processing error:', error);
			this.terminateWorkers();
			throw error;
		}
	}

	/**
	 * Terminate all workers
	 * @private
	 */
	terminateWorkers() {
		for (const worker of this.workers) {
			try {
				worker.postMessage({ type: 'TERMINATE' });
				worker.terminate();
			} catch (error) {
				console.error('Error terminating worker:', error);
			}
		}
		this.workers = [];
		this.workerPool = [];
		this.frameQueue = [];
		this.pendingFrames.clear();
	}


	static _WORKER_FILE = './faceWorker.js';
}

export default VideoV8Multithreaded;
