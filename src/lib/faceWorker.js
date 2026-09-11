/* eslint-disable no-restricted-globals */
/**
 * faceWorker.js — Web Worker for parallel AI effect processing
 *
 * Uses @mediapipe/tasks-vision (FaceLandmarker or ImageSegmenter) which runs
 * entirely in Web Workers via WebAssembly (no DOM, no window required).
 *
 * Selected by INIT payload.aiEffect:
 *   'facemesh'      — FaceLandmarker + sunglasses overlay (default)
 *   'segmentation'  — selfie ImageSegmenter + pattern fill in the character
 */

import { FaceLandmarker, ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { applyPatternFill, createSegmentationAux, loadPatternBitmaps, pickPatternIndex, SELFIE_SEGMENTER_MODEL } from './segmentationEffect.js';

// ─── Worker state ─────────────────────────────────────────────────────────────
let aiEffect = 'facemesh';
let faceLandmarker = null;
let imageSegmenter = null;
let detectCtx = null;   // small OffscreenCanvas — used ONLY for inference
let drawCtx = null;     // full-res OffscreenCanvas — used to composite effects (draw-in-worker mode)
let segmentationAux = null;
let patternBitmaps = [];
let workerWidth = 0;
let workerHeight = 0;
let cachedLandmarks = null;
let cachedCategoryMask = null;
let cachedInferenceFrameIndex = Number.NEGATIVE_INFINITY;

// ─── Message dispatcher ───────────────────────────────────────────────────────
self.onmessage = async (event) => {
	const { type, payload } = event.data;
	try {
		switch (type) {
			case 'INIT':      await handleInit(payload);        break;
			case 'PROCESS_BATCH': await handleProcessBatch(payload); break;
			case 'TERMINATE': handleTerminate(); break;
			default: break;
		}
	} catch (error) {
		self.postMessage({ type: 'ERROR', error: { message: error.message, stack: error.stack } });
	}
};

// ─── Initialise ───────────────────────────────────────────────────────────────
async function handleInit({ width, height, wasmBasePath, delegate = 'GPU', aiEffect: effect = 'facemesh' }) {
	aiEffect = effect === 'segmentation' ? 'segmentation' : 'facemesh';
	workerWidth = width;
	workerHeight = height;
	cachedLandmarks = null;
	cachedCategoryMask = null;
	cachedInferenceFrameIndex = Number.NEGATIVE_INFINITY;

	// Segmentation is cheaper at 256px; landmarks stay at 512px.
	const DETECT_W = aiEffect === 'segmentation' ? 256 : 512;
	const DETECT_H = Math.round(DETECT_W * height / width);
	const detectOffscreen = new OffscreenCanvas(DETECT_W, DETECT_H);
	detectCtx = detectOffscreen.getContext('2d');

	// Full-resolution compositing canvas. When the main thread runs the
	// draw-in-worker pipeline (renderStride === 1), each worker receives the
	// full-res frame, draws the effect here, and transfers a finished bitmap
	// back. One per worker so all N composite in parallel.
	const drawOffscreen = new OffscreenCanvas(width, height);
	drawCtx = drawOffscreen.getContext('2d', { willReadFrequently: false });

	if (aiEffect === 'segmentation') {
		segmentationAux = createSegmentationAux();
		try {
			patternBitmaps = await loadPatternBitmaps();
		} catch (err) {
			console.warn('faceWorker: failed to load pattern images:', err.message);
			patternBitmaps = [];
		}
	}

	try {
		// Use locally-served wasm files (public/mediapipe-wasm/) — no CDN dependency.
		// wasmBasePath is passed from the main thread as window.location.origin + '/mediapipe-wasm'.
		const filesetResolver = await FilesetResolver.forVisionTasks(wasmBasePath);

		if (aiEffect === 'segmentation') {
			imageSegmenter = await ImageSegmenter.createFromOptions(filesetResolver, {
				baseOptions: {
					modelAssetPath: SELFIE_SEGMENTER_MODEL,
					delegate,
				},
				runningMode: 'VIDEO',
				outputCategoryMask: true,
				outputConfidenceMasks: false,
			});
			console.log(`faceWorker: ImageSegmenter ready (${delegate}, ${DETECT_W}w, VIDEO mode)`);
		} else {
			faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
				baseOptions: {
					// Model is ~4 MB, downloaded once and browser-cached.
					modelAssetPath:
						'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
					// Multiple workers competing for one GPU usually serialize throughput.
					// CPU delegate scales better across workers because each worker gets its
					// own core instead of contending for the same GPU queue.
					delegate,
				},
				runningMode: 'VIDEO',
				numFaces: 1,
				outputFaceBlendshapes: false,
				outputFacialTransformationMatrixes: false,
			});
			console.log(`faceWorker: FaceLandmarker ready (${delegate}, ${DETECT_W}w, VIDEO mode)`);
		}
	} catch (err) {
		// AI init failure is non-fatal — frames are passed through without effects
		console.warn(`faceWorker: ${aiEffect} init failed, running without AI:`, err.message);
		faceLandmarker = null;
		imageSegmenter = null;
	}

	self.postMessage({ type: 'INIT_COMPLETE', ready: true, aiEffect });
}

// ─── Batch processing ─────────────────────────────────────────────────────────
async function handleProcessBatch({ frames, width, height, inferenceStride = 1, drawInWorker = false }) {
	const processedFrames = [];
	const transferables = [];
	const useSegmentation = aiEffect === 'segmentation';

	for (const frame of frames) {
		const { frameIndex, timestamp } = frame;
		// In draw-in-worker mode the worker receives the full-res bitmap and
		// downscales it here for inference (so the main thread only makes one
		// bitmap per frame). Otherwise it receives a pre-made small detect bitmap.
		const detectSource = drawInWorker ? frame.fullBitmap : frame.detectBitmap;

		if (detectSource) {
			try {
				const shouldInfer = (useSegmentation ? !cachedCategoryMask : !cachedLandmarks)
					|| (frameIndex - cachedInferenceFrameIndex) >= inferenceStride;
				if (shouldInfer) {
					detectCtx.clearRect(0, 0, detectCtx.canvas.width, detectCtx.canvas.height);
					detectCtx.drawImage(detectSource, 0, 0, detectCtx.canvas.width, detectCtx.canvas.height);

					if (useSegmentation && imageSegmenter) {
						cachedCategoryMask = runSegmenter(detectCtx.canvas, timestamp);
						cachedInferenceFrameIndex = frameIndex;
					} else if (!useSegmentation && faceLandmarker) {
						const result = faceLandmarker.detectForVideo(detectCtx.canvas, timestamp);
						cachedInferenceFrameIndex = frameIndex;
						cachedLandmarks = result.faceLandmarks && result.faceLandmarks.length > 0
							? result.faceLandmarks[0]
							: null;
					}
				}
			} catch (e) {
				// Frame passes through with last-known result if inference throws
			}
		}

		if (drawInWorker) {
			// Composite the finished frame entirely inside the worker, then hand a
			// ready-to-encode ImageBitmap back to the main thread (transferable).
			if (useSegmentation) {
				const pattern = patternBitmaps.length > 0
					? patternBitmaps[pickPatternIndex(timestamp, patternBitmaps.length)]
					: null;
				applyPatternFill(drawCtx, frame.fullBitmap, cachedCategoryMask, segmentationAux, pattern, width, height);
				drawOverlay(drawCtx, frameIndex, cachedCategoryMask ? '| Pattern \u2713' : '| No Person');
			} else {
				drawCtx.clearRect(0, 0, width, height);
				if (frame.fullBitmap) {
					drawCtx.drawImage(frame.fullBitmap, 0, 0, width, height);
				}
				if (cachedLandmarks) {
					drawSunglasses(drawCtx, cachedLandmarks, width, height);
				} else {
					drawOverlay(drawCtx, frameIndex, '| No Face');
				}
				drawOverlay(drawCtx, frameIndex, cachedLandmarks ? '| AI Worker \u2713' : '| Worker');
			}

			const rendered = drawCtx.canvas.transferToImageBitmap();
			if (frame.fullBitmap) frame.fullBitmap.close();

			processedFrames.push({ frameIndex, timestamp, rendered });
			transferables.push(rendered);
		} else {
			if (frame.detectBitmap) frame.detectBitmap.close();
			processedFrames.push({
				frameIndex,
				hasFace: Boolean(cachedLandmarks),
				landmarks: cachedLandmarks,
				timestamp,
			});
		}
	}

	self.postMessage({ type: 'PROCESS_BATCH_COMPLETE', payload: { processedFrames } }, transferables);
}

function runSegmenter(canvas, timestamp) {
	let copied = null;
	imageSegmenter.segmentForVideo(canvas, timestamp, (result) => {
		copied = copyCategoryMask(result.categoryMask);
	});
	return copied;
}

function copyCategoryMask(mask) {
	if (!mask) return null;
	const src = mask.getAsUint8Array();
	const data = new Uint8Array(src.length);
	data.set(src);
	return {
		width: mask.width,
		height: mask.height,
		getAsUint8Array: () => data,
	};
}

// ─── Effect drawing (runs inside the worker in draw-in-worker mode) ─────────────
function drawSunglasses(ctx, landmarks, width, height) {
	const lx = (lm) => lm.x * width;
	const ly = (lm) => lm.y * height;

	const leftCX = (lx(landmarks[133]) + lx(landmarks[33])) / 2;
	const leftCY = (ly(landmarks[160]) + ly(landmarks[145])) / 2;
	const leftRX = Math.abs(lx(landmarks[33]) - lx(landmarks[133])) / 2 + 15;
	const leftRY = Math.abs(ly(landmarks[145]) - ly(landmarks[160])) / 2 + 12;

	const rightCX = (lx(landmarks[263]) + lx(landmarks[362])) / 2;
	const rightCY = (ly(landmarks[387]) + ly(landmarks[374])) / 2;
	const rightRX = Math.abs(lx(landmarks[362]) - lx(landmarks[263])) / 2 + 15;
	const rightRY = Math.abs(ly(landmarks[374]) - ly(landmarks[387])) / 2 + 12;

	const grad = ctx.createLinearGradient(0, leftCY - leftRY, 0, leftCY + leftRY);
	grad.addColorStop(0, 'rgba(40,40,40,0.85)');
	grad.addColorStop(0.5, 'rgba(20,20,20,0.9)');
	grad.addColorStop(1, 'rgba(40,40,40,0.85)');

	ctx.fillStyle = grad;
	ctx.strokeStyle = '#1a1a1a';
	ctx.lineWidth = 4;

	ctx.beginPath();
	ctx.ellipse(leftCX, leftCY, leftRX, leftRY, 0, 0, 2 * Math.PI);
	ctx.fill();
	ctx.stroke();

	ctx.beginPath();
	ctx.ellipse(rightCX, rightCY, rightRX, rightRY, 0, 0, 2 * Math.PI);
	ctx.fill();
	ctx.stroke();

	ctx.beginPath();
	ctx.moveTo(leftCX + leftRX, leftCY);
	ctx.lineTo(rightCX - rightRX, rightCY);
	ctx.stroke();

	ctx.fillStyle = 'rgba(255,255,255,0.15)';
	for (const [cx, cy, rx, ry] of [[leftCX, leftCY, leftRX, leftRY], [rightCX, rightCY, rightRX, rightRY]]) {
		ctx.beginPath();
		ctx.ellipse(cx - rx / 3, cy - ry / 2.5, rx / 4, ry / 4, -0.3, 0, 2 * Math.PI);
		ctx.fill();
	}
}

function drawOverlay(ctx, frameIndex, suffix) {
	ctx.font = 'bold 14px Arial';
	ctx.shadowColor = 'rgba(0,0,0,0.7)';
	ctx.shadowBlur = 4;
	ctx.fillStyle = '#ffffff';
	ctx.fillText(`Frame: ${frameIndex} ${suffix}`, 10, 30);
	ctx.shadowColor = 'transparent';
}

// ─── Terminate ────────────────────────────────────────────────────────────────
function handleTerminate() {
	if (faceLandmarker) {
		try { faceLandmarker.close(); } catch (_) {}
		faceLandmarker = null;
	}
	if (imageSegmenter) {
		try { imageSegmenter.close(); } catch (_) {}
		imageSegmenter = null;
	}
	cachedLandmarks = null;
	cachedCategoryMask = null;
	cachedInferenceFrameIndex = Number.NEGATIVE_INFINITY;
	for (const bitmap of patternBitmaps) {
		try { bitmap.close(); } catch (_) {}
	}
	patternBitmaps = [];
	self.postMessage({ type: 'TERMINATED' });
}
