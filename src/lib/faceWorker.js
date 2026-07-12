/* eslint-disable no-restricted-globals */
/**
 * faceWorker.js — Web Worker for parallel AI-powered face effect processing
 *
 * Uses @mediapipe/tasks-vision FaceLandmarker which runs entirely in Web Workers
 * via WebAssembly (no DOM, no window required).
 *
 * Pipeline per frame:
 *   receive small detect bitmap  →  FaceLandmarker.detectForVideo()
 *   →  return lightweight overlay metadata to main thread
 */

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ─── Worker state ─────────────────────────────────────────────────────────────
let faceLandmarker = null;
let detectCtx = null;   // small OffscreenCanvas — used ONLY for inference
let drawCtx = null;     // full-res OffscreenCanvas — used to composite effects (draw-in-worker mode)
let workerWidth = 0;
let workerHeight = 0;
let cachedLandmarks = null;
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
async function handleInit({ width, height, wasmBasePath, delegate = 'GPU' }) {
	workerWidth = width;
	workerHeight = height;
	cachedLandmarks = null;
	cachedInferenceFrameIndex = Number.NEGATIVE_INFINITY;

	// Small detect canvas — AI inference runs here (9× fewer pixels than 1080p)
	// FaceLandmarker returns normalised landmarks (0-1), so resolution doesn't
	// affect landmark accuracy — only inference speed.
	const DETECT_W = 512;
	const DETECT_H = Math.round(DETECT_W * height / width);
	const detectOffscreen = new OffscreenCanvas(DETECT_W, DETECT_H);
	detectCtx = detectOffscreen.getContext('2d');

	// Full-resolution compositing canvas. When the main thread runs the
	// draw-in-worker pipeline (renderStride === 1), each worker receives the
	// full-res frame, draws the effect here, and transfers a finished bitmap
	// back. One per worker so all N composite in parallel.
	const drawOffscreen = new OffscreenCanvas(width, height);
	drawCtx = drawOffscreen.getContext('2d', { willReadFrequently: false });

	try {
		// Use locally-served wasm files (public/mediapipe-wasm/) — no CDN dependency.
		// wasmBasePath is passed from the main thread as window.location.origin + '/mediapipe-wasm'.
		const filesetResolver = await FilesetResolver.forVisionTasks(wasmBasePath);

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
	} catch (err) {
		// AI init failure is non-fatal — frames are passed through without effects
		console.warn('faceWorker: FaceLandmarker init failed, running without AI:', err.message);
		faceLandmarker = null;
	}

	self.postMessage({ type: 'INIT_COMPLETE', ready: true });
}

// ─── Batch processing ─────────────────────────────────────────────────────────
async function handleProcessBatch({ frames, width, height, inferenceStride = 1, drawInWorker = false }) {
	const processedFrames = [];
	const transferables = [];

	for (const frame of frames) {
		const { frameIndex, timestamp } = frame;
		// In draw-in-worker mode the worker receives the full-res bitmap and
		// downscales it here for inference (so the main thread only makes one
		// bitmap per frame). Otherwise it receives a pre-made small detect bitmap.
		const detectSource = drawInWorker ? frame.fullBitmap : frame.detectBitmap;

		// AI inference — normalised landmarks (0-1) are resolution-independent, so
		// tracking runs on a small frame while effects draw at full resolution.
		if (faceLandmarker && detectSource) {
			try {
				const shouldInfer = !cachedLandmarks || (frameIndex - cachedInferenceFrameIndex) >= inferenceStride;
				if (shouldInfer) {
					detectCtx.clearRect(0, 0, detectCtx.canvas.width, detectCtx.canvas.height);
					detectCtx.drawImage(detectSource, 0, 0, detectCtx.canvas.width, detectCtx.canvas.height);
					const result = faceLandmarker.detectForVideo(detectCtx.canvas, timestamp);
					cachedInferenceFrameIndex = frameIndex;
					cachedLandmarks = result.faceLandmarks && result.faceLandmarks.length > 0
						? result.faceLandmarks[0]
						: null;
				}
			} catch (e) {
				// Frame passes through with last-known landmarks if inference throws
			}
		}

		if (drawInWorker) {
			// Composite the finished frame entirely inside the worker, then hand a
			// ready-to-encode ImageBitmap back to the main thread (transferable).
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
	cachedLandmarks = null;
	cachedInferenceFrameIndex = Number.NEGATIVE_INFINITY;
	self.postMessage({ type: 'TERMINATED' });
}
