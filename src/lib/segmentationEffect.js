/**
 * Selfie-segmentation compositor — pattern fill inside the character shape.
 *
 * A random image from src/Patterns/ is cover-fitted into the masked region
 * and held for PATTERN_HOLD_MS. The same timestamp slot always picks the
 * same image so every worker stays in sync.
 */

const MASK_BLUR_PX = 3;
export const PATTERN_HOLD_MS = 500;

const PATTERN_URLS = [
	new URL('../Patterns/1.png', import.meta.url).href,
	new URL('../Patterns/2.png', import.meta.url).href,
	new URL('../Patterns/3.png', import.meta.url).href,
];

function makeCanvas(width, height) {
	if (typeof OffscreenCanvas !== 'undefined') {
		return new OffscreenCanvas(width, height);
	}
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	return canvas;
}

/**
 * Allocate the small auxiliary canvases used by applyPatternFill.
 * Call once per worker / once on the main thread and reuse.
 */
export function createSegmentationAux() {
	const maskCanvas = makeCanvas(256, 256);
	const featherCanvas = makeCanvas(256, 256);

	return {
		maskCtx: maskCanvas.getContext('2d', { willReadFrequently: true }),
		featherCtx: featherCanvas.getContext('2d'),
	};
}

export async function loadPatternBitmaps() {
	return Promise.all(PATTERN_URLS.map(async (url) => {
		const response = await fetch(url);
		const blob = await response.blob();
		return createImageBitmap(blob);
	}));
}

/**
 * Deterministic pick so workers agree: one image per 0.5s slot, not the
 * same as the previous slot when more than one pattern exists.
 */
export function pickPatternIndex(timestampMs, count) {
	if (count <= 1) return 0;
	const slot = Math.max(0, Math.floor(timestampMs / PATTERN_HOLD_MS));
	const current = hashSlot(slot) % count;
	const previous = hashSlot(slot - 1) % count;
	return current === previous ? (current + 1) % count : current;
}

function hashSlot(slot) {
	let x = (slot + 1) * 2654435761;
	x = Math.imul(x ^ (x >>> 16), 2246822519);
	x = Math.imul(x ^ (x >>> 13), 3266489917);
	return (x >>> 0);
}

function resizeCanvas(ctx, width, height) {
	if (ctx.canvas.width !== width || ctx.canvas.height !== height) {
		ctx.canvas.width = width;
		ctx.canvas.height = height;
	}
}

/**
 * Paint a category mask (0 = background, non-zero = person) into maskCtx as
 * white-with-alpha, then feather the edges so the cutout isn't a hard line.
 * Polarity is left as-is to match the current segmentation output.
 */
function paintFeatheredMask(aux, categoryMask) {
	const { maskCtx, featherCtx } = aux;
	const mw = categoryMask.width;
	const mh = categoryMask.height;
	const data = categoryMask.getAsUint8Array();

	resizeCanvas(maskCtx, mw, mh);
	resizeCanvas(featherCtx, mw, mh);

	const imageData = maskCtx.createImageData(mw, mh);
	const pixels = imageData.data;
	for (let i = 0; i < data.length; i++) {
		const alpha = data[i] !== 0 ? 255 : 0;
		const o = i * 4;
		pixels[o] = 255;
		pixels[o + 1] = 255;
		pixels[o + 2] = 255;
		pixels[o + 3] = alpha;
	}
	maskCtx.putImageData(imageData, 0, 0);

	featherCtx.clearRect(0, 0, mw, mh);
	featherCtx.filter = `blur(${MASK_BLUR_PX}px)`;
	featherCtx.drawImage(maskCtx.canvas, 0, 0);
	featherCtx.filter = 'none';
}

function drawCover(ctx, image, width, height) {
	const iw = image.width || image.naturalWidth || width;
	const ih = image.height || image.naturalHeight || height;
	const scale = Math.max(width / iw, height / ih);
	const dw = iw * scale;
	const dh = ih * scale;
	ctx.drawImage(image, (width - dw) / 2, (height - dh) / 2, dw, dh);
}

/**
 * Keep the original frame everywhere except the character hole, which is
 * filled with a cover-fitted pattern image (same mask polarity as before).
 */
export function applyPatternFill(drawCtx, frameSource, categoryMask, aux, patternImage, width, height) {
	if (!frameSource) return;

	if (!categoryMask || !patternImage) {
		drawCtx.clearRect(0, 0, width, height);
		drawCtx.drawImage(frameSource, 0, 0, width, height);
		return;
	}

	paintFeatheredMask(aux, categoryMask);

	drawCtx.clearRect(0, 0, width, height);
	drawCtx.drawImage(frameSource, 0, 0, width, height);
	drawCtx.globalCompositeOperation = 'destination-in';
	drawCtx.drawImage(aux.featherCtx.canvas, 0, 0, width, height);
	drawCtx.globalCompositeOperation = 'destination-over';
	drawCover(drawCtx, patternImage, width, height);
	drawCtx.globalCompositeOperation = 'source-over';
}

export const SELFIE_SEGMENTER_MODEL =
	'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite';
