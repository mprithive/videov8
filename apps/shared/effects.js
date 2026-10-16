/**
 * Demo effects shared by the core and React examples.
 *
 * `setupEffect` and `drawEffect` are sent to workers as source text, so
 * everything they need lives inside those two functions. Asset URLs arrive
 * through `params`.
 */

export const SAMPLE_VIDEOS = [
	{ label: 'Portrait 4K', url: '/videos/demo_portrait_4k_25.mp4' },
	{ label: 'Cine 4K 1', url: '/videos/demo_cine_4k_1.mp4' },
	{ label: 'Cine 4K 2', url: '/videos/demo_cine_4k_2.mp4' },
	{ label: 'Cine 4K 6', url: '/videos/demo_cine_4k_6.mp4' },
];

export const EFFECT_ASSETS = {
	coolersUrl: '/patterns/coolers.png',
	patternUrls: ['/patterns/1.png', '/patterns/2.png', '/patterns/3.png'],
};

export function inferenceForEffect(effect) {
	if (effect === 'coolers') return { task: 'face-landmarker' };
	if (effect === 'pattern') return { task: 'image-segmenter' };
	return undefined;
}

export async function fileFromSampleUrl(url) {
	const response = await fetch(url);
	if (!response.ok) throw new Error('Could not load the sample video');
	const blob = await response.blob();
	const name = url.split('/').pop();
	return new File([blob], name, { type: blob.type || 'video/mp4' });
}

export function setupEffect(params) {
	function loadBitmap(url) {
		return fetch(url).then(function (response) {
			if (!response.ok) throw new Error('Failed to load ' + url);
			return response.blob();
		}).then(function (blob) {
			return createImageBitmap(blob);
		});
	}

	var jobs = [];
	if (params.effect === 'coolers' && params.coolersUrl) {
		jobs.push(loadBitmap(params.coolersUrl).then(function (image) {
			return { coolersImage: image };
		}));
	}
	if (params.effect === 'pattern' && params.patternUrls && params.patternUrls.length) {
		jobs.push(Promise.all(params.patternUrls.map(loadBitmap)).then(function (images) {
			var maskCanvas = new OffscreenCanvas(8, 8);
			var featherCanvas = new OffscreenCanvas(8, 8);
			return {
				patternImages: images,
				maskCtx: maskCanvas.getContext('2d', { willReadFrequently: true }),
				featherCtx: featherCanvas.getContext('2d'),
			};
		}));
	}

	return Promise.all(jobs).then(function (parts) {
		var merged = {};
		var i;
		var key;
		for (i = 0; i < parts.length; i++) {
			for (key in parts[i]) merged[key] = parts[i][key];
		}
		return merged;
	});
}

export function drawEffect(ctx, frame, inference, params) {
	function drawCoolers(image) {
		if (!image || inference.type !== 'facemesh') return;
		var faces = inference.faceLandmarks || [];
		var i;
		for (i = 0; i < faces.length; i++) {
			var landmarks = faces[i];
			if (!landmarks || !landmarks[33] || !landmarks[263]) continue;
			var leftX = landmarks[33].x * frame.width;
			var leftY = landmarks[33].y * frame.height;
			var rightX = landmarks[263].x * frame.width;
			var rightY = landmarks[263].y * frame.height;
			var eyeDist = Math.hypot(rightX - leftX, rightY - leftY);
			if (eyeDist < 2) continue;
			var drawWidth = eyeDist * 2.15;
			var drawHeight = drawWidth * (image.height / image.width);
			ctx.save();
			ctx.translate((leftX + rightX) / 2, (leftY + rightY) / 2);
			ctx.rotate(Math.atan2(rightY - leftY, rightX - leftX));
			ctx.drawImage(image, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
			ctx.restore();
		}
	}

	function hashSlot(slot) {
		var x = (slot + 1) * 2654435761;
		x = Math.imul(x ^ (x >>> 16), 2246822519);
		x = Math.imul(x ^ (x >>> 13), 3266489917);
		return x >>> 0;
	}

	function pickPatternIndex(timestampMs, count) {
		if (count <= 1) return 0;
		var slot = Math.max(0, Math.floor(timestampMs / 500));
		var current = hashSlot(slot) % count;
		var previous = hashSlot(slot - 1) % count;
		return current === previous ? (current + 1) % count : current;
	}

	function drawCover(image, width, height) {
		var iw = image.width || width;
		var ih = image.height || height;
		var scale = Math.max(width / iw, height / ih);
		var dw = iw * scale;
		var dh = ih * scale;
		ctx.drawImage(image, (width - dw) / 2, (height - dh) / 2, dw, dh);
	}

	function drawPattern() {
		var mask = inference.categoryMask;
		var images = params.patternImages || [];
		if (inference.type !== 'imagesegmenter' || !mask || !mask.data || !images.length) return;
		if (!params.maskCtx || !params.featherCtx) return;
		var patternImage = images[pickPatternIndex(frame.timestamp, images.length)];
		var maskCtx = params.maskCtx;
		var featherCtx = params.featherCtx;
		var mw = mask.width;
		var mh = mask.height;
		if (maskCtx.canvas.width !== mw || maskCtx.canvas.height !== mh) {
			maskCtx.canvas.width = mw;
			maskCtx.canvas.height = mh;
		}
		if (featherCtx.canvas.width !== mw || featherCtx.canvas.height !== mh) {
			featherCtx.canvas.width = mw;
			featherCtx.canvas.height = mh;
		}
		var imageData = maskCtx.createImageData(mw, mh);
		var pixels = imageData.data;
		var src = mask.data;
		var p;
		for (p = 0; p < src.length; p++) {
			var alpha = src[p] !== 0 ? 255 : 0;
			var offset = p * 4;
			pixels[offset] = 255;
			pixels[offset + 1] = 255;
			pixels[offset + 2] = 255;
			pixels[offset + 3] = alpha;
		}
		maskCtx.putImageData(imageData, 0, 0);
		featherCtx.clearRect(0, 0, mw, mh);
		featherCtx.filter = 'blur(3px)';
		featherCtx.drawImage(maskCtx.canvas, 0, 0);
		featherCtx.filter = 'none';

		ctx.globalCompositeOperation = 'destination-in';
		ctx.drawImage(featherCtx.canvas, 0, 0, frame.width, frame.height);
		ctx.globalCompositeOperation = 'destination-over';
		drawCover(patternImage, frame.width, frame.height);
		ctx.globalCompositeOperation = 'source-over';
	}

	if (params.effect === 'coolers') drawCoolers(params.coolersImage);
	else if (params.effect === 'pattern') drawPattern();
}
