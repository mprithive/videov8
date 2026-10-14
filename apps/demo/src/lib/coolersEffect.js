/**
 * Overlay src/Patterns/coolers.png on FaceMesh eye landmarks.
 * Shared by faceWorker.js (OffscreenCanvas) and the single-thread path.
 */

export const COOLERS_URL = new URL('../Patterns/coolers.png', import.meta.url).href;

export async function loadCoolersBitmap() {
	const response = await fetch(COOLERS_URL);
	const blob = await response.blob();
	return createImageBitmap(blob);
}

/**
 * Draw the coolers image over both eyes, scaled to the face and rotated
 * to match head tilt. Landmarks are MediaPipe FaceMesh (normalized 0–1).
 */
export function drawCoolers(ctx, landmarks, width, height, image) {
	if (!image || !landmarks || !landmarks[33] || !landmarks[263]) return;

	const leftX = landmarks[33].x * width;
	const leftY = landmarks[33].y * height;
	const rightX = landmarks[263].x * width;
	const rightY = landmarks[263].y * height;

	const centerX = (leftX + rightX) / 2;
	const centerY = (leftY + rightY) / 2;
	const eyeDist = Math.hypot(rightX - leftX, rightY - leftY);
	if (eyeDist < 2) return;

	const drawWidth = eyeDist * 2.15;
	const drawHeight = drawWidth * (image.height / image.width);
	const angle = Math.atan2(rightY - leftY, rightX - leftX);

	ctx.save();
	ctx.translate(centerX, centerY);
	ctx.rotate(angle);
	ctx.drawImage(image, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
	ctx.restore();
}
