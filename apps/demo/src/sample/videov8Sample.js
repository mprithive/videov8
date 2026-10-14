import React, { useRef, useState } from 'react';
import { VideoV8 } from 'videov8';

const DEMO_VIDEO = new URL('../videos/demo_portrait_4k_25.mp4', import.meta.url).href;
const COOLERS_URL = new URL('../Patterns/coolers.png', import.meta.url).href;

/**
 * User effect. Must stay self-contained: no closures, no imports, no optional
 * chaining, no async/await. CRA transpiles this file, and these functions are
 * sent to the worker via Function.prototype.toString().
 *
 * setup runs once per worker and loads coolers.png. draw places it on the
 * FaceMesh eye landmarks (33 and 263), same placement as coolersEffect.js.
 */
function setupCoolers(params) {
	return fetch(params.coolersUrl).then(function (response) {
		return response.blob();
	}).then(function (blob) {
		return createImageBitmap(blob);
	}).then(function (coolers) {
		return { coolers: coolers };
	});
}

function drawCoolersEffect(ctx, frame, inference, params) {
	if (inference.type !== 'face') return;

	const landmarks = inference.faceLandmarks && inference.faceLandmarks[0];
	const image = params.coolers;
	if (!image || !landmarks || !landmarks[33] || !landmarks[263]) return;

	const width = frame.width;
	const height = frame.height;
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

async function fileFromUrl(url, name) {
	const response = await fetch(url);
	if (!response.ok) throw new Error('Failed to fetch demo video');
	const blob = await response.blob();
	return new File([blob], name, { type: blob.type || 'video/mp4' });
}

export default function VideoV8Sample() {
	const playerRef = useRef(null);
	const [file, setFile] = useState(null);
	const [meta, setMeta] = useState(null);
	const [workerCount, setWorkerCount] = useState(2);
	const [task, setTask] = useState('face-landmarker');
	const [maxFrames, setMaxFrames] = useState('');
	const [status, setStatus] = useState('Pick a video, then run.');
	const [progress, setProgress] = useState(0);
	const [busy, setBusy] = useState(false);
	const [resultUrl, setResultUrl] = useState(null);
	const [error, setError] = useState(null);

	const load = async (nextFile) => {
		setError(null);
		setResultUrl((prev) => {
			if (prev) URL.revokeObjectURL(prev);
			return null;
		});
		setStatus('Loading video…');
		const player = new VideoV8({ workerCount: Number(workerCount) || 1 });
		playerRef.current = player;
		const ok = await player.loadVideo(nextFile);
		if (!ok) {
			setStatus('Failed to load video.');
			setMeta(null);
			return;
		}
		setFile(nextFile);
		setMeta(player.getMetadata());
		setStatus('Ready.');
	};

	const onFile = async (event) => {
		const next = event.target.files && event.target.files[0];
		if (next) await load(next);
	};

	const onDemo = async () => {
		setBusy(true);
		setError(null);
		try {
			setStatus('Fetching demo video…');
			const demo = await fileFromUrl(DEMO_VIDEO, 'demo_portrait_4k_25.mp4');
			await load(demo);
		} catch (err) {
			setError(err.message || String(err));
			setStatus('Demo load failed.');
		} finally {
			setBusy(false);
		}
	};

	const onRun = async () => {
		const player = playerRef.current;
		if (!player || !file) {
			setError('Load a video first.');
			return;
		}
		setBusy(true);
		setError(null);
		setProgress(0);
		setResultUrl((prev) => {
			if (prev) URL.revokeObjectURL(prev);
			return null;
		});
		const started = performance.now();
		try {
			// Recreate so a changed worker count takes effect.
			const next = new VideoV8({ workerCount: Number(workerCount) || 1 });
			const ok = await next.loadVideo(file);
			if (!ok) throw new Error('Failed to reload video');
			playerRef.current = next;

			setStatus(Number(workerCount) > 1 ? 'Running coordinator + inference workers…' : 'Running on the main thread…');
			const frameLimit = Number(maxFrames);
			const blob = await next.process({
				...(frameLimit > 0 ? { maxFrames: frameLimit } : {}),
				inferenceStride: 4,
				...(task === 'none' ? {} : {
					inference: {
						engine: 'mediapipe',
						task,
						delegate: Number(workerCount) > 1 ? 'CPU' : 'GPU',
						options: task === 'face-landmarker' ? { numFaces: 1 } : {},
					},
				}),
				workerEffect: {
					params: { coolersUrl: COOLERS_URL },
					setup: setupCoolers,
					draw: drawCoolersEffect,
				},
				onProgress: (data) => {
					setProgress(data.progress || 0);
					setStatus('Frame ' + data.current + ' / ' + data.total);
				},
			});
			const url = URL.createObjectURL(blob);
			setResultUrl(url);
			const seconds = ((performance.now() - started) / 1000).toFixed(1);
			setStatus('Done in ' + seconds + 's. ' + Math.round(blob.size / 1024) + ' KB');
		} catch (err) {
			setError(err.message || String(err));
			setStatus('Failed.');
		} finally {
			setBusy(false);
		}
	};

	return (
		<div style={styles.page}>
			<h1 style={styles.h1}>VideoV8 library sample</h1>
			<p style={styles.note}>
				Leave max frames empty to process every frame.
				Worker count 1 stays on the main thread. 2 or more uses the coordinator plus inference workers.
			</p>

			<label style={styles.row}>
				Video
				<input type="file" accept="video/*" onChange={onFile} disabled={busy} />
			</label>

			<label style={styles.row}>
				Workers
				<input
					type="number"
					min="1"
					max="8"
					value={workerCount}
					onChange={(e) => setWorkerCount(e.target.value)}
					disabled={busy}
					style={styles.num}
				/>
			</label>

			<label style={styles.row}>
				Task
				<select value={task} onChange={(e) => setTask(e.target.value)} disabled={busy}>
					<option value="none">none</option>
					<option value="face-landmarker">face-landmarker</option>
					<option value="image-segmenter">image-segmenter</option>
				</select>
			</label>

			<label style={styles.row}>
				Max frames
				<input
					type="number"
					min="1"
					placeholder="all"
					value={maxFrames}
					onChange={(e) => setMaxFrames(e.target.value)}
					disabled={busy}
					style={styles.num}
				/>
				<span style={styles.meta}>empty = full video</span>
			</label>

			<button type="button" onClick={onRun} disabled={busy || !file} style={styles.run}>
				{busy ? 'Working…' : 'Run'}
			</button>

			<div style={styles.bar}><div style={{ ...styles.fill, width: Math.round(progress * 100) + '%' }} /></div>
			<p style={styles.status}>{status}</p>
			{error ? <p style={styles.error}>{error}</p> : null}
			{meta ? (
				<p style={styles.meta}>
					{meta.width}×{meta.height} · {meta.frameRate.toFixed(2)} fps · {meta.duration.toFixed(2)}s
				</p>
			) : null}
			{resultUrl ? (
				<video src={resultUrl} controls autoPlay loop style={styles.video} />
			) : null}
		</div>
	);
}

const styles = {
	page: {
		maxWidth: 720,
		margin: '32px auto',
		padding: 24,
		textAlign: 'left',
		fontFamily: 'sans-serif',
		color: '#111',
		background: '#f6f6f6',
		borderRadius: 12,
		paddingRight: 280,
		position: 'relative',
		zIndex: 1,
	},
	h1: { fontSize: 22, marginBottom: 8 },
	note: { color: '#444', fontSize: 14, lineHeight: 1.4 },
	row: { display: 'flex', gap: 12, alignItems: 'center', margin: '10px 0' },
	num: { width: 72 },
	run: { marginTop: 8, padding: '8px 16px' },
	bar: { marginTop: 16, height: 8, background: '#e5e5e5', borderRadius: 4, overflow: 'hidden' },
	fill: { height: '100%', background: '#111' },
	status: { fontSize: 14 },
	error: { color: '#b00020', fontSize: 14 },
	meta: { fontSize: 13, color: '#555' },
	video: { width: '100%', marginTop: 12, background: '#000' },
};
