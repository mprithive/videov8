import { useEffect, useState } from 'react';
import { useVideoProcess } from 'videov8-react';
import {
	EFFECT_ASSETS,
	SAMPLE_VIDEOS,
	drawEffect,
	fileFromSampleUrl,
	inferenceForEffect,
	setupEffect,
} from '../../shared/effects.js';

function Processor({ workerCount }) {
	const { loadVideo, process, progress, status, error, resultUrl } = useVideoProcess({
		workerCount: workerCount,
	});
	const [sampleUrl, setSampleUrl] = useState(SAMPLE_VIDEOS[0].url);
	const [uploadUrl, setUploadUrl] = useState(null);
	const [effect, setEffect] = useState('coolers');

	useEffect(() => {
		return () => {
			if (uploadUrl) URL.revokeObjectURL(uploadUrl);
		};
	}, [uploadUrl]);

	async function onSubmit(event) {
		event.preventDefault();
		const file = sampleUrl
			? await fileFromSampleUrl(sampleUrl)
			: event.target.elements.file.files[0];
		if (!file) return;
		const loaded = await loadVideo(file);
		if (!loaded) return;
		await process({
			inference: inferenceForEffect(effect),
			workerEffect: effect === 'none' ? undefined : {
				params: {
					effect: effect,
					coolersUrl: EFFECT_ASSETS.coolersUrl,
					patternUrls: EFFECT_ASSETS.patternUrls,
				},
				setup: setupEffect,
				draw: drawEffect,
			},
			inferenceStride: 1,
		});
	}

	function onFileChange(event) {
		const file = event.target.files[0];
		if (!file) return;
		if (uploadUrl) URL.revokeObjectURL(uploadUrl);
		setUploadUrl(URL.createObjectURL(file));
	}

	const statusText = error
		? (error.message || String(error))
		: status === 'running'
			? 'Encoding… ' + Math.round(progress * 100) + '%'
			: status === 'done'
				? 'Done'
				: 'Choose a video and run.';

	const sourceUrl = sampleUrl || uploadUrl;

	return (
		<form onSubmit={onSubmit}>
			<label>
				Video
				<select value={sampleUrl} onChange={(event) => setSampleUrl(event.target.value)}>
					{SAMPLE_VIDEOS.map((video) => (
						<option key={video.url} value={video.url}>{video.label}</option>
					))}
					<option value="">Upload a file</option>
				</select>
			</label>
			{sampleUrl === '' ? (
				<label>
					File
					<input name="file" type="file" accept="video/*" required onChange={onFileChange} />
				</label>
			) : null}
			<label>
				Effect
				<select value={effect} onChange={(event) => setEffect(event.target.value)}>
					<option value="coolers">coolers</option>
					<option value="pattern">pattern</option>
					<option value="none">none</option>
				</select>
			</label>
			<button type="submit" disabled={status === 'running'}>Run</button>
			<p>{statusText}</p>
			<progress value={progress} max="1" />
			<p className="caption">Source</p>
			{sourceUrl ? <video src={sourceUrl} controls playsInline /> : <video controls playsInline />}
			<p className="caption">Result</p>
			{resultUrl ? <video src={resultUrl} controls playsInline /> : <video controls playsInline />}
		</form>
	);
}

export default function App() {
	const [workerCount, setWorkerCount] = useState(2);

	return (
		<main>
			<h1>videov8-react</h1>
			<p className="lede">
				React example. Pick a clip from <code>apps/videos</code> and an effect that uses <code>apps/patterns</code>.
			</p>
			<label>
				Workers
				<select
					value={workerCount}
					onChange={(event) => setWorkerCount(Number(event.target.value))}
				>
					<option value={1}>1 (main thread)</option>
					<option value={2}>2</option>
					<option value={4}>4</option>
				</select>
			</label>
			<Processor key={workerCount} workerCount={workerCount} />
		</main>
	);
}
