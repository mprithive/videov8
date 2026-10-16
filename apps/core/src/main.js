import { VideoV8 } from 'videov8';
import {
	EFFECT_ASSETS,
	SAMPLE_VIDEOS,
	drawEffect,
	fileFromSampleUrl,
	inferenceForEffect,
	setupEffect,
} from '../../shared/effects.js';
import './style.css';

const form = document.querySelector('#form');
const sampleSelect = document.querySelector('#sample');
const fileLabel = document.querySelector('#file-label');
const fileInput = document.querySelector('#file');
const effectSelect = document.querySelector('#effect');
const workersSelect = document.querySelector('#workers');
const statusEl = document.querySelector('#status');
const progressEl = document.querySelector('#progress');
const sourceEl = document.querySelector('#source');
const resultEl = document.querySelector('#result');
const runButton = document.querySelector('#run');

let player = null;
let resultUrl = null;
let uploadUrl = null;

SAMPLE_VIDEOS.forEach(function (video) {
	const option = document.createElement('option');
	option.value = video.url;
	option.textContent = video.label;
	sampleSelect.appendChild(option);
});
const uploadOption = document.createElement('option');
uploadOption.value = '';
uploadOption.textContent = 'Upload a file';
sampleSelect.appendChild(uploadOption);

function setStatus(text) {
	statusEl.textContent = text;
}

function showSource(url) {
	sourceEl.src = url;
}

sampleSelect.addEventListener('change', function () {
	const sampleUrl = sampleSelect.value;
	fileLabel.classList.toggle('is-hidden', sampleUrl !== '');
	if (uploadUrl) {
		URL.revokeObjectURL(uploadUrl);
		uploadUrl = null;
	}
	if (sampleUrl) showSource(sampleUrl);
	else sourceEl.removeAttribute('src');
});

fileInput.addEventListener('change', function () {
	const file = fileInput.files && fileInput.files[0];
	if (!file) return;
	if (uploadUrl) URL.revokeObjectURL(uploadUrl);
	uploadUrl = URL.createObjectURL(file);
	showSource(uploadUrl);
});

showSource(sampleSelect.value);

form.addEventListener('submit', async (event) => {
	event.preventDefault();
	if (player) player.terminateWorkers();
	if (resultUrl) URL.revokeObjectURL(resultUrl);

	const workerCount = Number(workersSelect.value);
	const effect = effectSelect.value;
	player = new VideoV8({ workerCount });
	runButton.disabled = true;
	progressEl.value = 0;
	resultEl.removeAttribute('src');
	setStatus('Loading video…');

	try {
		const sampleUrl = sampleSelect.value;
		const file = sampleUrl
			? await fileFromSampleUrl(sampleUrl)
			: fileInput.files && fileInput.files[0];
		if (!file) throw new Error('Choose a video');

		const loaded = await player.loadVideo(file);
		if (!loaded) throw new Error('Could not load that video');

		const meta = player.getMetadata();
		setStatus(
			meta.width + '×' + meta.height + ' · ' + meta.frameRate.toFixed(2) + ' fps · encoding…'
		);

		const blob = await player.process({
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
			onProgress: function (data) {
				progressEl.value = data.progress || 0;
			},
		});

		resultUrl = URL.createObjectURL(blob);
		resultEl.src = resultUrl;
		setStatus('Done · ' + Math.round(blob.size / 1024) + ' KB');
	} catch (error) {
		setStatus(error && error.message ? error.message : String(error));
	} finally {
		runButton.disabled = false;
		if (player) player.terminateWorkers();
	}
});
