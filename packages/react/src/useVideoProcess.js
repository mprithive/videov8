import { useCallback, useEffect, useRef, useState } from 'react';
import { useVideoV8 } from './useVideoV8.js';

/**
 * Same player as useVideoV8, plus progress, status, and an object URL for the
 * encoded video. The object URL is revoked when the component unmounts.
 *
 * workerEffect.draw is still sent to the worker as source text, so it must
 * stay self-contained. Do not close over React state inside draw or setup.
 */
export function useVideoProcess(options = {}) {
	const player = useVideoV8(options);
	const [progress, setProgress] = useState(0);
	const [status, setStatus] = useState('idle');
	const [error, setError] = useState(null);
	const [resultUrl, setResultUrl] = useState(null);
	const urlRef = useRef(null);

	useEffect(() => {
		return () => {
			if (urlRef.current) URL.revokeObjectURL(urlRef.current);
		};
	}, []);

	const loadVideo = useCallback((file) => player.loadVideo(file), [player]);

	const process = useCallback(async (processOptions = {}) => {
		setStatus('running');
		setError(null);
		setProgress(0);
		try {
			const blob = await player.process({
				...processOptions,
				onProgress: (data) => {
					setProgress(data.progress || 0);
					if (processOptions.onProgress) processOptions.onProgress(data);
				},
			});
			if (urlRef.current) URL.revokeObjectURL(urlRef.current);
			const url = URL.createObjectURL(blob);
			urlRef.current = url;
			setResultUrl(url);
			setStatus('done');
			return blob;
		} catch (err) {
			setError(err);
			setStatus('error');
			throw err;
		}
	}, [player]);

	const getMetadata = useCallback(() => player.getMetadata(), [player]);

	return { player, loadVideo, process, progress, status, error, resultUrl, getMetadata };
}
