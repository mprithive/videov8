import { useEffect, useRef } from 'react';
import { VideoV8 } from 'videov8';

/**
 * Holds one VideoV8 player for the lifetime of the component.
 * Options are read when the player is first created.
 */
export function useVideoV8(options = {}) {
	const playerRef = useRef(null);

	if (playerRef.current == null) {
		playerRef.current = new VideoV8(options);
	}

	useEffect(() => {
		const player = playerRef.current;
		return () => {
			if (player) player.terminateWorkers();
		};
	}, []);

	return playerRef.current;
}
