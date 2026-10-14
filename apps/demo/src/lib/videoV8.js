import { Input, ALL_FORMATS, BlobSource, VideoSampleSink, Output, BufferTarget, Mp4OutputFormat, CanvasSource, QUALITY_HIGH } from 'mediabunny';

/**
 * VideoV8 - A video player library using MediaBunny
 * Provides methods to load and play video files on HTML5 canvas
 */
class VideoV8 {
	constructor() {
		this.input = null;
		this.videoTrack = null;
		this.sink = null;
		this.isPlaying = false;
		this.currentTime = 0;
		this.duration = 0;
		this.frameRate = 24; // default frame rate
		this.playbackRate = 1.0;
		this.animationFrameId = null;
		this.onTimeUpdate = null;
		this.onEnded = null;
		this.onError = null;
		this.frameCache = new Map(); // Cache decoded frames
		this.maxFrameCacheSize = 10; // Keep last 10 frames in cache
	}

	/**
	 * Load a video file
	 * @param {File} file - The video file to load
	 * @returns {Promise<boolean>} True if video loaded successfully
	 */
	async loadVideo(file) {
		try {
			// Create input from the file
			this.input = new Input({
				formats: ALL_FORMATS,
				source: new BlobSource(file),
			});

			// Get the primary video track
			this.videoTrack = await this.input.getPrimaryVideoTrack();

			if (!this.videoTrack) {
				throw new Error('No video track found in the file');
			}

			// Check if the video can be decoded
			const canDecode = await this.videoTrack.canDecode();
			if (!canDecode) {
				throw new Error('Video codec is not supported');
			}

			// Create a video sample sink for decoding
			this.sink = new VideoSampleSink(this.videoTrack);

			// Get video metadata
			this.duration = await this.videoTrack.computeDuration();
			this.width = await this.videoTrack.getDisplayWidth();
			this.height = await this.videoTrack.getDisplayHeight();

			// Estimate frame rate
			const packetStats = await this.videoTrack.computePacketStats(100);
			this.frameRate = packetStats.averagePacketRate || 24;

			console.log('Video loaded successfully', {
				duration: this.duration,
				width: this.width,
				height: this.height,
				frameRate: this.frameRate,
			});

			return true;
		} catch (error) {
			console.error('Failed to load video:', error);
			if (this.onError) {
				this.onError(error);
			}
			return false;
		}
	}

	/**
	 * Get video metadata
	 * @returns {Object} Metadata object
	 */
	getMetadata() {
		return {
			duration: this.duration,
			width: this.width,
			height: this.height,
			frameRate: this.frameRate,
			currentTime: this.currentTime,
			isPlaying: this.isPlaying,
		};
	}

	/**
	 * Extract a thumbnail from the video at a specific time
	 * @param {number} time - Time in seconds (default: 0)
	 * @returns {Promise<HTMLCanvasElement>} Canvas element with the thumbnail
	 */
	async getThumbnail(time = 0) {
		if (!this.sink || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}

		try {
			// Clamp time to video duration
			time = Math.max(0, Math.min(time, this.duration));

			// Get the video sample at the specified time
			const sample = await this.sink.getSample(time);

			// Create a canvas for the thumbnail
			const canvas = document.createElement('canvas');
			canvas.width = this.width;
			canvas.height = this.height;

			const ctx = canvas.getContext('2d');
			if (!ctx) {
				throw new Error('Unable to get 2D context from canvas');
			}

			// Draw the sample to the canvas
			sample.draw(ctx, 0, 0);

			// Close the sample to prevent memory leaks
			sample.close();

			return canvas;
		} catch (error) {
			console.error('Error extracting thumbnail:', error);
			throw error;
		}
	}

	/**
	 * Process video frames with a custom renderer function
	 * @deprecated Use processAndEncodeFramesStreaming. This keeps every frame in memory.
	 * @param {Function} renderFunction - Function(ctx, frameIndex, time) that processes each frame
	 * @param {Object} options - Processing options
	 * @returns {Promise<Array>} Array of processed canvas elements
	 */
	async processFrames(renderFunction, options = {}) {
		if (!this.sink || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}

		const {
			maxFrames = null, // null = all frames
			startTime = 0,
			endTime = this.duration,
			onProgress = null,
		} = options;

		const processedFrames = [];
		const frameInterval = 1 / this.frameRate;
		let frameCount = 0;

		try {
			for (let time = startTime; time < endTime; time += frameInterval) {
				// Check if we've reached max frames
				if (maxFrames && frameCount >= maxFrames) {
					break;
				}

				// Get the video sample at this time
				const sample = await this.sink.getSample(time);

				// Create a canvas for this frame
				const canvas = document.createElement('canvas');
				canvas.width = this.width;
				canvas.height = this.height;

				const ctx = canvas.getContext('2d');
				if (!ctx) {
					throw new Error('Unable to get 2D context from canvas');
				}

				// Draw the original frame
				sample.draw(ctx, 0, 0);

				// Call the render function to allow custom processing
				await renderFunction(ctx, frameCount, time, {
					width: this.width,
					height: this.height,
					duration: this.duration,
				});

				// Store the processed frame
				processedFrames.push({
					canvas,
			});

			// Close the sample to prevent memory leaks
			sample.close();

			// Call progress callback
			if (onProgress) {
				const progress = (time - startTime) / (endTime - startTime);
				onProgress({
					current: frameCount + 1,
					progress: Math.min(1, progress),
					time,
				});
			}

			frameCount++;
			}

			return processedFrames;
		} catch (error) {
			console.error('Error processing frames:', error);
			throw error;
		}
	}

	/**
	 * Preview processed frame
	 * @deprecated Unused. Remove in the next version.
	 * @param {Function} renderFunction - Function to process the frame
	 * @param {number} time - Time in seconds to preview
	 * @returns {Promise<HTMLCanvasElement>} Processed canvas
	 */
	async previewProcessedFrame(renderFunction, time = 0) {
		if (!this.sink || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}

		try {
			time = Math.max(0, Math.min(time, this.duration));

			// Get the video sample
			const sample = await this.sink.getSample(time);

			// Create canvas
			const canvas = document.createElement('canvas');
			canvas.width = this.width;
			canvas.height = this.height;

			const ctx = canvas.getContext('2d');
			if (!ctx) {
				throw new Error('Unable to get 2D context from canvas');
			}

			// Draw original frame
			sample.draw(ctx, 0, 0);

			// Apply render function
			await renderFunction(ctx, 0, time, {
				width: this.width,
				height: this.height,
				duration: this.duration,
			});

			// Close sample
			sample.close();

			return canvas;
		} catch (error) {
			console.error('Error previewing processed frame:', error);
			throw error;
		}
	}

	/**
	 * Encode processed frames using MediaBunny
	 * @deprecated Use processAndEncodeFramesStreaming. This keeps every frame in memory.
	 * @param {Array} processedFrames - Array of processed frame objects
	 * @param {Object} options - Encoding options
	 * @returns {Promise<Blob>} Video file blob
	 */
	async encodeFramesSimple(processedFrames, options = {}) {
		if (!processedFrames || processedFrames.length === 0) {
			throw new Error('No frames to encode');
		}

		const { onProgress = null } = options;

		try {
			const firstFrame = processedFrames[0];
			const output = new Output({
				format: new Mp4OutputFormat(),
				target: new BufferTarget(),
			});

			// Create a dummy canvas with the first frame dimensions
			const dummyCanvas = document.createElement('canvas');
			dummyCanvas.width = firstFrame.canvas.width;
			dummyCanvas.height = firstFrame.canvas.height;

			const videoSource = new CanvasSource(dummyCanvas, {
				codec: 'avc',
				bitrate: QUALITY_HIGH,
			});

			output.addVideoTrack(videoSource);
			await output.start();

			const frameRate = this.frameRate || 24;
			const frameDuration = 1 / frameRate;

			// Add each processed frame
			for (let i = 0; i < processedFrames.length; i++) {
				const frame = processedFrames[i];
				const timestamp = i * frameDuration;

				// Copy the processed frame canvas to the dummy canvas
				const ctx = dummyCanvas.getContext('2d');
				ctx.drawImage(frame.canvas, 0, 0);

				await videoSource.add(timestamp, frameDuration);

				if (onProgress) {
					const progress = (i + 1) / processedFrames.length;
					onProgress({
						current: i + 1,
						total: processedFrames.length,
						progress,
					});
				}
			}

			await output.finalize();
			const buffer = output.target.buffer;
			const blob = new Blob([buffer], { type: 'video/mp4' });

			return blob;
		} catch (error) {
			console.error('Error encoding frames:', error);
			throw error;
		}
	}

	/**
	 * Process and encode video frames in a streaming fashion (frame-by-frame, no intermediate storage)
	 * Memory efficient: decodes, processes, encodes, and releases each frame immediately
	 * @param {Function} renderFunction - Function(ctx, frameIndex, time, frameInfo) that processes each frame
	 * @param {Object} options - Processing and encoding options
	 * @returns {Promise<Blob>} Encoded video file blob
	 */
	async processAndEncodeFramesStreaming(renderFunction, options = {}) {
		if (!this.sink || !this.videoTrack) {
			throw new Error('Video not loaded. Call loadVideo first.');
		}

		const {
			maxFrames = null,
			startTime = 0,
			endTime = this.duration,
			onProgress = null,
		} = options;

		try {
			// Initialize output encoder
			const output = new Output({
				format: new Mp4OutputFormat(),
				target: new BufferTarget(),
			});

			// Create a single reusable canvas
			const canvas = document.createElement('canvas');
			canvas.width = this.width;
			canvas.height = this.height;

			const videoSource = new CanvasSource(canvas, {
				codec: 'avc',
				bitrate: QUALITY_HIGH,
			});

			output.addVideoTrack(videoSource);
			await output.start();

			const frameInterval = 1 / this.frameRate;
			const frameDuration = 1 / this.frameRate;
			let frameCount = 0;
			const totalFrames = Math.ceil((endTime - startTime) * this.frameRate);

			// Process frames one at a time and encode immediately
			for (let time = startTime; time < endTime; time += frameInterval) {
				// Check if we've reached max frames
				if (maxFrames && frameCount >= maxFrames) {
					break;
				}

				// Decode frame
				const sample = await this.sink.getSample(time);

				// Get canvas context
				const ctx = canvas.getContext('2d');
				if (!ctx) {
					throw new Error('Unable to get 2D context from canvas');
				}

				// Draw original frame
				sample.draw(ctx, 0, 0);

				// Process frame with custom render function
				await renderFunction(ctx, frameCount, time, {
					width: this.width,
					height: this.height,
					duration: this.duration,
				});

				// Add processed frame to encoder
				const timestamp = frameCount * frameDuration;
				await videoSource.add(timestamp, frameDuration);

				// Close sample immediately to release memory
				sample.close();

				// Report progress
				if (onProgress) {
					const processingProgress = (time - startTime) / (endTime - startTime);
					onProgress({
						current: frameCount + 1,
						total: totalFrames,
						progress: Math.min(1, processingProgress),
						time,
					});
				}

				frameCount++;
			}

			// Finalize encoding
			await output.finalize();
			const buffer = output.target.buffer;
			const blob = new Blob([buffer], { type: 'video/mp4' });

			console.log(`Streamed and encoded ${frameCount} frames successfully`);
			return blob;
		} catch (error) {
			console.error('Error processing and encoding frames:', error);
			throw error;
		}
	}
}

export default VideoV8;
