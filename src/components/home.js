import React, { useState, useRef } from 'react';
import './home.css';
import VideoV8 from '../lib/videoV8';
import VideoV8Multithreaded from '../lib/videoV8_multithreaded';
import { applyPatternFill, createSegmentationAux, loadPatternBitmaps, pickPatternIndex, SELFIE_SEGMENTER_MODEL } from '../lib/segmentationEffect';

// ── Worker count ─────────────────────────────────────────────────────────────
// Change this number to experiment. Each worker loads its own FaceLandmarker
// WASM runtime (~100 MB RAM each). Recommended range: 2–6.
// Your machine has navigator.hardwareConcurrency cores available.
const WORKER_COUNT = Math.min(navigator.hardwareConcurrency || 4, 8);
const TARGET_AI_INFERENCE_FRAMES = 160;
const APPROX_DEVICE_MEMORY_GB = navigator.deviceMemory || 8;
const MEMORY_BUDGET_FRACTION = 0.4;
// renderStride = 1 renders every frame (no frames skipped / interpolated) and
// unlocks the draw-in-worker pipeline: each worker does AI inference AND effect
// compositing in parallel, so the main thread only decodes and encodes.
// Increase (e.g. 3) to trade quality for speed via keyframe interpolation.
const RENDER_FRAME_STRIDE = 1;
const AI_EFFECT_FACEMESH = 'facemesh';
const AI_EFFECT_SEGMENTATION = 'segmentation';

function createPipelineState(workerCount) {
  return {
    workers: Array.from({ length: workerCount }, (_, id) => ({
      id,
      status: 'idle',
      pending: 0,
      lastFrom: null,
      lastTo: null,
      processed: 0,
    })),
    decoder: { active: false, decoded: 0 },
    encoder: { active: false, frame: 0, encoded: 0 },
  };
}

function Home() {
  const [videoFile, setVideoFile] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [thumbnailUrl, setThumbnailUrl] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [downloadFileName, setDownloadFileName] = useState(null);
  const [totalFrames, setTotalFrames] = useState(0);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  const [useMultithreaded, setUseMultithreaded] = useState(false);
  const [aiEffect, setAiEffect] = useState(AI_EFFECT_FACEMESH);
  const [exportStartTime, setExportStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [showBts, setShowBts] = useState(false);
  const [pipelineView, setPipelineView] = useState(() => createPipelineState(WORKER_COUNT));
  const fileInputRef = useRef(null);
  const playerRef = useRef(null);
  const pipelineRef = useRef(createPipelineState(WORKER_COUNT));
  const pipelineFlushRef = useRef(false);

  const flushPipelineView = () => {
    if (pipelineFlushRef.current) return;
    pipelineFlushRef.current = true;
    requestAnimationFrame(() => {
      pipelineFlushRef.current = false;
      const next = pipelineRef.current;
      setPipelineView({
        decoder: { ...next.decoder },
        encoder: { ...next.encoder },
        workers: next.workers.map((worker) => ({ ...worker })),
      });
    });
  };

  const handlePipelineEvent = (event) => {
    const state = pipelineRef.current;
    const worker = event.workerIndex != null ? state.workers[event.workerIndex] : null;

    switch (event.type) {
      case 'pipeline-ready':
        pipelineRef.current = createPipelineState(event.workerCount || WORKER_COUNT);
        break;
      case 'decoder-start':
        state.decoder.active = true;
        break;
      case 'decoder-idle':
        state.decoder.active = false;
        if (event.decoded != null) state.decoder.decoded = event.decoded;
        break;
      case 'worker-receive':
        if (worker) {
          worker.status = 'processing';
          worker.pending += event.frameCount || 0;
          worker.lastFrom = event.from;
          worker.lastTo = event.to;
        }
        break;
      case 'worker-processed':
        if (worker) {
          worker.status = worker.pending > 0 ? 'handoff' : 'idle';
          worker.processed += event.frameCount || 0;
          worker.lastFrom = event.from;
          worker.lastTo = event.to;
        }
        break;
      case 'encoder-start':
        state.encoder.active = true;
        state.encoder.frame = event.frameIndex ?? state.encoder.frame;
        break;
      case 'encoder-frame':
        state.encoder.active = true;
        state.encoder.frame = event.frameIndex ?? state.encoder.frame;
        state.encoder.encoded += 1;
        if (worker && worker.pending > 0) {
          worker.pending -= 1;
          if (worker.pending === 0) worker.status = 'idle';
        }
        break;
      case 'encoder-idle':
        state.encoder.active = false;
        break;
      default:
        break;
    }

    flushPipelineView();
  };

  // Sample render function that draws circles on frames
  const sampleRenderFunction = async (ctx, frameIndex, time, frameInfo) => {
    const { width, height } = frameInfo;

    // Draw a circle that changes over time
    ctx.fillStyle = 'rgba(127, 216, 190, 0.6)';
    ctx.beginPath();
    
    // Circle radius grows and shrinks over time
    const radius = 30 + 20 * Math.sin(time * 2 * Math.PI);
    const x = width / 2;
    const y = height / 2;
    
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();

    // Add text showing frame info
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px Arial';
    ctx.fillText(`Frame: ${frameIndex} | Time: ${time.toFixed(2)}s`, 10, 30);
  };

  // AI-powered cooler rendering function using MediaPipe FaceMesh
  // Detects face landmarks and draws cool sunglasses on detected faces
  const aiCoolerRenderFunction = async (ctx, frameIndex, time, frameInfo) => {
    const { width, height } = frameInfo;

    try {
      // Initialize FaceMesh only once (cache it globally)
      if (!window.faceMeshInstance) {
        // Load MediaPipe drawing utilities and FaceMesh
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js';
          script.onload = () => {
            // Initialize FaceMesh
            window.faceMeshInstance = new window.FaceMesh({
              locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
            });
            window.faceMeshInstance.setOptions({
              maxNumFaces: 1,
              refineLandmarks: true,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5,
            });
            resolve();
          };
          script.onerror = () => reject(new Error('Failed to load FaceMesh'));
          document.body.appendChild(script);
        });
      }

      const faceMesh = window.faceMeshInstance;
      const canvas = ctx.canvas;

      // Process the canvas with FaceMesh
      const results = await new Promise((resolve) => {
        const tempResults = { multiFaceLandmarks: [] };
        faceMesh.onResults((res) => {
          tempResults.multiFaceLandmarks = res.multiFaceLandmarks || [];
          resolve(tempResults);
        });
        faceMesh.send({ image: canvas });
      });

      // Draw coolers if face is detected
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];

        // Key eye landmarks (normalized coordinates)
        const leftEyeOuter = landmarks[33];
        const leftEyeInner = landmarks[133];
        const leftEyeTop = landmarks[160];
        const leftEyeBottom = landmarks[145];

        const rightEyeOuter = landmarks[362];
        const rightEyeInner = landmarks[263];
        const rightEyeTop = landmarks[387];
        const rightEyeBottom = landmarks[374];

        // Convert normalized coordinates to pixel coordinates
        const toPixelX = (x) => x * width;
        const toPixelY = (y) => y * height;

        // Calculate eye centers and radii - LARGER for cooler look
        const leftEyeCenterX = (toPixelX(leftEyeInner.x) + toPixelX(leftEyeOuter.x)) / 2;
        const leftEyeCenterY = (toPixelY(leftEyeTop.y) + toPixelY(leftEyeBottom.y)) / 2;
        const leftEyeRadiusX = Math.abs(toPixelX(leftEyeOuter.x) - toPixelX(leftEyeInner.x)) / 2 + 15;
        const leftEyeRadiusY = Math.abs(toPixelY(leftEyeBottom.y) - toPixelY(leftEyeTop.y)) / 2 + 12;

        const rightEyeCenterX = (toPixelX(rightEyeInner.x) + toPixelX(rightEyeOuter.x)) / 2;
        const rightEyeCenterY = (toPixelY(rightEyeTop.y) + toPixelY(rightEyeBottom.y)) / 2;
        const rightEyeRadiusX = Math.abs(toPixelX(rightEyeOuter.x) - toPixelX(rightEyeInner.x)) / 2 + 15;
        const rightEyeRadiusY = Math.abs(toPixelY(rightEyeBottom.y) - toPixelY(rightEyeTop.y)) / 2 + 12;

        // Draw cooler frames with dark, trendy style
        ctx.strokeStyle = '#1a1a1a'; // Very dark/black frame color
        ctx.lineWidth = 4; // Thicker frames for cooler look
        
        // Dark gradient lens for sunglasses effect
        const gradient = ctx.createLinearGradient(0, leftEyeCenterY - leftEyeRadiusY, 0, leftEyeCenterY + leftEyeRadiusY);
        gradient.addColorStop(0, 'rgba(40, 40, 40, 0.85)');
        gradient.addColorStop(0.5, 'rgba(20, 20, 20, 0.9)');
        gradient.addColorStop(1, 'rgba(40, 40, 40, 0.85)');
        ctx.fillStyle = gradient;

        // Left cooler lens
        ctx.beginPath();
        ctx.ellipse(leftEyeCenterX, leftEyeCenterY, leftEyeRadiusX, leftEyeRadiusY, 0, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Right cooler lens
        ctx.beginPath();
        ctx.ellipse(rightEyeCenterX, rightEyeCenterY, rightEyeRadiusX, rightEyeRadiusY, 0, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Bridge connecting the lenses - thicker and cooler
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(leftEyeCenterX + leftEyeRadiusX, leftEyeCenterY);
        ctx.lineTo(rightEyeCenterX - rightEyeRadiusX, rightEyeCenterY);
        ctx.stroke();

        // Add shine effect on left lens
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.beginPath();
        ctx.ellipse(leftEyeCenterX - leftEyeRadiusX / 3, leftEyeCenterY - leftEyeRadiusY / 2.5, leftEyeRadiusX / 4, leftEyeRadiusY / 4, -0.3, 0, 2 * Math.PI);
        ctx.fill();

        // Add shine effect on right lens
        ctx.beginPath();
        ctx.ellipse(rightEyeCenterX - rightEyeRadiusX / 3, rightEyeCenterY - rightEyeRadiusY / 2.5, rightEyeRadiusX / 4, rightEyeRadiusY / 4, -0.3, 0, 2 * Math.PI);
        ctx.fill();

        // Draw text overlay with detection info
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Arial';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
        ctx.shadowBlur = 4;
        ctx.fillText(`Frame: ${frameIndex} | Cooler Applied`, 10, 30);
        ctx.shadowColor = 'transparent';
      } else {
        // Draw text if no face detected
        ctx.fillStyle = '#ffcccc';
        ctx.font = 'bold 14px Arial';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
        ctx.shadowBlur = 4;
        ctx.fillText(`Frame: ${frameIndex} | No Face Detected`, 10, 30);
        ctx.shadowColor = 'transparent';
      }
    } catch (error) {
      console.error('Error in FaceMesh cooler render:', error);
      // Draw error message on canvas
      ctx.fillStyle = '#ff6666';
      ctx.font = 'bold 14px Arial';
      ctx.fillText('Error processing face', 10, 30);
    }
  };

  // AI-powered pattern fill: selfie mask + random src/Patterns image every 0.5s
  const aiSegmentationRenderFunction = async (ctx, frameIndex, time, frameInfo) => {
    const { width, height } = frameInfo;

    try {
      if (!window.imageSegmenterInstance) {
        const { ImageSegmenter, FilesetResolver } = await import('@mediapipe/tasks-vision');
        const wasmBasePath = `${window.location.origin}${process.env.PUBLIC_URL || ''}/mediapipe-wasm`;
        const filesetResolver = await FilesetResolver.forVisionTasks(wasmBasePath);
        window.imageSegmenterInstance = await ImageSegmenter.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: SELFIE_SEGMENTER_MODEL,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          outputCategoryMask: true,
          outputConfidenceMasks: false,
        });
        window.segmentationAux = createSegmentationAux();
        window.patternBitmaps = await loadPatternBitmaps();
        window.segmentationDetectCanvas = document.createElement('canvas');
        window.segmentationDetectCanvas.width = 256;
        window.segmentationDetectCanvas.height = Math.max(1, Math.round(256 * height / width));
        window.segmentationDetectCtx = window.segmentationDetectCanvas.getContext('2d');
      }

      const detectCtx = window.segmentationDetectCtx;
      detectCtx.clearRect(0, 0, detectCtx.canvas.width, detectCtx.canvas.height);
      detectCtx.drawImage(ctx.canvas, 0, 0, detectCtx.canvas.width, detectCtx.canvas.height);

      const timestampMs = Math.round(time * 1000);
      let copiedMask = null;
      window.imageSegmenterInstance.segmentForVideo(detectCtx.canvas, timestampMs, (result) => {
        if (!result || !result.categoryMask) return;
        const src = result.categoryMask.getAsUint8Array();
        const data = new Uint8Array(src.length);
        data.set(src);
        copiedMask = {
          width: result.categoryMask.width,
          height: result.categoryMask.height,
          getAsUint8Array: () => data,
        };
      });

      const patterns = window.patternBitmaps || [];
      const pattern = patterns.length > 0
        ? patterns[pickPatternIndex(timestampMs, patterns.length)]
        : null;
      const frameBitmap = await createImageBitmap(ctx.canvas);
      applyPatternFill(ctx, frameBitmap, copiedMask, window.segmentationAux, pattern, width, height);
      frameBitmap.close();

      ctx.font = 'bold 14px Arial';
      ctx.shadowColor = 'rgba(0,0,0,0.7)';
      ctx.shadowBlur = 4;
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`Frame: ${frameIndex} | Pattern`, 10, 30);
      ctx.shadowColor = 'transparent';
    } catch (error) {
      console.error('Error in segmentation pattern render:', error);
      ctx.fillStyle = '#ff6666';
      ctx.font = 'bold 14px Arial';
      ctx.fillText('Error processing segmentation', 10, 30);
    }
  };

  // Handle file upload
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.includes('video/mp4') && !file.type.includes('video')) {
      setError('Please upload a valid video file (MP4)');
      return;
    }

    try {
      setError(null);
      setVideoFile(file);
      setThumbnailUrl(null);
      setProgress(0);
      setDownloadUrl(null);
      setDownloadFileName(null);
      setTotalFrames(0);
      setCurrentFrameIndex(0);
      setProcessingStatus('');

      // Initialize VideoV8 player (single-threaded or multithreaded)
      const player = useMultithreaded
        ? new VideoV8Multithreaded({
            workerCount: WORKER_COUNT,
            approxDeviceMemoryGB: APPROX_DEVICE_MEMORY_GB,
            memoryBudgetFraction: MEMORY_BUDGET_FRACTION,
            aiEffect,
          })
        : new VideoV8();

      player.onError = (err) => {
        setError(err.message);
      };

      // Load the video
      const loaded = await player.loadVideo(file);
      if (!loaded) {
        setError('Failed to load video');
        return;
      }

      playerRef.current = player;

      // Extract and display thumbnail
      const thumbnailCanvas = await player.getThumbnail(0);
      const thumbUrl = thumbnailCanvas.toDataURL('image/png');
      setThumbnailUrl(thumbUrl);

      // Store metadata
      setMetadata(player.getMetadata());
    } catch (err) {
      console.error('Error loading video:', err);
      setError(err.message);
    }
  };

  // Handle export/processing
  const handleExport = async () => {
    if (!playerRef.current || !videoFile) return;

    let timerInterval = null;

    try {
      setError(null);
      setIsProcessing(true);
      setProgress(0);
      setDownloadUrl(null);
      setProcessingStatus('Initializing...');

      // Start timer
      const startTime = Date.now();
      setExportStartTime(startTime);
      setElapsedTime(0);
      timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        setElapsedTime(elapsed);
      }, 100);

      let player = playerRef.current;

      // Check if player mode matches desired mode, reinitialize if needed
      const isMultithreadedInstance = player.constructor.name === 'VideoV8Multithreaded';
      if (isMultithreadedInstance !== useMultithreaded) {
        console.log(`Mode changed: reinitializing player (${useMultithreaded ? 'multithreaded' : 'single-threaded'})`);
        
        // Create new player with correct mode
        player = useMultithreaded
          ? new VideoV8Multithreaded({
              workerCount: WORKER_COUNT,
              approxDeviceMemoryGB: APPROX_DEVICE_MEMORY_GB,
              memoryBudgetFraction: MEMORY_BUDGET_FRACTION,
              aiEffect,
            })
          : new VideoV8();

        player.onError = (err) => {
          setError(err.message);
        };

        // Reload the video
        const loaded = await player.loadVideo(videoFile);
        if (!loaded) {
          setError('Failed to reload video with new mode');
          setIsProcessing(false);
          return;
        }

        playerRef.current = player;
      }

      const frameRate = player.frameRate || 24;
      const duration = player.duration || 0;
      const totalFrameCount = Math.ceil(duration * frameRate);
      setTotalFrames(totalFrameCount);
      setCurrentFrameIndex(0);
      const workerSlots = useMultithreaded ? WORKER_COUNT : 1;
      pipelineRef.current = createPipelineState(workerSlots);
      setPipelineView(createPipelineState(workerSlots));
      setShowBts(false);

      // Process and encode ALL frames (memory efficient)
      const processingMode = useMultithreaded ? `AI parallel (${WORKER_COUNT} workers)` : 'streaming';
      console.log(`Starting video processing: ${totalFrameCount} frames (${processingMode})...`);
      setProcessingStatus(`Processing and encoding video (${processingMode})...`);

      let videoBlob;
      if (useMultithreaded) {
        // Multithreaded: faceWorker.js runs @mediapipe/tasks-vision FaceLandmarker inside each
        // worker thread — true parallel AI inference across 3 CPU cores.
        // The render function is not used here; workers own the full AI + drawing pipeline.
        videoBlob = await player.processAndEncodeFramesWithWorkers(null, {
          targetInferenceFrames: TARGET_AI_INFERENCE_FRAMES,
          renderStride: RENDER_FRAME_STRIDE,
          aiEffect,
          onProgress: (data) => {
            setCurrentFrameIndex(data.current);
            const currentElapsed = Math.floor((Date.now() - startTime) / 1000);
            const effectLabel = aiEffect === AI_EFFECT_SEGMENTATION ? 'Pattern fill' : 'FaceMesh';
            setProcessingStatus(`${effectLabel} Worker: frame ${data.current}/${data.total} - Elapsed: ${currentElapsed}s`);
            setProgress(data.progress);
          },
          onPipelineEvent: handlePipelineEvent,
        });
      } else {
        // Single-threaded: FaceMesh runs on the main thread (requires window/DOM).
        videoBlob = await player.processAndEncodeFramesStreaming(
          async (ctx, frameIndex, time, frameInfo) => {
            setCurrentFrameIndex(frameIndex);
            if (aiEffect === AI_EFFECT_SEGMENTATION) {
              await aiSegmentationRenderFunction(ctx, frameIndex, time, frameInfo);
            } else {
              await aiCoolerRenderFunction(ctx, frameIndex, time, frameInfo);
            }
          },
          {
            onProgress: (data) => {
              setCurrentFrameIndex(data.current);
              const currentElapsed = Math.floor((Date.now() - startTime) / 1000);
              setProcessingStatus(`Processing & encoding frame ${data.current}/${data.total} (${processingMode}) - Elapsed: ${currentElapsed}s`);
              setProgress(data.progress);
              handlePipelineEvent({ type: 'decoder-start' });
              handlePipelineEvent({ type: 'worker-receive', workerIndex: 0, frameCount: 1, from: data.current, to: data.current });
              handlePipelineEvent({ type: 'worker-processed', workerIndex: 0, frameCount: 1, from: data.current, to: data.current });
              handlePipelineEvent({ type: 'encoder-start', frameIndex: data.current });
              handlePipelineEvent({ type: 'encoder-frame', workerIndex: 0, frameIndex: data.current });
              handlePipelineEvent({ type: 'encoder-idle' });
              handlePipelineEvent({ type: 'decoder-idle', decoded: data.current });
            },
          }
        );
      }

      // Create download URL
      const url = URL.createObjectURL(videoBlob);
      setDownloadUrl(url);
      
      // Create filename with timestamp
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
      const fileName = `rendered-video-${timestamp}.mp4`;
      setDownloadFileName(fileName);

      setProgress(1);
      const totalTime = Math.floor((Date.now() - startTime) / 1000);
      setProcessingStatus(`✓ Complete! Total time: ${totalTime}s. Ready to download.`);
      console.log('Video processing and encoding complete! Ready to download.');
    } catch (err) {
      console.error('Error processing frames:', err);
      setError(err.message);
      setProcessingStatus('✗ Error during processing');
    } finally {
      if (timerInterval) {
        clearInterval(timerInterval);
      }
      setIsProcessing(false);
    }
  };

  // Format time display
  const formatTime = (seconds) => {
    if (!seconds && seconds !== 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="home-container">
      <h1 className="home-title">VideoV8</h1>
      {videoFile && (
        <div className="home-content">
          <div className="upload-section">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/*"
              onChange={handleUpload}
              style={{ display: 'none' }}
            />
            <button
              className="upload-button"
              onClick={() => fileInputRef.current?.click()}
            >
              Upload
            </button>
          </div>
          <div className="divider"></div>
          <div className="preview-section">
            {thumbnailUrl && <img src={thumbnailUrl} alt="Video thumbnail" className="video-thumbnail" />}
            <div className="preview-overlay">
              <button
                className="export-button"
                onClick={handleExport}
                disabled={isProcessing}
              >
                {isProcessing ? `Processing... ${Math.round(progress * 100)}%` : 'Export'}
              </button>
            </div>
          </div>
        </div>
      )}

      {!videoFile && (
        <div className="home-content">
          <div className="upload-section">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/*"
              onChange={handleUpload}
              style={{ display: 'none' }}
            />
            <button
              className="upload-button"
              onClick={() => fileInputRef.current?.click()}
            >
              Upload
            </button>
          </div>
          <div className="divider"></div>
          <div className="preview-section">
            {!videoFile && <div className="preview-placeholder">Select a video to preview</div>}
          </div>
        </div>
      )}

      {videoFile && metadata && (
        <div className="video-info-section">
          <div className="video-details">
            <span className="file-name">{videoFile.name}</span>
            <span className="duration">Duration: {formatTime(metadata.duration)}</span>
            <span className="dimensions">{metadata.width} × {metadata.height}px</span>
          </div>
          <div className="processing-mode-toggle">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={useMultithreaded}
                onChange={(e) => setUseMultithreaded(e.target.checked)}
                disabled={isProcessing}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-text">
                {useMultithreaded ? '⚡ Multithreaded' : '⚙️ Single-threaded'}
              </span>
            </label>
            <div className="toggle-info">
              {useMultithreaded 
                ? `Parallel processing with ${WORKER_COUNT} workers (faster rendering)`
                : 'Single-threaded streaming mode'
              }
            </div>
          </div>
          <div className="ai-effect-picker">
            <label className="ai-effect-label" htmlFor="ai-effect-select">AI effect</label>
            <select
              id="ai-effect-select"
              className="ai-effect-select"
              value={aiEffect}
              onChange={(e) => setAiEffect(e.target.value)}
              disabled={isProcessing}
            >
              <option value={AI_EFFECT_FACEMESH}>FaceMesh coolers</option>
              <option value={AI_EFFECT_SEGMENTATION}>Pattern fill (segmentation)</option>
            </select>
            <p className="ai-effect-hint">
              {aiEffect === AI_EFFECT_SEGMENTATION
                ? 'Fills the character shape with a random image from src/Patterns every 0.5s.'
                : 'FaceMesh detects landmarks and draws sunglasses on the face.'}
            </p>
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>
      )}

      {isProcessing && processingStatus && (
        <div className="processing-status-section">
          <div className="processing-status-header">
            <h3>{showBts ? 'Behind the scenes' : 'Processing Status'}</h3>
            <label className="bts-toggle">
              <input
                type="checkbox"
                checked={showBts}
                onChange={(e) => setShowBts(e.target.checked)}
              />
              <span className="bts-toggle-slider"></span>
              <span className="bts-toggle-text">{showBts ? 'BTS on' : 'View BTS'}</span>
            </label>
          </div>

          {!showBts && (
            <div className="status-content">
              <p className="status-message">{processingStatus}</p>
              <p className="frame-count">
                Frame {currentFrameIndex + 1} / {totalFrames}
              </p>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress * 100}%` }}></div>
              </div>
              <p className="progress-percentage">{Math.round(progress * 100)}%</p>
            </div>
          )}

          {showBts && (
            <div className="bts-dashboard">
              <p className="bts-legend">
                {useMultithreaded
                  ? 'Blocks glow while they hold a frame. A worker stays lit until its batch is handed to the encoder.'
                  : 'Single-threaded: decode, AI inference, and encode all run on the main thread.'}
              </p>
              <div className="bts-workflow">
                <div className={`bts-node ${pipelineView.decoder.active ? 'bts-node-glow' : ''}`}>
                  <span className="bts-node-label">Decoder</span>
                  <span className="bts-node-meta">
                    {pipelineView.decoder.active ? 'Decoding chunk' : 'Idle'}
                  </span>
                  <span className="bts-node-count">{pipelineView.decoder.decoded} frames</span>
                </div>

                <div className="bts-arrow" aria-hidden="true">→</div>

                {useMultithreaded ? (
                  <div className="bts-workers">
                    {pipelineView.workers.map((worker) => {
                      const glowing = worker.status === 'processing' || worker.status === 'handoff';
                      const statusLabel = worker.status === 'processing'
                        ? 'Received · processing'
                        : worker.status === 'handoff'
                          ? 'Sending to encoder'
                          : 'Idle';
                      return (
                        <div
                          key={worker.id}
                          className={`bts-node bts-worker ${glowing ? 'bts-node-glow' : ''}`}
                        >
                          <span className="bts-node-label">Worker {worker.id + 1}</span>
                          <span className="bts-node-meta">{statusLabel}</span>
                          {worker.lastFrom != null && (
                            <span className="bts-node-count">
                              frames {worker.lastFrom}–{worker.lastTo}
                            </span>
                          )}
                          <span className="bts-node-count">{worker.processed} processed</span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="bts-node bts-ai-inference">
                    <span className="bts-node-label">AI Inference</span>
                    <span className="bts-node-meta">Main thread</span>
                  </div>
                )}

                <div className="bts-arrow" aria-hidden="true">→</div>

                <div className={`bts-node bts-encoder ${pipelineView.encoder.active ? 'bts-node-glow' : ''}`}>
                  <span className="bts-node-label">Encoder</span>
                  <span className="bts-node-meta">
                    {pipelineView.encoder.active ? `Encoding frame ${pipelineView.encoder.frame}` : 'Idle'}
                  </span>
                  <span className="bts-node-count">{pipelineView.encoder.encoded} encoded</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {downloadUrl && downloadFileName && (
        <div className="download-section">
          <h3>✓ Video Rendering Complete!</h3>
          <p>Your rendered video is ready to download.</p>
          <a
            href={downloadUrl}
            download={downloadFileName}
            className="download-button"
          >
            📥 Download {downloadFileName}
          </a>
        </div>
      )}
    </div>
  );
}

export default Home;
