/**
 * Inference task registry shared by the main-thread library and the worker.
 *
 * Add new models here. Each adapter owns the default model URL, the inference
 * resolution, how to build the MediaPipe task, and how to normalise its output
 * into a plain object handed to `workerEffect.draw`.
 */

export const DEFAULT_MODEL_URLS = {
	'face-landmarker':
		'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
	'image-segmenter':
		'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite',
};

const SUPPORTED_TASKS = Object.keys(DEFAULT_MODEL_URLS);

/** Result handed to workerEffect.draw when no model ran for this frame. */
export function emptyInferenceResult() {
	return {
		type: 'none',
		task: null,
		faceLandmarks: [],
		faceBlendshapes: [],
		categoryMask: null,
	};
}

/**
 * Build the adapter map. `vision` is the imported @mediapipe/tasks-vision module.
 * Kept as a factory so the heavy dependency is only imported where it's used.
 */
export function createInferenceAdapters(vision) {
	const { FaceLandmarker, ImageSegmenter } = vision;

	return {
		'face-landmarker': {
			detectWidth: 512,
			async create(resolver, { modelUrl, delegate, options }) {
				return FaceLandmarker.createFromOptions(resolver, {
					baseOptions: {
						modelAssetPath: modelUrl || DEFAULT_MODEL_URLS['face-landmarker'],
						delegate,
					},
					runningMode: 'VIDEO',
					numFaces: 1,
					outputFaceBlendshapes: false,
					outputFacialTransformationMatrixes: false,
					...options,
				});
			},
			infer(model, canvas, timestamp) {
				const result = model.detectForVideo(canvas, timestamp);
				return {
					type: 'facemesh',
					task: 'face-landmarker',
					faceLandmarks: result.faceLandmarks || [],
					faceBlendshapes: result.faceBlendshapes || [],
					categoryMask: null,
				};
			},
		},

		'image-segmenter': {
			detectWidth: 256,
			async create(resolver, { modelUrl, delegate, options }) {
				return ImageSegmenter.createFromOptions(resolver, {
					baseOptions: {
						modelAssetPath: modelUrl || DEFAULT_MODEL_URLS['image-segmenter'],
						delegate,
					},
					runningMode: 'VIDEO',
					outputCategoryMask: true,
					outputConfidenceMasks: false,
					...options,
				});
			},
			infer(model, canvas, timestamp) {
				let categoryMask = null;
				model.segmentForVideo(canvas, timestamp, (result) => {
					const mask = result.categoryMask;
					if (mask) {
						// Copy out of the WASM-owned buffer so it survives the frame
						// and is structured-cloneable.
						const src = mask.getAsUint8Array();
						const data = new Uint8Array(src.length);
						data.set(src);
						categoryMask = { width: mask.width, height: mask.height, data };
					}
				});
				return {
					type: 'imagesegmenter',
					task: 'image-segmenter',
					faceLandmarks: [],
					faceBlendshapes: [],
					categoryMask,
				};
			},
		},
	};
}

/**
 * Normalise + validate the `inference` option.
 * Omitted inference means "no model": frames are still encoded and workerEffect
 * still runs, with an empty inference result. A provided but unknown task throws.
 */
export function resolveInferenceConfig(inference, { defaultDelegate }) {
	if (!inference) return null;
	if (!inference.task) {
		throw new Error(
			`[VideoV8] "inference.task" is required when inference is set. Supported tasks: ${SUPPORTED_TASKS.join(', ')}`
		);
	}
	if (!SUPPORTED_TASKS.includes(inference.task)) {
		throw new Error(
			`[VideoV8] Unsupported inference task "${inference.task}". Supported: ${SUPPORTED_TASKS.join(', ')}`
		);
	}
	const engine = inference.engine || 'mediapipe';
	if (engine !== 'mediapipe') {
		throw new Error(`[VideoV8] Unsupported inference engine "${engine}" (only "mediapipe" in v1).`);
	}
	return {
		engine,
		task: inference.task,
		modelUrl: inference.modelUrl || DEFAULT_MODEL_URLS[inference.task],
		delegate: inference.delegate || defaultDelegate,
		options: inference.options || {},
	};
}
