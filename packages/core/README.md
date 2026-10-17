# VideoV8

Browser video processing. Decode a video, optionally run a MediaPipe model, draw on each frame, and encode an MP4.

`videov8` is the library. `videov8-react` is a set of React hooks around the same class.

## Install

### Vanilla JavaScript

```bash
npm install videov8
```

### React

Install the hooks and React 18 or newer. `videov8-react` depends on `videov8`.

```bash
npm install videov8-react react react-dom
```

## Setup

Copy the MediaPipe wasm runtime into your app's public folder. The library loads it from `/mediapipe-wasm` by default.

```bash
mkdir -p public/mediapipe-wasm
cp -R node_modules/@mediapipe/tasks-vision/wasm/. public/mediapipe-wasm
```

Pass `wasmBasePath` to `new VideoV8()` if those files live somewhere else.

Use a bundler that understands `new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })`. Vite and Webpack 5 do. With Vite, keep the package out of dependency prebundling so the worker file still resolves:

```js
// vite.config.js
export default {
  optimizeDeps: {
    exclude: ['videov8', 'videov8-react'],
  },
  worker: { format: 'es' },
};
```

`workerCount: 1` runs decode, inference, drawing, and encode on the calling thread. `workerCount` greater than 1 moves that work into workers. Chrome, Firefox, and Safari 16.4+ can run the worker path. Older Safari should stay on `workerCount: 1`.

## Quick start — vanilla JavaScript

```js
import { VideoV8 } from 'videov8';

const player = new VideoV8({ workerCount: 2 });

const input = document.querySelector('#file');
input.addEventListener('change', async () => {
  const file = input.files[0];
  const loaded = await player.loadVideo(file);
  if (!loaded) return;

  const blob = await player.process({
    inference: { task: 'face-landmarker' }, // or 'image-segmenter'; omit to skip the model
    workerEffect: {
      params: { tint: 'rgba(255, 90, 40, 0.28)' },
      draw: function (ctx, frame, inference, params) {
        if (inference.type === 'facemesh') {
          var faces = inference.faceLandmarks || [];
          // draw from faces[i], using frame.width and frame.height
          return;
        }
        if (inference.type === 'imagesegmenter' && inference.categoryMask) {
          // inference.categoryMask is { width, height, data }
          return;
        }
        ctx.fillStyle = params.tint;
        ctx.fillRect(0, 0, frame.width, frame.height);
      },
    },
    onProgress: function (data) {
      console.log(Math.round(data.progress * 100) + '%');
    },
  });

  const url = URL.createObjectURL(blob);
  document.querySelector('#result').src = url;
  player.terminateWorkers();
});
```

`loadVideo(file)` accepts a `File` or `Blob`. `process()` returns an MP4 `Blob`.

`inference.task` is `face-landmarker` or `image-segmenter`. Inside `draw`, `inference.type` is `facemesh`, `imagesegmenter`, or `none` when no model ran.

## Draw and setup

On the worker path, `setup` and `draw` are rebuilt from `Function.prototype.toString()`. Keep them self-contained:

- No closures. Pass values through `params`.
- No `import` and no DOM. Load images inside `setup` with `fetch` and `createImageBitmap`.
- `setup` runs once per worker. Its return value is merged into `params`.
- `draw(ctx, frame, inference, params)` runs after inference. `ctx` already has the source frame on it.

`frame` is `{ frameIndex, timestamp, width, height }`.

## Examples in this repo

```bash
npm install
npm run start:core    # http://localhost:5173
npm start             # React example, http://localhost:5174
```

The core example imports `videov8`. The React example imports `videov8-react`. Both can run the coolers and pattern effects on the clips in `apps/videos`.
