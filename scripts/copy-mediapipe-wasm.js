/**
 * Copy MediaPipe's wasm runtime into an example's public folder.
 * FilesetResolver loads these from /mediapipe-wasm at runtime.
 *
 * Usage (from an app directory):
 *   node ../../scripts/copy-mediapipe-wasm.js public/mediapipe-wasm
 */
import { cpSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const destArg = process.argv[2];
if (!destArg) {
	console.error('Usage: node copy-mediapipe-wasm.js <dest-dir>');
	process.exit(1);
}

let dir = process.cwd();
let wasmDir = null;
for (let i = 0; i < 6; i++) {
	const candidate = join(dir, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
	if (existsSync(candidate)) {
		wasmDir = candidate;
		break;
	}
	const parent = dirname(dir);
	if (parent === dir) break;
	dir = parent;
}

if (!wasmDir) {
	console.error('Could not find @mediapipe/tasks-vision/wasm. Run npm install from the repo root first.');
	process.exit(1);
}

const dest = resolve(process.cwd(), destArg);
cpSync(wasmDir, dest, { recursive: true });
console.log('Copied MediaPipe wasm to', dest);
