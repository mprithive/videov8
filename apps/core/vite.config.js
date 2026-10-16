import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const repoRoot = fileURLToPath(new URL('../..', import.meta.url));

export default defineConfig({
	server: {
		port: 5173,
		fs: { allow: [repoRoot] },
	},
	optimizeDeps: {
		// Keep the library as source so `new Worker(new URL('./worker.js', import.meta.url))` still resolves.
		exclude: ['videov8'],
	},
	worker: {
		format: 'es',
	},
});
