import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const repoRoot = fileURLToPath(new URL('../..', import.meta.url));

export default defineConfig({
	plugins: [react()],
	server: {
		port: 5174,
		fs: { allow: [repoRoot] },
	},
	optimizeDeps: {
		exclude: ['videov8', 'videov8-react'],
	},
	worker: {
		format: 'es',
	},
});
