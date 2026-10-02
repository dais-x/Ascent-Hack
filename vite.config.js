import { defineConfig } from 'vite';

export default defineConfig({
  // Allow SharedArrayBuffer (needed by ONNX WASM multi-thread backend)
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Ensure large ONNX files are not inlined
  build: {
    assetsInlineLimit: 0,
  },
  optimizeDeps: {
    // Exclude transformers.js from pre-bundling (it uses dynamic imports)
    exclude: ['@huggingface/transformers'],
  },
  // Serve .onnx files from public/models with correct MIME type
  plugins: [
    {
      name: 'onnx-mime',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.endsWith('.onnx')) {
            res.setHeader('Content-Type', 'application/octet-stream');
          }
          next();
        });
      },
    },
  ],
});
