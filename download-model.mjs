/**
 * download-model.mjs
 * Downloads all Nagafi/moondream2-q4-onnx model files from HuggingFace
 * into public/models/Nagafi/moondream2-q4-onnx/ so the app can serve
 * them locally without any internet connection during inference.
 *
 * Run with:  node download-model.mjs
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const MODEL_ID  = 'Nagafi/moondream2-q4-onnx';
const HF_BASE   = 'https://huggingface.co';
const LOCAL_DIR = path.join(__dirname, 'public', 'models', MODEL_ID);

// All files in the repo (from HF API)
const FILES = [
  'added_tokens.json',
  'config.json',
  'generation_config.json',
  'merges.txt',
  'onnx/decoder_model_merged_q4.onnx',
  'onnx/decoder_model_merged_q4f16.onnx',
  'onnx/embed_tokens_q4.onnx',
  'onnx/vision_encoder_q4.onnx',
  'preprocessor_config.json',
  'quantize_config.json',
  'special_tokens_map.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'vocab.json',
];

// ── Helpers ────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024)         return `${bytes} B`;
  if (bytes < 1024 ** 2)   return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3)   return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(path.dirname(destPath), { recursive: true });

    // If file already exists and is non-empty, skip
    if (fs.existsSync(destPath) && fs.statSync(destPath).size > 0) {
      console.log(`  ⏭  Already exists, skipping: ${path.relative(__dirname, destPath)}`);
      return resolve();
    }

    const tmpPath = destPath + '.tmp';
    const file = fs.createWriteStream(tmpPath);
    let downloaded = 0;
    let total = 0;
    let lastPct = -1;

    function doRequest(reqUrl) {
      // Ensure we always have an absolute URL
      const parsedUrl = new URL(reqUrl);
      const options = {
        hostname: parsedUrl.hostname,
        path: parsedUrl.pathname + parsedUrl.search,
        headers: { 'User-Agent': 'node-fetch/1.0' },
      };

      https.get(options, (res) => {
        // Follow redirects (HF uses CDN redirects, sometimes relative)
        if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307) {
          const location = res.headers.location;
          // Resolve relative redirects against current host
          const nextUrl = location.startsWith('http')
            ? location
            : `https://${parsedUrl.hostname}${location}`;
          res.resume(); // drain the response
          doRequest(nextUrl);
          return;
        }
        if (res.statusCode !== 200) {
          file.destroy();
          if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
          reject(new Error(`HTTP ${res.statusCode} for ${reqUrl}`));
          return;
        }

        total = parseInt(res.headers['content-length'] || '0', 10);

        res.on('data', (chunk) => {
          downloaded += chunk.length;
          if (total > 0) {
            const pct = Math.floor((downloaded / total) * 100);
            if (pct !== lastPct && pct % 5 === 0) {
              process.stdout.write(`\r  ↓  ${formatBytes(downloaded)} / ${formatBytes(total)} (${pct}%)   `);
              lastPct = pct;
            }
          }
        });

        res.pipe(file);
        file.on('finish', () => {
          file.close(() => {
            fs.renameSync(tmpPath, destPath);
            process.stdout.write(`\r  ✅ ${formatBytes(downloaded)} saved                          \n`);
            resolve();
          });
        });
      }).on('error', (err) => {
        file.destroy();
        if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
        reject(err);
      });
    }

    doRequest(url);
  });
}

// ── Main ────────────────────────────────────────────────────────────

async function main() {
  console.log(`\n🌙 Moondream2 Q4 ONNX – Model Downloader`);
  console.log(`   Source : ${HF_BASE}/${MODEL_ID}`);
  console.log(`   Target : ${LOCAL_DIR}\n`);

  let totalSize = 0;
  let skipped = 0;

  for (const file of FILES) {
    const url      = `${HF_BASE}/${MODEL_ID}/resolve/main/${file}`;
    const destPath = path.join(LOCAL_DIR, file);

    const alreadyExists =
      fs.existsSync(destPath) && fs.statSync(destPath).size > 0;

    console.log(`📄 ${file}`);

    if (alreadyExists) {
      const sz = fs.statSync(destPath).size;
      console.log(`  ⏭  Already exists (${formatBytes(sz)}), skipping`);
      skipped++;
      totalSize += sz;
      continue;
    }

    try {
      await downloadFile(url, destPath);
      totalSize += fs.statSync(destPath).size;
    } catch (err) {
      console.error(`  ❌ Failed: ${err.message}`);
    }
  }

  console.log(`\n✨ Done!`);
  console.log(`   Files downloaded : ${FILES.length - skipped} new, ${skipped} skipped`);
  console.log(`   Total on disk    : ${formatBytes(totalSize)}`);
  console.log(`\n   Model will be served from: /models/${MODEL_ID}/\n`);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
