/**
 * preprocessImage.js
 *
 * Manual image preprocessing pipeline for Nagafi/moondream2-q4-onnx.
 * Replaces AutoProcessor to guarantee a correctly formatted pixel_values tensor.
 *
 * Pipeline (from preprocessor_config.json):
 *   processor_class : SiglipImageProcessor
 *   do_resize       : true  → 378 × 378  (bilinear, resample: 3)
 *   do_rescale      : true  → ÷ 255  (rescale_factor = 1/255)
 *   do_normalize    : true  → (pixel - 0.5) / 0.5   per channel
 *   output shape    : [1, 3, 378, 378]  float32  (NCHW)
 *
 * Steps:
 *   1. Draw original image onto an OffscreenCanvas at 378×378 (bilinear by default)
 *   2. Read RGBA pixels → strip Alpha → keep RGB
 *   3. Rescale  : uint8 [0,255] → float32 [0.0, 1.0]  (÷ 255)
 *   4. Normalize: (value - 0.5) / 0.5  per channel
 *   5. Transpose: HWC [378,378,3] → NCHW [1,3,378,378]
 *   6. Wrap in a Tensor object compatible with transformers.js
 */

import { Tensor } from '@huggingface/transformers';

/** Exact values from preprocessor_config.json */
const TARGET_H = 378;
const TARGET_W = 378;
const MEAN  = [0.5, 0.5, 0.5];   // R, G, B
const STD   = [0.5, 0.5, 0.5];   // R, G, B
const RESCALE = 1 / 255;

/**
 * Preprocesses an HTMLImageElement / ImageBitmap / Blob / data-URL string
 * into the pixel_values Tensor required by the vision encoder.
 *
 * @param {string|Blob|ImageBitmap} source  Image source (data URL, Blob, or ImageBitmap)
 * @returns {Promise<{ pixel_values: Tensor }>}
 */
export async function preprocessImage(source) {
  // ── Step 0: resolve the source to an ImageBitmap ──────────────────
  let bitmap;
  if (typeof source === 'string') {
    // data URL
    const blob = await fetch(source).then((r) => r.blob());
    bitmap = await createImageBitmap(blob);
  } else if (source instanceof Blob) {
    bitmap = await createImageBitmap(source);
  } else {
    bitmap = source; // already an ImageBitmap
  }

  // ── Step 1: Resize to 378×378 via OffscreenCanvas (bilinear) ──────
  const canvas = new OffscreenCanvas(TARGET_W, TARGET_H);
  const ctx = canvas.getContext('2d');

  // imageSmoothingQuality = 'high' enables bilinear / bicubic depending on browser
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, TARGET_W, TARGET_H);

  // ── Step 2: Extract RGBA pixels → strip Alpha ─────────────────────
  const imageData = ctx.getImageData(0, 0, TARGET_W, TARGET_H);
  const rgba      = imageData.data; // Uint8ClampedArray, length = 378*378*4

  const numPixels  = TARGET_H * TARGET_W;
  // Output buffer: [1, 3, 378, 378] float32
  const float32    = new Float32Array(3 * numPixels);

  // Channel offsets in NCHW layout
  const rOffset = 0;
  const gOffset = numPixels;
  const bOffset = numPixels * 2;

  // ── Step 3+4+5: Rescale → Normalize → Transpose ───────────────────
  for (let i = 0; i < numPixels; i++) {
    const base = i * 4;            // RGBA stride

    const r = rgba[base]     * RESCALE;   // 0→1
    const g = rgba[base + 1] * RESCALE;
    const b = rgba[base + 2] * RESCALE;
    // rgba[base + 3] = Alpha — DISCARDED

    // Normalize: (value - mean) / std
    float32[rOffset + i] = (r - MEAN[0]) / STD[0];
    float32[gOffset + i] = (g - MEAN[1]) / STD[1];
    float32[bOffset + i] = (b - MEAN[2]) / STD[2];
  }

  // ── Step 6: Wrap in Tensor [1, 3, 378, 378] ───────────────────────
  const pixel_values = new Tensor('float32', float32, [1, 3, TARGET_H, TARGET_W]);

  return { pixel_values };
}
