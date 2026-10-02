/**
 * moondream.worker.js
 *
 * FIX FOR "BLIND MODEL" / PARK HALLUCINATION:
 *
 * Why the model was "blind":
 * Moondream's vision encoder outputs 729 feature vectors (378x378 image -> 27x27 grid = 729 tokens).
 * Transformers.js's native `model.generate({ pixel_values, input_ids })` requires `input_ids` to contain
 * EXACTLY 729 `<image>` tokens (token ID -200).
 *
 * When `<image>` is repeated 729 times in the prompt, `transformers.js` automatically merges the 729 vision
 * embeddings into the 729 `<image>` token slots in `inputs_embeds`.
 *
 * If `<image>` was missing or only present once, the vision embeddings were skipped, making the LLM "blind"
 * so it defaulted to its base pretraining weights (hallucinating a generic "serene park with a lake").
 */

import {
  AutoTokenizer,
  Moondream1ForConditionalGeneration,
  env,
} from '@huggingface/transformers';

import { preprocessImage } from './preprocessImage.js';

// Configure WASM and environment for stable Chromium Android WebView execution
env.wasm.numThreads = 1; // Prevents pthread scheduling policy crashes (policy 1073741825) in Chromium Crashpad
env.wasm.simd       = true;
env.wasm.proxy      = false;

const HF_MODEL_ID = 'Nagafi/moondream2-q4-onnx';
const LOCAL_ROOT  = '/models/';
const LOCAL_PROBE = `${LOCAL_ROOT}${HF_MODEL_ID}/config.json`;

let tokenizer = null;
let model     = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function localModelAvailable() {
  try {
    const res = await fetch(LOCAL_PROBE, { method: 'HEAD' });
    return res.ok;
  } catch { return false; }
}

async function detectDevice() {
  // Mobile Android WebViews trigger Chromium Crashpad dumps under WebGPU shader compilation & multithreading
  const isAndroid = typeof navigator !== 'undefined' && /Android/i.test(navigator.userAgent);
  if (isAndroid) {
    post('debug', { message: 'Android WebView detected -> defaulting to WASM CPU (single-thread mode) for Crashpad stability.' });
    return 'wasm';
  }

  try {
    if (!navigator.gpu) return 'wasm';
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch { return 'wasm'; }
}

async function loadModel() {
  const useLocal = await localModelAvailable();

  if (useLocal) {
    env.localModelPath    = LOCAL_ROOT;
    env.allowLocalModels  = true;
    env.allowRemoteModels = false;
    post('status', { text: '📁 Local model found – loading from disk…', state: 'loading' });
  } else {
    env.allowLocalModels  = false;
    env.allowRemoteModels = true;
    post('status', { text: '🌐 Downloading from HuggingFace…', state: 'loading' });
  }

  const device = await detectDevice();
  post('status', {
    text: device === 'webgpu' ? '🖥️ WebGPU acceleration active' : '⚙️ WASM (CPU) mode active',
    state: 'loading',
  });

  const src = HF_MODEL_ID;

  post('status', { text: 'Loading tokenizer…', state: 'loading' });
  tokenizer = await AutoTokenizer.from_pretrained(src, {
    progress_callback: (p) => post('progress', { progress: p }),
  });

  post('status', { text: `Loading model weights (${device.toUpperCase()})…`, state: 'loading' });

  try {
    model = await Moondream1ForConditionalGeneration.from_pretrained(src, {
      dtype: 'q4',
      device,
      progress_callback: (p) => post('progress', { progress: p }),
    });
  } catch (err) {
    if (device === 'webgpu') {
      post('status', { text: '⚠️ WebGPU initialization failed – falling back to WASM…', state: 'loading' });
      model = await Moondream1ForConditionalGeneration.from_pretrained(src, {
        dtype: 'q4',
        device: 'wasm',
        progress_callback: (p) => post('progress', { progress: p }),
      });
    } else throw err;
  }

  post('ready', { source: useLocal ? 'local' : 'remote', device: model.device ?? device });
}

async function runInference({ imageDataUrl, question }) {
  if (!model || !tokenizer) {
    post('error', { message: 'Model not loaded yet.' });
    return;
  }

  // 1. Preprocess image into NCHW float32 [1, 3, 378, 378] tensor
  post('status', { text: 'Processing image…', state: 'running' });
  post('debug', { message: 'Preprocessing image to 378×378 RGB Float32 [1, 3, 378, 378]...' });

  let pixel_values;
  try {
    const prep = await preprocessImage(imageDataUrl);
    pixel_values = prep.pixel_values;
    post('debug', { message: 'Image preprocessed successfully ✅' });
  } catch (e) {
    post('error', { message: `Image preprocessing failed: ${e?.message ?? e}` });
    return;
  }

  // 2. Format prompt with EXACTLY 729 <image> tokens
  // This satisfies default_merge_input_ids_with_image_features (n_tokens === n_features === 729)
  post('status', { text: 'Encoding visual features & prompt…', state: 'running' });
  const imageTokens = '<image>'.repeat(729);
  const prompt = `${imageTokens}\n\nQuestion: ${question}\n\nAnswer:`;

  post('debug', { message: 'Tokenizing prompt with 729 <image> tokens...' });
  const text_inputs = tokenizer(prompt);

  // Verify token count
  const tokenList = text_inputs.input_ids.tolist()[0];
  const imageTokenCount = tokenList.filter((id) => id === -200 || id === -200n).length;
  post('debug', { message: `Verified <image> (-200) tokens in prompt: ${imageTokenCount} / 729` });

  post('status', { text: 'Generating response…', state: 'running' });
  post('debug', { message: 'Running model.generate() with vision + text embeddings...' });

  const t0 = performance.now();
  let output;

  try {
    // Native generate() merges pixel_values and input_ids seamlessly
    output = await model.generate({
      ...text_inputs,
      pixel_values,
      max_new_tokens: 128,
      do_sample: false,
    });
  } catch (err) {
    post('debug', { message: `Error during model.generate(): ${err?.message ?? err}` });
    post('error', { message: `Inference error: ${err?.message ?? err}` });
    return;
  }

  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

  // Decode output tokens
  const decoded = tokenizer.batch_decode(output, { skip_special_tokens: true });
  const raw = decoded[0] ?? '';

  // Extract answer text after 'Answer:'
  let answer = raw;
  if (raw.includes('Answer:')) {
    answer = raw.split('Answer:').pop().trim();
  } else {
    answer = raw.replace(prompt, '').trim();
  }

  post('debug', { message: `Generation completed in ${elapsed}s ✅` });
  post('result', { answer: answer || raw.trim(), elapsed });
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'load')      await loadModel();
    if (data.type === 'inference') await runInference(data);
  } catch (err) {
    post('error', { message: err?.message ?? String(err) });
  }
});
