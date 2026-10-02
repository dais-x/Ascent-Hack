/**
 * main.js – UI controller
 * Spawns the Moondream2 Web Worker and wires up all UI interactions.
 */

// ── DOM refs ────────────────────────────────────────────────────────
const statusBar        = document.getElementById('status-bar');
const statusText       = document.getElementById('status-text');
const loadBtn          = document.getElementById('load-btn');
const progressContainer= document.getElementById('progress-container');
const progressFill     = document.getElementById('progress-fill');
const progressLabel    = document.getElementById('progress-label');
const progressLog      = document.getElementById('progress-log');
const progressLogInner = document.getElementById('progress-log-inner');
const inferenceSection = document.getElementById('inference-section');
const imageUpload      = document.getElementById('image-upload');
const dropZone         = document.getElementById('drop-zone');
const dropText         = document.getElementById('drop-text');
const preview          = document.getElementById('preview');
const questionInput    = document.getElementById('question-input');
const runBtn           = document.getElementById('run-btn');
const outputArea       = document.getElementById('output-area');
const outputText       = document.getElementById('output-text');
const timing           = document.getElementById('timing');

// ── State ───────────────────────────────────────────────────────────
let worker        = null;
let modelReady    = false;
let currentImageUrl = null;

// Per-file tracking for the log panel
// key = filename, value = { el, barEl, bytesEl, loaded, total }
const fileEntries = new Map();
let totalLoaded   = 0;
let totalSize     = 0;

// ── Worker setup ────────────────────────────────────────────────────
function initWorker() {
  worker = new Worker(new URL('./moondream.worker.js', import.meta.url), {
    type: 'module',
  });

  worker.addEventListener('message', ({ data }) => {
    switch (data.type) {
      case 'status':
        setStatus(data.text, data.state);
        break;

      case 'progress':
        handleProgress(data.progress);
        break;

      case 'ready': {
        modelReady = true;
        const src    = data.source === 'local' ? '📁 disk' : '🌐 HuggingFace';
        const device = data.device === 'webgpu'  ? '🖥️ WebGPU' : '⚙️ WASM/CPU';
        setStatus(`Model ready ✓  (${src} · ${device})`, 'ready');
        progressContainer.hidden = true;
        // keep log visible so user can review what loaded
        loadBtn.textContent = '✅ Loaded';
        loadBtn.disabled = true;
        inferenceSection.hidden = false;
        updateRunBtn();
        break;
      }

      case 'result':
        showResult(data.answer, data.elapsed);
        setStatus('Model ready ✓', 'ready');
        runBtn.disabled = false;
        break;

      case 'error':
        setStatus(`Error: ${data.message}`, 'error');
        runBtn.disabled = false;
        break;

      case 'debug': {
        const debugLog = document.getElementById('debug-log');
        if (debugLog) {
          const memStr = getMemInfo();
          const prefix = memStr ? `[${memStr}] ` : '';
          debugLog.textContent += prefix + data.message + '\n';
          debugLog.scrollTop = debugLog.scrollHeight;
        }
        break;
      }
    }
  });
}

// ── Memory Stats Tracker ─────────────────────────────────────────────
const memStatsEl = document.getElementById('mem-stats');

function getMemInfo() {
  if (performance && performance.memory) {
    const used = formatBytes(performance.memory.usedJSHeapSize);
    const total = formatBytes(performance.memory.totalJSHeapSize);
    return `${used} / ${total}`;
  }
  return null;
}

function updateMemStats() {
  if (!memStatsEl) return;
  const mem = getMemInfo();
  if (mem) {
    memStatsEl.textContent = `RAM: ${mem}`;
  } else {
    memStatsEl.textContent = `RAM: Active`;
  }
}

setInterval(updateMemStats, 1000);
updateMemStats();

// ── Helpers ─────────────────────────────────────────────────────────
function setStatus(text, state = '') {
  statusText.textContent = text;
  statusBar.className = state;
}

function formatBytes(bytes = 0) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1024 ** 2)  return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3)  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

/** Create or retrieve a log row for a given filename. */
function getOrCreateEntry(name) {
  if (fileEntries.has(name)) return fileEntries.get(name);

  // Build DOM row
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `
    <div class="log-entry-header">
      <span class="log-file" title="${name}">${name}</span>
      <span class="log-bytes">—</span>
    </div>
    <div class="log-mini-bar-wrap">
      <div class="log-mini-bar" style="width:0%"></div>
    </div>`;
  progressLogInner.appendChild(entry);

  const obj = {
    el:      entry,
    fileEl:  entry.querySelector('.log-file'),
    bytesEl: entry.querySelector('.log-bytes'),
    barEl:   entry.querySelector('.log-mini-bar'),
    loaded:  0,
    total:   0,
  };
  fileEntries.set(name, obj);

  // Auto-scroll to bottom
  progressLog.scrollTop = progressLog.scrollHeight;
  return obj;
}

function handleProgress(p) {
  if (!p || !p.status) return;

  // Show panels on first event
  progressContainer.hidden = false;
  progressLog.hidden = false;

  const name = p.file ?? p.name ?? 'unknown';

  if (p.status === 'initiate') {
    // File is about to start – create the row early so user sees it queued
    getOrCreateEntry(name);
    return;
  }

  if (p.status === 'download' || p.status === 'progress') {
    const entry = getOrCreateEntry(name);
    const loaded = p.loaded ?? 0;
    const total  = p.total  ?? 0;
    const pct    = total > 0 ? Math.round((loaded / total) * 100) : 0;

    // Update per-file delta for overall progress
    const delta = loaded - entry.loaded;
    entry.loaded = loaded;
    entry.total  = total;
    totalLoaded += delta;

    // Track total size once per file
    if (total > 0 && entry.total !== total) {
      totalSize += (total - entry.total);
    }

    // Update row
    entry.barEl.style.width  = `${pct}%`;
    entry.bytesEl.textContent = total > 0
      ? `${formatBytes(loaded)} / ${formatBytes(total)}  (${pct}%)`
      : `${formatBytes(loaded)}`;

    setStatus(`Loading ${name}…`, 'loading');
  }

  if (p.status === 'done') {
    const entry = getOrCreateEntry(name);
    entry.barEl.style.width  = '100%';
    entry.barEl.classList.add('done');
    entry.fileEl.classList.add('done');
    entry.fileEl.textContent = '✅ ' + name;

    // Snap loaded to total
    if (entry.total > 0) {
      const delta = entry.total - entry.loaded;
      totalLoaded += delta;
      entry.loaded = entry.total;
    }
    entry.bytesEl.textContent = formatBytes(entry.total);
  }

  // ── Overall progress bar ─────────────────────────────────────────
  const overallPct = totalSize > 0
    ? Math.min(100, Math.round((totalLoaded / totalSize) * 100))
    : 0;
  progressFill.style.width   = `${overallPct}%`;
  progressLabel.textContent  = `${overallPct}%`;

  // Auto-scroll log to latest entry
  progressLog.scrollTop = progressLog.scrollHeight;
}

function updateRunBtn() {
  runBtn.disabled = !(modelReady && currentImageUrl);
}

function showResult(answer, elapsed) {
  outputText.textContent = answer || '(no output)';
  timing.textContent = `⏱ Generated in ${elapsed}s`;
  outputArea.hidden = false;
}

function loadImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    currentImageUrl = e.target.result;
    preview.src = currentImageUrl;
    preview.hidden = false;
    dropText.hidden = true;
    outputArea.hidden = true;
    updateRunBtn();
  };
  reader.readAsDataURL(file);
}

// ── Event listeners ─────────────────────────────────────────────────
loadBtn.addEventListener('click', () => {
  if (!worker) initWorker();
  loadBtn.disabled = true;
  // Reset log
  progressLogInner.innerHTML = '';
  fileEntries.clear();
  totalLoaded = 0;
  totalSize   = 0;
  setStatus('Initialising…', 'loading');
  worker.postMessage({ type: 'load' });
});

// File input
imageUpload.addEventListener('change', (e) => loadImageFile(e.target.files[0]));

// Drag-and-drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = 'var(--accent)';
});
dropZone.addEventListener('dragleave', () => {
  dropZone.style.borderColor = '';
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  loadImageFile(e.dataTransfer.files[0]);
});

// Paste image from clipboard
document.addEventListener('paste', (e) => {
  const items = Array.from(e.clipboardData.items);
  const imgItem = items.find((i) => i.type.startsWith('image/'));
  if (imgItem) loadImageFile(imgItem.getAsFile());
});

// Run button
runBtn.addEventListener('click', () => {
  if (!currentImageUrl || !modelReady) return;
  const question = questionInput.value.trim() || 'Describe this image.';
  runBtn.disabled = true;
  setStatus('Running inference…', 'running');
  outputArea.hidden = true;
  worker.postMessage({ type: 'inference', imageDataUrl: currentImageUrl, question });
});

// Allow Enter key to trigger run
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') runBtn.click();
});
