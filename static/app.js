// Meeting Intelligence — browser app.
// Pipeline: decode/resample -> sherpa-onnx diarization (WASM) ->
//           Whisper ASR per segment (transformers.js) -> speaker alignment ->
//           POST /api/analyze -> render.
// Audio never leaves the device; only transcript text goes to the API.

'use strict';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const WASM_DIR = 'wasm';
const WHISPER_MODEL = 'Xenova/whisper-tiny.en'; // weights self-hosted under models/
const ANALYZE_URL = '/api/analyze';
const MAX_SEGMENT_SEC = 28;   // split long diarization segments for Whisper
const MERGE_GAP_SEC = 0.6;    // merge same-speaker utterances closer than this

const $ = (id) => document.getElementById(id);

const state = {
  samples: null,        // Float32Array @16k mono
  sampleRate: 16000,
  fileName: '',
  durationSec: 0,
  diarSegments: [],
  utterances: [],
  analysis: null,
  model: '',
  timings: {},
  speakerNames: {},
};

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
function setModelStatus(id, cls, text) {
  const el = $(id);
  el.className = 'ms-item ' + cls;
  el.textContent = text;
}

// 1) sherpa-onnx WASM (diarization). The prebuilt bundle preloads the
//    segmentation + embedding models into emscripten's virtual FS.
let sherpaReady = null;
function loadSherpa() {
  setModelStatus('ms-wasm', 'loading', '⏳ Engine (WASM) loading…');
  setModelStatus('ms-seg', 'loading', '⏳ Segmentation model…');
  setModelStatus('ms-emb', 'loading', '⏳ Speaker embedding model…');
  sherpaReady = new Promise((resolve, reject) => {
    window.Module = {
      onRuntimeInitialized() {
        try {
          const sd = createOfflineSpeakerDiarization(window.Module);
          setModelStatus('ms-wasm', 'ok', '✅ Engine (WASM)');
          setModelStatus('ms-seg', 'ok', '✅ Segmentation model');
          setModelStatus('ms-emb', 'ok', '✅ Speaker embedding model');
          resolve(sd);
        } catch (e) { reject(e); }
      },
      onAbort(why) { reject(new Error('WASM aborted: ' + why)); },
    };
    const s1 = document.createElement('script');
    s1.src = `${WASM_DIR}/sherpa-onnx-speaker-diarization.js`;
    s1.onload = () => {
      const s2 = document.createElement('script');
      s2.src = `${WASM_DIR}/sherpa-onnx.js`;
      s2.onerror = () => reject(new Error('failed to load WASM runtime'));
      document.head.appendChild(s2);
    };
    s1.onerror = () => reject(new Error('failed to load diarization bindings'));
    document.head.appendChild(s1);
  });
  return sherpaReady;
}

// 2) Whisper via transformers.js (self-hosted weights).
let whisperReady = null;
async function loadWhisper() {
  setModelStatus('ms-asr', 'loading', '⏳ Whisper ASR model…');
  const { pipeline, env } = await import('./vendor/transformers.min.js');
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  // Self-hosted weights: point the hub at our own static dir.
  env.remoteHost = `${location.origin}/models/`;
  env.remotePathTemplate = '{model_id}/';
  const asr = await pipeline('automatic-speech-recognition', WHISPER_MODEL, {
    quantized: true,
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        setModelStatus('ms-asr', 'loading', `⏳ Whisper ${pct}%`);
      }
    },
  });
  setModelStatus('ms-asr', 'ok', '✅ Whisper ASR model');
  return asr;
}

// ---------------------------------------------------------------------------
// Audio decode
// ---------------------------------------------------------------------------
async function decodeAudioFile(file) {
  const arrayBuf = await file.arrayBuffer();
  const ctx = new AudioContext({ sampleRate: 16000 });
  try {
    const decoded = await ctx.decodeAudioData(arrayBuf);
    // mono mixdown
    const n = decoded.length;
    const mono = new Float32Array(n);
    const chs = decoded.numberOfChannels;
    for (let c = 0; c < chs; c++) {
      const d = decoded.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += d[i] / chs;
    }
    return { samples: mono, sampleRate: decoded.sampleRate, durationSec: decoded.duration };
  } finally {
    ctx.close();
  }
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------
function setStage(id, cls, timeText) {
  const el = $(id);
  el.classList.remove('active', 'done', 'error');
  if (cls) el.classList.add(cls);
  if (timeText !== undefined) el.querySelector('.stage-time').textContent = timeText;
}

function setProgress(pct, text) {
  $('progress-fill').style.width = pct + '%';
  if (text) $('progress-text').textContent = text;
}

async function yieldUI() {
  return new Promise((r) => setTimeout(r, 0));
}

async function runPipeline(opts) {
  const t = {};

  // --- Stage 1: decode ---
  setStage('stage-decode', 'active');
  setProgress(5, 'Decoding audio…');
  let t0 = performance.now();
  const decoded = await decodeAudioFile(opts.file);
  state.samples = decoded.samples;
  state.sampleRate = decoded.sampleRate;
  state.durationSec = decoded.durationSec;
  state.fileName = opts.file.name;
  t.decode = performance.now() - t0;
  setStage('stage-decode', 'done', (t.decode / 1000).toFixed(1) + 's');
  await yieldUI();

  // --- Stage 2: diarization ---
  setStage('stage-diar', 'active');
  setProgress(15, `Running speaker diarization on ${decoded.durationSec.toFixed(0)}s of audio…`);
  const sd = await sherpaReady;
  const cfg = sd.config;
  cfg.clustering = {
    numClusters: opts.numSpeakers,
    threshold: opts.threshold,
  };
  sd.setConfig(cfg);
  t0 = performance.now();
  const segments = sd.process(state.samples);
  t.diar = performance.now() - t0;
  if (!segments || !segments.length) throw new Error('No speech detected in the audio.');
  state.diarSegments = segments;
  const nSpk = new Set(segments.map((s) => s.speaker)).size;
  setStage('stage-diar', 'done', (t.diar / 1000).toFixed(1) + 's');
  setProgress(35, `Found ${nSpk} speaker(s), ${segments.length} segments`);
  await yieldUI();

  // --- Stage 3: VAD-like refinement (merge/split diarization segments) ---
  setStage('stage-vad', 'active');
  const asrSegments = refineSegments(segments);
  setStage('stage-vad', 'done', asrSegments.length + ' chunks');
  setProgress(40, `Prepared ${asrSegments.length} chunks for transcription`);
  await yieldUI();

  // --- Stage 4: Whisper ASR per segment ---
  setStage('stage-asr', 'active');
  const asr = await whisperReady;
  t0 = performance.now();
  const results = [];
  for (let i = 0; i < asrSegments.length; i++) {
    const seg = asrSegments[i];
    const audio = state.samples.subarray(
      Math.floor(seg.start * 16000),
      Math.min(state.samples.length, Math.floor(seg.end * 16000))
    );
    if (audio.length < 1600) continue; // <0.1s — skip
    const out = await asr(audio, {
      language: 'english',
      task: 'transcribe',
      chunk_length_s: 30,
      return_timestamps: false,
    });
    const text = (out.text || '').trim();
    if (text && !/^\[.*\]$/.test(text)) { // drop [BLANK_AUDIO] etc.
      results.push({ start: seg.start, end: seg.end, speaker: seg.speaker, text });
    }
    const pct = 40 + Math.round(((i + 1) / asrSegments.length) * 45);
    setProgress(pct, `Transcribing chunk ${i + 1}/${asrSegments.length}…`);
    if (i % 3 === 0) await yieldUI();
  }
  t.asr = performance.now() - t0;
  if (!results.length) throw new Error('Whisper produced no transcription.');
  setStage('stage-asr', 'done', (t.asr / 1000).toFixed(1) + 's');
  await yieldUI();

  // --- Stage 5: alignment / merge ---
  setStage('stage-align', 'active');
  state.utterances = mergeUtterances(results);
  setStage('stage-align', 'done', state.utterances.length + ' turns');
  setProgress(88, 'Transcript assembled');
  t.total = t.decode + t.diar + t.asr;
  state.timings = t;
  await yieldUI();

  // --- Stage 6: LLM analysis ---
  if (opts.useLlm) {
    setStage('stage-llm', 'active');
    setProgress(92, 'Generating AI summary & action items…');
    t0 = performance.now();
    try {
      const resp = await fetch(ANALYZE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          utterances: state.utterances.map((u) => ({
            speaker: u.speaker, start: +u.start.toFixed(2), end: +u.end.toFixed(2), text: u.text,
          })),
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.analysis) {
        throw new Error(data.error || ('HTTP ' + resp.status));
      }
      state.analysis = data.analysis;
      state.model = data.model || '';
      t.llm = performance.now() - t0;
      setStage('stage-llm', 'done', (t.llm / 1000).toFixed(1) + 's');
    } catch (e) {
      t.llm = performance.now() - t0;
      setStage('stage-llm', 'error', 'failed');
      console.error('LLM analysis failed:', e);
      state.analysis = null;
    }
  } else {
    setStage('stage-llm', 'done', 'skipped');
  }
  setProgress(100, 'Done');
}

// Split long segments (Whisper handles ≤30s well) and drop tiny ones.
function refineSegments(segments) {
  const out = [];
  for (const s of segments) {
    const dur = s.end - s.start;
    if (dur < 0.2) continue;
    if (dur <= MAX_SEGMENT_SEC) { out.push({ ...s }); continue; }
    const n = Math.ceil(dur / MAX_SEGMENT_SEC);
    const step = dur / n;
    for (let i = 0; i < n; i++) {
      out.push({ start: s.start + i * step, end: s.start + (i + 1) * step, speaker: s.speaker });
    }
  }
  return out;
}

function mergeUtterances(chunks) {
  const utts = [];
  for (const c of chunks) {
    const last = utts[utts.length - 1];
    if (last && last.speaker === c.speaker && c.start - last.end <= MERGE_GAP_SEC) {
      last.text += ' ' + c.text;
      last.end = c.end;
    } else {
      utts.push({ ...c });
    }
  }
  return utts;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
const SPK_COLORS = ['spk-0', 'spk-1', 'spk-2', 'spk-3', 'spk-4', 'spk-5'];
const SPK_BG = ['spk-bg-0', 'spk-bg-1', 'spk-bg-2', 'spk-bg-3', 'spk-bg-4', 'spk-bg-5'];

function speakerLabel(id) {
  return state.speakerNames[id] || `Speaker ${id + 1}`;
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderResults() {
  const a = state.analysis;
  const nSpk = new Set(state.utterances.map((u) => u.speaker)).size;

  // Map participant names from analysis onto speaker ids when possible.
  state.speakerNames = {};
  if (a && a.participants && a.participants.length === nSpk) {
    // Heuristic: participants are usually listed in order of first speech.
    const order = [];
    for (const u of state.utterances) {
      if (!order.includes(u.speaker)) order.push(u.speaker);
    }
    order.forEach((spkId, i) => {
      if (a.participants[i] && !/^speaker\s*\d+$/i.test(a.participants[i])) {
        state.speakerNames[spkId] = a.participants[i];
      }
    });
  }

  // summary card
  if (a) {
    $('meeting-title').textContent = a.title || 'Meeting';
    $('meeting-summary').textContent = a.summary || '';
    $('model-badge').textContent = state.model;
    $('participants').innerHTML = (a.participants || [])
      .map((p, i) => `<span class="chip ${SPK_BG[i % 6]}">${esc(p)}</span>`).join('');
  } else {
    $('meeting-title').textContent = 'Meeting';
    $('meeting-summary').textContent = '(AI analysis unavailable — transcript below is still complete.)';
    $('model-badge').textContent = '';
    $('participants').innerHTML = '';
  }

  // action items
  const items = a?.actionItems || [];
  $('action-count').textContent = items.length + ' items';
  $('action-items').innerHTML = items.length ? items.map((it) => {
    const spkIdx = ownerIndex(it.owner);
    const cls = SPK_BG[spkIdx >= 0 ? spkIdx % 6 : 3];
    return `<div class="action-item">
      <span class="ai-owner ${cls}">${esc(it.owner)}</span>
      <div class="ai-body">
        <div class="ai-task">${esc(it.task)}</div>
        <div class="ai-meta">${it.due && it.due !== 'Not stated' ? `<span class="ai-due">📅 ${esc(it.due)}</span> · ` : ''}${esc(it.context || '')}</div>
      </div>
    </div>`;
  }).join('') : '<p class="muted">No explicit action items detected.</p>';

  // decisions / keypoints / questions
  renderList('decisions-list', a?.decisions);
  renderList('keypoints-list', a?.keyPoints);
  renderList('questions-list', a?.openQuestions);
  $('decisions-card').classList.toggle('hidden', !(a?.decisions?.length));
  $('keypoints-card').classList.toggle('hidden', !(a?.keyPoints?.length));
  $('questions-card').classList.toggle('hidden', !(a?.openQuestions?.length));

  // transcript
  $('speaker-legend').innerHTML = [...new Set(state.utterances.map((u) => u.speaker))]
    .map((s) => `<span class="chip ${SPK_BG[s % 6]}">${esc(speakerLabel(s))}</span>`).join('');
  $('transcript').innerHTML = state.utterances.map((u) => `
    <div class="utt">
      <span class="utt-time">${fmtTime(u.start)}</span>
      <span class="utt-speaker ${SPK_COLORS[u.speaker % 6]}">${esc(speakerLabel(u.speaker))}</span>
      <span class="utt-text">${esc(u.text)}</span>
    </div>`).join('');

  // stats
  const words = state.utterances.reduce((n, u) => n + u.text.split(/\s+/).length, 0);
  const stats = [
    [fmtTime(state.durationSec), 'Duration'],
    [nSpk, 'Speakers'],
    [state.utterances.length, 'Turns'],
    [words.toLocaleString(), 'Words'],
    [state.diarSegments.length, 'Diarization segments'],
    [(state.timings.diar / 1000).toFixed(1) + 's', 'Diarization time'],
    [(state.timings.asr / 1000).toFixed(1) + 's', 'Transcription time'],
    [state.timings.llm ? (state.timings.llm / 1000).toFixed(1) + 's' : '—', 'Analysis time'],
  ];
  $('stats').innerHTML = stats.map(([v, k]) =>
    `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');

  $('pipeline-section').classList.add('hidden');
  $('results-section').classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function ownerIndex(owner) {
  if (!owner) return -1;
  const o = owner.toLowerCase();
  for (const [id, name] of Object.entries(state.speakerNames)) {
    if (name.toLowerCase() === o) return +id;
  }
  const m = o.match(/speaker\s*(\d+)/);
  if (m) return +m[1] - 1;
  return -1;
}

function renderList(id, arr) {
  $(id).innerHTML = (arr || []).map((x) => `<li>${esc(x)}</li>`).join('') || '<li class="muted">None detected.</li>';
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
function transcriptText() {
  return state.utterances.map((u) =>
    `[${fmtTime(u.start)}] ${speakerLabel(u.speaker)}: ${u.text}`).join('\n');
}

function markdownExport() {
  const a = state.analysis;
  let md = `# ${a?.title || 'Meeting'}\n\n`;
  md += `**Duration:** ${fmtTime(state.durationSec)} · **Speakers:** ${new Set(state.utterances.map((u) => u.speaker)).size} · **File:** ${state.fileName}\n\n`;
  if (a?.summary) md += `## Summary\n\n${a.summary}\n\n`;
  if (a?.actionItems?.length) {
    md += `## Action Items\n\n| Owner | Task | Due |\n|---|---|---|\n`;
    md += a.actionItems.map((i) => `| ${i.owner} | ${i.task} | ${i.due} |`).join('\n') + '\n\n';
  }
  if (a?.decisions?.length) md += `## Decisions\n\n${a.decisions.map((d) => `- ${d}`).join('\n')}\n\n`;
  if (a?.keyPoints?.length) md += `## Key Points\n\n${a.keyPoints.map((k) => `- ${k}`).join('\n')}\n\n`;
  md += `## Transcript\n\n` + state.utterances.map((u) =>
    `**[${fmtTime(u.start)}] ${speakerLabel(u.speaker)}:** ${u.text}`).join('\n\n') + '\n';
  return md;
}

function srtExport() {
  const pad = (n, w = 3) => String(n).padStart(w, '0');
  const ts = (sec) => {
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60),
      s = Math.floor(sec % 60), ms = Math.round((sec % 1) * 1000);
    return `${pad(h, 2)}:${pad(m)}:${pad(s)},${pad(ms)}`;
  };
  return state.utterances.map((u, i) =>
    `${i + 1}\n${ts(u.start)} --> ${ts(u.end)}\n${speakerLabel(u.speaker)}: ${u.text}\n`).join('\n');
}

function download(name, content, type = 'text/plain') {
  const blob = new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function showError(msg) {
  $('error-text').textContent = msg;
  $('pipeline-section').classList.add('hidden');
  $('input-section').classList.add('hidden');
  $('results-section').classList.add('hidden');
  $('error-section').classList.remove('hidden');
}

function resetUI() {
  ['stage-decode', 'stage-diar', 'stage-vad', 'stage-asr', 'stage-align', 'stage-llm']
    .forEach((id) => { setStage(id, null, ''); });
  setProgress(0, '');
  $('error-section').classList.add('hidden');
  $('results-section').classList.add('hidden');
  $('input-section').classList.remove('hidden');
}

async function start(file) {
  resetUI();
  $('input-section').classList.add('hidden');
  $('pipeline-section').classList.remove('hidden');
  try {
    await runPipeline({
      file,
      numSpeakers: parseInt($('opt-speakers').value, 10),
      threshold: parseFloat($('opt-threshold').value) || 0.5,
      useLlm: $('opt-llm').checked,
    });
    renderResults();
  } catch (e) {
    console.error(e);
    showError(e.message || String(e));
  }
}

async function init() {
  // file input
  const dz = $('dropzone');
  const fi = $('file-input');
  $('btn-browse').addEventListener('click', (e) => { e.stopPropagation(); fi.click(); });
  dz.addEventListener('click', () => fi.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter') fi.click(); });
  fi.addEventListener('change', () => { if (fi.files[0]) start(fi.files[0]); });
  ['dragover', 'dragenter'].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.remove('dragover');
  }));
  dz.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) start(f);
  });

  // demo meeting
  $('btn-demo').addEventListener('click', async (e) => {
    e.stopPropagation();
    setProgress(0, 'Loading demo meeting…');
    try {
      const resp = await fetch('demo/meeting1.wav');
      const blob = await resp.blob();
      start(new File([blob], 'demo-meeting.wav', { type: 'audio/wav' }));
    } catch (err) {
      showError('Demo audio not available: ' + err.message);
    }
  });

  // exports
  $('btn-copy').addEventListener('click', async () => {
    await navigator.clipboard.writeText(transcriptText());
    $('btn-copy').textContent = 'Copied!';
    setTimeout(() => ($('btn-copy').textContent = 'Copy'), 1500);
  });
  $('btn-md').addEventListener('click', () => download('meeting-notes.md', markdownExport(), 'text/markdown'));
  $('btn-json').addEventListener('click', () => download('meeting.json', JSON.stringify({
    file: state.fileName, durationSec: state.durationSec,
    utterances: state.utterances, analysis: state.analysis, model: state.model,
  }, null, 2), 'application/json'));
  $('btn-srt').addEventListener('click', () => download('meeting.srt', srtExport(), 'application/x-subrip'));

  $('btn-restart').addEventListener('click', resetUI);
  $('btn-retry').addEventListener('click', resetUI);

  // preload models
  try {
    loadSherpa().catch((e) => {
      setModelStatus('ms-wasm', 'err', '❌ Engine failed');
      console.error(e);
    });
    whisperReady = loadWhisper();
    whisperReady.catch((e) => {
      setModelStatus('ms-asr', 'err', '❌ Whisper failed');
      console.error(e);
    });
    setModelStatus('ms-vad', 'ok', '✅ VAD (diarization-based)');
  } catch (e) {
    console.error(e);
  }
}

init();
