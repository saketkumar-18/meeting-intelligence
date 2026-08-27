// Meeting Intelligence — core pipeline (Node).
// audio (16kHz mono WAV) -> diarization -> VAD-chunked Whisper ASR -> speaker alignment
//
// Everything runs locally via sherpa-onnx WASM. No audio leaves the machine.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const sherpa = require('sherpa-onnx');

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const MODELS_DIR = path.resolve(__dirname, '..', 'models');

export const MODEL_PATHS = {
  // int8 segmentation is ~2x faster than fp32 with equivalent output quality.
  segmentation: path.join(MODELS_DIR, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.int8.onnx'),
  // ERes2Net embeddings: 100% speaker attribution on the eval meeting vs
  // 78.6% for CAM++ (Bob/Carol voices were merging with CAM++).
  embedding: path.join(MODELS_DIR, '3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx'),
  whisperEncoder: path.join(MODELS_DIR, 'sherpa-onnx-whisper-base.en', 'base.en-encoder.onnx'),
  whisperDecoder: path.join(MODELS_DIR, 'sherpa-onnx-whisper-base.en', 'base.en-decoder.onnx'),
  whisperTokens: path.join(MODELS_DIR, 'sherpa-onnx-whisper-base.en', 'base.en-tokens.txt'),
  vad: path.join(MODELS_DIR, 'silero_vad.onnx'),
};

export function checkModels() {
  const missing = [];
  for (const [k, p] of Object.entries(MODEL_PATHS)) {
    if (!fs.existsSync(p)) missing.push(`${k}: ${p}`);
  }
  return missing;
}

// ---------------------------------------------------------------------------
// Diarization: pyannote segmentation + CAM++ embeddings + fast clustering
// ---------------------------------------------------------------------------
export function diarize(samples, opts = {}) {
  const config = {
    segmentation: {
      // windowShiftRatio 0.25 = 2x faster than the 0.1 default with
      // equivalent segment quality on meeting audio (benchmarked).
      pyannote: { model: MODEL_PATHS.segmentation, windowShiftRatio: opts.windowShiftRatio ?? 0.25 },
      numThreads: opts.numThreads || 1,
      debug: 0,
      provider: 'cpu',
    },
    embedding: {
      model: MODEL_PATHS.embedding,
      numThreads: opts.numThreads || 1,
      debug: 0,
      provider: 'cpu',
    },
    clustering: {
      numClusters: opts.numSpeakers > 0 ? opts.numSpeakers : -1,
      threshold: opts.threshold ?? 0.5,
    },
    minDurationOn: 0.3,
    minDurationOff: 0.5,
  };
  const diarizer = sherpa.createOfflineSpeakerDiarization(config);
  const segments = diarizer.process(samples);
  diarizer.free();
  return segments; // [{start, end, speaker}]
}

// ---------------------------------------------------------------------------
// VAD: Silero — split audio into speech chunks for Whisper
// ---------------------------------------------------------------------------
export function vadSplit(samples, sampleRate = 16000, opts = {}) {
  const config = {
    sileroVad: {
      model: MODEL_PATHS.vad,
      threshold: opts.threshold ?? 0.5,
      minSilenceDuration: opts.minSilenceDuration ?? 0.5,
      minSpeechDuration: 0.25,
      maxSpeechDuration: opts.maxSpeechDuration ?? 20.0,
      windowSize: 512,
    },
    sampleRate,
    numThreads: 1,
    debug: 0,
    provider: 'cpu',
    bufferSizeInSeconds: 600,
  };
  const vad = sherpa.createVad(config);
  vad.acceptWaveform(samples);
  vad.flush();
  const chunks = [];
  while (!vad.isEmpty()) {
    const s = vad.front();
    chunks.push({ start: s.start / sampleRate, samples: s.samples });
    vad.pop();
  }
  vad.free();
  return chunks;
}

// ---------------------------------------------------------------------------
// Whisper ASR (offline, per-chunk)
// ---------------------------------------------------------------------------
export function createWhisperRecognizer(opts = {}) {
  const config = {
    featConfig: { sampleRate: 16000, featureDim: 80 },
    modelConfig: {
      whisper: {
        encoder: MODEL_PATHS.whisperEncoder,
        decoder: MODEL_PATHS.whisperDecoder,
        language: opts.language || 'en',
        task: 'transcribe',
        tailPaddings: -1,
      },
      tokens: MODEL_PATHS.whisperTokens,
      numThreads: opts.numThreads || 1,
      debug: 0,
      provider: 'cpu',
      modelType: 'whisper',
    },
    decodingMethod: 'greedy_search',
    maxActivePaths: 4,
  };
  return sherpa.createOfflineRecognizer(config);
}

export function transcribeChunks(recognizer, chunks) {
  const results = [];
  for (const chunk of chunks) {
    const stream = recognizer.createStream();
    stream.acceptWaveform(16000, chunk.samples);
    recognizer.decode(stream);
    const r = recognizer.getResult(stream);
    stream.free();
    const text = (r.text || '').trim();
    if (text) {
      results.push({
        start: chunk.start,
        end: chunk.start + chunk.samples.length / 16000,
        text,
        timestamps: r.timestamps || [],
        tokens: r.tokens || [],
      });
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Speaker alignment: assign each ASR chunk to the diarization speaker with
// the greatest temporal overlap.
// ---------------------------------------------------------------------------
export function alignSpeakers(asrChunks, diarSegments) {
  if (!diarSegments.length) {
    return asrChunks.map((c) => ({ ...c, speaker: 0 }));
  }
  return asrChunks.map((chunk) => {
    const overlap = new Map();
    for (const seg of diarSegments) {
      const o = Math.min(chunk.end, seg.end) - Math.max(chunk.start, seg.start);
      if (o > 0) overlap.set(seg.speaker, (overlap.get(seg.speaker) || 0) + o);
    }
    let best = 0, bestOv = -1;
    for (const [spk, o] of overlap) {
      if (o > bestOv) { best = spk; bestOv = o; }
    }
    return { ...chunk, speaker: best };
  });
}

// Merge consecutive chunks from the same speaker into utterances.
export function mergeUtterances(aligned, gapTolerance = 0.75) {
  const utterances = [];
  for (const c of aligned) {
    const last = utterances[utterances.length - 1];
    if (last && last.speaker === c.speaker && c.start - last.end <= gapTolerance) {
      last.text += ' ' + c.text;
      last.end = c.end;
    } else {
      utterances.push({ speaker: c.speaker, start: c.start, end: c.end, text: c.text });
    }
  }
  return utterances;
}

// ---------------------------------------------------------------------------
// Full pipeline.
//
// Architecture (mirrors the browser app): diarization segments ARE the ASR
// chunks. pyannote already localizes speech, so a separate VAD pass is
// redundant. Long diarization segments are split to stay under Whisper's
// comfortable window.
// ---------------------------------------------------------------------------
const MAX_ASR_SEGMENT_SEC = 28;

export function refineSegments(segments, maxSec = MAX_ASR_SEGMENT_SEC) {
  const out = [];
  for (const s of segments) {
    const dur = s.end - s.start;
    if (dur < 0.2) continue;
    if (dur <= maxSec) { out.push({ ...s }); continue; }
    const n = Math.ceil(dur / maxSec);
    const step = dur / n;
    for (let i = 0; i < n; i++) {
      out.push({ start: s.start + i * step, end: s.start + (i + 1) * step, speaker: s.speaker });
    }
  }
  return out;
}

export function runPipeline(samples, opts = {}) {
  const t0 = Date.now();
  const timings = {};

  let t = Date.now();
  const diarSegments = diarize(samples, opts);
  timings.diarizationMs = Date.now() - t;

  const asrSegments = refineSegments(diarSegments);

  t = Date.now();
  const recognizer = createWhisperRecognizer(opts);
  const asrChunks = transcribeSegments(recognizer, samples, asrSegments);
  recognizer.free();
  timings.asrMs = Date.now() - t;

  const utterances = mergeUtterances(asrChunks);
  const numSpeakers = new Set(diarSegments.map((s) => s.speaker)).size;

  timings.totalMs = Date.now() - t0;
  return {
    durationSec: samples.length / 16000,
    numSpeakers,
    diarSegments,
    utterances,
    timings,
  };
}

// Transcribe diarization segments directly (each carries its speaker label).
export function transcribeSegments(recognizer, samples, segments) {
  const results = [];
  for (const seg of segments) {
    const startIdx = Math.floor(seg.start * 16000);
    const endIdx = Math.min(samples.length, Math.floor(seg.end * 16000));
    const audio = samples.subarray(startIdx, endIdx);
    if (audio.length < 1600) continue; // <0.1s
    const stream = recognizer.createStream();
    stream.acceptWaveform(16000, audio);
    recognizer.decode(stream);
    const r = recognizer.getResult(stream);
    stream.free();
    const text = (r.text || '').trim();
    if (text && !/^\[.*\]$/.test(text)) {
      results.push({ start: seg.start, end: seg.end, speaker: seg.speaker, text });
    }
  }
  return results;
}

// Read a 16-bit PCM WAV via sherpa's reader; returns {sampleRate, samples}.
export function readWave(file) {
  const wave = sherpa.readWave(file);
  return { sampleRate: wave.sampleRate, samples: wave.samples };
}
