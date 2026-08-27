// Isolated tests: auto-clustering crash check + eres2net k=3.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { readWave } from '../lib/pipeline.mjs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const sherpa = require('sherpa-onnx');
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const M = path.resolve(__dirname, '..', 'models');

const gt = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'eval', 'meeting1.gt.json'), 'utf8'));
const { samples } = readWave(path.join(__dirname, '..', 'eval', 'meeting1.wav'));

function score(segs) {
  let correct = 0, total = 0;
  for (const turn of gt.turns) {
    const overlap = new Map();
    for (const s of segs) {
      const o = Math.min(turn.end, s.end) - Math.max(turn.start, s.start);
      if (o > 0) overlap.set(s.speaker, (overlap.get(s.speaker) || 0) + o);
    }
    let best = -1, bestO = 0;
    for (const [spk, o] of overlap) if (o > bestO) { best = spk; bestO = o; }
    const dur = turn.end - turn.start;
    total += dur;
    if (best === turn.speakerId) correct += dur;
  }
  return correct / total;
}

function diarize(embModel, threshold, numClusters) {
  const cfg = {
    segmentation: { pyannote: { model: path.join(M, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.int8.onnx'), windowShiftRatio: 0.25 }, numThreads: 1, debug: 0, provider: 'cpu' },
    embedding: { model: embModel, numThreads: 1, debug: 0, provider: 'cpu' },
    clustering: { numClusters, threshold },
    minDurationOn: 0.3, minDurationOff: 0.5,
  };
  const d = sherpa.createOfflineSpeakerDiarization(cfg);
  const segs = d.process(samples);
  d.free();
  return segs;
}

const eres2net = path.join(M, '3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx');

// eres2net with fixed 3 clusters
for (const t of [0.5]) {
  try {
    const segs = diarize(eres2net, t, 3);
    const acc = score(segs);
    const nSpk = new Set(segs.map((s) => s.speaker)).size;
    console.log(`eres2net k=3 thr=${t}: ${segs.length} segs, ${nSpk} spk, attribution=${(acc * 100).toFixed(1)}%`);
  } catch (e) {
    console.log('eres2net k=3 FAILED:', e.message);
  }
}

// eres2net auto clustering
for (const t of [0.5, 0.6]) {
  try {
    const segs = diarize(eres2net, t, -1);
    const acc = score(segs);
    const nSpk = new Set(segs.map((s) => s.speaker)).size;
    console.log(`eres2net auto thr=${t}: ${segs.length} segs, ${nSpk} spk, attribution=${(acc * 100).toFixed(1)}%`);
  } catch (e) {
    console.log(`eres2net auto thr=${t} FAILED:`, e.message);
  }
}
