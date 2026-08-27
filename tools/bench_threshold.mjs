// Threshold sweep for clustering, scored against GT attribution.
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

const campplus = path.join(M, '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx');
const eres2net = path.join(M, '3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx');

const configs = [];
// campplus with fixed 3 clusters, vary threshold (threshold ignored when numClusters set, but test)
for (const t of [0.3, 0.4, 0.5, 0.6, 0.7]) configs.push(['campplus k=3', campplus, t, 3]);
// auto clusters via threshold
for (const t of [0.4, 0.5, 0.6, 0.7]) configs.push(['campplus auto', campplus, t, -1]);

for (const [label, emb, thr, k] of configs) {
  const segs = diarize(emb, thr, k);
  const acc = score(segs);
  const nSpk = new Set(segs.map((s) => s.speaker)).size;
  console.log(`${label} thr=${thr}: ${segs.length} segs, ${nSpk} spk, attribution=${(acc * 100).toFixed(1)}%`);
}

// eres2net if present
if (fs.existsSync(eres2net)) {
  for (const t of [0.4, 0.5, 0.6]) {
    const segs = diarize(eres2net, t, 3);
    const acc = score(segs);
    const nSpk = new Set(segs.map((s) => s.speaker)).size;
    console.log(`eres2net k=3 thr=${t}: ${segs.length} segs, ${nSpk} spk, attribution=${(acc * 100).toFixed(1)}%`);
  }
} else {
  console.log('eres2net not downloaded yet');
}
