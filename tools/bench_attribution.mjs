// Compare diarization speaker-purity across windowShiftRatio settings,
// scored against synthetic-meeting ground truth.
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
console.log('audio:', (samples.length / 16000).toFixed(1), 's, gt speakers:', gt.numSpeakers);

function diarize(wsr, segModel) {
  const cfg = {
    segmentation: { pyannote: { model: segModel, windowShiftRatio: wsr }, numThreads: 1, debug: 0, provider: 'cpu' },
    embedding: { model: path.join(M, '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx'), numThreads: 1, debug: 0, provider: 'cpu' },
    clustering: { numClusters: gt.numSpeakers, threshold: 0.5 },
    minDurationOn: 0.3, minDurationOff: 0.5,
  };
  const t0 = Date.now();
  const d = sherpa.createOfflineSpeakerDiarization(cfg);
  const segs = d.process(samples);
  const ms = Date.now() - t0;
  d.free();
  return { segs, ms };
}

// Score: for each GT turn, find the predicted speaker with max temporal overlap;
// correct if it matches the GT speakerId. Weighted by turn duration.
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

const int8 = path.join(M, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.int8.onnx');
const fp32 = path.join(M, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.onnx');

for (const [label, wsr, model] of [
  ['int8 wsr=0.10', 0.1, int8],
  ['int8 wsr=0.25', 0.25, int8],
  ['fp32 wsr=0.10', 0.1, fp32],
]) {
  const { segs, ms } = diarize(wsr, model);
  const acc = score(segs);
  const nSpk = new Set(segs.map((s) => s.speaker)).size;
  console.log(`${label}: ${(ms / 1000).toFixed(1)}s, ${segs.length} segs, ${nSpk} spk, attribution=${(acc * 100).toFixed(1)}%`);
}
