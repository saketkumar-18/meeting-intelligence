// Test the official reuse pattern: one diarizer instance, setConfig between runs.
import path from 'path';
import { fileURLToPath } from 'url';
import { readWave } from '../lib/pipeline.mjs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const sherpa = require('sherpa-onnx');
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const M = path.resolve(__dirname, '..', 'models');

const { samples } = readWave(path.join(__dirname, '..', 'eval', 'short20.wav'));
const eres2net = path.join(M, '3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx');

const cfg = {
  segmentation: { pyannote: { model: path.join(M, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.int8.onnx'), windowShiftRatio: 0.25 }, numThreads: 1, debug: 0, provider: 'cpu' },
  embedding: { model: eres2net, numThreads: 1, debug: 0, provider: 'cpu' },
  clustering: { numClusters: 3, threshold: 0.5 },
  minDurationOn: 0.3, minDurationOff: 0.5,
};

const d = sherpa.createOfflineSpeakerDiarization(cfg);

let t0 = Date.now();
let segs = d.process(samples);
console.log(`run1 (k=3): ${Date.now() - t0}ms, ${segs.length} segs, ${new Set(segs.map(s => s.speaker)).size} spk`);

// reuse with setConfig -> auto clustering
const cfg2 = { ...cfg, clustering: { numClusters: -1, threshold: 0.5 } };
d.setConfig(cfg2);
t0 = Date.now();
segs = d.process(samples);
console.log(`run2 (auto via setConfig): ${Date.now() - t0}ms, ${segs.length} segs, ${new Set(segs.map(s => s.speaker)).size} spk`);

// third run back to k=2
const cfg3 = { ...cfg, clustering: { numClusters: 2, threshold: 0.5 } };
d.setConfig(cfg3);
t0 = Date.now();
segs = d.process(samples);
console.log(`run3 (k=2 via setConfig): ${Date.now() - t0}ms, ${segs.length} segs, ${new Set(segs.map(s => s.speaker)).size} spk`);

d.free();
console.log('REUSE PATTERN OK');
