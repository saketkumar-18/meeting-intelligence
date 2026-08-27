// Test auto-clustering (numClusters=-1) on the 20s clip — does it hang?
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
  clustering: { numClusters: -1, threshold: 0.5 },
  minDurationOn: 0.3, minDurationOff: 0.5,
};
console.log('starting auto-clustering on 20s clip...');
const t0 = Date.now();
const d = sherpa.createOfflineSpeakerDiarization(cfg);
const segs = d.process(samples);
d.free();
const nSpk = new Set(segs.map((s) => s.speaker)).size;
console.log(`auto done in ${Date.now() - t0}ms: ${segs.length} segs, ${nSpk} speakers`);
