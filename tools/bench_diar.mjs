// Benchmark diarization configs on a short clip.
import { readWave } from '../lib/pipeline.mjs';
import { createRequire } from 'module';
import path from 'path';
import { fileURLToPath } from 'url';
const require = createRequire(import.meta.url);
const sherpa = require('sherpa-onnx');
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const M = path.resolve(__dirname, '..', 'models');

const { samples } = readWave(path.join(__dirname, '..', 'eval', 'short20.wav'));
console.log('samples:', samples.length, '=', (samples.length / 16000).toFixed(1), 's');

function bench(label, cfg) {
  const t0 = Date.now();
  const d = sherpa.createOfflineSpeakerDiarization(cfg);
  const segs = d.process(samples);
  const ms = Date.now() - t0;
  d.free();
  const spk = new Set(segs.map((s) => s.speaker)).size;
  console.log(`${label}: ${ms}ms, ${segs.length} segments, ${spk} speakers`);
  return ms;
}

const base = (segModel, wsr) => ({
  segmentation: { pyannote: { model: segModel, windowShiftRatio: wsr }, numThreads: 1, debug: 0, provider: 'cpu' },
  embedding: { model: path.join(M, '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx'), numThreads: 1, debug: 0, provider: 'cpu' },
  clustering: { numClusters: 3, threshold: 0.5 },
  minDurationOn: 0.3, minDurationOff: 0.5,
});

const fp32 = path.join(M, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.onnx');
const int8 = path.join(M, 'sherpa-onnx-pyannote-segmentation-3-0', 'model.int8.onnx');

bench('fp32 wsr=0.1 (default)', base(fp32, 0.1));
bench('int8 wsr=0.1', base(int8, 0.1));
bench('int8 wsr=0.25', base(int8, 0.25));
bench('int8 wsr=0.5', base(int8, 0.5));
