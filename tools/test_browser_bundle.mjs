// Verify the rebuilt browser WASM bundle (data file with ERes2Net + int8 seg)
// by running it under Node against the 20s clip.
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { readWave } from '../lib/pipeline.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WASM_DIR = path.resolve(__dirname, '..', 'static', 'wasm');

// The browser bundle expects to fetch its .data/.wasm relative to the page.
// Under Node we point it at the local files via Module overrides.
const { samples } = readWave(path.join(__dirname, '..', 'eval', 'short20.wav'));

// Load the bindings (defines createOfflineSpeakerDiarization)
const bindingsSrc = fs.readFileSync(path.join(WASM_DIR, 'sherpa-onnx-speaker-diarization.js'), 'utf8');
// eslint-disable-next-line no-new-func
const bindingsFn = new Function('module', 'exports', bindingsSrc + '\nreturn module.exports;');
const bmod = { exports: {} };
const bindings = bindingsFn(bmod, bmod.exports) || bmod.exports;

// Load the emscripten runtime. It attaches to global Module.
const runtimeSrc = fs.readFileSync(path.join(WASM_DIR, 'sherpa-onnx-wasm-main-speaker-diarization.js'), 'utf8');

const result = await new Promise((resolve, reject) => {
  globalThis.Module = {
    locateFile: (f) => path.join(WASM_DIR, f),
    onRuntimeInitialized() {
      try {
        const sd = bindings.createOfflineSpeakerDiarization(globalThis.Module, {
          segmentation: { pyannote: { model: './segmentation.onnx', windowShiftRatio: 0.25 }, numThreads: 1, debug: 0, provider: 'cpu' },
          embedding: { model: './embedding.onnx', numThreads: 1, debug: 0, provider: 'cpu' },
          clustering: { numClusters: 2, threshold: 0.5 },
          minDurationOn: 0.3, minDurationOff: 0.5,
        });
        const t0 = Date.now();
        const segs = sd.process(samples);
        const ms = Date.now() - t0;
        const nSpk = new Set(segs.map((s) => s.speaker)).size;
        resolve({ ms, segs: segs.length, nSpk, segList: segs });
      } catch (e) { reject(e); }
    },
  };
  // eslint-disable-next-line no-new-func
  new Function(runtimeSrc)();
});

console.log(`browser-bundle diarization: ${result.ms}ms, ${result.segs} segments, ${result.nSpk} speakers`);
result.segList.forEach((s) => console.log(`  ${s.start.toFixed(1)}-${s.end.toFixed(1)} spk${s.speaker}`));
console.log('BROWSER BUNDLE OK');
