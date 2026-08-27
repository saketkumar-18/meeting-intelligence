# 🎙️ Meeting Intelligence

**Speaker diarization → transcription → auto summary + action items + who-owns-what.**

A production-ready meeting intelligence pipeline. Drop in a meeting recording and get:

- **Who spoke when** — speaker diarization (pyannote segmentation + CAM++ embeddings + clustering)
- **What was said** — Whisper transcription aligned to speakers
- **What it means** — LLM-generated executive summary, key points, decisions
- **Who owns what** — action items with owner, task, and deadline extraction
- **Exports** — Markdown meeting notes, JSON, SRT subtitles, clipboard

## Architecture

```
┌─────────────────────────── Browser (100% private) ───────────────────────────┐
│                                                                              │
│  audio file ──► decode/resample 16kHz mono                                   │
│                     │                                                        │
│                     ▼                                                        │
│  ┌─────────────────────────────────────────────┐                             │
│  │ sherpa-onnx WASM                            │                             │
│  │  • pyannote 3.0 segmentation (int8)         │                             │
│  │  • ERes2Net speaker embeddings (VoxCeleb)   │                             │
│  │  • fast clustering (how many speakers)      │                             │
│  └─────────────────────────────────────────────┘                             │
│                     │ speaker-labeled segments (double as ASR chunks)        │
│                     ▼                                                        │
│  ┌─────────────────────────────────────────────┐                             │
│  │ Whisper tiny.en (transformers.js, ONNX)     │                             │
│  │  • transcription per diarized segment       │                             │
│  └─────────────────────────────────────────────┘                             │
│                     │ speaker-labeled transcript (TEXT only)                 │
└─────────────────────┼────────────────────────────────────────────────────────┘
                      ▼
┌─────────── Vercel serverless (Python/FastAPI) ───────────┐
│  POST /api/analyze                                        │
│   • multi-provider LLM fallback chain (free tier)         │
│   • structured JSON: summary, action items, owners, dues  │
└───────────────────────────────────────────────────────────┘
```

**Privacy:** audio never leaves the device. All ML (diarization + ASR) runs
in-browser via WebAssembly/ONNX. Only the final text transcript is sent to the
analysis API.

## Live demo

https://meeting-intelligence-lime.vercel.app

## Repo layout

```
static/            # frontend (vanilla JS, no build step)
  index.html       # UI
  app.js           # pipeline orchestration
  style.css
  vendor/          # transformers.js
  models/          # self-hosted Whisper tiny.en ONNX weights
  wasm/            # sherpa-onnx WASM (diarization)
  demo/            # built-in demo meeting WAV
api/
  index.py         # FastAPI serverless: /api/analyze, /api/health
  requirements.txt
lib/
  pipeline.mjs     # Node pipeline (diarize + VAD + Whisper + alignment)
  analyze.mjs      # Node LLM analysis (same chain as the API)
cli/
  run.mjs          # CLI: node cli/run.mjs meeting.wav --out result.json
eval/
  run_eval.mjs     # eval harness: WER, speaker attribution, action-item P/R
tools/
  make_meeting.py  # synthetic meeting generator (edge-tts, ground truth)
  meeting1.json    # eval meeting script (3 speakers, 9 planted action items)
models/            # local Node models (gitignored, ~350MB)
```

## Local development

### 1. Install

```bash
npm install                 # sherpa-onnx (Node WASM)
pip install edge-tts        # only for generating synthetic eval meetings
```

### 2. Download models (Node pipeline)

```bash
cd models
# diarization
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
# ASR (whisper base.en for quality; ~208MB)
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2
tar xjf sherpa-onnx-whisper-base.en.tar.bz2
# VAD
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

### 3. Run the CLI

```bash
# any audio -> 16kHz mono wav first
ffmpeg -i my-meeting.mp3 -ar 16000 -ac 1 meeting.wav

node cli/run.mjs meeting.wav --out result.json
# options: --speakers 3   (known speaker count)
#          --no-llm       (skip AI analysis)
```

### 4. Run the web app locally

```bash
cd static && python -m http.server 8080
# open http://localhost:8080  (LLM analysis needs the API or a proxy)
```

## Evaluation

Synthetic meetings with known ground truth (distinct TTS voices per speaker,
planted action items):

```bash
python tools/make_meeting.py --script tools/meeting1.json --out eval/meeting1.wav
node eval/run_eval.mjs eval/meeting1.wav --llm
```

Metrics: overall WER, mean per-turn WER, speaker attribution accuracy,
speaker-count accuracy, and action-item precision/recall/F1 (owner + task match).

See `eval/RESULTS.md` for the latest numbers.

## API

### `POST /api/analyze`

```json
{
  "utterances": [
    {"speaker": 0, "start": 1.2, "end": 4.5, "text": "Let's kick off..."}
  ]
}
```

Response:

```json
{
  "analysis": {
    "title": "...", "summary": "...",
    "participants": ["..."], "keyPoints": ["..."],
    "decisions": ["..."],
    "actionItems": [{"owner": "...", "task": "...", "due": "...", "context": "..."}],
    "openQuestions": ["..."]
  },
  "model": "openrouter:minimax/minimax-m3:free",
  "fallbackUsed": false
}
```

LLM chain (free tier, falls through on 429/timeout/refusal):
`minimax-m3 → qwen3.8-max → nemotron-3-super → gemma-4`.

### `GET /api/health` — provider key status.

## Deploy (Vercel)

```bash
vercel link
vercel env add TOKENROUTER_API_KEY production    # and/or OPENROUTER_API_KEY
vercel --prod
```

`vercel.json` routes `/api/*` to the Python function and everything else to
`static/`.

## Tech notes

- **Diarization**: sherpa-onnx offline speaker diarization = pyannote 3.0
  segmentation (int8, `windowShiftRatio=0.25` for 2x speed) → ERes2Net
  (VoxCeleb) embeddings → fast agglomerative clustering. ERes2Net scored
  100% speaker attribution on the eval meeting vs 78.6% for CAM++ (similar
  female voices were merging). Threshold 0.5 default; pass expected speaker
  count when known.
- **Segmentation doubles as VAD**: diarization segments are fed directly to
  Whisper as ASR chunks (long ones split at 28s) — no separate VAD pass.
- **Browser ASR**: Whisper tiny.en quantized ONNX via transformers.js,
  self-hosted weights (~40MB total, cached by the browser after first load).
- **Node ASR**: Whisper base.en via sherpa-onnx for higher-quality eval.
- **Alignment**: each ASR chunk inherits its diarization speaker label;
  consecutive same-speaker chunks are merged into turns.
- **No GPU, no cloud ML bills**: everything runs on CPU/WASM.

## License

MIT (project code). Models: sherpa-onnx models are Apache-2.0 (k2-fsa),
Whisper weights are MIT (OpenAI).
