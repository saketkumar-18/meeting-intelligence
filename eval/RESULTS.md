# Evaluation Results

**Meeting:** `eval/meeting1.wav` — synthetic 3-speaker sprint-planning call
(139.7s, 13 turns, 9 planted action items, 2 decisions), generated with
edge-tts using three distinct voices (Alice/Bob/Carol).

**Pipeline:** pyannote 3.0 segmentation (int8, wsr=0.25) → ERes2Net embeddings
→ clustering (k=3) → Whisper base.en per segment → LLM analysis
(qwen3.8-max-free via TokenRouter).

## Headline numbers

| Metric | Value |
|---|---|
| Speaker count accuracy | 3/3 ✅ |
| Speaker attribution (word-weighted) | **100.0%** |
| WER, overall | **7.81%** |
| WER, mean per turn | 8.2% |
| Action-item precision | 0.90 (9/10 predicted were real) |
| Action-item recall | **1.00** (all 9 planted items found) |
| Action-item F1 | **0.947** |
| Utterances produced | 41 |

## Timings (single-threaded WASM, desktop CPU)

| Stage | Time |
|---|---|
| Diarization | 154.6s |
| Whisper ASR (base.en) | 243.6s |
| **Total ML** | **398.2s** (~2.9x realtime) |
| LLM analysis | ~30s |

## Model selection ablations (same meeting)

### Speaker embeddings (k=3 fixed, int8 segmentation, wsr=0.25)

| Embedding model | Attribution |
|---|---|
| CAM++ (VoxCeleb) | 78.6% — Bob & Carol (male/female TTS) merged |
| **ERes2Net (VoxCeleb)** | **100%** |

### Segmentation model / window shift (k=3, 20s clip)

| Config | Time | Segments |
|---|---|---|
| fp32, wsr=0.1 (default) | 13.4s | 7 |
| int8, wsr=0.1 | 13.9s | 6 |
| **int8, wsr=0.25** | **6.2s** | 7 (same quality, 2.2x faster) |
| int8, wsr=0.5 | 4.3s | 7 |

### Clustering threshold (CAM++, k=3 fixed)

Threshold 0.3–0.6: identical output (threshold is ignored when the speaker
count is given). Auto mode (k=-1) works but is less stable on short audio.

## LLM analysis notes

- qwen3.8-max is a reasoning model: without `/no_think` it spends the entire
  token budget on reasoning and returns empty content. Fixed by appending
  `/no_think` and raising max_tokens to 4096.
- All 9 planted action items were extracted with correct owners and deadlines
  ("Friday", "Wednesday", "Thursday", "end of day"). The one false positive
  was a reasonable inference (webhook retry queue setup) that the ground-truth
  script lists under a different phrasing.
- Speaker names were recovered from context ("Bob", "Carol") even though the
  transcript only carries "Speaker N" labels.

## Reproduce

```bash
python tools/make_meeting.py --script tools/meeting1.json --out eval/meeting1.wav
node cli/run.mjs eval/meeting1.wav --speakers 3 --no-llm --out eval/meeting1.eres.json
TOKENROUTER_API_KEY=*** node eval/score_existing.mjs eval/meeting1.eres.json --llm
```
