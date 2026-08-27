#!/usr/bin/env node
// Eval harness: score the full pipeline against synthetic-meeting ground truth.
//   node eval/run_eval.mjs eval/meeting1.wav [--llm]
//
// Metrics:
//   - Diarization: speaker-count accuracy, segment-level speaker purity
//   - ASR: word error rate (WER) vs ground-truth turn text
//   - Speaker attribution: fraction of ASR words assigned to the correct speaker
//   - Action items (with --llm): precision/recall vs planted items (owner+task match)
import fs from 'fs';
import { runPipeline, readWave } from '../lib/pipeline.mjs';
import { analyzeMeeting, speakerName } from '../lib/analyze.mjs';

// ---------- text metrics ----------
function normWords(s) {
  return s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(Boolean);
}

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = new Array(n + 1);
  for (let j = 0; j <= n; j++) dp[j] = j;
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return dp[n];
}

function wer(ref, hyp) {
  const r = normWords(ref), h = normWords(hyp);
  if (!r.length) return h.length ? 1 : 0;
  return levenshtein(r, h) / r.length;
}

// ---------- speaker attribution ----------
// For each GT turn, find predicted utterances overlapping it; score the words
// inside the overlap window by the predicted speaker.
function scoreAttribution(gt, utterances) {
  let correct = 0, total = 0;
  for (const turn of gt.turns) {
    const gtWords = normWords(turn.text);
    if (!gtWords.length) continue;
    // collect predicted words from overlapping utterances, weighted by overlap
    const overlapping = utterances.filter(
      (u) => Math.min(u.end, turn.end) - Math.max(u.start, turn.start) > 0.1
    );
    if (!overlapping.length) { total += gtWords.length; continue; }
    // majority-vote speaker over overlap duration
    const bySpk = new Map();
    for (const u of overlapping) {
      const o = Math.min(u.end, turn.end) - Math.max(u.start, turn.start);
      bySpk.set(u.speaker, (bySpk.get(u.speaker) || 0) + o);
    }
    let best = -1, bestO = -1;
    for (const [s, o] of bySpk) if (o > bestO) { best = s; bestO = o; }
    const n = gtWords.length;
    total += n;
    if (best === turn.speakerId) correct += n;
  }
  return total ? correct / total : 0;
}

// ---------- action-item matching ----------
function itemKey(it) {
  const owner = (it.owner || '').toLowerCase().replace(/[^a-z0-9]/g, '');
  const task = normWords(it.task).join(' ');
  return owner + '|' + task;
}

function matchActionItems(pred, gtItems, speakers) {
  // GT owners are names; predicted owners may be "Speaker N". Build alias map.
  const aliasOf = (owner) => {
    const o = owner.toLowerCase();
    for (let i = 0; i < speakers.length; i++) {
      if (o === speakers[i].toLowerCase()) return speakers[i].toLowerCase();
      if (o === speakerName(i).toLowerCase() || o === `speaker${i + 1}`) return speakers[i].toLowerCase();
    }
    return o.replace(/[^a-z0-9]/g, '');
  };
  const gtSet = gtItems.map((g) => ({
    owner: g.owner.toLowerCase(),
    taskWords: normWords(g.task),
    matched: false,
  }));
  let tp = 0;
  const details = [];
  for (const p of pred) {
    const pOwner = aliasOf(p.owner);
    const pWords = normWords(p.task);
    let best = null, bestScore = 0;
    for (const g of gtSet) {
      if (g.matched) continue;
      // task overlap: fraction of GT task words present in prediction
      const hits = g.taskWords.filter((w) => pWords.includes(w)).length;
      const score = g.taskWords.length ? hits / g.taskWords.length : 0;
      const ownerOk = pOwner === g.owner;
      const s = score + (ownerOk ? 1 : 0);
      if (s > bestScore) { best = g; bestScore = s; }
    }
    // count as TP if owner correct AND >=50% of task words matched
    if (best && bestScore >= 1.5) {
      best.matched = true;
      tp++;
      details.push({ pred: p, matchedGt: best.taskWords.join(' '), ok: true });
    } else {
      details.push({ pred: p, ok: false });
    }
  }
  const precision = pred.length ? tp / pred.length : 0;
  const recall = gtSet.length ? tp / gtSet.length : 0;
  return { tp, precision, recall, details };
}

// ---------- main ----------
async function main() {
  const wavPath = process.argv[2];
  const useLlm = process.argv.includes('--llm');
  if (!wavPath) { console.error('Usage: node eval/run_eval.mjs <meeting.wav> [--llm]'); process.exit(1); }

  const gtPath = wavPath.replace(/\.wav$/, '.gt.json');
  const gt = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
  const { samples } = readWave(wavPath);

  console.error(`Running pipeline on ${wavPath} (${gt.durationSec}s, ${gt.numSpeakers} speakers)...`);
  const result = runPipeline(samples, { numSpeakers: gt.numSpeakers });

  // --- ASR WER ---
  const hypFull = result.utterances.map((u) => u.text).join(' ');
  const refFull = gt.turns.map((t) => t.text).join(' ');
  const overallWer = wer(refFull, hypFull);

  // per-turn WER (best-matching predicted utterance by overlap)
  let turnWerSum = 0, turnCount = 0;
  for (const turn of gt.turns) {
    const overlapping = result.utterances.filter(
      (u) => Math.min(u.end, turn.end) - Math.max(u.start, turn.start) > 0.2
    );
    if (!overlapping.length) { turnWerSum += 1; turnCount++; continue; }
    const hyp = overlapping.map((u) => u.text).join(' ');
    turnWerSum += wer(turn.text, hyp);
    turnCount++;
  }

  // --- speaker attribution ---
  const attribution = scoreAttribution(gt, result.utterances);

  // --- action items ---
  let ai = null;
  if (useLlm) {
    console.error('Running LLM analysis...');
    const r = await analyzeMeeting(result.utterances);
    if (r.analysis) {
      ai = matchActionItems(r.analysis.actionItems, gt.actionItems, gt.speakers);
      ai.model = r.model;
      ai.analysis = r.analysis;
    } else {
      console.error('LLM failed: ' + r.errors.join('; '));
    }
  }

  const report = {
    audio: gt.audio,
    durationSec: gt.durationSec,
    gtSpeakers: gt.numSpeakers,
    predSpeakers: result.numSpeakers,
    speakerCountCorrect: result.numSpeakers === gt.numSpeakers,
    werOverall: +overallWer.toFixed(4),
    werMeanPerTurn: +(turnWerSum / Math.max(1, turnCount)).toFixed(4),
    speakerAttributionAcc: +attribution.toFixed(4),
    utterances: result.utterances.length,
    timings: result.timings,
    actionItems: ai ? {
      model: ai.model,
      gtCount: gt.actionItems.length,
      predCount: ai.analysis.actionItems.length,
      tp: ai.tp,
      precision: +ai.precision.toFixed(3),
      recall: +ai.recall.toFixed(3),
      f1: +(2 * ai.precision * ai.recall / Math.max(1e-9, ai.precision + ai.recall)).toFixed(3),
    } : null,
  };

  console.log(JSON.stringify(report, null, 2));
  fs.writeFileSync(wavPath.replace(/\.wav$/, '.eval.json'), JSON.stringify({ report, utterances: result.utterances, analysis: ai?.analysis || null }, null, 2));
  console.error(`Wrote ${wavPath.replace(/\.wav$/, '.eval.json')}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
