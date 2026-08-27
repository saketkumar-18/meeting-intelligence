#!/usr/bin/env node
// Score an EXISTING pipeline output (meeting*.eres.json / .pipeline.json)
// against ground truth, optionally running the LLM for action items.
// Avoids re-running the ~7min ML pipeline.
//   node eval/score_existing.mjs eval/meeting1.eres.json [--llm]
import fs from 'fs';
import { analyzeMeeting, speakerName } from '../lib/analyze.mjs';

function normWords(s) {
  return s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(Boolean);
}
function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = new Array(n + 1);
  for (let j = 0; j <= n; j++) dp[j] = j;
  for (let i = 1; i <= m; i++) {
    let prev = dp[0]; dp[0] = i;
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

function scoreAttribution(gt, utterances) {
  let correct = 0, total = 0;
  for (const turn of gt.turns) {
    const gtWords = normWords(turn.text);
    if (!gtWords.length) continue;
    const overlapping = utterances.filter(
      (u) => Math.min(u.end, turn.end) - Math.max(u.start, turn.start) > 0.1);
    if (!overlapping.length) { total += gtWords.length; continue; }
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

function matchActionItems(pred, gtItems, speakers) {
  const aliasOf = (owner) => {
    const o = owner.toLowerCase();
    for (let i = 0; i < speakers.length; i++) {
      if (o === speakers[i].toLowerCase()) return speakers[i].toLowerCase();
      if (o === speakerName(i).toLowerCase() || o === `speaker${i + 1}`) return speakers[i].toLowerCase();
    }
    return o.replace(/[^a-z0-9]/g, '');
  };
  const gtSet = gtItems.map((g) => ({ owner: g.owner.toLowerCase(), taskWords: normWords(g.task), matched: false }));
  let tp = 0;
  for (const p of pred) {
    const pOwner = aliasOf(p.owner);
    const pWords = normWords(p.task);
    let best = null, bestScore = 0;
    for (const g of gtSet) {
      if (g.matched) continue;
      const hits = g.taskWords.filter((w) => pWords.includes(w)).length;
      const score = g.taskWords.length ? hits / g.taskWords.length : 0;
      const ownerOk = pOwner === g.owner;
      const s = score + (ownerOk ? 1 : 0);
      if (s > bestScore) { best = g; bestScore = s; }
    }
    if (best && bestScore >= 1.5) { best.matched = true; tp++; }
  }
  const precision = pred.length ? tp / pred.length : 0;
  const recall = gtSet.length ? tp / gtSet.length : 0;
  return { tp, precision, recall };
}

async function main() {
  const jsonPath = process.argv[2];
  const useLlm = process.argv.includes('--llm');
  if (!jsonPath) { console.error('Usage: node eval/score_existing.mjs <pipeline.json> [--llm]'); process.exit(1); }

  const result = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  const gtPath = jsonPath.replace(/\.eres\.json$|\.pipeline\.json$/, '.gt.json').replace(/meeting1\./, 'meeting1.');
  // ground truth lives next to the wav: eval/meeting1.gt.json
  const gt = JSON.parse(fs.readFileSync(new URL('../eval/meeting1.gt.json', import.meta.url), 'utf8'));
  const utterances = result.utterances;

  const hypFull = utterances.map((u) => u.text).join(' ');
  const refFull = gt.turns.map((t) => t.text).join(' ');
  const overallWer = wer(refFull, hypFull);

  let turnWerSum = 0, turnCount = 0;
  for (const turn of gt.turns) {
    const overlapping = utterances.filter((u) => Math.min(u.end, turn.end) - Math.max(u.start, turn.start) > 0.2);
    if (!overlapping.length) { turnWerSum += 1; turnCount++; continue; }
    turnWerSum += wer(turn.text, overlapping.map((u) => u.text).join(' '));
    turnCount++;
  }

  const attribution = scoreAttribution(gt, utterances);

  let ai = null;
  if (useLlm) {
    console.error('Running LLM analysis...');
    const r = await analyzeMeeting(utterances);
    if (r.analysis) {
      const m = matchActionItems(r.analysis.actionItems, gt.actionItems, gt.speakers);
      ai = { ...m, model: r.model, analysis: r.analysis };
    } else {
      console.error('LLM failed: ' + r.errors.join('; '));
    }
  }

  const report = {
    source: jsonPath,
    durationSec: result.durationSec,
    gtSpeakers: gt.numSpeakers,
    predSpeakers: result.numSpeakers,
    speakerCountCorrect: result.numSpeakers === gt.numSpeakers,
    werOverall: +overallWer.toFixed(4),
    werMeanPerTurn: +(turnWerSum / Math.max(1, turnCount)).toFixed(4),
    speakerAttributionAcc: +attribution.toFixed(4),
    utterances: utterances.length,
    timings: result.timings,
    actionItems: ai ? {
      model: ai.model, gtCount: gt.actionItems.length, predCount: ai.analysis.actionItems.length,
      tp: ai.tp, precision: +ai.precision.toFixed(3), recall: +ai.recall.toFixed(3),
      f1: +(2 * ai.precision * ai.recall / Math.max(1e-9, ai.precision + ai.recall)).toFixed(3),
    } : null,
  };
  console.log(JSON.stringify(report, null, 2));
  fs.writeFileSync(jsonPath.replace(/\.json$/, '.eval.json'),
    JSON.stringify({ report, analysis: ai?.analysis || null }, null, 2));
}
main().catch((e) => { console.error(e); process.exit(1); });
