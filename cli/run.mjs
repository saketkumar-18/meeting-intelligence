#!/usr/bin/env node
// Meeting Intelligence CLI
//   node cli/run.mjs <audio.wav> [--speakers N] [--no-llm] [--out result.json]
//
// Runs the full pipeline: diarization -> VAD -> Whisper ASR -> speaker alignment
// -> (optional) LLM analysis. Prints a human-readable report and writes JSON.
import fs from 'fs';
import path from 'path';
import { runPipeline, readWave, checkModels } from '../lib/pipeline.mjs';
import { analyzeMeeting, speakerName } from '../lib/analyze.mjs';

function parseArgs(argv) {
  const args = { audio: null, speakers: -1, llm: true, out: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--speakers') args.speakers = parseInt(argv[++i], 10);
    else if (a === '--no-llm') args.llm = false;
    else if (a === '--out') args.out = argv[++i];
    else if (!a.startsWith('--')) args.audio = a;
  }
  return args;
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.audio) {
    console.error('Usage: node cli/run.mjs <audio.wav> [--speakers N] [--no-llm] [--out result.json]');
    process.exit(1);
  }
  const missing = checkModels();
  if (missing.length) {
    console.error('Missing models:\n  ' + missing.join('\n  '));
    process.exit(1);
  }

  console.error(`[1/3] Loading ${args.audio} ...`);
  const { sampleRate, samples } = readWave(args.audio);
  if (sampleRate !== 16000) {
    console.error(`Warning: expected 16kHz, got ${sampleRate}Hz. Resample with ffmpeg first.`);
  }
  console.error(`      ${(samples.length / sampleRate).toFixed(1)}s of audio`);

  console.error('[2/3] Running diarization + VAD + Whisper ASR ...');
  const result = runPipeline(samples, { numSpeakers: args.speakers });
  console.error(`      ${result.numSpeakers} speaker(s), ${result.utterances.length} utterances`);
  console.error(`      timings: diar=${result.timings.diarizationMs}ms vad=${result.timings.vadMs}ms asr=${result.timings.asrMs}ms total=${result.timings.totalMs}ms`);

  let analysis = null;
  if (args.llm) {
    console.error('[3/3] Running LLM analysis ...');
    const r = await analyzeMeeting(result.utterances);
    analysis = r.analysis;
    if (analysis) {
      console.error(`      model=${r.model} fallback=${r.fallbackUsed}`);
    } else {
      console.error('      LLM analysis FAILED: ' + r.errors.join('; '));
    }
  }

  // ---- human-readable report ----
  console.log('\n' + '='.repeat(70));
  if (analysis) {
    console.log(`MEETING: ${analysis.title}`);
    console.log('='.repeat(70));
    console.log('\nSUMMARY\n' + analysis.summary);
    if (analysis.participants.length) console.log('\nPARTICIPANTS: ' + analysis.participants.join(', '));
    if (analysis.keyPoints.length) {
      console.log('\nKEY POINTS');
      analysis.keyPoints.forEach((k) => console.log('  • ' + k));
    }
    if (analysis.decisions.length) {
      console.log('\nDECISIONS');
      analysis.decisions.forEach((d) => console.log('  ✓ ' + d));
    }
    if (analysis.actionItems.length) {
      console.log('\nACTION ITEMS (who owns what)');
      analysis.actionItems.forEach((a) => {
        console.log(`  [${a.owner}] ${a.task}${a.due && a.due !== 'Not stated' ? ' — due ' + a.due : ''}`);
      });
    }
    if (analysis.openQuestions.length) {
      console.log('\nOPEN QUESTIONS');
      analysis.openQuestions.forEach((q) => console.log('  ? ' + q));
    }
  }
  console.log('\n' + '-'.repeat(70));
  console.log('TRANSCRIPT');
  console.log('-'.repeat(70));
  for (const u of result.utterances) {
    console.log(`[${fmtTime(u.start)}] ${speakerName(u.speaker)}: ${u.text}`);
  }

  const payload = {
    audio: path.basename(args.audio),
    durationSec: result.durationSec,
    numSpeakers: result.numSpeakers,
    timings: result.timings,
    utterances: result.utterances,
    analysis,
  };
  if (args.out) {
    fs.writeFileSync(args.out, JSON.stringify(payload, null, 2));
    console.error(`\nWrote ${args.out}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
