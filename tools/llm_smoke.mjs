// Quick live check of the LLM analysis chain using the ground-truth transcript
// (bypasses ASR). Verifies provider chain + JSON extraction + normalization.
import fs from 'fs';
import { analyzeMeeting } from '../lib/analyze.mjs';

const gt = JSON.parse(fs.readFileSync(new URL('../eval/meeting1.gt.json', import.meta.url), 'utf8'));
const utterances = gt.turns.map((t) => ({
  speaker: t.speakerId, start: t.start, end: t.end, text: t.text,
}));

console.error(`Analyzing ${utterances.length} ground-truth turns...`);
const r = await analyzeMeeting(utterances);
if (!r.analysis) {
  console.error('FAILED:', r.errors.join('\n'));
  process.exit(1);
}
console.error(`model=${r.model} fallback=${r.fallbackUsed}`);
if (r.errors.length) console.error('chain errors:', r.errors);
console.log(JSON.stringify(r.analysis, null, 2));
fs.writeFileSync(new URL('../eval/llm_smoke.json', import.meta.url), JSON.stringify(r, null, 2));
