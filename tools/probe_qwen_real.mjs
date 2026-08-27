// Probe the REAL analysis prompt against qwen3.8-max to see what breaks.
import fs from 'fs';
import { SYSTEM_PROMPT, JSON_SCHEMA_HINT } from '../lib/analyze.mjs';

const gt = JSON.parse(fs.readFileSync(new URL('../eval/meeting1.gt.json', import.meta.url), 'utf8'));
const lines = gt.turns.map((t) => `[${t.start.toFixed(1)}s] Speaker ${t.speakerId + 1}: ${t.text}`);
const userMsg = `Diarized meeting transcript:\n\n${lines.join('\n')}\n\nAnalyze this meeting and respond with ONLY valid JSON matching this schema:\n${JSON_SCHEMA_HINT}`;

const base = 'https://api.tokenrouter.com/v1';
const key = process.env.TOKENROUTER_API_KEY;
const t0 = Date.now();
const res = await fetch(base + '/chat/completions', {
  method: 'POST',
  headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen/qwen3.8-max-free',
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userMsg + '\n/no_think' },
    ],
    max_tokens: 2048,
    temperature: 0.1,
  }),
});
console.log('status', res.status, 'latency', Date.now() - t0, 'ms');
const data = await res.json();
const ch = data.choices?.[0] || {};
const msg = ch.message || {};
console.log('finish_reason:', ch.finish_reason);
console.log('usage:', JSON.stringify(data.usage));
console.log('content length:', (msg.content || '').length);
console.log('reasoning length:', (msg.reasoning_content || '').length);
console.log('--- content head ---');
console.log((msg.content || '').slice(0, 400));
console.log('--- content tail ---');
console.log((msg.content || '').slice(-400));
fs.writeFileSync(new URL('../eval/probe_qwen_real.json', import.meta.url), JSON.stringify(data, null, 2));
