// Probe: what does qwen3.8-max actually return for our analysis prompt?
const base = 'https://api.tokenrouter.com/v1';
const key = process.env.TOKENROUTER_API_KEY;
const t0 = Date.now();
const res = await fetch(base + '/chat/completions', {
  method: 'POST',
  headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen/qwen3.8-max-free',
    messages: [
      { role: 'system', content: 'Respond with ONLY valid JSON. No commentary.' },
      { role: 'user', content: 'Return exactly: {"ok": true, "n": 3}' },
    ],
    max_tokens: 2048,
    temperature: 0.1,
  }),
});
console.log('status', res.status, 'latency', Date.now() - t0, 'ms');
const data = await res.json();
const msg = data.choices?.[0]?.message || {};
console.log('finish_reason:', data.choices?.[0]?.finish_reason);
console.log('content:', JSON.stringify(msg.content)?.slice(0, 500));
console.log('reasoning_content:', JSON.stringify(msg.reasoning_content)?.slice(0, 300));
console.log('usage:', JSON.stringify(data.usage));
