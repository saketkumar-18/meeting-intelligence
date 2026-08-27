// Meeting Intelligence — LLM analysis layer.
// transcript -> { summary, actionItems[{owner,task,due}], decisions, participants }
//
// Provider-agnostic with an ordered free-tier fallback chain, mirroring the
// proven DocVQA pattern. Runs in Node (CLI/eval) and is mirrored in Python for
// the Vercel serverless API.

const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';
const TOKENROUTER_BASE = 'https://api.tokenrouter.com/v1';

// (provider, model) chains. Order = preference.
// minimax-m3:free first (non-reasoning, fast ~5-10s), qwen3.8 as fallback
export const TEXT_CHAIN = [
  ['openrouter', 'minimax/minimax-m3:free'],
  ['tokenrouter', 'minimax/minimax-m3:free'],
  ['tokenrouter', 'qwen/qwen3.8-max-free'],
  ['openrouter', 'nvidia/nemotron-3-super-120b-a12b:free'],
  ['openrouter', 'google/gemma-4-31b-it:free'],
];

// Smart transcript truncation: keep first N + last N + action-rich middle
export function truncateTranscript(utterances, maxChars = 12000) {
  if (!utterances.length) return '';
  
  const lines = utterances.map((u) => {
    const spk = `Speaker ${u.speaker + 1}`;
    return `[${u.start.toFixed(1)}s] ${spk}: ${u.text}`;
  });
  
  let full = lines.join('\n');
  if (full.length <= maxChars) return full;
  
  // Keep first 3 + last 3 + estimate middle budget
  const first = lines.slice(0, 3).join('\n');
  const last = lines.slice(-3).join('\n');
  const fixed = first + '\n...\n' + last;
  const remaining = maxChars - fixed.length - 20;
  
  if (remaining > 200) {
    // Find action-rich utterances in the middle (contain commitment keywords)
    const actionKeywords = /\b(will|finish|complete|send|deliver|deploy|launch|review|meet|call|email|schedule|deadline|by\s+\w+day|friday|monday|wednesday|thursday|tuesday)\b/i;
    const middle = lines.slice(3, -3);
    const actionLines = middle.filter((l) => actionKeywords.test(l));
    const otherLines = middle.filter((l) => !actionKeywords.test(l));
    
    let selected = '';
    for (const l of actionLines) {
      if (selected.length + l.length + 1 <= remaining) {
        selected += l + '\n';
      }
    }
    for (const l of otherLines) {
      if (selected.length + l.length + 1 <= remaining) {
        selected += l + '\n';
      } else break;
    }
    return first + '\n...\n' + selected + '\n...\n' + last;
  }
  
  return fixed;
}

export const SYSTEM_PROMPT = `You are a precise meeting-intelligence engine. You are given a diarized meeting transcript with speaker labels and timestamps. Produce a structured analysis. Rules:
1. Use ONLY information present in the transcript. Never invent facts, names, dates, or tasks.
2. Owners must be speakers who actually appear in the transcript. Use their exact label (e.g. "Speaker 1") or their name if stated.
3. An action item is only something a speaker explicitly agrees/commits to do, or is clearly assigned. Copy deadlines and specifics exactly.
4. If a field has no supporting evidence, use an empty array or "Not stated".
5. Respond with ONLY valid JSON matching the schema. No markdown fences, no commentary.`;

export const JSON_SCHEMA_HINT = `{
  "title": "short meeting title",
  "summary": "3-6 sentence executive summary",
  "participants": ["speaker labels or names"],
  "keyPoints": ["bullet point", "..."],
  "decisions": ["decision made", "..."],
  "actionItems": [
    {"owner": "speaker label/name", "task": "what to do", "due": "deadline or 'Not stated'", "context": "one-line why"}
  ],
  "openQuestions": ["unresolved question", "..."]
}`;

function providerCfg(provider) {
  if (provider === 'tokenrouter') {
    return [TOKENROUTER_BASE, (process.env.TOKENROUTER_API_KEY || '').trim()];
  }
  return [OPENROUTER_BASE, (process.env.OPENROUTER_API_KEY || '').trim()];
}

function headers(provider, key) {
  const h = { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };
  if (provider === 'openrouter') {
    h['HTTP-Referer'] = 'https://meeting-intelligence.vercel.app';
    h['X-Title'] = 'Meeting Intelligence';
  }
  return h;
}

async function chat(provider, model, messages, { timeout = 55000, maxTokens = 1024 } = {}) {
  const [base, key] = providerCfg(provider);
  if (!key) return { content: null, error: `${provider}: no api key configured` };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(`${base.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: headers(provider, key),
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0.1 }),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const body = (await res.text()).slice(0, 160);
      return { content: null, error: `${provider}/${model} HTTP ${res.status}: ${body}` };
    }
    const data = await res.json();
    if (data.error) return { content: null, error: `${provider}/${model}: ${String(data.error).slice(0, 160)}` };
    const content = (data.choices?.[0]?.message?.content || '').trim();
    if (!content) return { content: null, error: `${provider}/${model}: empty completion` };
    return { content, error: null };
  } catch (e) {
    return { content: null, error: `${provider}/${model}: ${String(e.message || e).slice(0, 160)}` };
  } finally {
    clearTimeout(timer);
  }
}

// Extract the first JSON object from a model reply (tolerates fences/prose).
export function extractJson(text) {
  if (!text) return null;
  let t = text.trim();
  // strip code fences
  t = t.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '');
  // find first { ... last }
  const start = t.indexOf('{');
  const end = t.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return null;
  const candidate = t.slice(start, end + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    // try to repair trailing commas
    try {
      return JSON.parse(candidate.replace(/,\s*([}\]])/g, '$1'));
    } catch {
      return null;
    }
  }
}

function formatTranscript(utterances, { maxChars = 12000 } = {}) {
  const lines = utterances.map((u) => {
    const t = `[${u.start.toFixed(1)}s] ${speakerName(u.speaker)}: ${u.text}`;
    return t;
  });
  let out = lines.join('\n');
  if (out.length > maxChars) out = out.slice(0, maxChars) + '\n[... transcript truncated ...]';
  return out;
}

export function speakerName(id) {
  return `Speaker ${id + 1}`;
}

export async function analyzeMeeting(utterances, opts = {}) {
  const transcript = format_transcript(utterances, { maxChars: 12000 });
  // Use smart truncation that preserves action items
  const smartTranscript = truncateTranscript(utterances, 12000);
  const userMsg = `Diarized meeting transcript:\n\n${smartTranscript}\n\nAnalyze this meeting and respond with ONLY valid JSON matching this schema:\n${JSON_SCHEMA_HINT}`;
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userMsg },
  ];

  const errors = [];
  for (let i = 0; i < TEXT_CHAIN.length; i++) {
    const [provider, model] = TEXT_CHAIN[i];
    const mt = model.includes('qwen3.8') ? 2048 : 1200;
    // qwen3.8 is a reasoning model: without /no_think it burns the whole
    // budget on reasoning tokens and returns empty content.
    const msgs = model.includes('qwen3.8')
      ? [messages[0], { role: 'user', content: messages[1].content + '\n/no_think' }]
      : messages;
    const { content, error } = await chat(provider, model, msgs, { maxTokens: mt });
    if (error) { errors.push(error); continue; }
    const parsed = extractJson(content);
    if (!parsed) { errors.push(`${provider}/${model}: unparseable JSON`); continue; }
    return { analysis: normalizeAnalysis(parsed), model: `${provider}:${model}`, fallbackUsed: i > 0, errors };
  }
  return { analysis: null, model: null, fallbackUsed: true, errors };
}

// Coerce/validate the analysis into a stable shape.
export function normalizeAnalysis(a) {
  const arr = (x) => (Array.isArray(x) ? x : []);
  return {
    title: typeof a.title === 'string' ? a.title : 'Meeting',
    summary: typeof a.summary === 'string' ? a.summary : '',
    participants: arr(a.participants).filter((x) => typeof x === 'string'),
    keyPoints: arr(a.keyPoints).filter((x) => typeof x === 'string'),
    decisions: arr(a.decisions).filter((x) => typeof x === 'string'),
    actionItems: arr(a.actionItems)
      .filter((x) => x && typeof x === 'object')
      .map((x) => ({
        owner: typeof x.owner === 'string' ? x.owner : 'Unassigned',
        task: typeof x.task === 'string' ? x.task : '',
        due: typeof x.due === 'string' ? x.due : 'Not stated',
        context: typeof x.context === 'string' ? x.context : '',
      }))
      .filter((x) => x.task),
    openQuestions: arr(a.openQuestions).filter((x) => typeof x === 'string'),
  };
}
