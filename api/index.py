"""Meeting Intelligence — Vercel serverless API.

POST /api/analyze
  body: {"utterances": [{"speaker": 0, "start": 1.2, "end": 4.5, "text": "..."}]}
  resp: {"analysis": {...}, "model": "provider:model", "fallbackUsed": bool}

GET /api/health
  resp: {"ok": true, "providers": {...}}

The heavy ML (diarization + ASR) runs in the user's browser via WASM; this API
only does the LLM analysis step, so payloads are small text transcripts.
"""
from __future__ import annotations

import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TOKENROUTER_BASE = "https://api.tokenrouter.com/v1"

TEXT_CHAIN = [
    ("openrouter", "minimax/minimax-m3:free"),
    ("tokenrouter", "qwen/qwen3.8-max-free"),
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("openrouter", "google/gemma-4-31b-it:free"),
]

SYSTEM_PROMPT = (
    "You are a precise meeting-intelligence engine. You are given a diarized "
    "meeting transcript with speaker labels and timestamps. Produce a "
    "structured analysis. Rules:\n"
    "1. Use ONLY information present in the transcript. Never invent facts, "
    "names, dates, or tasks.\n"
    "2. Owners must be speakers who actually appear in the transcript. Use "
    'their exact label (e.g. "Speaker 1") or their name if stated.\n'
    "3. An action item is only something a speaker explicitly agrees/commits "
    "to do, or is clearly assigned. Copy deadlines and specifics exactly.\n"
    "4. If a field has no supporting evidence, use an empty array or "
    "'Not stated'.\n"
    "5. Respond with ONLY valid JSON matching the schema. No markdown fences, "
    "no commentary."
)

JSON_SCHEMA_HINT = """{
  "title": "short meeting title",
  "summary": "3-6 sentence executive summary",
  "participants": ["speaker labels or names"],
  "keyPoints": ["bullet point", "..."],
  "decisions": ["decision made", "..."],
  "actionItems": [
    {"owner": "speaker label/name", "task": "what to do", "due": "deadline or 'Not stated'", "context": "one-line why"}
  ],
  "openQuestions": ["unresolved question", "..."]
}"""

MAX_TRANSCRIPT_CHARS = 12000

app = FastAPI(title="Meeting Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _provider_cfg(provider: str):
    if provider == "tokenrouter":
        return TOKENROUTER_BASE, os.environ.get("TOKENROUTER_API_KEY", "").strip()
    return OPENROUTER_BASE, os.environ.get("OPENROUTER_API_KEY", "").strip()


def _headers(provider: str, key: str) -> dict:
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        h["HTTP-Referer"] = "https://meeting-intelligence.vercel.app"
        h["X-Title"] = "Meeting Intelligence"
    return h


def _chat(provider, model, messages, timeout=55.0, max_tokens=1200):
    base, key = _provider_cfg(provider)
    if not key:
        return None, f"{provider}: no api key configured"
    try:
        r = httpx.post(
            base.rstrip("/") + "/chat/completions",
            headers=_headers(provider, key),
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.1},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None, f"{provider}/{model} HTTP {r.status_code}: {r.text[:160]}"
        data = r.json()
        if data.get("error"):
            return None, f"{provider}/{model}: {str(data['error'])[:160]}"
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content or not content.strip():
            return None, f"{provider}/{model}: empty completion"
        return content.strip(), None
    except httpx.TimeoutException:
        return None, f"{provider}/{model}: timeout"
    except Exception as e:  # noqa: BLE001
        return None, f"{provider}/{model}: {str(e)[:160]}"


def extract_json(text: str):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    cand = t[start:end + 1]
    try:
        return json.loads(cand)
    except Exception:  # noqa: BLE001
        import re
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
        except Exception:  # noqa: BLE001
            return None


def normalize_analysis(a: dict) -> dict:
    def arr(x):
        return x if isinstance(x, list) else []

    items = []
    for x in arr(a.get("actionItems")):
        if not isinstance(x, dict):
            continue
        task = x.get("task") if isinstance(x.get("task"), str) else ""
        if not task:
            continue
        items.append({
            "owner": x.get("owner") if isinstance(x.get("owner"), str) else "Unassigned",
            "task": task,
            "due": x.get("due") if isinstance(x.get("due"), str) else "Not stated",
            "context": x.get("context") if isinstance(x.get("context"), str) else "",
        })
    return {
        "title": a.get("title") if isinstance(a.get("title"), str) else "Meeting",
        "summary": a.get("summary") if isinstance(a.get("summary"), str) else "",
        "participants": [x for x in arr(a.get("participants")) if isinstance(x, str)],
        "keyPoints": [x for x in arr(a.get("keyPoints")) if isinstance(x, str)],
        "decisions": [x for x in arr(a.get("decisions")) if isinstance(x, str)],
        "actionItems": items,
        "openQuestions": [x for x in arr(a.get("openQuestions")) if isinstance(x, str)],
    }


def format_transcript(utterances) -> str:
    lines = []
    for u in utterances:
        spk = u.get("speaker", 0)
        name = u.get("speakerName") or f"Speaker {int(spk) + 1}"
        lines.append(f"[{float(u.get('start', 0)):.1f}s] {name}: {u.get('text', '')}")
    out = "\n".join(lines)
    if len(out) > MAX_TRANSCRIPT_CHARS:
        out = out[:MAX_TRANSCRIPT_CHARS] + "\n[... transcript truncated ...]"
    return out


@app.get("/api/health")
def health():
    providers = {}
    for p in ("openrouter", "tokenrouter"):
        _, key = _provider_cfg(p)
        providers[p] = "configured" if key else "missing"
    return {"ok": True, "providers": providers}


@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    utterances = body.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        return JSONResponse({"error": "utterances[] required"}, status_code=400)
    if len(utterances) > 2000:
        return JSONResponse({"error": "too many utterances (max 2000)"}, status_code=400)

    transcript = format_transcript(utterances)
    user_msg = (
        "Diarized meeting transcript:\n\n" + transcript +
        "\n\nAnalyze this meeting and respond with ONLY valid JSON matching "
        "this schema:\n" + JSON_SCHEMA_HINT
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    errors = []
    for i, (provider, model) in enumerate(TEXT_CHAIN):
        mt = 4096 if "qwen3.8" in model else 1200
        # qwen3.8 is a reasoning model: without /no_think it burns the whole
        # budget on reasoning tokens and returns empty content.
        if "qwen3.8" in model:
            msgs = [messages[0],
                    {"role": "user", "content": messages[1]["content"] + "\n/no_think"}]
        else:
            msgs = messages
        # qwen3.8 free tier can take 30-60s. The Vercel function is capped at
        # 60s (Hobby), so give the model 58s — a retry cannot fit in the
        # remaining budget. The UI degrades gracefully if analysis times out.
        content, err = _chat(provider, model, msgs, timeout=58.0, max_tokens=mt)
        if err:
            errors.append(err)
            continue
        parsed = extract_json(content)
        if parsed is None:
            errors.append(f"{provider}/{model}: unparseable JSON")
            continue
        return {
            "analysis": normalize_analysis(parsed),
            "model": f"{provider}:{model}",
            "fallbackUsed": i > 0,
            "errors": errors,
        }
    return JSONResponse({"error": "all models failed", "details": errors}, status_code=502)


# Vercel handler
handler = app
