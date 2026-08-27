"""Tests for the serverless API (no network calls — LLM layer is monkeypatched)."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
import index as api  # noqa: E402


@pytest.fixture
def client():
    return TestClient(api.app)


SAMPLE_UTTERANCES = [
    {"speaker": 0, "start": 0.0, "end": 3.0, "text": "Let's start the meeting."},
    {"speaker": 1, "start": 3.2, "end": 7.0, "text": "I will send the report by Friday."},
]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "providers" in body


def test_analyze_rejects_empty(client):
    r = client.post("/api/analyze", json={"utterances": []})
    assert r.status_code == 400


def test_analyze_rejects_bad_body(client):
    r = client.post("/api/analyze", content=b"not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_analyze_rejects_too_many(client):
    r = client.post("/api/analyze", json={"utterances": [{}] * 2001})
    assert r.status_code == 400


def test_extract_json_variants():
    assert api.extract_json('{"a": 1}') == {"a": 1}
    assert api.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert api.extract_json('sure: {"a": [1,]} done') == {"a": [1]}
    assert api.extract_json("no json") is None
    assert api.extract_json("") is None


def test_normalize_analysis():
    a = api.normalize_analysis({
        "title": "T", "summary": "S",
        "participants": ["A", 5],
        "actionItems": [
            {"owner": "A", "task": "do it", "due": "Fri"},
            {"owner": "B", "task": ""},
            "junk",
        ],
    })
    assert a["title"] == "T"
    assert a["participants"] == ["A"]
    assert len(a["actionItems"]) == 1
    assert a["actionItems"][0]["owner"] == "A"


def test_normalize_analysis_empty():
    a = api.normalize_analysis({})
    assert a["title"] == "Meeting"
    assert a["actionItems"] == []


def test_format_transcript_truncates():
    utts = [{"speaker": 0, "start": i, "end": i + 1, "text": "word " * 500}
            for i in range(100)]
    out = api.format_transcript(utts)
    assert len(out) <= api.MAX_TRANSCRIPT_CHARS + 100
    assert "truncated" in out


def test_analyze_success_with_mocked_llm(client, monkeypatch):
    canned = json.dumps({
        "title": "Test Meeting",
        "summary": "A test.",
        "participants": ["Speaker 1", "Speaker 2"],
        "keyPoints": ["kp"],
        "decisions": ["d"],
        "actionItems": [{"owner": "Speaker 2", "task": "send report", "due": "Friday"}],
        "openQuestions": [],
    })

    def fake_chat(provider, model, messages, timeout=55.0, max_tokens=1200):
        return canned, None

    monkeypatch.setattr(api, "_chat", fake_chat)
    r = client.post("/api/analyze", json={"utterances": SAMPLE_UTTERANCES})
    assert r.status_code == 200
    body = r.json()
    assert body["analysis"]["title"] == "Test Meeting"
    assert body["analysis"]["actionItems"][0]["owner"] == "Speaker 2"
    assert body["fallbackUsed"] is False


def test_analyze_falls_through_chain(client, monkeypatch):
    canned = json.dumps({"title": "T", "summary": "S", "actionItems": []})
    calls = []

    def fake_chat(provider, model, messages, timeout=55.0, max_tokens=1200):
        calls.append((provider, model))
        if len(calls) < 2:
            return None, "fake 429"
        return canned, None

    monkeypatch.setattr(api, "_chat", fake_chat)
    r = client.post("/api/analyze", json={"utterances": SAMPLE_UTTERANCES})
    assert r.status_code == 200
    body = r.json()
    assert body["fallbackUsed"] is True
    assert len(calls) == 2
    assert "fake 429" in body["errors"][0]


def test_analyze_all_fail(client, monkeypatch):
    def fake_chat(provider, model, messages, timeout=55.0, max_tokens=1200):
        return None, "down"

    monkeypatch.setattr(api, "_chat", fake_chat)
    r = client.post("/api/analyze", json={"utterances": SAMPLE_UTTERANCES})
    assert r.status_code == 502
    assert "all models failed" in r.json()["error"]
