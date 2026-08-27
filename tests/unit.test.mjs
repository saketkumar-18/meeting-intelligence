// Unit tests for pure pipeline logic (no models required).
// Run: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert';
import { alignSpeakers, mergeUtterances } from '../lib/pipeline.mjs';
import { extractJson, normalizeAnalysis, speakerName } from '../lib/analyze.mjs';

// ---------- alignSpeakers ----------
test('alignSpeakers assigns chunk to max-overlap speaker', () => {
  const chunks = [{ start: 0, end: 5, text: 'hello' }];
  const segs = [
    { start: 0, end: 2, speaker: 0 },
    { start: 2, end: 5, speaker: 1 },
  ];
  const out = alignSpeakers(chunks, segs);
  assert.equal(out[0].speaker, 1); // 3s overlap vs 2s
});

test('alignSpeakers with no diarization defaults to speaker 0', () => {
  const chunks = [{ start: 0, end: 3, text: 'x' }];
  const out = alignSpeakers(chunks, []);
  assert.equal(out[0].speaker, 0);
});

test('alignSpeakers handles non-overlapping chunk', () => {
  const chunks = [{ start: 10, end: 12, text: 'x' }];
  const segs = [{ start: 0, end: 5, speaker: 2 }];
  const out = alignSpeakers(chunks, segs);
  assert.equal(out[0].speaker, 0); // no overlap -> default
});

// ---------- mergeUtterances ----------
test('mergeUtterances merges consecutive same-speaker chunks', () => {
  const chunks = [
    { speaker: 0, start: 0, end: 2, text: 'hello' },
    { speaker: 0, start: 2.3, end: 4, text: 'world' },
  ];
  const out = mergeUtterances(chunks);
  assert.equal(out.length, 1);
  assert.equal(out[0].text, 'hello world');
  assert.equal(out[0].end, 4);
});

test('mergeUtterances splits on speaker change', () => {
  const chunks = [
    { speaker: 0, start: 0, end: 2, text: 'a' },
    { speaker: 1, start: 2.1, end: 4, text: 'b' },
  ];
  const out = mergeUtterances(chunks);
  assert.equal(out.length, 2);
});

test('mergeUtterances splits on large gap', () => {
  const chunks = [
    { speaker: 0, start: 0, end: 2, text: 'a' },
    { speaker: 0, start: 10, end: 12, text: 'b' },
  ];
  const out = mergeUtterances(chunks);
  assert.equal(out.length, 2);
});

// ---------- extractJson ----------
test('extractJson parses clean JSON', () => {
  const r = extractJson('{"a": 1}');
  assert.deepEqual(r, { a: 1 });
});

test('extractJson strips code fences', () => {
  const r = extractJson('```json\n{"a": 1}\n```');
  assert.deepEqual(r, { a: 1 });
});

test('extractJson finds JSON inside prose', () => {
  const r = extractJson('Here you go: {"a": [1,2]} hope that helps');
  assert.deepEqual(r, { a: [1, 2] });
});

test('extractJson repairs trailing commas', () => {
  const r = extractJson('{"a": 1, "b": [2,],}');
  assert.deepEqual(r, { a: 1, b: [2] });
});

test('extractJson returns null on garbage', () => {
  assert.equal(extractJson('no json here'), null);
  assert.equal(extractJson(''), null);
  assert.equal(extractJson(null), null);
});

// ---------- normalizeAnalysis ----------
test('normalizeAnalysis coerces full shape', () => {
  const a = normalizeAnalysis({
    title: 'Standup',
    summary: 'We met.',
    participants: ['Alice', 42, 'Bob'],
    keyPoints: ['k1', null],
    decisions: ['d1'],
    actionItems: [
      { owner: 'Alice', task: 'ship it', due: 'Friday', context: 'demo' },
      { task: 'no owner' },
      { owner: 'Bob', task: '', due: 'x' }, // empty task -> dropped
      'garbage',
    ],
    openQuestions: ['q1'],
  });
  assert.equal(a.title, 'Standup');
  assert.deepEqual(a.participants, ['Alice', 'Bob']);
  assert.deepEqual(a.keyPoints, ['k1']);
  assert.equal(a.actionItems.length, 2);
  assert.equal(a.actionItems[0].owner, 'Alice');
  assert.equal(a.actionItems[1].owner, 'Unassigned');
  assert.equal(a.actionItems[1].due, 'Not stated');
});

test('normalizeAnalysis handles empty/missing input', () => {
  const a = normalizeAnalysis({});
  assert.equal(a.title, 'Meeting');
  assert.equal(a.summary, '');
  assert.deepEqual(a.actionItems, []);
});

test('speakerName is 1-indexed', () => {
  assert.equal(speakerName(0), 'Speaker 1');
  assert.equal(speakerName(2), 'Speaker 3');
});
