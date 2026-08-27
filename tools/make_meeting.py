#!/usr/bin/env python3
"""Generate a synthetic meeting WAV with known ground truth.

Each turn is synthesized with a distinct edge-tts voice, concatenated with
short silences, and resampled to 16 kHz mono. Emits a ground-truth JSON with
per-turn speaker labels, timestamps, and the planted action items so the eval
harness can score diarization, WER, and action-item extraction.

Usage:
  python tools/make_meeting.py --script tools/meeting1.json --out eval/meeting1.wav
"""
import argparse
import asyncio
import json
import subprocess
import tempfile
import os
import wave
import struct

import edge_tts

# Distinct, clearly separable voices.
VOICES = [
    "en-US-GuyNeural",      # male
    "en-US-JennyNeural",    # female
    "en-GB-RyanNeural",     # male, British
    "en-AU-NatashaNeural",  # female, Australian
    "en-IN-NeerjaNeural",   # female, Indian
    "en-US-ChristopherNeural",  # male, deep
]


async def synth(text: str, voice: str, out_mp3: str):
    communicate = edge_tts.Communicate(text, voice, rate="+0%")
    await communicate.save(out_mp3)


def mp3_to_wav16k(mp3: str, wav: str):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
         "-ar", "16000", "-ac", "1", "-f", "wav", wav],
        check=True,
    )


def read_wav_samples(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit pcm"
        n = w.getnframes()
        raw = w.readframes(n)
        sr = w.getframerate()
    samples = struct.unpack(f"<{n}h", raw)
    return sr, samples


def write_wav16k(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="JSON: {turns:[{speaker,text}], actionItems:[...]}")
    ap.add_argument("--out", required=True, help="output 16kHz mono WAV")
    ap.add_argument("--silence-ms", type=int, default=450, help="silence between turns")
    args = ap.parse_args()

    with open(args.script) as f:
        script = json.load(f)

    turns = script["turns"]
    speakers = []
    for t in turns:
        if t["speaker"] not in speakers:
            speakers.append(t["speaker"])
    voice_of = {s: VOICES[i % len(VOICES)] for i, s in enumerate(speakers)}

    silence = [0] * int(16000 * args.silence_ms / 1000)
    all_samples = []
    gt_turns = []
    cursor = 0.0

    tmpdir = tempfile.mkdtemp()
    for i, t in enumerate(turns):
        spk = t["speaker"]
        text = t["text"]
        mp3 = os.path.join(tmpdir, f"turn{i}.mp3")
        wav = os.path.join(tmpdir, f"turn{i}.wav")
        await synth(text, voice_of[spk], mp3)
        mp3_to_wav16k(mp3, wav)
        sr, samples = read_wav_samples(wav)
        start = cursor
        all_samples.extend(samples)
        cursor += len(samples) / 16000.0
        all_samples.extend(silence)
        cursor += len(silence) / 16000.0
        gt_turns.append({
            "speaker": spk,
            "speakerId": speakers.index(spk),
            "start": round(start, 3),
            "end": round(start + len(samples) / 16000.0, 3),
            "text": text,
        })
        print(f"  turn {i}: [{spk}] {len(samples)/16000.0:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_wav16k(args.out, all_samples)

    gt = {
        "audio": os.path.basename(args.out),
        "durationSec": round(cursor, 3),
        "numSpeakers": len(speakers),
        "speakers": speakers,
        "turns": gt_turns,
        "actionItems": script.get("actionItems", []),
        "decisions": script.get("decisions", []),
    }
    gt_path = os.path.splitext(args.out)[0] + ".gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"Wrote {args.out} ({cursor:.1f}s, {len(speakers)} speakers)")
    print(f"Wrote {gt_path}")


if __name__ == "__main__":
    asyncio.run(main())
