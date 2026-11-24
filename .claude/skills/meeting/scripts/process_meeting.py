#!/usr/bin/env python
"""Turn meeting screen recording(s) into a timestamped transcript + key frames.

Everything runs locally: ffmpeg for audio/frames, faster-whisper on CPU for speech.
Multiple recordings are treated as consecutive parts of one meeting; timestamps
continue across parts.

Usage:
    python process_meeting.py VIDEO [VIDEO ...] --out meetings/2026-10-10
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PROMPT = (
    "OXAI hardware team meeting. LeRobot, SO-100, SO-101, URDF, MuJoCo, "
    "inverse kinematics, calibration, servo, Feetech, gripper, sim2real, "
    "teleoperation, policy, dataset, chessboard, camera."
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ffmpeg_bin(name):
    """Prefer the binary next to this interpreter (conda env), then PATH."""
    local = Path(sys.executable).parent / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        sys.exit(f"{name} not found; install it in this env (conda install -c conda-forge ffmpeg)")
    return found


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def duration(video):
    out = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(out.strip())


def has_audio(video):
    out = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    return bool(out.strip())


def extract_audio(video, wav):
    subprocess.run(
        [ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(video),
         # dynaudnorm evens out quiet and loud speakers so whisper doesn't drop the quiet ones
         "-vn", "-ac", "1", "-ar", "16000", "-af", "dynaudnorm=f=150:g=15", str(wav)],
        check=True,
    )


def transcribe(parts, out, model_name, language, prompt):
    from faster_whisper import WhisperModel

    log(f"loading whisper model '{model_name}' (first run downloads it)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    segments_out = []
    for part in parts:
        if not part["has_audio"]:
            log(f"part {part['index']}: no audio track, skipping transcription")
            continue
        wav = out / f"audio_part{part['index']}.wav"
        log(f"part {part['index']}: extracting audio")
        extract_audio(part["path"], wav)
        log(f"part {part['index']}: transcribing {hms(part['duration'])} of audio")
        segments, info = model.transcribe(
            str(wav), language=language, initial_prompt=prompt,
            # Defaults silently skipped long stretches of quieter speech on a test
            # recording; these settings recovered them.
            vad_filter=True, vad_parameters=dict(threshold=0.25, min_silence_duration_ms=1000),
            condition_on_previous_text=False, no_speech_threshold=0.8, log_prob_threshold=-1.5,
            beam_size=5,
        )
        last_report = 0.0
        for seg in segments:
            start = part["offset"] + seg.start
            segments_out.append({"part": part["index"], "start": round(start, 2),
                                 "end": round(part["offset"] + seg.end, 2),
                                 "text": seg.text.strip()})
            if seg.end - last_report >= 120:
                last_report = seg.end
                log(f"  ... {hms(seg.end)} / {hms(part['duration'])}")
        wav.unlink()

    with open(out / "transcript.txt", "w") as f:
        for s in segments_out:
            f.write(f"[{hms(s['start'])}] {s['text']}\n")
    with open(out / "transcript.json", "w") as f:
        json.dump(segments_out, f, indent=1)
    log(f"transcript: {len(segments_out)} segments -> {out / 'transcript.txt'}")


def signature(path):
    """Map of strong edges. Text, slides and UI produce sharp edges; webcam tiles of
    people moving mostly don't, so diffing edges ignores talking heads."""
    g = np.asarray(Image.open(path).convert("L").resize((480, 278)), dtype=np.float32)
    gx = np.abs(np.diff(g, axis=1))[:-1]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    return np.maximum(gx, gy) > 40


def extract_frames(parts, out, interval, change_thresh, max_gap, width):
    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    index_lines = []
    for part in parts:
        tmp = out / f"_frames_tmp_part{part['index']}"
        tmp.mkdir(exist_ok=True)
        log(f"part {part['index']}: sampling a frame every {interval}s")
        subprocess.run(
            [ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(part["path"]),
             "-vf", f"fps=1/{interval},scale={width}:-2", "-q:v", "3",
             str(tmp / "f_%05d.jpg")],
            check=True,
        )
        # Compare against several recently kept frames, not just the last one, so
        # alt-tabbing between the call and another window doesn't keep both every time.
        recent, last_t = deque(maxlen=8), -1e9
        candidates = sorted(tmp.glob("f_*.jpg"))
        for i, path in enumerate(candidates):
            t = part["offset"] + i * interval
            sig = signature(path)
            changed = min((float((sig ^ r).mean()) for r in recent), default=1.0)
            if changed >= change_thresh or t - last_t >= max_gap:
                name = f"t{hms(t).replace(':', '-')}.jpg"
                shutil.move(str(path), frames_dir / name)
                reason = "new" if changed >= change_thresh else "periodic"
                index_lines.append(f"{hms(t)}  frames/{name}  ({reason}, edge diff {changed:.3f})")
                recent.append(sig)
                last_t = t
        log(f"part {part['index']}: kept {sum(1 for _ in frames_dir.glob('*.jpg'))} frames "
            f"so far from {len(candidates)} samples")
        shutil.rmtree(tmp)
    (out / "frames_index.txt").write_text("\n".join(index_lines) + "\n")
    log(f"frames: {len(index_lines)} kept -> {frames_dir}")
    contact_sheets(sorted(frames_dir.glob("*.jpg")), out / "contact_sheets")


def contact_sheets(frames, sheets_dir, cols=4, rows=4, thumb_w=480):
    """Grids of labelled thumbnails, for skimming many frames in one image."""
    from PIL import ImageDraw

    if sheets_dir.exists():
        shutil.rmtree(sheets_dir)
    sheets_dir.mkdir()
    per = cols * rows
    for s in range(0, len(frames), per):
        batch = frames[s:s + per]
        thumbs = []
        for f in batch:
            im = Image.open(f).convert("RGB")
            im = im.resize((thumb_w, round(im.height * thumb_w / im.width)))
            ImageDraw.Draw(im).rectangle([0, 0, 110, 22], fill="black")
            ImageDraw.Draw(im).text((6, 5), f.stem[1:].replace("-", ":"), fill="yellow")
            thumbs.append(im)
        th = thumbs[0].height
        sheet = Image.new("RGB", (cols * thumb_w, ((len(thumbs) - 1) // cols + 1) * th), "white")
        for i, im in enumerate(thumbs):
            sheet.paste(im, ((i % cols) * thumb_w, (i // cols) * th))
        sheet.save(sheets_dir / f"sheet_{s // per + 1:02d}.jpg", quality=85)
    log(f"contact sheets: {(len(frames) - 1) // per + 1} -> {sheets_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path, help="recording(s), in meeting order")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--model", default="small",
                    help="whisper model: tiny/base/small/medium/large-v3 (default: small)")
    ap.add_argument("--language", default="en", help="spoken language code, or 'auto'")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="vocabulary hint for whisper (names, jargon)")
    ap.add_argument("--frame-interval", type=float, default=5, help="seconds between sampled frames")
    ap.add_argument("--change-thresh", type=float, default=0.05,
                    help="edge-map difference vs recent kept frames needed to keep a frame")
    ap.add_argument("--max-gap", type=float, default=120,
                    help="keep a frame at least this often (seconds) even if unchanged")
    ap.add_argument("--frame-width", type=int, default=1280)
    ap.add_argument("--skip-transcript", action="store_true")
    ap.add_argument("--skip-frames", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    parts, offset = [], 0.0
    for i, v in enumerate(args.videos, start=1):
        if not v.exists():
            sys.exit(f"not found: {v}")
        d = duration(v)
        parts.append({"index": i, "path": v, "duration": d, "offset": offset, "has_audio": has_audio(v)})
        log(f"part {i}: {v.name}  {hms(d)}  audio={'yes' if parts[-1]['has_audio'] else 'NO'}  "
            f"starts at {hms(offset)}")
        offset += d

    with open(args.out / "sources.txt", "w") as f:
        for p in parts:
            f.write(f"part {p['index']}: {p['path']}  duration {hms(p['duration'])}  "
                    f"meeting time {hms(p['offset'])}-{hms(p['offset'] + p['duration'])}  "
                    f"audio={'yes' if p['has_audio'] else 'no'}\n")

    if not args.skip_frames:
        extract_frames(parts, args.out, args.frame_interval, args.change_thresh,
                       args.max_gap, args.frame_width)
    if not args.skip_transcript:
        lang = None if args.language == "auto" else args.language
        transcribe(parts, args.out, args.model, lang, args.prompt)
    log("done")


if __name__ == "__main__":
    main()
