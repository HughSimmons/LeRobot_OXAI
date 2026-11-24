---
name: meeting
description: Digest an OXAI hardware team meeting from a screen recording (Google Meet via Windows Snipping Tool) into a transcript, key screenshots and a summary with decisions and action items. Use when the user asks to summarise, transcribe, review or take notes from a meeting recording or video, or runs /meeting.
argument-hint: "[recording path(s) or date, default: latest recordings]"
---

# Meeting recording → notes

Everything runs locally (ffmpeg + faster-whisper on CPU). Never upload recordings or
transcripts to an external service.

## 1. Find the recording(s)

- Default location: `D:\Videos\Screen Recordings` = `/mnt/d/Videos/Screen Recordings/`.
  Snipping Tool names files `Screen Recording YYYY-MM-DD HHMMSS.mp4`.
- Convert Windows paths with `wslpath 'D:\...'`. Quote paths: they contain spaces.
- If `$ARGUMENTS` gives paths or a date, use those; otherwise take the most recent
  day's recordings. A meeting may be split into several parts: pass them in time
  order (filename order). If it's unclear which files belong to one meeting, ask.
- Output dir (`$OUT` below): the user's Obsidian vault,
  `/mnt/c/Users/Oliver/Documents/Second Brain/Careers/Robotics/OXAI Hardware Lab/Meetings/<YYYY-MM-DD>/`
  (date of the meeting, not today). Quote it: it has spaces. Read the vault's `CLAUDE.md` before
  writing there. Never delete vault files without asking, and never write anything unverified.
  If it already has a `transcript.txt`, don't redo the processing unless asked.

## 2. Process (slow: run in the background)

Environment: conda env `meeting` (python 3.11, ffmpeg, pillow, numpy, faster-whisper).
The Bash tool doesn't load `~/.bashrc`, so use the full interpreter path. If the env is
missing, create it:

```bash
~/miniforge3/bin/mamba create -y -n meeting -c conda-forge python=3.11 ffmpeg pillow numpy && ~/miniforge3/envs/meeting/bin/pip install faster-whisper
```

Run with `run_in_background: true`. On this laptop (8 cores, no GPU) `small` takes about as long as the recording itself, plus ~3 min for frames:

```bash
~/miniforge3/envs/meeting/bin/python .claude/skills/meeting/scripts/process_meeting.py "<part1.mp4>" "<part2.mp4>" --out "$OUT"
```

Useful flags: `--model medium` (more accurate jargon, ~3x slower), `--prompt "..."` (add
attendee names / new jargon to the default vocabulary hint), `--change-thresh 0.08` (fewer
frames if webcam tiles trigger too many), `--skip-frames` / `--skip-transcript` (redo one half).

Outputs in `$OUT`:
- `sources.txt` – parts, durations, where each sits on the meeting timeline
- `transcript.txt` – `[HH:MM:SS] text` lines (no speaker labels); `transcript.json` same data
- `frames/tHH-MM-SS.jpg` + `frames_index.txt` – frames kept when the screen's sharp edges
  (text, slides, UI) differ from all of the last 8 kept frames, plus one at least every
  2 min. Webcam tiles and window-switching back and forth don't trigger new frames.
- `contact_sheets/sheet_NN.jpg` – 4×4 grids of labelled thumbnails, 16 frames per image

If the script reports `audio=NO`, or the transcript only contains one voice, tell the user
(Snipping Tool's system-audio / mic toggles) and continue with frames only.

Recording tip for the user: in Snipping Tool's record mode, drag the capture area over just
the browser window with the Meet call. That keeps personal windows out of the recording and
avoids stray frames.

## 3. Read it

- Read **all** of `transcript.txt` (chunk with offset/limit if long).
- Skim every contact sheet first to map the meeting: who presented what, and when.
  Meet's header shows "<Name> (Presenting)". Then open individual frames at full size only
  where detail matters: slides, code, plots, CAD, robot footage, and moments where the
  transcript says "as you can see" or "this graph".
- Snipping Tool records the **whole screen**, so frames also capture whatever else the
  user had open: email, logins, banking, social media, chats. Ignore anything that isn't
  the meeting or work shown in it. Never describe, quote or link those frames in the
  summary. If the user was looking at their own work (IDE, Claude app) during the call,
  mention it only when the transcript shows it was discussed.
- Whisper mishears jargon and names. Fix obvious ones using repo vocabulary (grep the repo
  if unsure, e.g. module names), but don't invent content you can't support.
- Speakers aren't labelled. Attribute statements only when the transcript makes it clear
  (names used, "I'll do X" + context); otherwise write "someone suggested…".
- Optionally check `git log --since=<previous meeting date>` to connect discussion to
  recent commits.

## 4. Write `$OUT/summary.md`

```markdown
# OXAI hardware meeting – <date>

Source: <parts, total length>. Transcript quality: <good / patchy / one-sided audio>.
Related: [[Project notes]] · previous meeting: [[Meetings/<prev date>/summary|<prev date>]]

## TL;DR
3–5 bullets.

## Discussion
### <Topic> (HH:MM:SS–HH:MM:SS)
What was said, what was shown (link frames: ![](frames/tHH-MM-SS.jpg) only for the
important ones), conclusions.

## Decisions
- ...

## Action items
| Owner | Task | Due / when | Ref |
|---|---|---|---|

## Open questions / blockers

## Uncertain bits
Parts of the transcript that were garbled or ambiguous, with timestamps, so the user can check.
```

Then give the user the TL;DR, decisions and action items in chat, plus a link to the summary.
The output holds private conversation and screenshots: keep it in the vault, never in the repo.
