# Gold-mask batch

Standalone app for building a gold-standard **piece silhouette** dataset with interactive SAM clicks. Images can be square crops **or** wider board / multi-piece views.

## Layout

```text
gold_mask_batch/
  input/
    squares/             # individual square crops
    boards/              # full / multi-piece scenes
  output/                # written pairs + index.jsonl
  run_gold_mask_batch.py
  requirements.txt
```

## Environments

Own `.venv` in this folder (no `helk` required). First run may download
`facebook/sam2.1-hiera-tiny` from Hugging Face (network once).

```bash
cd gold_mask_batch
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_gold_mask_batch.py
```

Optional: `--input`, `--output`, `--overwrite`.

When placed under OXAI as `LeRobot_OXAI/lowlevel/gold_mask_batch/`, shared SAM
code is imported from `../coord_finder/` (`piece_mask_sam.py`).

## Multi-piece workflow

For each image, mask **every** visible piece:

1. LMB / RMB — refine the **current** piece proposal (green dots FG, red BG; cyan tint = proposal).
2. **`a`** or **Space** — add that proposal into the committed union (green tint = all pieces so far).
3. Repeat for other pieces. **`b`** undoes the last committed piece; **`u`** undoes the last click on the current proposal.
4. **Enter** / **`s`** — write the union mask and advance (auto-adds the current proposal if it is ok).

Controls are also drawn on the live window.

| Key / click | Action |
| --- | --- |
| LMB / RMB | FG / BG on current piece |
| `a` / Space | Commit current piece into union |
| `b` | Undo last committed piece |
| `u` / `r` | Undo last point / reset everything |
| Enter / `s` | Accept union → `image.png` + `mask.png` |
| `n` / `p` | Next / previous |
| `x` | Skip |
| `q` | Quit |

## Output

```text
output/<stem>/
  image.png
  mask.png       # union of all committed piece silhouettes
  overlay.png
  meta.json      # includes n_pieces
output/index.jsonl
```

Already-labeled stems (with `mask.png`) are skipped unless `--overwrite`.
