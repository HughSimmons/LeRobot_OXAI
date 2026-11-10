#!/usr/bin/env python3
"""Before/after table for bullseye runs from one source square to its 3x3 neighbourhood.

Each --condition is LABEL=TAG, where TAG is the run-directory tag (the newest run
per target with exactly that tag is used, same lookup as plot_bullseye_neighbourhood.py).
Writes a Markdown table and a CSV with, per target and condition:
within 5 mm, within 2 mm, missed grasps, fell/dropped, median error, and the
median landing point (the bias a place-offset calibration is meant to remove).

Example:
  python lowlevel/drafts/summarise_bullseye_conditions.py --from-square e4 \
      --condition before=e4ring --condition clean=e4cal_clean \
      --condition "blind mean=e4cal_mean" --condition "blind median=e4cal_median"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bullseye_neighbourhood import LOOKUP_DIR, find_run, neighbourhood  # noqa: E402


def run_stats(run: Path) -> dict:
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    samples = [s for s in data["samples"] if "error" not in s]
    landed = [s for s in samples if s.get("landed_upright") and s.get("xy_error_mm") is not None]
    err = np.array([s["xy_error_mm"] for s in landed]) if landed else np.array([])
    vec = np.array([s["final_error_xy_mm"] for s in landed], dtype=float) if landed else np.zeros((0, 2))
    missed = sum(not s.get("pickup_success") for s in samples)
    return {
        "run": run.name,
        "n": len(samples),
        "within_5mm": int((err < 5).sum()),
        "within_2mm": int((err < 2).sum()),
        "missed": missed,
        "fell": len(samples) - len(landed) - missed,
        "median_err_mm": float(np.median(err)) if len(err) else float("nan"),
        "median_landing_xy_mm": np.median(vec, axis=0).tolist() if len(vec) else [float("nan")] * 2,
        "place_offset_mm": [round(v * 1000, 2) for v in data["config"]["place_offset"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-square", default="e4")
    parser.add_argument("--centre", default=None, help="Centre of the 3x3 block of targets (default: source).")
    parser.add_argument("--mode", default="blind")
    parser.add_argument("--condition", action="append", required=True, help="LABEL=TAG")
    parser.add_argument("--output-stem", type=Path, default=None)
    args = parser.parse_args()
    conditions = [tuple(c.split("=", 1)) for c in args.condition]
    targets = [sq for row in neighbourhood(args.centre or args.from_square) for sq in row if sq]

    rows = []
    for dst in targets:
        for label, tag in conditions:
            run = find_run(args.from_square, dst, args.mode, tag)
            if run is not None:
                rows.append({"target": dst, "condition": label, **run_stats(run)})

    stem = args.output_stem or LOOKUP_DIR / f"{args.from_square}_neighbourhood_before_after"
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def cell(r: dict) -> str:
        if r["n"] == 0:
            return "–"
        bx, by = r["median_landing_xy_mm"]
        bias = "–" if np.isnan(bx) else f"({bx:+.1f}, {by:+.1f})"
        med = "–" if np.isnan(r["median_err_mm"]) else f"{r['median_err_mm']:.2f}"
        return (f"{r['within_5mm']}/{r['n']} · {r['within_2mm']}/{r['n']} · "
                f"M{r['missed']} F{r['fell']} · {med} · {bias}")

    lines = [
        f"# {args.from_square} → 3×3 neighbourhood, {args.mode} mode: before vs after",
        "",
        "Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**",
        "",
        "| target | " + " | ".join(label for label, _ in conditions) + " |",
        "|---|" + "---|" * len(conditions),
    ]
    by_key = {(r["target"], r["condition"]): r for r in rows}
    for dst in targets:
        cells = [cell(by_key[(dst, label)]) if (dst, label) in by_key else "not run" for label, _ in conditions]
        lines.append(f"| {dst} | " + " | ".join(cells) + " |")
    # Totals only over targets every condition ran, so the columns compare like with like.
    common = [dst for dst in targets if all((dst, label) in by_key for label, _ in conditions)]
    totals = []
    for label, _ in conditions:
        rs = [by_key[(dst, label)] for dst in common]
        n = sum(r["n"] for r in rs)
        totals.append(
            "–" if not n else
            f"**{sum(r['within_5mm'] for r in rs)}/{n}** · {sum(r['within_2mm'] for r in rs)}/{n} · "
            f"M{sum(r['missed'] for r in rs)} F{sum(r['fell'] for r in rs)}"
        )
    lines.append(f"| **total ({', '.join(common) or 'none'})** | " + " | ".join(totals) + " |")
    lines += ["", "Place offset used (mm, gripper frame):", ""]
    lines.append("| target | " + " | ".join(label for label, _ in conditions) + " |")
    lines.append("|---|" + "---|" * len(conditions))
    for dst in targets:
        cells = [str(by_key[(dst, l)]["place_offset_mm"]) if (dst, l) in by_key else "–" for l, _ in conditions]
        lines.append(f"| {dst} | " + " | ".join(cells) + " |")
    lines += ["", "Run folders:", ""]
    lines += [f"- {r['target']} / {r['condition']}: `{r['run']}`" for r in rows]
    Path(f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(targets) + 7]))
    print(f"\nSaved {stem}.md and {stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
