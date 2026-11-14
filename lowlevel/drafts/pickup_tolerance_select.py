#!/usr/bin/env python3
"""Pick the most error-tolerant grasp from a run_pickup_tolerance.py search folder.

Score per grasp (at one error level k): `held` = rook lifted and both jaws still touching at the
end of the carry (the pickup-success definition). Grasps are ranked by the held rate smoothed over the
grasp and its 6 face neighbours on the grid (neighbours outside the grid count as 0, so edge points need
a margin we did not test). With ~40 draws per grasp one sample is 2.5%, so the raw count alone is noise
(first run picked a z-edge point on 40/40 vs 39/40). Ties: clean (strict) passes, then distance to baseline.
Writes <folder>/best.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASELINE = (-20.0, -1.0, -3.0)


def held(r: dict) -> bool:
    m = r.get("metrics") or {}
    ce = m.get("carry_end")
    return bool(m.get("lifted")) and ce is not None and min(ce["jaw_contacts"].values()) > 0


def load(folder: Path) -> list[dict]:
    return [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines() if l.strip()]


def table(rows: list[dict], k: float) -> dict:
    out = {}
    for r in rows:
        if "error" in r or float(r["k"]) != k:
            continue
        g = tuple(round(v * 1000, 1) for v in r["grasp_offset"])
        d = out.setdefault(g, {"n": 0, "held": 0, "strict": 0})
        d["n"] += 1
        d["held"] += held(r)
        d["strict"] += bool(r["pass"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--k", type=float, default=2.0)
    args = ap.parse_args()
    t = table(load(args.folder), args.k)
    xs, ys, zs = (sorted({g[i] for g in t}) for i in range(3))

    def neighbours(g):
        ix, iy, iz = xs.index(g[0]), ys.index(g[1]), zs.index(g[2])
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = ix + dx, iy + dy, iz + dz
            if 0 <= a < len(xs) and 0 <= b < len(ys) and 0 <= c < len(zs):
                yield (xs[a], ys[b], zs[c])
            else:
                yield None  # outside the searched grid: margin unknown, counted as a failure

    ranked = []
    for g, d in t.items():
        nb = [t[n]["held"] / t[n]["n"] if n in t else 0.0 for n in neighbours(g)]
        ranked.append({"grasp_mm": list(g), "held": d["held"], "n": d["n"], "strict": d["strict"],
                       "smoothed_held_rate": float(np.mean([d["held"] / d["n"]] + nb)),
                       "neighbour_held_rate": float(np.mean(nb)) if nb else 0.0,
                       "dist_to_baseline_mm": float(np.linalg.norm(np.subtract(g, BASELINE)))})
    ranked.sort(key=lambda r: (-round(r["smoothed_held_rate"], 9), -r["strict"], r["dist_to_baseline_mm"]))
    base = next((r for r in ranked if tuple(r["grasp_mm"]) == BASELINE), None)
    out = {"k": args.k, "best": ranked[0], "baseline": base, "top10": ranked[:10]}
    (args.folder / "best.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"best": ranked[0], "baseline": base}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
