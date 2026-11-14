#!/usr/bin/env python3
"""Tables and figures for a `run_stability_sweep.py` output folder.

Pickup phase -> summary.{csv,md}, grasp_area_heatmap.png, start_maps_small_multiples.png,
best_vs_baseline.png. Set-down phase -> summary.{csv,md}, setdown_heatmap.png,
failure_breakdown.png (and tilt_traces.png when traces were stored).

Stable area = the 4-connected region of passing start offsets that contains the square centre,
in mm^2 (cells x cell area). Ranking: area, then pass count, then mean in-jaw tilt (lower = more
margin), then mean in-jaw slip.

  python -B lowlevel/drafts/stability_report.py lowlevel/rook_kiri_xy_lookup/stability/pickup/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

# Reference palette (dataviz skill): categorical slots in fixed order, sequential blue, ink, surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
PASS_FILL = "#dcdad3"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PICKUP_REASONS = ["knocked_before_grasp", "missed_grasp", "dropped_in_carry", "slipped_in_jaws"]
SETDOWN_REASONS = ["slipped_before_release", "toppled", "leaning", "dragged_on_retreat", "slid_on_board", "not_settled"]
BASELINE_GRASP_MM = (-20.0, -1.0, -3.0)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10,
})


def load(run: Path):
    cfg = json.loads((run / "config.json").read_text())
    rows = {}
    for line in (run / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["setting"], r["name"])
        if "error" in r and key in rows and "error" not in rows[key]:
            continue  # keep a finished row over a later error
        rows[key] = r
    return cfg, list(rows.values())


def connected_area(grid_pass: np.ndarray) -> int:
    n = grid_pass.shape[0]
    c = n // 2
    if not grid_pass[c, c]:
        return 0
    seen = {(c, c)}
    q = deque([(c, c)])
    while q:
        r, k = q.popleft()
        for dr, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, kk = r + dr, k + dk
            if 0 <= rr < n and 0 <= kk < n and grid_pass[rr, kk] and (rr, kk) not in seen:
                seen.add((rr, kk))
                q.append((rr, kk))
    return len(seen)


def summarise(cfg, rows, reasons_order):
    n = cfg["grid_size"]
    cell_mm2 = (2 * cfg["inset_m"] * 1000 / (n - 1)) ** 2
    by = {}
    for r in rows:
        by.setdefault(r["setting"], []).append(r)
    out = []
    for sid, rs in by.items():
        grid = np.zeros((n, n), dtype=bool)
        done = [r for r in rs if "error" not in r]
        for r in done:
            grid[r["row"], r["col"]] = r["pass"]
        reasons = Counter(x for r in done for x in r["reasons"][:1])  # primary reason
        jt = [r["metrics"]["jaw_tilt_max_deg"] for r in done if r["metrics"].get("lifted")]
        js = [r["metrics"]["jaw_slip_max_mm"] for r in done if r["metrics"].get("lifted")]
        r0 = rs[0]
        row = {
            "setting": sid,
            "grasp_x_mm": round(r0["grasp_offset"][0] * 1000, 2),
            "grasp_y_mm": round(r0["grasp_offset"][1] * 1000, 2),
            "grasp_z_mm": round(r0["grasp_offset"][2] * 1000, 2),
            "place_z_mm": round(r0["place_offset"][2] * 1000, 2),
            "lower_steps": r0["lower_steps"],
            "release_clearance_mm": None if r0.get("release_clearance") is None else round(r0["release_clearance"] * 1000, 1),
            "n": len(done),
            "errors": len(rs) - len(done),
            "pass": int(grid.sum()),
            "stable_area_mm2": connected_area(grid) * cell_mm2,
            "mean_jaw_tilt_deg": float(np.mean(jt)) if jt else np.nan,
            "mean_jaw_slip_mm": float(np.mean(js)) if js else np.nan,
            **{k: reasons.get(k, 0) for k in reasons_order},
            "_grid": grid,
            "_rows": done,
        }
        if cfg["phase"] == "setdown":
            for k in ("drop_height_mm", "slide_mm"):
                v = [r["metrics"].get(k) for r in done if r["metrics"].get(k) is not None]
                row[f"mean_{k}"] = float(np.mean(v)) if v else np.nan
            ft = [r["metrics"]["final"]["tilt_deg"] for r in done]
            row["max_final_tilt_deg"] = float(np.max(ft)) if ft else np.nan
            xe = [r["xy_error_mm"] for r in done if r.get("xy_error_mm") is not None]
            row["median_xy_error_mm (info)"] = float(np.median(xe)) if xe else np.nan
        out.append(row)
    add_margin(cfg, out)
    out.sort(key=lambda s: (-s["stable_area_mm2"], -s["margin_mm"], -s["pass"], s["mean_jaw_tilt_deg"]))
    return out, cell_mm2


def add_margin(cfg, out):
    """Pickup: distance (mm, grasp x/y plane) from each grasp to the nearest grasp with a smaller stable
    area, or to just outside the swept grid (untested = assumed failing). 0 for grasps below the best area."""
    if cfg["phase"] != "pickup":
        for s in out:
            s["margin_mm"] = 0.0
        return
    top = max(s["stable_area_mm2"] for s in out)
    worse = [(s["grasp_x_mm"], s["grasp_y_mm"]) for s in out if s["stable_area_mm2"] < top]
    xs = [s["grasp_x_mm"] for s in out]
    ys = [s["grasp_y_mm"] for s in out]
    step = cfg["grasp_step_m"] * 1000
    for s in out:
        if s["stable_area_mm2"] < top:
            s["margin_mm"] = 0.0
            continue
        x, y = s["grasp_x_mm"], s["grasp_y_mm"]
        edge = min(x - min(xs), max(xs) - x, y - min(ys), max(ys) - y) + step
        near = min((np.hypot(x - a, y - b) for a, b in worse), default=np.inf)
        s["margin_mm"] = float(min(edge, near))


def is_baseline(s) -> bool:
    return (s["grasp_x_mm"], s["grasp_y_mm"], s["grasp_z_mm"]) == BASELINE_GRASP_MM


def write_tables(run: Path, cfg, summ, cell_mm2, reasons_order):
    cols = [k for k in summ[0] if not k.startswith("_")]
    with (run / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in summ:
            w.writerow({k: s[k] for k in cols})
    n = cfg["grid_size"]
    total = n * n
    lines = [
        f"# {cfg['from_square']} -> {cfg['to_square']} {cfg['phase']} stability (known mode)",
        "",
        f"Start grid {n}x{n}, +-{cfg['inset_m'] * 1000:.0f} mm (cell {cell_mm2:.0f} mm^2, max area {total * cell_mm2:.0f} mm^2). "
        "Stable area = connected passing region around the centre. Ranked by area, then margin "
        "(distance to the nearest grasp with a smaller area, or just outside the grid), then pass count, in-jaw tilt.",
        "Accuracy is not scored." + (" Baseline grasp (-20, -1, -3) mm marked **B**." if cfg["phase"] == "pickup" else ""),
        "",
    ]
    show = [c for c in cols if c not in ("setting", "n", "errors")]
    lines.append("| rank | " + " | ".join(show) + " |")
    lines.append("|" + "---|" * (len(show) + 1))
    for i, s in enumerate(summ, 1):
        tag = f"{i}" + (" **B**" if cfg["phase"] == "pickup" and is_baseline(s) else "")
        vals = []
        for c in show:
            v = s[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append(f"| {tag} | " + " | ".join(vals) + " |")
    (run / "summary.md").write_text("\n".join(lines) + "\n")


def reason_cmap(reasons_order):
    return ListedColormap([PASS_FILL] + CATEGORICAL[: len(reasons_order)])


def reason_grid(s, n, reasons_order):
    g = np.full((n, n), np.nan)
    for r in s["_rows"]:
        g[r["row"], r["col"]] = 0 if r["pass"] else 1 + reasons_order.index(r["reasons"][0])
    return g


def pickup_figures(run: Path, cfg, summ, cell_mm2):
    xs = sorted({s["grasp_x_mm"] for s in summ})
    ys = sorted({s["grasp_y_mm"] for s in summ})
    n = cfg["grid_size"]
    idx = {(s["grasp_x_mm"], s["grasp_y_mm"]): s for s in summ}
    best = summ[0]
    base = next((s for s in summ if is_baseline(s)), None)

    # 1. grasp heatmaps: area | mean in-jaw tilt | mean in-jaw slip (separate scales, separate panels)
    panels = [("stable_area_mm2", "Stable area (mm²)", False), ("mean_jaw_tilt_deg", "Mean max in-jaw tilt (°)", True),
              ("mean_jaw_slip_mm", "Mean max in-jaw slip (mm)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    cmap = ListedColormap(SEQ_BLUE)
    for ax, (key, title, lower_better) in zip(axes, panels):
        def val(x, y):
            s = idx.get((x, y))
            if s is None or (key != "stable_area_mm2" and s["stable_area_mm2"] == 0):
                return np.nan  # no stable region: in-jaw numbers are meaningless (rook not really held)
            return s[key]

        m = np.array([[val(x, y) for x in xs] for y in ys])
        im = ax.imshow(m, origin="lower", cmap=cmap if not lower_better else cmap.reversed(), aspect="equal")
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                v = m[j, i]
                if not np.isfinite(v):
                    ax.text(i, j, "—", ha="center", va="center", fontsize=7, color=INK2)
                elif (x, y) in idx:
                    txt = f"{v:.0f}" if key == "stable_area_mm2" else f"{v:.1f}"
                    lum = (im.norm(v) if not lower_better else 1 - im.norm(v))
                    ax.text(i, j, txt, ha="center", va="center", fontsize=6.5, color="white" if lum > 0.55 else INK)
        bi, bj = xs.index(best["grasp_x_mm"]), ys.index(best["grasp_y_mm"])
        ax.plot(bi, bj, marker="*", ms=16, mfc="none", mec=INK, mew=1.5)
        if base is not None:
            ax.plot(xs.index(base["grasp_x_mm"]), ys.index(base["grasp_y_mm"]), marker="o", ms=17, mfc="none", mec=INK2, mew=1.5, ls="none")
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
        ax.set_xlabel("grasp x (mm, gripper frame)")
        ax.set_ylabel("grasp y (mm, gripper frame)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"{cfg['from_square']} pickup stability by grasp offset (z {best['grasp_z_mm']:g} mm), "
        f"starts {n}×{n} ±{cfg['inset_m'] * 1000:.0f} mm · ★ best {best['grasp_x_mm']:g}, {best['grasp_y_mm']:g}"
        + (" · ○ baseline −20, −1" if base else ""), fontsize=11)
    fig.savefig(run / "grasp_area_heatmap.png", dpi=150)
    plt.close(fig)

    # 2. small multiples: start map per grasp, coloured by primary fail reason
    cm = reason_cmap(PICKUP_REASONS)
    fig, axes = plt.subplots(len(ys), len(xs), figsize=(1.35 * len(xs) + 1, 1.35 * len(ys) + 1.2), squeeze=False)
    for j, y in enumerate(reversed(ys)):
        for i, x in enumerate(xs):
            ax = axes[j, i]
            ax.set_xticks([])
            ax.set_yticks([])
            s = idx.get((x, y))
            if s is None:
                ax.axis("off")
                continue
            ax.imshow(reason_grid(s, n, PICKUP_REASONS), origin="lower", cmap=cm, vmin=-0.5, vmax=len(PICKUP_REASONS) + 0.5)
            ax.set_title(f"{x:g},{y:g} · {s['stable_area_mm2']:.0f}", fontsize=6.5, color=INK if s is not best else CATEGORICAL[0])
            for sp in ax.spines.values():
                sp.set_color(INK if s is best else GRID)
                sp.set_linewidth(1.6 if s is best or (base is not None and s is base) else 0.6)
    handles = [Patch(color=PASS_FILL, label="pass")] + [Patch(color=CATEGORICAL[k], label=r.replace("_", " ")) for k, r in enumerate(PICKUP_REASONS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"{cfg['from_square']} pickup: start-offset maps per grasp (x, y mm · stable area mm²); "
                 f"each map {n}×{n} starts ±{cfg['inset_m'] * 1000:.0f} mm, x right, y up", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07, wspace=0.15, hspace=0.35)
    fig.savefig(run / "start_maps_small_multiples.png", dpi=150)
    plt.close(fig)

    # 3. best vs baseline: in-jaw tilt per start, failures marked
    pair = [("baseline (−20, −1, −3)", base), (f"best ({best['grasp_x_mm']:g}, {best['grasp_y_mm']:g}, {best['grasp_z_mm']:g})", best)]
    pair = [(t, s) for t, s in pair if s is not None]
    if len(pair) == 2 and pair[0][1] is pair[1][1]:
        pair = [(f"best = baseline (−20, −1, −3)", base)]
    ticks = np.linspace(-cfg["inset_m"] * 1000, cfg["inset_m"] * 1000, n)
    allt = [r["metrics"]["jaw_tilt_max_deg"] for _, s in pair for r in s["_rows"] if r["metrics"].get("lifted")]
    vmin, vmax = (min(allt), max(allt)) if allt else (0, 1)
    fig, axes = plt.subplots(1, len(pair), figsize=(5.2 * len(pair), 4.6), constrained_layout=True, squeeze=False)
    for ax, (title, s) in zip(axes[0], pair):
        g = np.full((n, n), np.nan)
        for r in s["_rows"]:
            if r["metrics"].get("lifted"):
                g[r["row"], r["col"]] = r["metrics"]["jaw_tilt_max_deg"]
        im = ax.imshow(g, origin="lower", cmap=ListedColormap(SEQ_BLUE), vmin=vmin, vmax=vmax)
        for r in s["_rows"]:
            if not r["pass"]:
                ax.plot(r["col"], r["row"], marker="x", ms=9, mew=2, color=CATEGORICAL[1 + PICKUP_REASONS.index(r["reasons"][0]) - 1])
        ax.set_xticks(range(n), [f"{t:g}" for t in ticks], fontsize=7)
        ax.set_yticks(range(n), [f"{t:g}" for t in ticks], fontsize=7)
        ax.set_xlabel("start dx (mm)")
        ax.set_ylabel("start dy (mm)")
        ax.set_title(f"{title}\nstable area {s['stable_area_mm2']:.0f} mm² · pass {s['pass']}/{s['n']} · "
                     f"mean tilt {s['mean_jaw_tilt_deg']:.1f}°")
        fig.colorbar(im, ax=ax, shrink=0.8, label="max in-jaw tilt (°)")
    fig.savefig(run / "best_vs_baseline.png", dpi=150)
    plt.close(fig)


def setdown_figures(run: Path, cfg, summ):
    # heatmap of pass share: place z (or release clearance) x lower steps
    xkey = "release_clearance_mm" if any(s["release_clearance_mm"] is not None for s in summ) else "place_z_mm"
    other = "place_z_mm" if xkey == "release_clearance_mm" else None
    groups = sorted({s[other] for s in summ}) if other else [None]
    xs = sorted({s[xkey] for s in summ})
    ls = sorted({s["lower_steps"] for s in summ})
    fig, axes = plt.subplots(1, len(groups), figsize=(1.0 * len(xs) * len(groups) + 3, 3.4), constrained_layout=True, squeeze=False)
    for ax, gval in zip(axes[0], groups):
        m = np.full((len(ls), len(xs)), np.nan)
        for s in summ:
            if other and s[other] != gval:
                continue
            m[ls.index(s["lower_steps"]), xs.index(s[xkey])] = s["pass"] / max(1, s["n"]) * 100
        im = ax.imshow(m, origin="lower", cmap=ListedColormap(SEQ_BLUE), vmin=0, vmax=100, aspect="auto")
        for j in range(len(ls)):
            for i in range(len(xs)):
                if np.isfinite(m[j, i]):
                    ax.text(i, j, f"{m[j, i]:.0f}%", ha="center", va="center", fontsize=8, color="white" if m[j, i] > 55 else INK)
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_yticks(range(len(ls)), [str(v) for v in ls])
        ax.set_xlabel(xkey.replace("_", " "))
        ax.set_ylabel("lowering steps")
        ax.set_title(f"place z {gval:g} mm" if other else "stable set-downs")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="stable set-downs (%)")
    fig.suptitle(f"{cfg['from_square']} → {cfg['to_square']} set-down stability", fontsize=11)
    fig.savefig(run / "setdown_heatmap.png", dpi=150)
    plt.close(fig)

    # failure breakdown, stacked horizontal bars (primary reason), settings in ranked order
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(summ) + 1.4), constrained_layout=True)
    reasons = PICKUP_REASONS + SETDOWN_REASONS
    labels = [s["setting"].split("__")[-1] for s in summ]
    left = np.zeros(len(summ))
    for k, reason in enumerate(reasons):
        v = np.array([s.get(reason, 0) for s in summ], dtype=float)
        if v.sum() == 0:
            continue
        ax.barh(labels, v, left=left, color=CATEGORICAL[k % len(CATEGORICAL)], edgecolor=SURFACE, linewidth=2,
                label=reason.replace("_", " "))
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("failed starts (primary reason)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.savefig(run / "failure_breakdown.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    cfg, rows = load(a.run)
    order = PICKUP_REASONS + (SETDOWN_REASONS if cfg["phase"] == "setdown" else [])
    summ, cell = summarise(cfg, rows, order)
    write_tables(a.run, cfg, summ, cell, order)
    if cfg["phase"] == "pickup":
        pickup_figures(a.run, cfg, summ, cell)
    else:
        setdown_figures(a.run, cfg, summ)
    print(f"{len(summ)} settings; best {summ[0]['setting']} area {summ[0]['stable_area_mm2']:.0f} mm² "
          f"pass {summ[0]['pass']}/{summ[0]['n']} -> {a.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
