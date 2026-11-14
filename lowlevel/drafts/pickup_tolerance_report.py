#!/usr/bin/env python3
"""Figures + slide-ready markdown for the pickup tolerance runs (run_pickup_tolerance_cde.sh).

  python -B lowlevel/drafts/pickup_tolerance_report.py lowlevel/rook_kiri_xy_lookup/stability/pickup_tolerance

Reads (whatever exists): fk_bound/, e4_search/, e4_scale/, squares/<sq>/, d4_search/.
Writes report/: PICKUP_TOLERANCE.md, fig*.png, *.csv.
Pickup success ("held") = rook lifted and both jaws still touching at the end of the carry.
"Clean" = also passes the strict probe rules (no knock > 3 deg / 3 mm before grasp, slip in jaws <= 2 mm).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

BASELINE = (-20.0, -1.0, -3.0)
SIGMA = [0.80, 0.62, 0.22, 0.27, 0.70]
FILES, RANKS = "cde", [7, 6, 5, 4, 3, 2]
# reference palette (dataviz skill): categorical slot 1 blue = best, slot 2 orange = baseline
C_BEST, C_BASE = "#2a78d6", "#eb6834"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ee", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "text.color": INK,
})


def held(r: dict) -> bool:
    m = r.get("metrics") or {}
    ce = m.get("carry_end")
    return bool(m.get("lifted")) and ce is not None and min(ce["jaw_contacts"].values()) > 0


def outcome(r: dict) -> str:
    if not held(r):
        return "missed grasp" if not (r.get("metrics") or {}).get("lifted") else "dropped in carry"
    if r.get("pass"):
        return "held, clean"
    if "knocked_before_grasp" in r["reasons"]:
        return "held, rook knocked first"
    return "held, slipped > 2 mm in jaws"


OUTCOMES = ["held, clean", "held, slipped > 2 mm in jaws", "held, rook knocked first", "dropped in carry", "missed grasp"]


def load(folder: Path) -> list[dict]:
    path = folder / "results.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        r["g"] = tuple(round(v * 1000, 1) for v in r["grasp_offset"])
        r["k"] = float(r["k"])
        r["held"] = held(r)
        r["outcome"] = outcome(r)
        rows.append(r)
    return rows


def rate(rows, **kw):
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    n = len(sel)
    h = sum(r["held"] for r in sel)
    c = sum(bool(r["pass"]) for r in sel)
    return h, c, n


def wilson(h, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def pct(h, n):
    return f"{100 * h / n:.0f}%" if n else "n/a"


def gstr(g):
    return f"({g[0]:+.0f}, {g[1]:+.0f}, {g[2]:+.0f}) mm"


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig_fk(out: Path, fk_csv: Path):
    rows = list(csv.DictReader(fk_csv.open()))
    rows.sort(key=lambda r: float(r["reach_mm"]))
    sq = [r["square"] for r in rows]
    comps = [("T1_rest_tangential_mm", "sideways (pan)"), ("T1_rest_z_mm", "height"), ("T1_rest_radial_mm", "reach (radial)")]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(sq))
    w = 0.26
    for i, (col, lab) in enumerate(comps):
        v = [float(r[col]) for r in rows]
        ax.bar(x + (i - 1) * w, v, w * 0.9, color=CAT[[0, 2, 3][i]], label=lab)
    ax.set_xticks(x, [f"{s}\n{float(r['reach_mm']):.0f}" for s, r in zip(sq, rows)], fontsize=8)
    ax.set_ylabel("1σ grasp-point error (mm)")
    ax.set_xlabel("square  /  reach from robot base (mm)", color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    ax.set_title("Measured joint error at rest (Oliver, 14 Nov) pushed through forward kinematics")
    save(fig, out / "fig1_fk_bound.png")
    return rows


def fig_search(out: Path, rows: list[dict], best: tuple, name: str, title: str):
    ks = sorted({r["k"] for r in rows})
    k = 2.0 if 2.0 in ks else ks[0]
    rows = [r for r in rows if r["k"] == k]
    xs, ys, zs = (sorted({r["g"][i] for r in rows}) for i in range(3))
    fig, axes = plt.subplots(1, len(zs), figsize=(2.6 * len(zs) + 0.8, 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    grid = {}
    for ax, z in zip(axes, zs):
        m = np.full((len(ys), len(xs)), np.nan)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                h, c, n = rate(rows, g=(x, y, z))
                if n:
                    m[iy, ix] = 100 * h / n
                    grid[(x, y, z)] = (h, c, n)
        im = ax.imshow(m, origin="lower", cmap=SEQ, vmin=0, vmax=100, aspect="equal")
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                if not np.isnan(m[iy, ix]):
                    ax.text(ix, iy, f"{m[iy, ix]:.0f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if m[iy, ix] > 60 else INK)
        for g, col, mk in ((BASELINE, C_BASE, "s"), (best, C_BEST, "o")):
            if g[2] == z and g[0] in xs and g[1] in ys:
                ax.plot(xs.index(g[0]), ys.index(g[1]), mk, ms=19, mfc="none", mec=col, mew=2.2)
        ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs], fontsize=8)
        ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys], fontsize=8)
        ax.set_title(f"grasp z = {z:g} mm", fontsize=9.5)
        ax.set_xlabel("grasp x (mm)", fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("grasp y (mm)")
    cb = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.01)
    cb.set_label("% picked up and held", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(title + f"  (error level k = {k:g}; orange square = baseline, blue circle = best)",
                 fontsize=10.5, fontweight="bold", x=0.02, ha="left", y=1.02)
    save(fig, out / name)
    with (out / name.replace(".png", ".csv")).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grasp_x_mm", "grasp_y_mm", "grasp_z_mm", "k", "held", "clean", "n"])
        for g, (h, c, n) in sorted(grid.items()):
            w.writerow([*g, k, h, c, n])
    return grid, k


def fig_scale(out: Path, rows: list[dict], best: tuple, e4_tangential_mm: float | None):
    ks = sorted({r["k"] for r in rows})
    grasps = [g for g in (BASELINE, best) if any(r["g"] == g for r in rows)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    table = []
    for g in grasps:
        col = C_BASE if g == BASELINE else C_BEST
        lab = ("baseline " if g == BASELINE else "best ") + gstr(g)
        v, lo, hi = [], [], []
        for k in ks:
            h, c, n = rate(rows, g=g, k=k)
            a, b = wilson(h, n)
            v.append(100 * h / n)
            lo.append(100 * a)
            hi.append(100 * b)
            table.append((g, k, h, c, n))
        ax.fill_between(ks, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(ks, v, "-o", color=col, lw=2, ms=7, mec=SURF, mew=1.5, label=lab)
        ax.annotate(f"{v[-1]:.0f}%", (ks[-1], v[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
    ax.axvline(1, color=MUTED, lw=1, ls=":")
    ax.text(1.04, 32, "measured\nat rest", fontsize=8, color=MUTED, va="bottom")
    ax.set_xticks(ks, [f"{k:g}×" + (f"\n±{k * e4_tangential_mm:.1f} mm" if e4_tangential_mm else "") for k in ks])
    ax.set_xlabel("joint error, multiple of Oliver's measured RMS  (sideways 1σ at e4)", color=INK2)
    ax.set_ylabel("% picked up and held (95% CI)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("e4: pickup success as the arm error grows (64 fresh draws per point)")
    save(fig, out / "fig3_scale_curve.png")
    return table


def fig_board(out: Path, sq_rows: dict, best: tuple, k: float):
    grasps = [BASELINE] + ([best] if best != BASELINE else [])
    fig, axes = plt.subplots(1, len(grasps), figsize=(2.9 * len(grasps) + 0.8, 5.2))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, grasps):
        m = np.full((len(RANKS), len(FILES)), np.nan)
        for iy, rk in enumerate(RANKS):
            for ix, f in enumerate(FILES):
                h, c, n = rate(sq_rows.get(f"{f}{rk}", []), g=g, k=k)
                if n:
                    m[iy, ix] = 100 * h / n
        im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=100)
        for iy in range(len(RANKS)):
            for ix in range(len(FILES)):
                if not np.isnan(m[iy, ix]):
                    ax.text(ix, iy, f"{m[iy, ix]:.0f}%", ha="center", va="center", fontsize=10,
                            color="#ffffff" if m[iy, ix] > 60 else INK)
        ax.set_xticks(range(len(FILES)), list(FILES))
        ax.set_yticks(range(len(RANKS)), [str(r) for r in RANKS])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(("baseline " if g == BASELINE else "best (from e4) ") + "\n" + gstr(g), fontsize=9.5,
                     color=C_BASE if g == BASELINE else C_BEST)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.7, pad=0.02)
    cb.set_label("% picked up and held", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Pickup success per square, joint error k = {k:g}× measured (64 draws each)", fontsize=10.5,
                 fontweight="bold", x=0.02, ha="left")
    save(fig, out / f"fig4_board_k{k:g}.png")


def fig_failures(out: Path, sq_rows: dict, best: tuple, k: float):
    grasps = [BASELINE] + ([best] if best != BASELINE else [])
    allrows = [r for rows in sq_rows.values() for r in rows if r["k"] == k]
    fig, ax = plt.subplots(figsize=(7.5, 1.0 + 0.7 * len(grasps)))
    counts = {}
    seen = set()
    for i, g in enumerate(grasps):
        sel = [r for r in allrows if r["g"] == g]
        n = len(sel)
        left = 0.0
        for j, oc in enumerate(OUTCOMES):
            c = sum(r["outcome"] == oc for r in sel)
            counts[(g, oc)] = (c, n)
            if n and c:
                wv = 100 * c / n
                ax.barh(i, wv, left=left, color=CAT[j], edgecolor=SURF, lw=2, height=0.6,
                        label=None if oc in seen else oc)
                seen.add(oc)
                if wv >= 6:
                    ax.text(left + wv / 2, i, f"{wv:.0f}%", ha="center", va="center", fontsize=8.5,
                            color=INK if CAT[j] == "#eda100" else "#ffffff")
                left += wv
    ax.set_yticks(range(len(grasps)), [("baseline" if g == BASELINE else "best") for g in grasps])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of all c/d/e samples")
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(oc) for oc in OUTCOMES if oc in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.5))
    ax.set_title(f"How pickups end, all c/d/e squares, k = {k:g}")
    save(fig, out / f"fig5_outcomes_k{k:g}.png")
    return counts


def paired(sq_rows: dict, best: tuple, k: float):
    """Paired held outcomes (same draw, same square): only-baseline, only-best, n, exact sign-test p."""
    ob = ow = n = 0
    for rows in sq_rows.values():
        d = {(r["g"], r["draw"]): r["held"] for r in rows if r["k"] == k}
        for (g, i), h in d.items():
            if g == BASELINE and (best, i) in d:
                n += 1
                ob += h and not d[(best, i)]
                ow += d[(best, i)] and not h
    m = ob + ow
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(min(ob, ow) + 1)) / 2 ** m) if m else 1.0
    return ob, ow, n, p


# ---------------------------------------------------------------------------------------------------
def main() -> int:
    root = Path(sys.argv[1]).resolve()
    out = root / "report"
    out.mkdir(exist_ok=True)
    md = []
    add = md.append

    best_info = json.loads((root / "e4_search" / "best.json").read_text()) if (root / "e4_search" / "best.json").exists() else None
    best = tuple(best_info["best"]["grasp_mm"]) if best_info else BASELINE

    fk_rows = fig_fk(out, root / "fk_bound" / "fk_bound.csv") if (root / "fk_bound" / "fk_bound.csv").exists() else []
    fk = {r["square"]: r for r in fk_rows}
    e4t = float(fk["e4"]["T1_rest_tangential_mm"]) if "e4" in fk else None

    search = load(root / "e4_search")
    scale = load(root / "e4_scale")
    sq_rows = {p.name: load(p) for p in sorted((root / "squares").glob("*")) if p.is_dir()} if (root / "squares").exists() else {}
    d4s = load(root / "d4_search")

    add("# Pickup tolerance to arm error: c, d, e files (sim)\n")
    add("_Auto-generated by `lowlevel/drafts/pickup_tolerance_report.py`. Each `##` section is one slide: "
        "the heading is the claim, then one figure and up to three bullets._\n")

    # ---- summary numbers
    summary = []
    if scale:
        for g in (BASELINE, best):
            for k in (1.0, 2.0):
                h, c, n = rate(scale, g=g, k=k)
                if n:
                    summary.append((g, k, h, n))
    sq_k = {}
    for k in (1.0, 2.0):
        for g in (BASELINE, best):
            tot = [rate(rows, g=g, k=k) for rows in sq_rows.values()]
            h = sum(t[0] for t in tot)
            n = sum(t[2] for t in tot)
            worst = min(((s, rate(rows, g=g, k=k)) for s, rows in sq_rows.items() if rate(rows, g=g, k=k)[2]),
                        key=lambda t: t[1][0] / t[1][2], default=None)
            sq_k[(g, k)] = (h, n, worst)

    add("## Summary\n")
    add(f"- **Question.** Oliver's real arm misses its joint targets by 0.2–0.8° RMS when it has stopped at the grasp. "
        "Does the rook still get picked up, and which grasp offset gives the most margin? Pickup success only; "
        "place accuracy is not scored.")
    if fk:
        add(f"- **Arm error at the grasp point** (FK of the measured joint error, 1σ): sideways "
            f"{min(float(r['T1_rest_tangential_mm']) for r in fk_rows):.1f}–{max(float(r['T1_rest_tangential_mm']) for r in fk_rows):.1f} mm "
            f"(almost all shoulder pan), height {min(float(r['T1_rest_z_mm']) for r in fk_rows):.1f}–{max(float(r['T1_rest_z_mm']) for r in fk_rows):.1f} mm, "
            f"reach {min(float(r['T1_rest_radial_mm']) for r in fk_rows):.1f}–{max(float(r['T1_rest_radial_mm']) for r in fk_rows):.1f} mm. "
            "Errors grow with reach (e file > d > c).")
    if best_info:
        b, bl = best_info["best"], best_info["baseline"]
        add(f"- **Best grasp at e4** (search, k = 2, i.e. twice the measured error): {gstr(best)}: "
            f"{b['held']}/{b['n']} held, vs baseline {gstr(BASELINE)}: "
            + (f"{bl['held']}/{bl['n']}." if bl else "n/a."))
    if scale:
        parts = []
        for g in (BASELINE, best) if best != BASELINE else (BASELINE,):
            vals = [f"{k:g}×: {pct(*rate(scale, g=g, k=k)[::2])}" for k in sorted({r['k'] for r in scale})]
            parts.append(("baseline" if g == BASELINE else "best") + " " + ", ".join(vals))
        add("- **e4 success vs error size** (fresh draws): " + "; ".join(parts) + ".")
    for k in (1.0, 2.0):
        hb, nb, wb = sq_k[(BASELINE, k)]
        if nb:
            txt = f"- **c/d/e ranks 2–7, k = {k:g}**: baseline {pct(hb, nb)} held"
            if wb:
                txt += f" (worst {wb[0]} {pct(wb[1][0], wb[1][2])})"
            if best != BASELINE:
                hB, nB, wB = sq_k[(best, k)]
                txt += f"; best {pct(hB, nB)}" + (f" (worst {wB[0]} {pct(wB[1][0], wB[1][2])})" if wB else "")
            add(txt + f" over {len(sq_rows)} squares.")
    add("")

    if sq_rows and best != BASELINE:
        add("## Takeaways\n")
        for k in (1.0, 2.0):
            ob, ow, n, p = paired(sq_rows, best, k)
            cb = sum(rate(rows, g=BASELINE, k=k)[1] for rows in sq_rows.values())
            cw = sum(rate(rows, g=best, k=k)[1] for rows in sq_rows.values())
            add(f"- **k = {k:g}** ({n} paired pickups over {len(sq_rows)} squares): held by baseline only {ob}, "
                f"by the lower grasp only {ow} (sign test p = {p:.2g}). Clean grasps: baseline {pct(cb, n)}, "
                f"lower grasp {pct(cw, n)}.")
        add("- **At the error Oliver measured (1×), the sim picks the rook up almost every time with either grasp.** "
            "The real-arm failure rate on 7 Nov (26% compensated, 42% baseline) is much higher, so most real "
            "failures come from things this error model leaves out: board/camera calibration (13 mm, 3° on 14 Nov), "
            "arm lag at the moment the jaws close, and the sim-to-real gap.")
        add("- **Margin is what the lower grasp buys.** Gripping 3 mm lower holds more often once the error doubles "
            "(e4 at 4×: see the scale curve). It costs cleanliness: the rook slips > 2 mm in the jaws more often. "
            "That does not lose the piece, but may matter for placing.")
        add("- **The e file and rank 7 are weakest** (longest reach, so the largest pan error). c is near-perfect.\n")

    add("## What is different from previous weeks\n")
    add("- **Before (Oct to 7 Nov):** the robot was given a *wrong rook position* (blind mode, σ = 3.3 mm) and we scored "
        "*where the rook landed* (≤ 5 mm), then tuned place offsets. The arm itself was assumed perfect.")
    add("- **Last pilot (10 Nov):** known mode, so the arm aimed at the true position. The start grid therefore "
        "never created a gripper-to-rook error, and 54 of 81 grasps scored 100%. That cannot rank grasps, and z was never varied.")
    add("- **Now:** the planner aims at the true position, and the *simulated arm is driven with joint errors* drawn "
        "from Oliver's hardware measurements (`run_sim_move(joint_offset_deg=…)`). This gives position, height and "
        "gripper-angle error together, in the same proportions as the real arm. The score is **pickup only** (lifted and held through the carry). "
        "The grasp search now varies height (z) as well as x/y, and every grasp sees the same error draws, so comparisons are paired.\n")

    if fk_rows:
        add("## The arm's error at the grasp is 1–3.5 mm (1σ), and mostly sideways\n")
        add("![](fig1_fk_bound.png)\n")
        add("- Oliver's measured joint error at rest, at the end of the grasp hold, RMS (pan / lift / elbow / wrist flex / wrist roll): "
            "0.80 / 0.62 / 0.22 / 0.27 / 0.70°. It was pushed through the arm's forward kinematics at each square's planned grasp pose.")
        add("- Shoulder pan dominates: about 3–4 mm sideways per degree, growing with reach. Lift and elbow set the height error.")
        if "e4" in fk:
            r = fk["e4"]
            add(f"- Tracking error *during motion* is much larger (e4: {float(r['T2_tracking_radial_mm']):.0f} / "
                f"{float(r['T2_tracking_tangential_mm']):.0f} / {float(r['T2_tracking_z_mm']):.0f} mm reach / sideways / height). "
                "It matters only if the jaws close before the arm settles, so the sweep tests up to 4× the at-rest error.\n")
        else:
            add("")

    if search:
        grid, k = fig_search(out, search, best, "fig2_e4_grasp_search.png",
                             "e4: % picked up per grasp offset, under arm error")
        add(f"## Best grasp at e4: {gstr(best)}\n")
        add("![](fig2_e4_grasp_search.png)\n")
        if best_info:
            b, bl = best_info["best"], best_info["baseline"]
            add(f"- {len(grid)} grasp offsets × {b['n']} arm-error draws at k = {k:g} (twice the measured error). Every grasp "
                "sees the same draws and rook starts. Picked by held rate smoothed over the grasp and its 6 grid neighbours (one sample = 2.5%, so the raw count alone is noise).")
            add(f"- Best {gstr(best)}: {b['held']}/{b['n']} held, {b['strict']}/{b['n']} clean. Baseline: "
                + (f"{bl['held']}/{bl['n']} held, {bl['strict']}/{bl['n']} clean." if bl else "n/a."))
            add("- Top 5: " + "; ".join(f"{gstr(tuple(t['grasp_mm']))} {t['held']}/{t['n']}" for t in best_info["top10"][:5]) + ".\n")

    if scale:
        table = fig_scale(out, scale, best, e4t)
        add("## How much arm error the pickup tolerates (e4)\n")
        add("![](fig3_scale_curve.png)\n")
        for g in (BASELINE, best) if best != BASELINE else (BASELINE,):
            ok = [k for (gg, k, h, c, n) in table if gg == g and n and h / n >= 0.95]
            add(f"- {'Baseline' if g == BASELINE else 'Best'}: ≥ 95% held up to "
                + (f"{max(ok):g}× the measured error." if ok else "none of the tested levels."))
        add("- New random draws (seed 1), independent of the search, so the best grasp is not scored on the draws it was chosen on.\n")

    if sq_rows:
        for k in (1.0, 2.0):
            fig_board(out, sq_rows, best, k)
        counts = fig_failures(out, sq_rows, best, 2.0)
        add("## Across the c, d and e files\n")
        add("![](fig4_board_k2.png)\n")
        add("- 64 arm-error draws per square and grasp, with random rook starts within ±16 mm of the square centre. "
            "Each grasp is carried one rank forward (rank 7 back).")
        for k in (1.0, 2.0):
            hb, nb, wb = sq_k[(BASELINE, k)]
            line = f"- k = {k:g}: baseline {pct(hb, nb)}"
            if best != BASELINE:
                hB, nB, wB = sq_k[(best, k)]
                line += f", best {pct(hB, nB)}"
            add(line + (f"; weakest square {wb[0]}." if wb else "."))
        add("- k = 1 board: `fig4_board_k1.png`.\n")
        add("## How the misses happen\n")
        add("![](fig5_outcomes_k2.png)\n")
        for g in [BASELINE] + ([best] if best != BASELINE else []):
            parts = [f"{oc} {pct(*counts[(g, oc)])}" for oc in OUTCOMES if counts.get((g, oc), (0, 0))[0]]
            add(f"- {'Baseline' if g == BASELINE else 'Best'}: " + ", ".join(parts) + ".")
        add("- 'Slipped' and 'knocked' still count as picked up. They are early warnings: the rook moved in the jaws, "
            "or was nudged before the jaws closed.\n")
        with (out / "squares_summary.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["square", "grasp", "k", "held", "clean", "n", "fk_tangential_1sigma_mm", "fk_z_1sigma_mm"])
            for s, rows in sorted(sq_rows.items()):
                for g in [BASELINE] + ([best] if best != BASELINE else []):
                    for k in (1.0, 2.0):
                        h, c, n = rate(rows, g=g, k=k)
                        w.writerow([s, gstr(g), k, h, c, n, fk.get(s, {}).get("T1_rest_tangential_mm", ""),
                                    fk.get(s, {}).get("T1_rest_z_mm", "")])

    if d4s and (root / "d4_search" / "best.json").exists():
        d4b = json.loads((root / "d4_search" / "best.json").read_text())
        fig_search(out, d4s, tuple(d4b["best"]["grasp_mm"]), "fig6_d4_grasp_search.png",
                   "d4: % picked up per grasp offset, under arm error")
        add("## Does the e4 optimum carry over to d4?\n")
        add("![](fig6_d4_grasp_search.png)\n")
        b = d4b["best"]
        e4_at_d4 = rate(d4s, g=best, k=2.0)
        add(f"- d4's own best: {gstr(tuple(b['grasp_mm']))}, {b['held']}/{b['n']} held. The e4 best at d4: "
            f"{e4_at_d4[0]}/{e4_at_d4[2]}. Baseline at d4: {d4b['baseline']['held'] if d4b['baseline'] else 'n/a'}/{b['n']}.\n")

    add("## Caveats and next steps\n")
    add("- Sim only. The hardware moves 1.9× slower than planned and lags 51–122 ms. If the jaws close before the arm "
        "settles, the error at closing is larger than the at-rest figure. Ask Oliver for hardware − plan at the *close command*.")
    add("- Camera and board calibration error is not included (board about 13 mm and 3° off on 14 Nov). It shares the same "
        "tolerance budget, so the margin here is an upper limit on what calibration may use.")
    add("- Joint errors are drawn independently. Gravity sag couples lift, elbow and wrist on the real arm.")
    add("- Next: validate the best grasp on hardware at d4 and e4 against baseline (paired, as on 7 Nov). Then extend "
        "to the b file, and leave f, g and h for last.\n")
    add("## Files\n")
    add("- Runs: `e4_search/`, `e4_scale/`, `squares/<sq>/`, `d4_search/` (`results.jsonl`, one line per simulated pickup). FK: `fk_bound/`.")
    add("- Code: `lowlevel/drafts/run_pickup_tolerance.py`, `run_pickup_tolerance_cde.sh`, `joint_error_fk_bound.py`, "
        "`pickup_tolerance_select.py`, `pickup_tolerance_report.py`. Sim hook: `multisim_chess_fast.run_sim_move(joint_offset_deg=…)`.")
    (out / "PICKUP_TOLERANCE.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'PICKUP_TOLERANCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
