#!/usr/bin/env python3
"""Render a success map from continuous-XY lookup result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_svg", type=Path)
    parser.add_argument(
        "--verification-summary",
        type=Path,
        help="Use strict replay outcomes from verify_continuous_xy_grid.py.",
    )
    parser.add_argument(
        "--unverified-point",
        nargs=2,
        type=float,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="Add a historical point with no reproducible saved trajectory.",
    )
    parser.add_argument(
        "--additional-verified-lookup",
        type=Path,
        action="append",
        default=[],
        help="Add a separately stored lookup that has passed strict replay.",
    )
    return parser.parse_args()


def sample_from_lookup(path: Path) -> dict | None:
    data = json.loads(path.read_text())
    if data.get("schema") != "continuous_xy_lookup_v1":
        return None
    attempts = data.get("attempts", [])
    if not attempts:
        return None
    attempt = attempts[-1]
    result = attempt.get("selected_result", {})
    from_xy = result.get("from_world_xy")
    if from_xy is None:
        from_xy = result.get("from_square", {}).get("x"), result.get("from_square", {}).get("y")
    if not from_xy or len(from_xy) < 2:
        return None
    return {
        "name": path.parent.name,
        "lookup_path": str(path.resolve()),
        "x": float(from_xy[0]),
        "y": float(from_xy[1]),
        "success": bool(data.get("success", attempt.get("success", False))),
        "xy_error": result.get("xy_error"),
        "tilt": result.get("final_tilt_deg"),
    }


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("**/*_to_*.json")):
        if path.name.endswith(".verified.json"):
            continue
        sample = sample_from_lookup(path)
        if sample is not None:
            samples.append(sample)
    return samples


def apply_verification(samples: list[dict], summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcomes = {
        str(entry["lookup_json"]): bool(entry["verified_success"])
        for entry in summary.get("entries", [])
    }
    for sample in samples:
        if sample.get("lookup_path") in outcomes:
            sample["success"] = outcomes[sample["lookup_path"]]
            sample["verification_state"] = "strict_replay"


def replace_sample_at_xy(samples: list[dict], sample: dict, tol: float = 1e-9) -> None:
    x = float(sample["x"])
    y = float(sample["y"])
    samples[:] = [
        existing
        for existing in samples
        if abs(float(existing["x"]) - x) > tol or abs(float(existing["y"]) - y) > tol
    ]
    samples.append(sample)


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.verification_summary is not None:
        apply_verification(samples, args.verification_summary.expanduser().resolve())
    for lookup_path in args.additional_verified_lookup:
        sample = sample_from_lookup(lookup_path.expanduser().resolve())
        if sample is None:
            raise SystemExit(f"Not a continuous XY lookup: {lookup_path}")
        sample["success"] = True
        sample["verification_state"] = "strict_replay"
        replace_sample_at_xy(samples, sample)
    for point in args.unverified_point:
        samples.append(
            {
                "name": "historical_unreplayable",
                "x": float(point[0]),
                "y": float(point[1]),
                "success": False,
                "verification_state": "unverified",
                "xy_error": None,
                "tilt": None,
            }
        )
    if not samples:
        raise SystemExit(f"No continuous-XY result JSONs found under {args.input_dir}")

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for sample in samples:
        color = (
            "#787878"
            if sample.get("verification_state") == "unverified"
            else ("#16803c" if sample["success"] else "#c62828")
        )
        ax.scatter(sample["x"], sample["y"], s=150, color=color, edgecolor="white", linewidth=1.2)
        ax.annotate(sample["name"], (sample["x"], sample["y"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.scatter([], [], s=150, color="#16803c", label="success")
    ax.scatter([], [], s=150, color="#c62828", label="failure")
    if any(sample.get("verification_state") == "unverified" for sample in samples):
        ax.scatter([], [], s=150, color="#787878", label="not replayable")
    successes = sum(sample["success"] for sample in samples)
    verified = [sample for sample in samples if sample.get("verification_state") != "unverified"]
    ax.set_title(f"Continuous XY pickup success map ({successes}/{len(verified)} verified)")
    ax.set_xlabel("Pickup world X (m)")
    ax.set_ylabel("Pickup world Y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_svg, format="svg")
    plt.close(fig)
    print(f"Wrote {args.output_svg}")
    print(f"Samples: {len(samples)}; successes: {successes}; failures: {len(samples) - successes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
