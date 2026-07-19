import argparse
import json
import sys
import types
from pathlib import Path

# Some import paths in the lookup stack expect this module to exist.
kitty_mod = types.ModuleType("IPython.core.kitty")
kitty_mod._supports_kitty_graphics = lambda: False
sys.modules.setdefault("IPython.core.kitty", kitty_mod)

import build_general_nonh_reverse_lookup as builder
import verify_nonh_lookup_moves as verifier


DEFAULT_LOOKUP_NAME = "e3_non_h_reverse_move_lookup.json"


def successful_h_moves_from_lookup(lookup_path):
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    moves = lookup.get("moves", {})
    h_moves = [
        move_key
        for move_key, entry in moves.items()
        if move_key.startswith("e3_to_h") and entry.get("success")
    ]
    return sorted(
        h_moves,
        key=lambda move_key: int(move_key.rsplit("h", 1)[-1]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replay and verify successful e3 -> h-rank lookup moves."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder to receive one video subfolder per move plus summary.json.",
    )
    parser.add_argument(
        "--lookup-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Folder containing e3_non_h_reverse_move_lookup.json.",
    )
    parser.add_argument(
        "moves",
        nargs="*",
        help="Optional explicit move keys. If omitted, uses successful e3 -> h-rank moves from the lookup.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    lookup_dir = args.lookup_dir.expanduser().resolve()
    lookup_path = lookup_dir / DEFAULT_LOOKUP_NAME

    if args.moves:
        move_keys = list(args.moves)
    else:
        move_keys = successful_h_moves_from_lookup(lookup_path)

    if not move_keys:
        raise RuntimeError(
            f"No successful e3 -> h-rank moves found in {lookup_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    builder.ensure_physics_connected()

    summary = {
        "lookup_path": str(lookup_path),
        "lookup_dir": str(lookup_dir),
        "output_dir": str(output_dir),
        "moves": [],
    }

    for move_key in move_keys:
        print(f"\n=== verifying {move_key} from {lookup_path} ===", flush=True)
        _, entry = verifier.load_lookup_entry(lookup_dir, move_key)
        row = verifier.run_saved_entry(move_key, entry, output_dir)
        row["lookup_path"] = str(lookup_path)
        summary["moves"].append(row)
        print(
            f"{move_key}: success={row['verified_success']} "
            f"mode={row['replay_mode']} donor={row['replayed_donor']} "
            f"fk={row['trajectory_fk_error']:.6f} "
            f"xy={row['xy_error']:.6f} tilt={row['final_tilt_deg']:.3f} "
            f"video={row['video_output_dir']}",
            flush=True,
        )

    summary["verified_count"] = sum(row["verified_success"] for row in summary["moves"])
    summary["total_count"] = len(summary["moves"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(builder.json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nsummary: {summary_path}")
    print(f"verified {summary['verified_count']}/{summary['total_count']}")


if __name__ == "__main__":
    main()
