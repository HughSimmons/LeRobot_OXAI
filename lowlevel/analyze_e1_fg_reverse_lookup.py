import argparse
import json
from pathlib import Path


DEFAULT_LOOKUP_PATH = (
    Path(__file__).resolve().parent / "e1_fg_reverse_move_lookup.json"
)


def load_lookup(path):
    with path.open("r", encoding="utf-8") as lookup_file:
        lookup = json.load(lookup_file)

    moves = lookup.get("moves", {})
    if not isinstance(moves, dict):
        raise ValueError(f"Lookup file has invalid moves object: {path}")
    return moves


def move_label(move_key, move):
    from_square = move.get("from_square")
    to_square = move.get("to_square")
    if from_square and to_square:
        return f"{from_square}->{to_square}"
    return move_key


def analyze_pickups(moves):
    rows = []
    for move_key, move in sorted(moves.items()):
        metrics = move.get("metrics", {})
        pickup_success = bool(metrics.get("pickup_success", False))
        rows.append({
            "move_key": move_key,
            "label": move_label(move_key, move),
            "pickup_success": pickup_success,
            "lookup_success": bool(move.get("success", False)),
            "reject_reason": metrics.get("reject_reason"),
            "xy_error": metrics.get("xy_error"),
            "final_tilt_deg": metrics.get("final_tilt_deg"),
        })
    return rows


def print_rows(title, rows):
    print(f"\n{title} ({len(rows)})")
    if not rows:
        print("  none")
        return

    for row in rows:
        print(
            "  "
            f"{row['label']} | "
            f"lookup_success={row['lookup_success']} | "
            f"xy_error={row['xy_error']} | "
            f"tilt={row['final_tilt_deg']} | "
            f"reject={row['reject_reason']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Report pickup success from an e1 f/g reverse lookup JSON."
    )
    parser.add_argument(
        "lookup_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOOKUP_PATH,
        help=f"Lookup JSON path. Defaults to {DEFAULT_LOOKUP_PATH}",
    )
    args = parser.parse_args()

    moves = load_lookup(args.lookup_path)
    rows = analyze_pickups(moves)
    successful = [row for row in rows if row["pickup_success"]]
    unsuccessful = [row for row in rows if not row["pickup_success"]]

    print(f"Lookup: {args.lookup_path}")
    print(f"Total saved moves: {len(rows)}")
    print(f"Pickup successful: {len(successful)}")
    print(f"Pickup unsuccessful: {len(unsuccessful)}")

    print_rows("Successful pickups", successful)
    print_rows("Unsuccessful pickups", unsuccessful)


if __name__ == "__main__":
    main()
