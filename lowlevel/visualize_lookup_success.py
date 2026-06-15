import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_LOOKUP_PATH = (
    Path(__file__).resolve().parent / "f4_non_h_reverse_move_lookup.json"
)
FILES = "abcdefgh"
RANKS = range(1, 9)


COLORS = {
    "success": "#4caf50",
    "missing": "#e57373",
    "source": "#42a5f5",
    "excluded": "#eeeeee",
    "border": "#263238",
    "text": "#1f2933",
    "muted": "#607d8b",
}


def move_key(from_square, to_square):
    return f"{from_square}_to_{to_square}"


def load_lookup(path):
    with path.open("r", encoding="utf-8") as lookup_file:
        lookup = json.load(lookup_file)

    moves = lookup.get("moves", {})
    if not isinstance(moves, dict):
        raise ValueError(f"Lookup file has invalid moves object: {path}")
    return lookup, moves


def infer_from_square(metadata, moves):
    from_square = metadata.get("from_square")
    if from_square:
        return from_square

    for move_key_value, move in moves.items():
        if isinstance(move, dict) and move.get("from_square"):
            return move["from_square"]
        if "_to_" in move_key_value:
            return move_key_value.split("_to_", 1)[0]
    return None


def expected_destinations(metadata, moves, from_square):
    files = metadata.get("destination_files")
    ranks = metadata.get("destination_ranks")
    if files and ranks:
        candidates = {
            f"{file}{rank}"
            for file in files
            for rank in ranks
        }
    else:
        candidates = {
            move.get("to_square", move_key_value.split("_to_", 1)[-1])
            for move_key_value, move in moves.items()
            if isinstance(move, dict)
        }

    return {
        square
        for square in candidates
        if square in board_squares() and square != from_square
    }


def board_squares():
    return {
        f"{file}{rank}"
        for file in FILES
        for rank in RANKS
    }


def move_is_successful(move):
    if not isinstance(move, dict) or not move.get("success"):
        return False

    metrics = move.get("metrics", {})
    if not isinstance(metrics, dict):
        return False

    return (
        bool(metrics.get("pickup_success", False))
        and metrics.get("reject_reason") is None
    )


def square_status(square, from_square, expected, moves):
    if square == from_square:
        return "source"
    if square not in expected:
        return "excluded"

    key = move_key(from_square, square)
    if move_is_successful(moves.get(key)):
        return "success"
    return "missing"


def square_tooltip(square, from_square, moves, status):
    key = move_key(from_square, square) if from_square else square
    move = moves.get(key, {})
    metrics = move.get("metrics", {}) if isinstance(move, dict) else {}
    parts = [square, status]
    if key in moves:
        parts.extend([
            f"xy_error={metrics.get('xy_error')}",
            f"tilt={metrics.get('final_tilt_deg')}",
            f"reject={metrics.get('reject_reason')}",
        ])
    return html.escape(" | ".join(str(part) for part in parts))


def render_svg(lookup_path, lookup, moves):
    metadata = lookup.get("metadata", {})
    from_square = infer_from_square(metadata, moves)
    expected = expected_destinations(metadata, moves, from_square)
    statuses = {
        square: square_status(square, from_square, expected, moves)
        for square in board_squares()
    }

    success_count = sum(1 for status in statuses.values() if status == "success")
    missing_count = sum(1 for status in statuses.values() if status == "missing")
    expected_count = len(expected)

    cell = 64
    margin_left = 64
    margin_top = 100
    board_size = cell * 8
    width = margin_left + board_size + 260
    height = margin_top + board_size + 80

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="32" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="{COLORS["text"]}">Lookup success map</text>',
        f'<text x="32" y="66" font-family="Arial, sans-serif" font-size="13" fill="{COLORS["muted"]}">{html.escape(str(lookup_path))}</text>',
        f'<text x="32" y="86" font-family="Arial, sans-serif" font-size="13" fill="{COLORS["muted"]}">from={from_square} | expected={expected_count} | successful={success_count} | missing_or_failed={missing_count}</text>',
    ]

    for col, file in enumerate(FILES):
        x = margin_left + col * cell + cell / 2
        lines.append(
            f'<text x="{x}" y="{margin_top - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="{COLORS["text"]}">{file}</text>'
        )

    for row, rank in enumerate(range(8, 0, -1)):
        y = margin_top + row * cell + cell / 2 + 5
        lines.append(
            f'<text x="{margin_left - 18}" y="{y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="{COLORS["text"]}">{rank}</text>'
        )

    for row, rank in enumerate(range(8, 0, -1)):
        for col, file in enumerate(FILES):
            square = f"{file}{rank}"
            status = statuses[square]
            x = margin_left + col * cell
            y = margin_top + row * cell
            fill = COLORS[status]
            label = square.upper() if status == "source" else square
            stroke_width = 3 if status == "source" else 1
            lines.extend([
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{COLORS["border"]}" stroke-width="{stroke_width}">',
                f'<title>{square_tooltip(square, from_square, moves, status)}</title>',
                '</rect>',
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="{COLORS["text"]}">{label}</text>',
            ])

    legend_x = margin_left + board_size + 42
    legend_y = margin_top + 8
    legend_items = (
        ("success", "successful move"),
        ("missing", "missing or failed"),
        ("source", "source square"),
        ("excluded", "not in lookup target set"),
    )
    lines.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="{COLORS["text"]}">Legend</text>'
    )
    for idx, (status, label) in enumerate(legend_items):
        y = legend_y + 28 + idx * 30
        lines.extend([
            f'<rect x="{legend_x}" y="{y - 16}" width="18" height="18" fill="{COLORS[status]}" stroke="{COLORS["border"]}" stroke-width="1"/>',
            f'<text x="{legend_x + 28}" y="{y - 2}" font-family="Arial, sans-serif" font-size="13" fill="{COLORS["text"]}">{label}</text>',
        ])

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def default_output_path(lookup_path):
    return lookup_path.with_name(f"{lookup_path.stem}_success_map.svg")


def open_in_ide(path, preferred_opener="auto"):
    if preferred_opener == "none":
        return False

    if preferred_opener == "auto":
        candidates = ("cursor", "code")
    else:
        candidates = (preferred_opener,)

    for candidate in candidates:
        command = shutil.which(candidate)
        if command is None:
            continue
        subprocess.Popen(
            [command, str(path.resolve())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Create an SVG board map showing lookup move successes."
    )
    parser.add_argument(
        "lookup_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOOKUP_PATH,
        help=f"Lookup JSON path. Defaults to {DEFAULT_LOOKUP_PATH}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output SVG path. Defaults to <lookup stem>_success_map.svg beside the lookup.",
    )
    parser.add_argument(
        "--open-with",
        choices=("auto", "cursor", "code", "none"),
        default="auto",
        help="Open the generated SVG in the IDE. Defaults to auto.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Only write the SVG; do not open it in the IDE.",
    )
    args = parser.parse_args()

    lookup, moves = load_lookup(args.lookup_path)
    output_path = args.output or default_output_path(args.lookup_path)
    output_path.write_text(
        render_svg(args.lookup_path, lookup, moves),
        encoding="utf-8"
    )
    print(f"Wrote lookup success map: {output_path}")
    opener = "none" if args.no_open else args.open_with
    if open_in_ide(output_path, opener):
        print(f"Opened lookup success map in IDE: {output_path}")
    elif opener != "none":
        print(
            "Could not find a supported IDE command. "
            f"Open this SVG manually: {output_path}"
        )


if __name__ == "__main__":
    main()
