import json
from pathlib import Path

import numpy as np


LOOKUP_DIR = Path(__file__).resolve().parent


def plot_place_offset_histogram(from_square, lookup_dir=LOOKUP_DIR, bins=20):
    import matplotlib.pyplot as plt

    lookup_path = Path(lookup_dir) / f"{from_square}_non_h_reverse_move_lookup.json"
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    place_offsets = []

    for move_key, move in sorted(lookup.get("moves", {}).items()):
        if move.get("from_square") != from_square or not move.get("success"):
            continue
        place_offsets.append(np.array(move["selected_place_offset"], dtype=float))

    if not place_offsets:
        raise ValueError(f"No successful place offsets found in {lookup_path}")

    place_offsets = np.array(place_offsets)

    plt.figure()
    plt.hist(place_offsets[:, 0], bins=bins, alpha=0.5, label="x")
    plt.hist(place_offsets[:, 1], bins=bins, alpha=0.5, label="y")
    plt.hist(place_offsets[:, 2], bins=bins, alpha=0.5, label="z")
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("selected_place_offset")
    plt.ylabel("count")
    plt.title(f"{from_square} placement offset histogram")
    plt.legend()
    plt.tight_layout()
    return place_offsets


def plot_all_place_offset_histogram(lookup_dir=LOOKUP_DIR, bins=40):
    import matplotlib.pyplot as plt

    place_offsets = []

    for lookup_path in sorted(Path(lookup_dir).glob("*_non_h_reverse_move_lookup.json")):
        source_label = lookup_path.name.removesuffix("_non_h_reverse_move_lookup.json")
        if len(source_label) != 2 or source_label[0] not in "abcdef" or source_label[1] not in "12345678":
            continue

        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        for move_key, move in sorted(lookup.get("moves", {}).items()):
            if not move.get("success"):
                continue
            place_offsets.append(np.array(move["selected_place_offset"], dtype=float))

    if not place_offsets:
        raise ValueError(f"No successful place offsets found in {lookup_dir}")

    place_offsets = np.array(place_offsets)

    plt.figure()
    plt.hist(place_offsets[:, 0], bins=bins, alpha=0.5, label="x")
    plt.hist(place_offsets[:, 1], bins=bins, alpha=0.5, label="y")
    plt.hist(place_offsets[:, 2], bins=bins, alpha=0.5, label="z")
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("selected_place_offset")
    plt.ylabel("count")
    plt.title(f"all placement offsets ({len(place_offsets)} moves)")
    plt.legend()
    plt.tight_layout()
    return place_offsets


def plot_all_grasp_offset_histogram(lookup_dir=LOOKUP_DIR, bins=40):
    import matplotlib.pyplot as plt

    grasp_offsets = []

    for lookup_path in sorted(Path(lookup_dir).glob("*_non_h_reverse_move_lookup.json")):
        source_label = lookup_path.name.removesuffix("_non_h_reverse_move_lookup.json")
        if len(source_label) != 2 or source_label[0] not in "abcdef" or source_label[1] not in "12345678":
            continue

        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        for move_key, move in sorted(lookup.get("moves", {}).items()):
            if not move.get("success"):
                continue
            grasp_offsets.append(np.array(move["source_grasp_offset"], dtype=float))

    if not grasp_offsets:
        raise ValueError(f"No successful grasp offsets found in {lookup_dir}")

    grasp_offsets = np.array(grasp_offsets)

    plt.figure()
    plt.hist(grasp_offsets[:, 0], bins=bins, alpha=0.5, label="x")
    plt.hist(grasp_offsets[:, 1], bins=bins, alpha=0.5, label="y")
    plt.hist(grasp_offsets[:, 2], bins=bins, alpha=0.5, label="z")
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("source_grasp_offset")
    plt.ylabel("count")
    plt.title(f"all grasp offsets ({len(grasp_offsets)} moves)")
    plt.legend()
    plt.tight_layout()
    return grasp_offsets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_all_place_offset_histogram()
    plt.savefig("all_place_offset_histogram.png", dpi=300)
    plot_all_grasp_offset_histogram()
    plt.savefig("all_grasp_offset_histogram.png", dpi=300)
    plt.show()
