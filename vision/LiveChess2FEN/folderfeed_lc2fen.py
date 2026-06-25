"""Simple folder subprocess feeder for lc2fen.py."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
import sys


IMAGE_FOLDER = "chess_board_ims"
IMAGE_PREFIX = "im"
IMAGE_EXT = ".jpeg"
A1_POS = "TR"
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
DISPLAY_VISUALISATION = True


def draw_fen_board(ax, fen):
    pieces = {
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    }
    rows = fen.split()[0].split("/")

    for row in range(8):
        for col in range(8):
            color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
            ax.add_patch(
                plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="none")
            )

    for row_index, row in enumerate(rows):
        col = 0
        y = 7 - row_index
        for char in row:
            if char.isdigit():
                col += int(char)
                continue

            ax.text(
                col + 0.5,
                y + 0.5,
                pieces[char],
                ha="center",
                va="center",
                fontsize=34,
            )
            col += 1

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(8)], list("abcdefgh"))
    ax.set_yticks([i + 0.5 for i in range(8)], list("12345678"))
    ax.set_title("Predicted board")


def rotate_image_to_a1_bottom_left(image, a1_pos):
    if a1_pos == "BL":
        return image
    if a1_pos == "BR":
        return np.rot90(image, 1)
    if a1_pos == "TL":
        return np.rot90(image, -1)
    if a1_pos == "TR":
        return np.rot90(image, 2)
    raise ValueError("A1_POS must be BL, BR, TL, or TR")


def show_prediction(image_path, fen, a1_pos):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    input_image = rotate_image_to_a1_bottom_left(plt.imread(image_path), a1_pos)
    axes[0].imshow(input_image)
    axes[0].set_title("Input image rotated to a1 bottom-left")
    axes[0].axis("off")

    draw_fen_board(axes[1], fen)

    fig.suptitle(fen)
    plt.tight_layout()
    plt.show()


def image_number(path):
    stem = path.stem
    if not stem.startswith(IMAGE_PREFIX):
        return None

    number_text = stem[len(IMAGE_PREFIX):]
    if not number_text.isdigit():
        return None

    return int(number_text)


def find_sequence_images():
    image_dir = Path(IMAGE_FOLDER)
    images = []

    for path in image_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_EXT}"):
        number = image_number(path)
        if number is not None:
            images.append((number, path))

    images.sort()
    return images


images = find_sequence_images()

if len(images) < 2:
    raise SystemExit("Need at least im1 plus one later image.")

if images[0][0] != 1:
    raise SystemExit("The first sequence image must be im1.")

previous_fen = STARTING_FEN

for number, image_path in images[1:]:
    cmd = [
        sys.executable,
        "lc2fen.py",
        str(image_path),
        A1_POS,
        previous_fen,
        "--onnx",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)

    predicted_fen = result.stdout.strip().splitlines()[-1]
    print(f"{image_path.name}: {predicted_fen}", flush=True)

    if DISPLAY_VISUALISATION:
        show_prediction(image_path, predicted_fen, A1_POS)

    previous_fen = predicted_fen
