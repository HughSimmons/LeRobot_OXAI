"""Simple subprocess feeder for lc2fen.py."""

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys


# IMAGE_PATH = "data/predictions/test1.jpg"
IMAGE_PATH = "chess_board_ims/im2.jpeg"
# A1_POS = "BL"
A1_POS = "TR"
# PREVIOUS_FEN = "r3r1k1/1pq2pp1/2p2n2/1PNn4/2QN2b1/4R1P1/4PP2/2R3KB"
PREVIOUS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
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


cmd = [
    sys.executable,
    "lc2fen.py",
    IMAGE_PATH,
    A1_POS,
]

if PREVIOUS_FEN:
    cmd.append(PREVIOUS_FEN)

cmd.append("--onnx")

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)

predicted_fen = result.stdout.strip().splitlines()[-1]
print(predicted_fen, flush=True)

if DISPLAY_VISUALISATION:
    show_prediction(IMAGE_PATH, predicted_fen, A1_POS)
