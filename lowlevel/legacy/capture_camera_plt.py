import argparse

import cv2
import matplotlib.pyplot as plt


def capture_frame(camera_index, warmup_frames):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        frame = None
        for _ in range(max(warmup_frames, 1)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read from camera index {camera_index}")
        return frame
    finally:
        capture.release()


def show_frame(frame, camera_index):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure("Camera capture")
    plt.imshow(rgb_frame)
    plt.title(f"Camera index {camera_index}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Capture one OpenCV camera frame and display it with matplotlib."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use. Defaults to 0.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Number of frames to discard before displaying one. Defaults to 5.",
    )
    args = parser.parse_args()

    frame = capture_frame(args.camera_index, args.warmup_frames)
    show_frame(frame, args.camera_index)


if __name__ == "__main__":
    main()
