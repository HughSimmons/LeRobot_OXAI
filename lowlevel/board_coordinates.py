"""Board coordinate types shared by square and continuous motion flows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


FILES = "abcdefgh"
DEFAULT_BOARD_ORIGIN = (0.25, 0.0, 0.0)
DEFAULT_SQUARE_SIZE = 0.04


@dataclass(frozen=True)
class XYPoint:
    """A named world-frame XY target in metres."""

    x: float
    y: float
    name: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("XYPoint coordinates must be finite")

    def __str__(self) -> str:
        return self.name or coordinate_label(self.x, self.y)

    def as_xy(self) -> tuple[float, float]:
        return float(self.x), float(self.y)

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "continuous_xy",
            "frame": "world",
            "x": float(self.x),
            "y": float(self.y),
            "name": self.name,
        }


def coordinate_label(x: float, y: float) -> str:
    """Return a deterministic filesystem-safe label at 0.1 mm precision."""

    def component(value: float) -> str:
        sign = "p" if value >= 0.0 else "m"
        return f"{sign}{abs(value):.4f}".replace(".", "p")

    return f"xy_{component(float(x))}_{component(float(y))}"


def validate_square(square: str) -> str:
    normalized = square.strip().lower()
    if (
        len(normalized) != 2
        or normalized[0] not in FILES
        or normalized[1] not in "12345678"
    ):
        raise ValueError(f"Invalid chess square {square!r}")
    return normalized


def square_center_world_xy(
    square: str,
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> tuple[float, float]:
    square = validate_square(square)
    file_index = FILES.index(square[0])
    rank_index = int(square[1]) - 1
    board_x, board_y = float(board_origin[0]), float(board_origin[1])
    board_size = 8.0 * float(square_size)
    return (
        board_x - board_size / 2.0 + (file_index + 0.5) * square_size,
        board_y - board_size / 2.0 + (rank_index + 0.5) * square_size,
    )


def point_from_xy(
    x: float,
    y: float,
    *,
    frame: str = "world",
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    name: str | None = None,
) -> XYPoint:
    """Create a world point from world or board-centred coordinates."""

    if frame == "world":
        world_x, world_y = float(x), float(y)
    elif frame == "board":
        world_x = float(board_origin[0]) + float(x)
        world_y = float(board_origin[1]) + float(y)
    else:
        raise ValueError("frame must be 'world' or 'board'")
    return XYPoint(world_x, world_y, name=name)


def location_world_xy(
    location: str | XYPoint | Sequence[float],
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> tuple[float, float]:
    if isinstance(location, XYPoint):
        return location.as_xy()
    if isinstance(location, str):
        return square_center_world_xy(
            location,
            board_origin=board_origin,
            square_size=square_size,
        )
    if len(location) < 2:
        raise ValueError("Continuous locations need x and y coordinates")
    x, y = float(location[0]), float(location[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Continuous locations must be finite")
    return x, y


def location_world_xyz(
    location: str | XYPoint | Sequence[float],
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
    z_offset: float = 0.04,
) -> tuple[float, float, float]:
    x, y = location_world_xy(
        location,
        board_origin=board_origin,
        square_size=square_size,
    )
    return x, y, float(board_origin[2]) + float(z_offset)


def location_label(location: str | XYPoint | Sequence[float]) -> str:
    if isinstance(location, str):
        return validate_square(location)
    if isinstance(location, XYPoint):
        return str(location)
    return coordinate_label(float(location[0]), float(location[1]))


def location_file(
    location: str | XYPoint | Sequence[float],
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> str:
    """Return the nearest board file for legacy reach-policy selection."""

    if isinstance(location, str):
        return validate_square(location)[0]
    x, _ = location_world_xy(
        location,
        board_origin=board_origin,
        square_size=square_size,
    )
    board_min_x = float(board_origin[0]) - 4.0 * square_size
    file_index = int(math.floor((x - board_min_x) / square_size))
    return FILES[min(max(file_index, 0), 7)]


def is_exact_square(
    location: object,
    square: str,
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
    tolerance: float = 1e-9,
) -> bool:
    """Match a square label or a continuous point at that square's centre."""

    if isinstance(location, str):
        return validate_square(location) == validate_square(square)
    if not isinstance(location, (XYPoint, list, tuple)):
        return False
    location_x, location_y = location_world_xy(
        location,
        board_origin=board_origin,
        square_size=square_size,
    )
    square_x, square_y = square_center_world_xy(
        square,
        board_origin=board_origin,
        square_size=square_size,
    )
    return math.hypot(location_x - square_x, location_y - square_y) <= tolerance


def location_distance_in_squares(
    start: str | XYPoint | Sequence[float],
    end: str | XYPoint | Sequence[float],
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> float:
    start_x, start_y = location_world_xy(start, board_origin=board_origin, square_size=square_size)
    end_x, end_y = location_world_xy(end, board_origin=board_origin, square_size=square_size)
    return (abs(start_x - end_x) + abs(start_y - end_y)) / square_size


def world_xy_within_square(
    world_xy: Sequence[float],
    square: str,
    *,
    board_origin: Sequence[float] = DEFAULT_BOARD_ORIGIN,
    square_size: float = DEFAULT_SQUARE_SIZE,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether a world XY coordinate is within a named square footprint."""

    if len(world_xy) < 2:
        return False
    x, y = float(world_xy[0]), float(world_xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    center_x, center_y = square_center_world_xy(
        square,
        board_origin=board_origin,
        square_size=square_size,
    )
    half_side = square_size / 2.0 + tolerance
    return abs(x - center_x) <= half_side and abs(y - center_y) <= half_side
