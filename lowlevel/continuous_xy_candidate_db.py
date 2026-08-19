"""Persistent candidate store for successful continuous XY pick-and-place runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


LOWLEVEL_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_DB_PATH = (
    LOWLEVEL_DIR / "rook_kiri_xy_lookup" / "continuous_xy_candidates.sqlite3"
)
# The lower-place segment can cross the release-height sentinel a few waypoints
# before the commanded jaw opening. Keep only genuinely early/deep transport drops
# out of the continuous donor pool.
PREMATURE_DROP_GRACE_WAYPOINTS = 4
PREMATURE_DROP_Z_TOLERANCE_M = 0.002


def json_safe(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def is_excessive_premature_drop(result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Separate transport drops from benign near-release threshold crossings."""

    if not bool(result.get("premature_drop", False)):
        return False, {
            "premature_drop": False,
            "allowed": True,
            "reason": "not_detected",
        }

    release_index = result.get("release_move_idx")
    drop_index = result.get("premature_drop_move_idx")
    threshold = result.get("premature_drop_z_threshold")
    drop_z = result.get("premature_drop_z")
    waypoint_gap = (
        None
        if release_index is None or drop_index is None
        else int(release_index) - int(drop_index)
    )
    drop_depth = (
        None
        if threshold is None or drop_z is None
        else float(threshold) - float(drop_z)
    )
    excessive = (
        waypoint_gap is None
        or drop_depth is None
        or waypoint_gap > PREMATURE_DROP_GRACE_WAYPOINTS
        or drop_depth > PREMATURE_DROP_Z_TOLERANCE_M
    )
    return excessive, {
        "premature_drop": True,
        "allowed": not excessive,
        "waypoint_gap_before_release": waypoint_gap,
        "drop_depth_below_threshold_m": drop_depth,
        "grace_waypoints": PREMATURE_DROP_GRACE_WAYPOINTS,
        "z_tolerance_m": PREMATURE_DROP_Z_TOLERANCE_M,
        "reason": (
            "excessive_early_or_deep_drop"
            if excessive
            else "near_release_threshold_crossing_allowed"
        ),
    }


def candidate_fingerprint(candidate: dict[str, Any]) -> str:
    canonical = json.dumps(json_safe(candidate), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def initialize_database(db_path: Path) -> Path:
    db_path = db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS continuous_xy_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL UNIQUE,
                first_saved_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                move_id TEXT NOT NULL,
                from_x REAL NOT NULL,
                from_y REAL NOT NULL,
                to_x REAL NOT NULL,
                to_y REAL NOT NULL,
                grasp_offset_json TEXT NOT NULL,
                place_offset_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                search_json TEXT NOT NULL,
                piece_config_json TEXT NOT NULL,
                source_lookup_json TEXT NOT NULL,
                drop_policy_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS continuous_xy_candidates_endpoints_idx
            ON continuous_xy_candidates (from_x, from_y, to_x, to_y)
            """
        )
    return db_path


def save_candidate(
    db_path: Path,
    *,
    move_id: str,
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
    grasp_offset: np.ndarray,
    place_offset: np.ndarray,
    metrics: dict[str, Any],
    search: dict[str, Any],
    piece_config: dict[str, Any],
    source_lookup: dict[str, Any],
    drop_policy: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "move_id": move_id,
        "from_xy": list(map(float, from_xy)),
        "to_xy": list(map(float, to_xy)),
        "grasp_offset": np.array(grasp_offset, dtype=float),
        "place_offset": np.array(place_offset, dtype=float),
        "search": search,
        "piece_config": piece_config,
        "drop_policy": drop_policy,
    }
    fingerprint = candidate_fingerprint(record)
    now = datetime.now(timezone.utc).isoformat()
    db_path = initialize_database(db_path)
    values = (
        fingerprint,
        now,
        now,
        move_id,
        float(from_xy[0]),
        float(from_xy[1]),
        float(to_xy[0]),
        float(to_xy[1]),
        json.dumps(json_safe(grasp_offset), sort_keys=True),
        json.dumps(json_safe(place_offset), sort_keys=True),
        json.dumps(json_safe(metrics), sort_keys=True),
        json.dumps(json_safe(search), sort_keys=True),
        json.dumps(json_safe(piece_config), sort_keys=True),
        json.dumps(json_safe(source_lookup), sort_keys=True),
        json.dumps(json_safe(drop_policy), sort_keys=True),
    )
    with sqlite3.connect(db_path) as connection:
        existing = connection.execute(
            "SELECT id, seen_count FROM continuous_xy_candidates WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if existing is None:
            cursor = connection.execute(
                """
                INSERT INTO continuous_xy_candidates (
                    fingerprint, first_saved_at, last_seen_at, move_id,
                    from_x, from_y, to_x, to_y,
                    grasp_offset_json, place_offset_json, metrics_json,
                    search_json, piece_config_json, source_lookup_json,
                    drop_policy_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return {"id": cursor.lastrowid, "inserted": True, "fingerprint": fingerprint}
        candidate_id, seen_count = existing
        connection.execute(
            """
            UPDATE continuous_xy_candidates
            SET last_seen_at = ?, seen_count = ?, metrics_json = ?, source_lookup_json = ?
            WHERE id = ?
            """,
            (
                now,
                int(seen_count) + 1,
                json.dumps(json_safe(metrics), sort_keys=True),
                json.dumps(json_safe(source_lookup), sort_keys=True),
                candidate_id,
            ),
        )
    return {
        "id": candidate_id,
        "inserted": False,
        "seen_count": int(seen_count) + 1,
        "fingerprint": fingerprint,
    }


def find_successful_candidates(
    db_path: Path = DEFAULT_CANDIDATE_DB_PATH,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return stored candidates for future nearest-neighbour reuse logic."""

    db_path = db_path.expanduser().resolve()
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, move_id, from_x, from_y, to_x, to_y,
                   grasp_offset_json, place_offset_json, metrics_json,
                   search_json, piece_config_json, source_lookup_json,
                   drop_policy_json, first_saved_at, last_seen_at, seen_count
            FROM continuous_xy_candidates
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    keys = (
        "id", "move_id", "from_x", "from_y", "to_x", "to_y",
        "grasp_offset", "place_offset", "metrics", "search", "piece_config",
        "source_lookup", "drop_policy", "first_saved_at", "last_seen_at", "seen_count",
    )
    records = []
    for row in rows:
        record = dict(zip(keys, row))
        for key in (
            "grasp_offset", "place_offset", "metrics", "search", "piece_config",
            "source_lookup", "drop_policy",
        ):
            record[key] = json.loads(record[key])
        records.append(record)
    return records
