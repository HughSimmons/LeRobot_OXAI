"""Pickup / set-down stability measurements for one simulated rook move.

A `StabilityProbe` is passed to `multisim_chess_fast.run_sim_move(step_callback=...)`. It only
reads simulation state (no physics change) and records, per move:

  before grasp   rook tilt and how far it was pushed from its spawn point by the approach
  pickup/carry   rook pose relative to the fixed jaw (gripper_link) from lift-off to the end of
                 the carry: slip (mm), relative rotation (deg), in-jaw tilt (deg, angle between the
                 rook axis and the gripper-fixed direction the rook axis had just before closing),
                 jaw contacts at the end of the carry
  set-down       first board contact, slip in the air before contact, push-in after contact,
                 pose at release, displacement during release settle and retreat, final pose
                 and velocity

`classify()` turns those into a pass/fail with reasons (`PICKUP_REASONS`, `SETDOWN_REASONS`).
Landing accuracy is recorded but never part of the pass.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS = {
    "before_tilt_deg": 3.0,  # rook must be upright before the grasp ...
    "before_push_mm": 3.0,  # ... and not shoved by the approaching jaws
    "jaw_slip_mm": 2.0,  # rook position change in the jaws, lift-off -> end of carry
    "jaw_tilt_deg": 8.0,  # in-jaw tilt margin (see module doc): reported as a flag, NOT a failure
    "air_slip_mm": 5.0,  # slip in the jaws while lowering (baseline e4->e5 ~2 mm), or a jaw loses contact
    "upright_deg": 3.0,  # final tilt for "upright"
    "toppled_deg": 15.0,  # final tilt above this = toppled (between: leaning)
    "slide_mm": 1.0,  # board-plane movement from first board contact to rest
    "retreat_mm": 1.0,  # part of that movement during the retreat = dragged
    "settled_lin_mm_s": 1.0,
    "settled_ang_deg_s": 3.0,
}

PICKUP_REASONS = ("knocked_before_grasp", "missed_grasp", "dropped_in_carry", "slipped_in_jaws")
SETDOWN_REASONS = (
    "slipped_before_release",
    "toppled",
    "leaning",
    "dragged_on_retreat",
    "slid_on_board",
    "not_settled",
)


def _rot(p, quat) -> np.ndarray:
    return np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _rot_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


class StabilityProbe:
    def __init__(
        self,
        sim,
        world: dict,
        *,
        closeidx: int,
        release_idx: int,
        segments: dict,
        stop_after_carry: bool,
        lifted_z: float,
        trace_every: int = 0,
    ):
        p = sim.p
        self.p = p
        self.piece = world["piece_ids"][0]
        self.robot = world["robot_id"]
        links = {p.getJointInfo(self.robot, j)[12].decode(): j for j in range(p.getNumJoints(self.robot))}
        self.fixed_jaw = links["gripper_link"]
        self.moving_jaw = links["moving_jaw_so101_v1_link"]
        self.closeidx = closeidx
        self.release_idx = release_idx
        self.lower_start = segments["destination_lower_place"]["start"]
        self.retreat_start = segments["retreat"]["start"]
        self.retreat_end = segments["retreat"]["end"]
        self.stop_after_carry = stop_after_carry
        self.lifted_z = lifted_z
        self.trace_every = trace_every

        pos, _ = p.getBasePositionAndOrientation(self.piece)
        self.spawn_xy = np.array(pos[:2])
        self.m: dict = {"lifted": False, "trace": [] if trace_every else None}
        self._axis_in_grip = None
        self._ref_rel_pos = None
        self._ref_rel_rot = None
        self._carry_done = False
        self._in_contact = False
        self._released = False
        self._retreat_started = False
        self._retreat_done = False
        self.m.update(
            jaw_slip_max_mm=0.0,
            jaw_rot_max_deg=0.0,
            jaw_tilt_max_deg=0.0,
            air_slip_max_mm=0.0,
            pushin_max_mm=0.0,
            post_release_tilt_max_deg=0.0,
        )

    # -- helpers ---------------------------------------------------------------------------
    def _poses(self):
        p = self.p
        pp, po = p.getBasePositionAndOrientation(self.piece)
        ls = p.getLinkState(self.robot, self.fixed_jaw)
        return np.array(pp), _rot(p, po), np.array(ls[4]), _rot(p, ls[5])

    def _touching_board(self) -> bool:
        # The board is many bodies (squares, base, edge supports): anything that is not the robot.
        return any(c[2] != self.robot for c in self.p.getContactPoints(bodyA=self.piece))

    def _jaw_contacts(self) -> dict:
        p = self.p
        return {
            "fixed": len(p.getContactPoints(bodyA=self.piece, bodyB=self.robot, linkIndexB=self.fixed_jaw)),
            "moving": len(p.getContactPoints(bodyA=self.piece, bodyB=self.robot, linkIndexB=self.moving_jaw)),
        }

    # -- callback --------------------------------------------------------------------------
    def __call__(self, step: int, move_idx: int) -> bool:
        m = self.m
        pp, rp, pg, rg = self._poses()
        axis = rp[:, 2]
        tilt = _angle_deg(axis, np.array([0.0, 0.0, 1.0]))
        tilt = min(tilt, 180.0 - tilt)

        if self._axis_in_grip is None and move_idx >= self.closeidx:
            # Just before the jaws close: upright? pushed by the approach?
            m["before_tilt_deg"] = tilt
            m["before_push_mm"] = float(np.linalg.norm(pp[:2] - self.spawn_xy) * 1000.0)
            m["jaw_xyz_at_close"] = pg.tolist()  # fixed-jaw link origin (world, m) when closing starts
            self._axis_in_grip = rg.T @ axis

        rel_pos = rg.T @ (pp - pg)
        rel_rot = rg.T @ rp
        jaw_tilt = _angle_deg(axis, rg @ self._axis_in_grip) if self._axis_in_grip is not None else 0.0

        if not m["lifted"] and move_idx > self.closeidx and pp[2] > self.lifted_z:
            m["lifted"] = True
            m["lift_step"] = step
            m["jaw_tilt_at_lift_deg"] = jaw_tilt
            self._ref_rel_pos, self._ref_rel_rot = rel_pos, rel_rot

        drift_mm = drift_deg = 0.0
        if self._ref_rel_pos is not None:
            drift_mm = float(np.linalg.norm(rel_pos - self._ref_rel_pos) * 1000.0)
            drift_deg = _rot_angle_deg(self._ref_rel_rot.T @ rel_rot)

        # Pickup + carry: lift-off until the arm starts lowering at the target.
        if m["lifted"] and not self._carry_done:
            if move_idx < self.lower_start:
                m["jaw_slip_max_mm"] = max(m["jaw_slip_max_mm"], drift_mm)
                m["jaw_rot_max_deg"] = max(m["jaw_rot_max_deg"], drift_deg)
                m["jaw_tilt_max_deg"] = max(m["jaw_tilt_max_deg"], jaw_tilt)
        if not self._carry_done and move_idx >= self.lower_start:
            self._carry_done = True
            m["carry_end"] = {
                "z_mm": pp[2] * 1000.0,
                "jaw_slip_mm": drift_mm,
                "jaw_tilt_deg": jaw_tilt,
                "world_tilt_deg": tilt,
                "jaw_contacts": self._jaw_contacts(),
            }
            if self.stop_after_carry:
                return True

        # Lowering (before release): slip in the air vs push-in after touching the board.
        # First board contact after the carry: while still held (pushed down) or after release (dropped).
        if self._carry_done and m["lifted"] and not self._in_contact and self._touching_board():
            self._in_contact = True
            m["first_contact"] = {
                "step": step,
                "move_idx": move_idx,
                "before_release": not self._released and move_idx < self.release_idx,
                "xy": pp[:2].tolist(),
                "z_mm": pp[2] * 1000.0,
                "tilt_deg": tilt,
                "jaw_slip_mm": drift_mm,
            }
        if self._carry_done and not self._released and m["lifted"]:
            if self._in_contact:
                m["pushin_max_mm"] = max(m["pushin_max_mm"], drift_mm)
            else:
                m["air_slip_max_mm"] = max(m["air_slip_max_mm"], drift_mm)

        if not self._released and move_idx >= self.release_idx:
            self._released = True
            m["release"] = {
                "step": step,
                "xy": pp[:2].tolist(),
                "z_mm": pp[2] * 1000.0,
                "tilt_deg": tilt,
                "jaw_slip_mm": drift_mm,
                "jaw_contacts": self._jaw_contacts(),
            }
        if self._released:
            m["post_release_tilt_max_deg"] = max(m["post_release_tilt_max_deg"], tilt)
        if not self._retreat_started and move_idx >= self.retreat_start:
            self._retreat_started = True
            m["retreat_start_xy"] = pp[:2].tolist()
        if not self._retreat_done and move_idx >= self.retreat_end:
            self._retreat_done = True
            m["retreat_end_xy"] = pp[:2].tolist()

        if self.trace_every and step % self.trace_every == 0:
            m["trace"].append([step, move_idx, round(pp[2] * 1000.0, 2), round(tilt, 2), round(drift_mm, 2)])
        return False

    def finish(self) -> dict:
        """Final pose and velocity (call after run_sim_move returns)."""
        p = self.p
        pp, po = p.getBasePositionAndOrientation(self.piece)
        lin, ang = p.getBaseVelocity(self.piece)
        axis = _rot(p, po)[:, 2]
        tilt = _angle_deg(axis, np.array([0.0, 0.0, 1.0]))
        m = self.m
        m["final"] = {
            "xyz": list(pp),
            "tilt_deg": min(tilt, 180.0 - tilt),
            "lin_speed_mm_s": float(np.linalg.norm(lin) * 1000.0),
            "ang_speed_deg_s": float(np.degrees(np.linalg.norm(ang))),
        }
        if "first_contact" in m:
            m["slide_mm"] = float(np.linalg.norm(np.array(pp[:2]) - np.array(m["first_contact"]["xy"])) * 1000.0)
        if "release" in m:
            m["drop_height_mm"] = m["release"]["z_mm"] - pp[2] * 1000.0  # rook centre at release minus at rest
        if "retreat_start_xy" in m and "retreat_end_xy" in m:
            m["retreat_move_mm"] = float(
                np.linalg.norm(np.array(m["retreat_end_xy"]) - np.array(m["retreat_start_xy"])) * 1000.0
            )
        return m


def classify(m: dict, phase: str, th: dict | None = None) -> list[str]:
    """Fail reasons in causal order (empty list = pass). phase: 'pickup' or 'setdown'."""
    th = {**DEFAULT_THRESHOLDS, **(th or {})}
    reasons = []
    if m.get("before_tilt_deg", np.inf) > th["before_tilt_deg"] or m.get("before_push_mm", np.inf) > th["before_push_mm"]:
        reasons.append("knocked_before_grasp")
    if not m.get("lifted"):
        reasons.append("missed_grasp")
        return reasons
    carry = m.get("carry_end")
    if carry is None or min(carry["jaw_contacts"].values()) == 0:
        reasons.append("dropped_in_carry")
    if m["jaw_slip_max_mm"] > th["jaw_slip_mm"]:
        reasons.append("slipped_in_jaws")
    if phase == "pickup":
        return reasons

    rel = m.get("release")
    if m["air_slip_max_mm"] > th["air_slip_mm"] or (rel is not None and min(rel["jaw_contacts"].values()) == 0):
        reasons.append("slipped_before_release")
    final = m.get("final", {})
    tilt = final.get("tilt_deg", np.inf)
    if tilt > th["toppled_deg"]:
        reasons.append("toppled")
    elif tilt > th["upright_deg"]:
        reasons.append("leaning")
    if tilt <= th["toppled_deg"]:
        if m.get("retreat_move_mm", 0.0) > th["retreat_mm"]:
            reasons.append("dragged_on_retreat")
        if m.get("slide_mm", np.inf) > th["slide_mm"]:
            reasons.append("slid_on_board")
        if (
            final.get("lin_speed_mm_s", np.inf) > th["settled_lin_mm_s"]
            or final.get("ang_speed_deg_s", np.inf) > th["settled_ang_deg_s"]
        ):
            reasons.append("not_settled")
    return reasons
