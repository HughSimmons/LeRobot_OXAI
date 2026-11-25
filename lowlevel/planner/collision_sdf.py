"""Vectorized analytical Capsule/Sphere-vs-Cylinder Signed Distance Field (SDF) for SO-101 chess.

Models:
1. Standing chess pieces on the board as finite vertical cylinders ``(x, y, z_base, height, radius)``
   from FEN piece types, simulation presets (``rook_kiri`` / ``cylinder``), or continuous
   ``coord_finder`` measurements.
2. SO-101 robot links, wrist servo, wrist camera mount, fixed jaw, and moving jaw (tracking
   gripper opening ``q[5]``) as a calibrated multi-sphere / capsule hierarchy calibrated directly
   against the V-HACD meshes in ``so101_with_mount_vhacd_gripper.urdf``.
3. Rigidly attached held piece (``HeldPieceAttachment``) transformed dynamically via
   ``T_world_tcp(q) @ T_world_tcp(q_pick)^(-1)`` during ``lift``, ``transit``, and ``lower_place``,
   so both tilted and vertical held pieces are checked against all standing pieces and the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .frames import BoardConfig
from .kinematics_so101 import SO101Kinematics

# Staunton tournament / standard chess piece dimensions in metres (height, base_radius)
STAUNTON_PIECE_DIMS_M: dict[str, tuple[float, float]] = {
    "p": (0.045, 0.0135),
    "r": (0.052, 0.0150),
    "n": (0.058, 0.0155),
    "b": (0.065, 0.0155),
    "q": (0.078, 0.0170),
    "k": (0.092, 0.0180),
}
SIM_ROOK_DIMS_M = (0.040, 0.0125)


@dataclass(frozen=True)
class PieceCylinder:
    """Vertical cylindrical obstacle for a standing chess piece in world coordinates (metres)."""

    label: str
    x_m: float
    y_m: float
    z_base_m: float
    height_m: float
    radius_m: float
    piece_type: str = "r"

    @property
    def z_top_m(self) -> float:
        return self.z_base_m + self.height_m


@dataclass(frozen=True)
class HeldPieceAttachment:
    """Rigid attachment of a grasped chess piece in ``gripper_frame_link`` (TCP) coordinates."""

    points_tcp_m: np.ndarray   # (K, 3) sphere centers along piece shaft from base to crown
    radius_m: float
    height_m: float

    @classmethod
    def from_pick_pose(
        cls,
        kin: SO101Kinematics,
        q_pick_deg: Sequence[float],
        piece_xy_world_m: Sequence[float],
        z_surface_m: float,
        height_m: float,
        radius_m: float,
        n_spheres: int = 4,
    ) -> "HeldPieceAttachment":
        T_pick, _ = kin.fk_tcp(q_pick_deg)
        T_inv = np.linalg.inv(T_pick)
        zs = np.linspace(
            z_surface_m + min(0.006, 0.2 * height_m),
            z_surface_m + max(height_m - 0.004, 0.010),
            n_spheres,
        )
        pts_world = np.column_stack(
            [
                np.full(n_spheres, float(piece_xy_world_m[0]), dtype=np.float64),
                np.full(n_spheres, float(piece_xy_world_m[1]), dtype=np.float64),
                zs,
                np.ones(n_spheres, dtype=np.float64),
            ]
        )
        pts_tcp = (T_inv @ pts_world.T).T[:, :3]
        return cls(points_tcp_m=pts_tcp, radius_m=float(radius_m), height_m=float(height_m))


# Calibrated robot collision spheres: (link_key, local_xyz_m, radius_m, is_jaw_tip)
# Measured directly against wrist_roll_follower_so101_v1_vhacd.obj, moving_jaw_so101_v1_vhacd.obj,
# and so101_wrist_cam_mount.stl.
ROBOT_COLLISION_SPHERES: tuple[tuple[str, tuple[float, float, float], float, bool], ...] = (
    # Lower arm / elbow body
    ("lower_arm", (-0.055, 0.0, 0.0), 0.022, False),
    ("lower_arm", (-0.115, 0.0, 0.0), 0.022, False),
    # Wrist body
    ("wrist", (0.0, -0.030, 0.010), 0.022, False),
    # Gripper palm & wrist roll servo
    ("gripper", (0.004, 0.000, -0.028), 0.023, False),
    ("gripper", (0.000, 0.000, -0.055), 0.018, False),
    # Wrist camera mount & lens housing (extends +Y up to +88.7 mm on gripper_link!)
    ("wrist_cam", (-0.018, 0.015, -0.005), 0.019, False),
    ("wrist_cam", (-0.048, 0.015, -0.005), 0.018, False),
    # Fixed jaw (in tcp = gripper_frame_link: +Z is downward tip at z=0.0, x in [0, +0.012])
    ("tcp", (0.006, 0.000, -0.024), 0.0090, False),
    ("tcp", (0.006, 0.000, -0.0085), 0.0065, True),
    # Moving jaw (in moving_jaw_so101_v1_link: tip sits near y ~ -0.064 m, outer back at x ~ -0.011 m)
    ("moving_jaw", (-0.009, -0.045, 0.018), 0.0095, False),
    ("moving_jaw", (-0.009, -0.062, 0.018), 0.0075, True),
)


class SceneCollisionChecker:
    """Fast vectorized signed-distance evaluator for arm + camera + jaws + held piece."""

    def __init__(
        self,
        kin: SO101Kinematics,
        board: BoardConfig,
        pieces: Sequence[PieceCylinder] = (),
        *,
        table_z_m: float = 0.0,
        piece_safety_margin_m: float = 0.0035,
        board_transit_floor_margin_m: float = 0.006,
        table_floor_margin_m: float = 0.0,
    ) -> None:
        self.kin = kin
        self.board = board
        self.pieces = list(pieces)
        self.table_z_m = float(table_z_m)
        self.piece_safety_margin_m = float(piece_safety_margin_m)
        self.board_transit_floor_margin_m = float(board_transit_floor_margin_m)
        self.table_floor_margin_m = float(table_floor_margin_m)

        # Pre-group robot spheres by link for single-pass batch matmul
        self._link_order = ("lower_arm", "wrist", "gripper", "wrist_cam", "tcp", "moving_jaw")
        # Highest active arm joint index (0..4) affecting each link
        link_max_joint = {
            "lower_arm": 2,
            "wrist": 3,
            "gripper": 4,
            "wrist_cam": 4,
            "tcp": 4,
            "moving_jaw": 4,
        }
        self._link_pts_homo: dict[str, np.ndarray] = {}
        radii_list: list[float] = []
        jaw_tip_flags: list[bool] = []
        max_joint_list: list[int] = []
        for lk in self._link_order:
            entries = [s for s in ROBOT_COLLISION_SPHERES if s[0] == lk]
            if not entries:
                continue
            pts = np.array([[e[1][0], e[1][1], e[1][2], 1.0] for e in entries], dtype=np.float64)
            self._link_pts_homo[lk] = pts
            radii_list.extend(e[2] for e in entries)
            jaw_tip_flags.extend(e[3] for e in entries)
            max_joint_list.extend([link_max_joint[lk]] * len(entries))

        self._robot_radii = np.array(radii_list, dtype=np.float64)          # (S_rob,)
        self._robot_is_jaw_tip = np.array(jaw_tip_flags, dtype=bool)       # (S_rob,)
        self._robot_max_joint = np.array(max_joint_list, dtype=np.int32)   # (S_rob,)

    def with_pieces(self, pieces: Sequence[PieceCylinder]) -> "SceneCollisionChecker":
        return SceneCollisionChecker(
            self.kin,
            self.board,
            pieces,
            table_z_m=self.table_z_m,
            piece_safety_margin_m=self.piece_safety_margin_m,
            board_transit_floor_margin_m=self.board_transit_floor_margin_m,
            table_floor_margin_m=self.table_floor_margin_m,
        )

    def active_obstacles(self, exclude_labels: set[str] | Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return vectorized arrays ``(xy (M,2), z0 (M,), z1 (M,), r (M,))`` for active standing pieces."""
        excl = set(exclude_labels)
        obs = [p for p in self.pieces if p.label not in excl]
        if not obs:
            empty = np.zeros((0,), dtype=np.float64)
            return np.zeros((0, 2), dtype=np.float64), empty, empty, empty
        xy = np.array([[p.x_m, p.y_m] for p in obs], dtype=np.float64)
        z0 = np.array([p.z_base_m for p in obs], dtype=np.float64)
        z1 = np.array([p.z_top_m for p in obs], dtype=np.float64)
        r = np.array([p.radius_m + self.piece_safety_margin_m for p in obs], dtype=np.float64)
        return xy, z0, z1, r

    def world_spheres_batch(
        self,
        Q_deg: np.ndarray,
        held_piece: HeldPieceAttachment | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ``(centers_world (N, S, 3), radii (S,), is_tip_or_held (S,))`` for batch ``Q_deg``."""
        Q = np.asarray(Q_deg, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        frames = self.kin.fk_all_links_batch(Q)

        blocks = []
        for lk, pts_h in self._link_pts_homo.items():
            # frames[lk]: (N, 4, 4), pts_h.T: (4, K_l) -> (N, 4, K_l) -> (N, K_l, 3)
            w = np.matmul(frames[lk], pts_h.T)[:, :3, :].transpose(0, 2, 1)
            blocks.append(w)

        radii = self._robot_radii
        tip_flags = self._robot_is_jaw_tip

        if held_piece is not None:
            k_h = held_piece.points_tcp_m.shape[0]
            pts_h = np.column_stack([held_piece.points_tcp_m, np.ones(k_h, dtype=np.float64)])
            w_held = np.matmul(frames["tcp"], pts_h.T)[:, :3, :].transpose(0, 2, 1)
            blocks.append(w_held)
            radii = np.concatenate([radii, np.full(k_h, held_piece.radius_m, dtype=np.float64)])
            tip_flags = np.concatenate([tip_flags, np.ones(k_h, dtype=bool)])

        centers = np.concatenate(blocks, axis=1)  # (N, S, 3)
        return centers, radii, tip_flags

    def evaluate_clearance_batch(
        self,
        Q_deg: np.ndarray,
        *,
        exclude_labels: Sequence[str] = (),
        held_piece: HeldPieceAttachment | None = None,
        allow_tip_near_board: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized signed clearance (m) for ``Q_deg`` of shape ``(N, 6)``.

        Returns:
            min_clearance_per_pose: ``(N,)`` worst signed distance (m); ``< 0`` means collision/margin violation.
            sphere_clearances:      ``(N, S)`` signed distance per collision sphere across obstacles + floor.
        """
        centers, radii, tip_flags = self.world_spheres_batch(Q_deg, held_piece=held_piece)
        n, s, _ = centers.shape

        # 1. Board & table floor clearance
        half_span = 0.5 * self.board.board_span_m + self.board.border_margin_m
        over_board = (
            (np.abs(centers[:, :, 0] - self.board.origin_xyz_m[0]) <= half_span)
            & (np.abs(centers[:, :, 1] - self.board.origin_xyz_m[1]) <= half_span)
        )
        board_floor = self.board.surface_z_m + self.board_transit_floor_margin_m
        table_floor = self.table_z_m + self.table_floor_margin_m

        req_floor = np.where(over_board, board_floor, table_floor)
        if allow_tip_near_board:
            # During pick/place strokes, jaw tips and held-piece base may approach to 1.5 mm below surface_z
            tip_floor = np.where(over_board, self.board.surface_z_m - 0.002, self.table_z_m - 0.002)
            req_floor = np.where(tip_flags[None, :], tip_floor, req_floor)

        floor_clear = centers[:, :, 2] - radii[None, :] - req_floor  # (N, S)
        worst_clear = floor_clear

        # 2. Standing piece cylinders
        obs_xy, obs_z0, obs_z1, obs_r = self.active_obstacles(exclude_labels)
        if obs_xy.shape[0] > 0:
            # Broadcast (N, S, 1, 2) vs (1, 1, M, 2)
            diff_xy = centers[:, :, None, :2] - obs_xy[None, None, :, :]
            dist_xy_center = np.linalg.norm(diff_xy, axis=-1)              # (N, S, M)
            d_xy = np.maximum(dist_xy_center - obs_r[None, None, :], 0.0)

            cz = centers[:, :, None, 2]
            d_z = np.maximum(
                np.maximum(obs_z0[None, None, :] - cz, cz - obs_z1[None, None, :]),
                0.0,
            )
            outside_dist = np.hypot(d_xy, d_z)
            inside_pen = -np.minimum(
                obs_r[None, None, :] - dist_xy_center,
                np.minimum(cz - obs_z0[None, None, :], obs_z1[None, None, :] - cz),
            )
            surf_dist = np.where((d_xy == 0.0) & (d_z == 0.0), inside_pen, outside_dist)
            piece_clear = surf_dist.min(axis=2) - radii[None, :]           # (N, S)
            worst_clear = np.minimum(worst_clear, piece_clear)

        return worst_clear.min(axis=1), worst_clear

    def hinge_penalty_and_grad_batch(
        self,
        Q_deg: np.ndarray,
        safe_clearance_m: float,
        *,
        exclude_labels: Sequence[str] = (),
        held_piece: HeldPieceAttachment | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Analytical hinge-squared collision loss ``0.5 * sum(max(0, safe - c)^2)`` and ``d(loss)/dQ_deg[:, :5]``.

        Computes exact world-space repulsive normals for every sphere violating ``safe_clearance_m``
        and projects them onto the 5 revolute joint axes via ``z_j x (c_s - o_j)`` in a single pass!
        Returns ``(total_loss, grad_Q_arm_deg (N, 5), min_clearance_per_pose (N,))``.
        """
        Q = np.asarray(Q_deg, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        n = Q.shape[0]
        frames = self.kin.fk_all_links_batch(Q)

        # Extract joint origins o_j (N, 5, 3) and axes z_j (N, 5, 3) BEFORE/AT each joint rotation
        # Note: in URDF Rz(q_i), the rotation axis in world frame is column 2 of frames[link_i]!
        joint_keys = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
        origins = np.stack([frames[k][:, :3, 3] for k in joint_keys], axis=1)  # (N, 5, 3)
        axes = np.stack([frames[k][:, :3, 2] for k in joint_keys], axis=1)     # (N, 5, 3)

        blocks = []
        for lk, pts_h in self._link_pts_homo.items():
            w = np.matmul(frames[lk], pts_h.T)[:, :3, :].transpose(0, 2, 1)
            blocks.append(w)

        radii = self._robot_radii
        max_j = self._robot_max_joint
        if held_piece is not None:
            k_h = held_piece.points_tcp_m.shape[0]
            pts_h = np.column_stack([held_piece.points_tcp_m, np.ones(k_h, dtype=np.float64)])
            w_held = np.matmul(frames["tcp"], pts_h.T)[:, :3, :].transpose(0, 2, 1)
            blocks.append(w_held)
            radii = np.concatenate([radii, np.full(k_h, held_piece.radius_m, dtype=np.float64)])
            max_j = np.concatenate([max_j, np.full(k_h, 4, dtype=np.int32)])

        centers = np.concatenate(blocks, axis=1)  # (N, S, 3)
        s_total = centers.shape[1]

        # 1. Floor clearance & normal [0, 0, 1]
        half_span = 0.5 * self.board.board_span_m + self.board.border_margin_m
        over_board = (
            (np.abs(centers[:, :, 0] - self.board.origin_xyz_m[0]) <= half_span)
            & (np.abs(centers[:, :, 1] - self.board.origin_xyz_m[1]) <= half_span)
        )
        req_floor = np.where(
            over_board,
            self.board.surface_z_m + self.board_transit_floor_margin_m,
            self.table_z_m + self.table_floor_margin_m,
        )
        floor_clear = centers[:, :, 2] - radii[None, :] - req_floor  # (N, S)
        v_floor = np.maximum(0.0, safe_clearance_m - floor_clear)    # (N, S)
        loss = 0.5 * float(np.sum(v_floor * v_floor))

        # World force on sphere centers: d(0.5 * v^2)/dc = -v * dc/dc_xyz
        f_world = np.zeros((n, s_total, 3), dtype=np.float64)
        f_world[:, :, 2] -= v_floor
        min_clear = floor_clear.min(axis=1)

        # 2. Piece cylinder clearances & exact outward surface normals
        obs_xy, obs_z0, obs_z1, obs_r = self.active_obstacles(exclude_labels)
        if obs_xy.shape[0] > 0:
            diff_xy = centers[:, :, None, :2] - obs_xy[None, None, :, :]      # (N, S, M, 2)
            dist_xy = np.linalg.norm(diff_xy, axis=-1)                         # (N, S, M)
            unit_xy = diff_xy / np.maximum(dist_xy[..., None], 1e-9)           # (N, S, M, 2)
            d_xy = np.maximum(dist_xy - obs_r[None, None, :], 0.0)

            cz = centers[:, :, None, 2]
            below = obs_z0[None, None, :] - cz
            above = cz - obs_z1[None, None, :]
            d_z = np.maximum(np.maximum(below, above), 0.0)

            outside_dist = np.hypot(d_xy, d_z)
            # Inside cylinder fallback: push radially outward + upward out of piece top
            rad_pen = obs_r[None, None, :] - dist_xy
            top_pen = obs_z1[None, None, :] - cz
            inside = (d_xy == 0.0) & (d_z == 0.0)
            surf_dist = np.where(inside, -np.minimum(rad_pen, top_pen), outside_dist)
            clear_all = surf_dist - radii[None, :, None]                       # (N, S, M)
            min_clear = np.minimum(min_clear, clear_all.min(axis=(1, 2)))

            v_obs = np.maximum(0.0, safe_clearance_m - clear_all)              # (N, S, M)
            if np.any(v_obs > 0.0):
                loss += 0.5 * float(np.sum(v_obs * v_obs))
                # Outward normal dc/dc_xyz for each (N, S, M)
                inv_out = 1.0 / np.maximum(outside_dist, 1e-9)
                n_xy = (d_xy * inv_out)[..., None] * unit_xy
                n_z = np.where(above > 0.0, d_z * inv_out, np.where(below > 0.0, -d_z * inv_out, 0.0))

                # Inside cylinder normal: push upward if near top, else radially outward
                use_top = inside & (top_pen <= rad_pen)
                use_rad = inside & (~use_top)
                n_xy = np.where(use_top[..., None], 0.0, np.where(use_rad[..., None], unit_xy, n_xy))
                n_z = np.where(use_top, 1.0, np.where(use_rad, 0.0, n_z))

                normals = np.concatenate([n_xy, n_z[..., None]], axis=-1)      # (N, S, M, 3)
                f_world -= np.sum(v_obs[..., None] * normals, axis=2)

        # Chain rule to arm joints (N, 5) in degrees: dc_s / dq_j_rad = z_j x (c_s - o_j) for j <= max_j[s]
        # r_lever: (N, S, 5, 3) = centers[:, :, None, :] - origins[:, None, :, :]
        r_lever = centers[:, :, None, :] - origins[:, None, :, :]
        jac_s = np.cross(axes[:, None, :, :], r_lever)                         # (N, S, 5, 3)
        active_mask = (np.arange(5, dtype=np.int32)[None, :] <= max_j[:, None])  # (S, 5)
        grad_rad = np.einsum("nsd,nsjd,sj->nj", f_world, jac_s, active_mask)   # (N, 5)
        grad_deg = grad_rad * (np.pi / 180.0)
        return loss, grad_deg, min_clear


def build_pieces_from_square_map(
    square_to_piece: Mapping[str, str],
    board: BoardConfig,
    *,
    use_staunton_heights: bool = True,
    xy_jitter_world_m: Mapping[str, tuple[float, float]] | None = None,
    height_overrides_m: Mapping[str, float] | None = None,
) -> list[PieceCylinder]:
    """Construct ``PieceCylinder`` obstacles from a ``{square: piece_symbol}`` dict (e.g. ``{'e4': 'Q', 'd5': 'p'}``)."""
    out: list[PieceCylinder] = []
    for sq, sym in square_to_piece.items():
        ptype = sym.strip().lower()[0]
        if use_staunton_heights and ptype in STAUNTON_PIECE_DIMS_M:
            h_m, r_m = STAUNTON_PIECE_DIMS_M[ptype]
        else:
            h_m, r_m = SIM_ROOK_DIMS_M
        if height_overrides_m and sq in height_overrides_m:
            h_m = float(height_overrides_m[sq])
        if xy_jitter_world_m and sq in xy_jitter_world_m:
            x_m, y_m = float(xy_jitter_world_m[sq][0]), float(xy_jitter_world_m[sq][1])
        else:
            xy = board.square_center_world_xy(sq)
            x_m, y_m = float(xy[0]), float(xy[1])
        out.append(
            PieceCylinder(
                label=sq.lower(),
                x_m=x_m,
                y_m=y_m,
                z_base_m=board.surface_z_m,
                height_m=h_m,
                radius_m=r_m,
                piece_type=ptype,
            )
        )
    return out


def build_pieces_from_coord_finder_detections(
    detections: Sequence[Mapping[str, object]],
    board: BoardConfig,
    *,
    use_staunton_heights: bool = True,
) -> list[PieceCylinder]:
    """Construct ``PieceCylinder`` obstacles from ``coord_finder`` detection dicts.

    Accepts dicts containing ``x_mm, y_mm`` (in ``a1``-origin mm) plus optional
    ``square``, ``piece`` / ``symbol``, ``height_mm`` / ``z_mm``, and ``radius_mm``.
    """
    from .frames import coord_finder_mm_to_world_m

    out: list[PieceCylinder] = []
    for idx, det in enumerate(detections):
        x_mm = float(det["x_mm"])  # type: ignore[arg-type]
        y_mm = float(det["y_mm"])  # type: ignore[arg-type]
        w_xyz = coord_finder_mm_to_world_m((x_mm, y_mm, 0.0), board=board)
        sym = str(det.get("piece", det.get("symbol", "p")))
        ptype = sym.strip().lower()[0] if sym.strip() else "p"
        if use_staunton_heights and ptype in STAUNTON_PIECE_DIMS_M:
            h_m, r_m = STAUNTON_PIECE_DIMS_M[ptype]
        else:
            h_m, r_m = SIM_ROOK_DIMS_M
        if det.get("height_mm") is not None:
            h_m = float(det["height_mm"]) / 1000.0  # type: ignore[arg-type]
        elif det.get("z_mm") is not None:
            # two_view_grasp z_mm is grasp pinch height (~55% of total height)
            h_m = max(h_m, float(det["z_mm"]) / 1000.0 / 0.56)  # type: ignore[arg-type]
        if det.get("radius_mm") is not None:
            r_m = float(det["radius_mm"]) / 1000.0  # type: ignore[arg-type]
        label = str(det.get("square", board.nearest_square(w_xyz[:2]))).lower() or f"det_{idx}"
        out.append(
            PieceCylinder(
                label=label,
                x_m=float(w_xyz[0]),
                y_m=float(w_xyz[1]),
                z_base_m=board.surface_z_m,
                height_m=h_m,
                radius_m=r_m,
                piece_type=ptype,
            )
        )
    return out

