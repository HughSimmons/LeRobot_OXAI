"""Production online collision-free trajectory planner for the SO-101 chess robot.

The single deployed algorithm is :class:`DLSNullspacePlanner` (``dls_nullspace``).
Alternative candidates and the PyBullet benchmark live in ``lowlevel/planner_benchmarks``.
"""

from .base import (
    CaptureSequenceResult,
    MoveRequest,
    PlannedTrajectory,
    TrajectoryPlanner,
    plan_move_or_capture,
)
from .collision_sdf import (
    HeldPieceAttachment,
    PieceCylinder,
    SceneCollisionChecker,
    build_pieces_from_coord_finder_detections,
    build_pieces_from_square_map,
)
from .compensation import JointCompensator, resolve_standard_urdf, resolve_vhacd_urdf
from .dls_nullspace import DLSNullspacePlanner
from .frames import (
    BoardConfig,
    FILES as BOARD_FILES,
    REAL_MEASURED_BOARD_CONFIG,
    SIM_BOARD_CONFIG,
    coord_finder_mm_to_world_m,
    grasp_dict_to_world_target,
    world_m_to_coord_finder_mm,
)
from .kinematics_so101 import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    CoupledIKSolution,
    DEFAULT_HOME_DEG,
    DEFAULT_PINCH_OFFSET_TCP_M,
    GRIPPER_CLOSED_DEG,
    GRIPPER_OPEN_DEG,
    SO101Kinematics,
)

__all__ = [
    "ALL_JOINT_NAMES",
    "ARM_JOINT_NAMES",
    "BOARD_FILES",
    "BoardConfig",
    "CaptureSequenceResult",
    "CoupledIKSolution",
    "DEFAULT_HOME_DEG",
    "DEFAULT_PINCH_OFFSET_TCP_M",
    "DLSNullspacePlanner",
    "GRIPPER_CLOSED_DEG",
    "GRIPPER_OPEN_DEG",
    "HeldPieceAttachment",
    "JointCompensator",
    "MoveRequest",
    "PieceCylinder",
    "PlannedTrajectory",
    "REAL_MEASURED_BOARD_CONFIG",
    "SIM_BOARD_CONFIG",
    "SO101Kinematics",
    "SceneCollisionChecker",
    "TrajectoryPlanner",
    "build_pieces_from_coord_finder_detections",
    "build_pieces_from_square_map",
    "coord_finder_mm_to_world_m",
    "grasp_dict_to_world_target",
    "plan_move_or_capture",
    "resolve_standard_urdf",
    "resolve_vhacd_urdf",
    "world_m_to_coord_finder_mm",
]
