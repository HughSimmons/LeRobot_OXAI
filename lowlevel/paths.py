"""Project paths and Hugging Face lerobot source bootstrap."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

# LeRobot_OXAI repo root (parent of lowlevel/)
LOWLEVEL_DIR = Path(__file__).resolve().parent
ROOT = LOWLEVEL_DIR.parent

# Sibling chessbot checkout — used when assets are not vendored in this tree
CHESSBOT_MIRROR = ROOT.parent / "chessbot" / "LeRobot_OXAI"


def _first_existing(*candidates: Path) -> Path:
    """Return the first existing path, or the primary candidate for error messages."""
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


SO101_DIR = _first_existing(
    ROOT / "SO-ARM100" / "Simulation" / "SO101",
    CHESSBOT_MIRROR / "SO-ARM100" / "Simulation" / "SO101",
)
URDF_PATH = SO101_DIR / "so101_new_calib.urdf"
URDF_WRIST_CAM_PATH = SO101_DIR / "so101_new_calib_wrist_cam.urdf"
MESH_DIR = SO101_DIR
SO101_XML = SO101_DIR / "so101_new_calib.xml"
SO101_ASSETS = SO101_DIR / "assets"
SO101_WITH_CHESS_XML = LOWLEVEL_DIR / "so101_with_chess.xml"

# Fallback URDF at repo root (meshes still need SO-ARM100/assets)
URDF_FALLBACK = _first_existing(
    ROOT / "so101_new_calib.urdf",
    CHESSBOT_MIRROR / "so101_new_calib.urdf",
)

LEROBOT_SRC = _first_existing(
    LOWLEVEL_DIR / "lerobot" / "src",
    CHESSBOT_MIRROR / "lowlevel" / "lerobot" / "src",
)

# Board origin in world frame (matches pybsim_chess.py)
BOARD_ORIGIN = (0.25, 0.0, 0.0)

# SO-101 follower defaults (hardware scripts)
FOLLOWER_ID = "my_awesome_follower_arm"
CALIBRATION_DIR = LOWLEVEL_DIR
SO101_PORT_FILE = _first_existing(
    LOWLEVEL_DIR / "so101_port.txt",
    CHESSBOT_MIRROR / "lowlevel" / "so101_port.txt",
)


def setup_lerobot_path() -> Path:
    """Prepend cloned huggingface/lerobot src to sys.path (sim / kinematics fallback)."""
    if not LEROBOT_SRC.is_dir():
        raise FileNotFoundError(
            f"LeRobot source not found at {LEROBOT_SRC}. Clone it:\n"
            "  cd lowlevel && git clone https://github.com/huggingface/lerobot.git\n"
            f"  or use assets under {CHESSBOT_MIRROR}"
        )
    src = str(LEROBOT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return LEROBOT_SRC


def _clear_lerobot_modules() -> None:
    """Drop cached lerobot imports (e.g. wrong package from lowlevel/lerobot repo dir)."""
    for name in list(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            del sys.modules[name]


def setup_lerobot_for_hardware() -> None:
    """
    Load full lerobot (robots, motors, etc.) for hardware scripts.

    Always prepends lowlevel/lerobot/src so ``import lerobot`` resolves to the real
    package, not the git checkout folder ``lowlevel/lerobot/`` on sys.path.
    """
    import importlib

    setup_lerobot_path()
    _clear_lerobot_modules()
    importlib.import_module("lerobot.robots.so_follower")


def import_robot_kinematics():
    """Load RobotKinematics from cloned lerobot without importing torch."""
    import importlib.util
    import types

    setup_lerobot_path()

    if "lerobot.model.kinematics" in sys.modules:
        return sys.modules["lerobot.model.kinematics"].RobotKinematics

    lerobot = types.ModuleType("lerobot")
    lerobot.utils = types.ModuleType("lerobot.utils")
    lerobot.model = types.ModuleType("lerobot.model")
    sys.modules.setdefault("lerobot", lerobot)
    sys.modules.setdefault("lerobot.utils", lerobot.utils)
    sys.modules.setdefault("lerobot.model", lerobot.model)

    import_utils_path = LEROBOT_SRC / "lerobot" / "utils" / "import_utils.py"
    spec = importlib.util.spec_from_file_location(
        "lerobot.utils.import_utils", import_utils_path
    )
    import_utils = importlib.util.module_from_spec(spec)
    sys.modules["lerobot.utils.import_utils"] = import_utils
    spec.loader.exec_module(import_utils)

    kin_path = LEROBOT_SRC / "lerobot" / "model" / "kinematics.py"
    spec = importlib.util.spec_from_file_location("lerobot.model.kinematics", kin_path)
    kinematics_mod = importlib.util.module_from_spec(spec)
    sys.modules["lerobot.model.kinematics"] = kinematics_mod
    spec.loader.exec_module(kinematics_mod)
    return kinematics_mod.RobotKinematics


def resolve_urdf() -> tuple[str, str]:
    """Return (urdf_path, mesh_dir) as strings for pinocchio/placo/pybullet."""
    if URDF_PATH.is_file():
        return str(URDF_PATH), str(MESH_DIR)
    if URDF_FALLBACK.is_file():
        # Meshes referenced as assets/ relative to mesh_dir
        mesh_root = _first_existing(ROOT, CHESSBOT_MIRROR)
        return str(URDF_FALLBACK), str(mesh_root)
    raise FileNotFoundError(
        f"URDF not found. Expected {URDF_PATH} or {URDF_FALLBACK}. "
        f"Clone SO-ARM100 under {ROOT} or use {CHESSBOT_MIRROR}."
    )


def resolve_wrist_cam_urdf() -> tuple[str, str]:
    """Return the SO-101 URDF that includes the Hex-Nut wrist camera link."""
    if URDF_WRIST_CAM_PATH.is_file():
        return str(URDF_WRIST_CAM_PATH), str(MESH_DIR)
    raise FileNotFoundError(
        f"Wrist-camera URDF not found at {URDF_WRIST_CAM_PATH}. "
        "Expected so101_new_calib_wrist_cam.urdf next to so101_new_calib.urdf."
    )


def resolve_mujoco_xml() -> tuple[str, str, str]:
    """Return (source_xml_path, assets_dir, output_xml_path) as strings."""
    if not SO101_XML.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {SO101_XML}")
    if not SO101_ASSETS.is_dir():
        raise FileNotFoundError(f"SO101 assets not found: {SO101_ASSETS}")
    return str(SO101_XML), str(SO101_ASSETS), str(SO101_WITH_CHESS_XML)


def _list_so101_port_candidates() -> list[str]:
    """USB serial ports that might be the SO-101 (prefer macOS cu.* over tty.* duplicates)."""
    candidates = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not candidates:
        candidates = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not candidates and sys.platform.startswith("linux"):
        candidates = sorted(glob.glob("/dev/ttyACM*"))
    return candidates


def _read_saved_so101_port() -> str | None:
    if not SO101_PORT_FILE.is_file():
        return None
    port = SO101_PORT_FILE.read_text().strip().splitlines()[0].strip()
    return port or None


def _save_so101_port(port: str) -> None:
    (LOWLEVEL_DIR / "so101_port.txt").write_text(port + "\n")


def _probe_so101_port(port: str) -> bool:
    """Return True if Feetech servos respond on this port (SO-101 bus)."""
    setup_lerobot_for_hardware()
    from lerobot.motors import Motor, MotorNormMode  # noqa: E402
    from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

    bus = FeetechMotorsBus(
        port=port,
        motors={"shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES)},
    )
    try:
        bus.connect(handshake=False)
        ping = bus.broadcast_ping()
        return bool(ping)
    except Exception:
        return False
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def resolve_so101_port() -> str:
    """
    USB serial port for the SO-101 arm.

    Resolution order: SO101_PORT env → lowlevel/so101_port.txt → single device →
    Feetech ping probe when multiple /dev/cu.usbmodem* are present.

    Save the last auto-detected port: export SO101_SAVE_PORT=1
    """
    env_port = os.environ.get("SO101_PORT", "").strip()
    if env_port:
        return env_port

    saved = _read_saved_so101_port()
    if saved:
        return saved

    candidates = _list_so101_port_candidates()
    if len(candidates) == 1:
        port = candidates[0]
        if os.environ.get("SO101_SAVE_PORT", "").lower() in ("1", "true", "yes"):
            _save_so101_port(port)
        return port

    if len(candidates) > 1:
        matches = [p for p in candidates if _probe_so101_port(p)]
        if len(matches) == 1:
            port = matches[0]
            if os.environ.get("SO101_SAVE_PORT", "1").lower() in ("1", "true", "yes"):
                _save_so101_port(port)
            return port
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple ports responded to Feetech ping. Set SO101_PORT explicitly:\n"
                + "\n".join(f"  export SO101_PORT={p}" for p in matches)
            )
        raise RuntimeError(
            "Multiple USB serial ports found but none responded as an SO-101 arm.\n"
            "Set SO101_PORT (run: lerobot-find-port, or unplug other USB serial devices):\n"
            + "\n".join(f"  export SO101_PORT={p}" for p in candidates)
        )

    raise RuntimeError(
        "No USB serial port found. Plug in the SO-101 and set SO101_PORT "
        "(macOS: ls /dev/cu.usb*)."
    )


def resolve_so101_calibration_path() -> Path:
    """
    JSON calibration for the follower arm.

    Override with SO101_CALIBRATION_FILE (path under lowlevel/ or absolute).
    Otherwise uses SO101_ID / my_awesome_follower_arm → lowlevel/{id}.json
    """
    env_file = os.environ.get("SO101_CALIBRATION_FILE", "").strip()
    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_absolute():
            path = CALIBRATION_DIR / path
        if not path.is_file():
            mirror = CHESSBOT_MIRROR / "lowlevel" / path.name
            if mirror.is_file():
                return mirror
            raise FileNotFoundError(f"SO101_CALIBRATION_FILE not found: {path}")
        return path

    cal_id = os.environ.get("SO101_ID", FOLLOWER_ID)
    local = CALIBRATION_DIR / f"{cal_id}.json"
    if local.is_file():
        return local
    mirror = CHESSBOT_MIRROR / "lowlevel" / f"{cal_id}.json"
    if mirror.is_file():
        return mirror
    return local


def make_so101_follower_config():
    """
    Build SO101FollowerConfig the same way as control2.py.

    Only ``port`` and ``id`` are set — no ``calibration_dir``. Loading the repo
    JSON via calibration_dir was causing the gripper to snap fully open on
    ``send_action`` because motor range mapping did not match the hardware.
    """
    setup_lerobot_for_hardware()
    from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: E402

    return SO101FollowerConfig(
        port=resolve_so101_port(),
        id=os.environ.get("SO101_ID", FOLLOWER_ID),
    )


def connect_so101_follower(follower, *, calibrate: bool = True) -> None:
    """Connect the SO-101 without pushing lowlevel/*.json calibration to the motors."""
    follower.connect(calibrate=calibrate)


def connect_so101_hardware(*, calibrate: bool = True):
    """Create SO101Follower, connect, and return it (control2-style config)."""
    setup_lerobot_for_hardware()
    from lerobot.robots.so_follower import SO101Follower  # noqa: E402

    follower = SO101Follower(make_so101_follower_config())
    connect_so101_follower(follower, calibrate=calibrate)
    return follower
