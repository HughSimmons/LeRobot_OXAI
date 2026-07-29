import argparse
import os
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser(description="Run a focused non-h lookup search.")
parser.add_argument("--source", default="b1")
parser.add_argument("--target-moves", default=None)
parser.add_argument("--lift-height", type=float, default=0.07)
home_group = parser.add_mutually_exclusive_group()
home_group.add_argument(
    "--home-preset",
    choices=("mirrored_compromise",),
    default=None,
    help="Use a named full-joint home preset.",
)
home_group.add_argument(
    "--home-shoulder-pan-delta-deg",
    type=float,
    default=None,
    help="Add this rotation to the default home shoulder-pan angle.",
)
home_group.add_argument(
    "--home-shoulder-pan-deg",
    type=float,
    default=None,
    help="Set an absolute home shoulder-pan angle.",
)
args = parser.parse_args()

python_exe = "/opt/miniconda3/envs/IKsim_mj/bin/python"
script = Path(__file__).resolve().parent / "build_general_nonh_reverse_lookup.py"
source = args.source
target_moves = args.target_moves or f"{source}_to_f5"

env = {
    **os.environ,
    "SOURCE_SQUARES": source,
    "TARGET_MOVES": target_moves,
    "LOOKUP_LIFT_HEIGHT_OVERRIDE": str(args.lift_height),
}
if args.home_shoulder_pan_deg is not None:
    env["LOOKUP_HOME_SHOULDER_PAN_OVERRIDE_DEG"] = str(
        args.home_shoulder_pan_deg
    )
if args.home_preset is not None:
    env["LOOKUP_HOME_PRESET"] = args.home_preset
if args.home_shoulder_pan_delta_deg is not None:
    env["LOOKUP_HOME_SHOULDER_PAN_DELTA_DEG"] = str(
        args.home_shoulder_pan_delta_deg
    )

print(
    f"Running {source} for {target_moves} "
    f"with lift={args.lift_height} "
    f"and home preset={args.home_preset}, "
    f"shoulder pan absolute={args.home_shoulder_pan_deg}, "
    f"delta={args.home_shoulder_pan_delta_deg}..."
)
subprocess.run(
    [python_exe, "-B", str(script)],
    env=env,
    check=True,
)
print("Done.")
