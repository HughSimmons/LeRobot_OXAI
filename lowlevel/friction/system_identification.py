"""Fit a static friction-and-sag model to the real-arm friction sweeps by least squares.

Where does a servo joint stop when commanded to an angle? It is pulled toward the
target like a spring (stiffness kp), held back by gravity and by gearbox friction,
and it sticks at the edge of the friction band. This script fits that model,
the load-dependent friction model M3 of Duclusaud et al. (ICRA 2025) with the
joint at rest, to the readings recorded by toggle_joint_calibration.py in
real_world_runs/friction/. Nothing talks to the arm.

Our model's input (x) is the target pose (set of joint angles) and our output (y) is the
error between the target pose and the actual pose.

Model, per reading:
  * "up" is the direction that opposes gravity. tau_hold, from inverse dynamics at
    the recorded pose, is the torque the motor needs to hold it; s = sign(tau_hold)
    is +1 if up is +theta and -1 if up is -theta, and the load is tau_g = |tau_hold|.
  * A move is "lifting" if it went up and "lowering" otherwise.
  * Friction at rest is K_c + K_l * |tau_m - tau_e| (M3), solved together with the
    torque balance. With e_up = s * (target - reading), positive when sagged:

        lifting:                    e_up = ( K_c + (1 + K_l) tau_g) / ((1 - K_l) kp)
        lowering, tau_g >= K_c/2:   e_up = (-K_c + (1 - K_l) tau_g) / ((1 + K_l) kp)
        lowering, tau_g <  K_c/2:   e_up = (-K_c + (1 + K_l) tau_g) / ((1 - K_l) kp)

    The last case is a lowering move where friction ends up larger than twice the
    gravity torque, so the motor pulls with gravity. The case is chosen from the
    parameters being tried, so the optimiser moves readings between cases as it goes;
    both lowering formulas give -K_c / (2 kp) at the switch, so the curve is continuous.

What is fitted:
  * kp, K_c and K_l for each joint that carries gravity (shoulder_lift, elbow_flex,
    wrist_flex). Without gravity load the three only appear as K_c / ((1 - K_l) kp),
    so for shoulder_pan and wrist_roll only the band half-width in degrees is reported.
  * The calibration offsets of those three joints (URDF angle = motor angle + offset),
    since they change tau_hold. They're shared by all sweeps, and each is pulled
    toward the value in multisim_chess_fast.py by a prior with that file's stated
    uncertainty (0.7 deg for elbow and wrist flex; the lift was never measured, so it
    gets a loose 5 deg). A fitted offset also absorbs any error in the URDF's mass
    model, so compare it with the board calibration rather than replacing it.
    --fix-offsets keeps them at the sim's values.

Errors use target - reading, positive when the joint stopped below its target in
theta (fit_servo_model_to_sweeps.py uses the opposite sign).

Limitation: it assumes every reading sits exactly at the edge of the friction band.
A soft joint may still be creeping when read (see fit_servo_model_to_sweeps.py),
which leaves it slightly short of the edge.
"""
import json
GRAVITY_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
NO_GRAVITY_JOINTS = ["shoulder_pan", "wrist_roll"]
ALL_JOINTS = ["shoulder_pan","shoulder_lift", "elbow_flex", "wrist_flex","wrist_roll"]

def getDataPoints(dir):
    dataset = {}
    for joint in ALL_JOINTS:
        dataset[joint] = [] 
        #for each joint there will be an array of datapoints

    for joint in ALL_JOINTS:
        with open(dir / f"toggle_{joint}_calibration.json") as f:
            calib = json.load(f)
            for result in calib["results"]:
                dataset[joint].append([result["target"],result["diff"]])
    
    return dataset



