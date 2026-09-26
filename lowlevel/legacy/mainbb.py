import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cma

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from vision import detect_and_annotate


# ========================
# Robot setup
# ========================
config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90659861",
    id="my_awesome_follower_arm"
)

follower = SO101Follower(config)
follower.connect()

obs = follower.get_observation()
print("Initial Observation:", obs)


# ========================
# Motor keys (ALL 6 DOF)
# ========================
MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos"
]


# ========================
# Camera
# ========================
cap = cv2.VideoCapture(0)


# ========================
# Parameters
# ========================
SIGMA0 = 2
POPSIZE = 6

N_AVG = 3
STEP_GAIN = 0.6        # proportional tracking toward target
MAX_STEP = 20           # max per update (smoothness)
MOVE_DELAY = 0.05      # fast loop


# ========================
# Helpers
# ========================
def get_state():
    obs = follower.get_observation()
    return np.array([obs[k] for k in MOTOR_KEYS])


def send_state(x):
    obs = follower.get_observation()
    action = obs.copy()
    for i, k in enumerate(MOTOR_KEYS):
        if i<2:
            action[k] = x[i]
    follower.send_action(action)


def smooth_step(target, current):
    delta = target - current
    delta = np.clip(delta, -MAX_STEP, MAX_STEP)
    return current + STEP_GAIN * delta


def evaluate_target(x):
    """
    Evaluate a 6D motor target with smoothing + averaging
    """

    current = get_state()

    # Smooth movement toward target
    new_state = smooth_step(x, current)
    send_state(new_state)

    distances = []

    for _ in range(N_AVG):
        ret, frame = cap.read()
        if not ret:
            continue

        output, cube, gripper, dist = detect_and_annotate(frame)

        if cube and gripper:
            distances.append(dist)

        cv2.imshow("Detection", output)
        cv2.waitKey(1)

        time.sleep(MOVE_DELAY)

    if len(distances) == 0:
        return 1e6

    avg_dist = np.mean(distances)

    # Motion penalty (reduces jerk)
    movement_penalty = 0.05 * np.linalg.norm(new_state - current)

    score = avg_dist #+ movement_penalty

    print(f"Dist: {avg_dist:.2f}, Penalty: {movement_penalty:.2f}")

    return score


# ========================
# CMA-ES init (6D)
# ========================
x0 = get_state()

es = cma.CMAEvolutionStrategy(
    x0,
    SIGMA0,
    {'popsize': POPSIZE}
)


# ========================
# Main loop (continuous)
# ========================
dist_history = []

try:
    while not es.stop():

        solutions = es.ask()
        values = []

        for x in solutions:
            val = evaluate_target(x)
            values.append(val)
            dist_history.append(val)

        es.tell(solutions, values)
        es.disp()

        best = es.result.xbest
        print("Best target:", best)

        # Optional plotting
        if len(dist_history) % 20 == 0:
            plt.plot(dist_history)
            plt.xlabel("Evaluations")
            plt.ylabel("Distance")
            plt.savefig("distances.png", dpi=300)
            plt.close()

except KeyboardInterrupt:
    print("Stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Final best:", es.result.xbest)

