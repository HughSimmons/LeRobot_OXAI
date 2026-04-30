import sys
import time
import math
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90659861",
    id="my_awesome_follower_arm"
)
follower = SO101Follower(config)
follower.connect()
# follower.setup_motors()

obs = follower.get_observation()
print("Observation:", obs)




# action = obs.copy()
# action["shoulder_pan.pos"] += 5
# print("Action", action)

# follower.send_action(action)



# --- parameters ---
joint = "wrist_flex.pos"
amplitude = 10.0      # degrees (keep small for safety)
frequency = 0.2       # Hz (cycles per second)
total_time = 10


# --- get initial position ---
obs = follower.get_observation()
offset = obs[joint]

start_time = time.time()


while True:
    t = time.time() - start_time

    obs = follower.get_observation()
    action = obs.copy()

    # sinusoidal motion
    action[joint] = offset + amplitude * math.sin(2 * math.pi * frequency * t)

    follower.send_action(action)

    time.sleep(0.02)   # ~50 Hz control loop

    if t>total_time:
        sys.exit()
