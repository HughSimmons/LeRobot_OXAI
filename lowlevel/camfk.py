import sys
import time
import math
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from vision import detect_and_annotate
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def move_smooth(target, steps=20, delay=0.1):
    obs = follower.get_observation()
    current = {k: obs[k] for k in target}

    for i in range(steps):
        alpha = (i + 1) / steps
        action = obs.copy()

        for k in target:
            action[k] = current[k] + alpha * (target[k] - current[k])

        follower.send_action(action)
        time.sleep(delay)



config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90659861",
    id="my_awesome_follower_arm"
)
follower = SO101Follower(config)
follower.connect()
# follower.setup_motors()

startobs = follower.get_observation()
print("Observation:", startobs)


k = 3  # start small
cap = cv2.VideoCapture(0)
timestep = 0.3
climit=2

start_time = time.time()
lastprinttime = start_time

olddist = 1000

newdistlist = []
cnt=0
ind = 0

steps = 10

startaction = startobs.copy()
# startaction["shoulder_pan.pos"] = 0
#70 -70
# startaction["shoulder_lift.pos"] = 0

# startaction["wrist_flex.pos"] = 0
# startaction["elbow_flex.pos"] = 20
# 20 -45


# startaction["shoulder_lift.pos"] = 0


# follower.send_action(startaction)
move_smooth(startaction)


cap = cv2.VideoCapture(0)



for ind1 in range(steps):
    for ind2 in range(steps):

        obs = follower.get_observation()
        action = startobs.copy()

        startaction["shoulder_pan.pos"] = -70 + 14*ind1
        #70 -70

        startaction["elbow_flex.pos"] = -45 + 6.5*ind2
        # 20 -45

        move_smooth(startaction)



        ret, frame = cap.read()
        plt.figure()
        plt.imshow(frame)
        plt.savefig(f"img{ind1}_{ind2}.png")
        plt.close()
        output, cube, gripper, newdist = detect_and_annotate(frame)
        time.sleep(2)






