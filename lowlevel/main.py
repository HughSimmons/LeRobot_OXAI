import sys
import time
import math
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from vision import detect_and_annotate
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


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

while True:

    ret, frame = cap.read()
    if not ret:
        break

    output, cube, gripper, newdist = detect_and_annotate(frame)


    if cube and gripper:


        delta_pan  =  -k * (newdist - olddist) #+ random.uniform(0,1)

        # delta_lift = -k * dy
        delta_lift = 0

        # Clamp (VERY important)
        def clamp(x, limit=climit):
            return max(min(x, limit), -limit)


        obs = follower.get_observation()
        action = obs.copy()


        
        action["shoulder_pan.pos"]+= clamp(delta_pan)
        action["shoulder_lift.pos"]+= clamp(delta_lift)


        if time.time() - lastprinttime>timestep:
            cnt=0
            print("Action", action)
            follower.send_action(action)

            # print(f"dx: {dx:.3f}, dy: {dy:.3f}, dist: {dist:.1f}")
            print("Delta Pan: ", delta_pan, "Delta Lift:", delta_lift)
            lastprinttime = time.time()


        if time.time() - lastprinttime>timestep*0.5:
            if cnt==0:
                ind+=1
                olddist = newdist.copy()

                if ind//20==0:
                    plt.plot(np.arange(len(newdistlist)), newdistlist)
                    plt.xlabel("Time")
                    plt.ylabel("Distance")
                    plt.savefig(f"distances{ind}.png", dpi=500)
                    plt.close()

                cnt+=1

    newdistlist.append(newdist)

    cv2.imshow("Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


