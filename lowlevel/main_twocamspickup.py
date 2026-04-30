import sys
import time
import math
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from vision_twocams import detect_and_annotate, detect_and_annotate_cam2
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
startingobs = {'shoulder_pan.pos': -3.120879120879121, 'shoulder_lift.pos': -91.34065934065934, 'elbow_flex.pos': 97.27472527472527, 'wrist_flex.pos': 11.296703296703297, 'wrist_roll.pos': 96.65934065934066, 'gripper.pos': 2.226588081204977}

# sys.exit()

def move_smooth(target, steps=10, delay=0.05):
    obs = follower.get_observation()
    current = {k: obs[k] for k in target}

    for i in range(steps):
        alpha = (i + 1) / steps
        action = obs.copy()

        for k in target:
            action[k] = current[k] + alpha * (target[k] - current[k])

        follower.send_action(action)
        time.sleep(delay)


def safevals(action):
    if action["shoulder_pan.pos"] < -70: 
        action["shoulder_pan.pos"] = -70
    if action["shoulder_pan.pos"] > 70: 
        action["shoulder_pan.pos"] = 70

    # if action["elbow_flex.pos"]>90:
    #     action["elbow_flex.pos"] = 90
    if action["elbow_flex.pos"]<-45:
        action["elbow_flex.pos"] = -45


    return(action)

import random

def sample_candidates(step, num_samples=8):
    candidates = [(0,0,0)]  # always include

    for _ in range(num_samples - 1):
        # scale = random.choice([0.5, 1.0])  # small + normal steps

        dp = random.choice([-step, 0, step]) #* scale
        dl = random.choice([-step, 0, step]) #* scale
        de = random.choice([-step, 0, step]) #* scale

        candidates.append((dp, dl, de))

    return candidates

# action = obs.copy()
# action["shoulder_pan.pos"] += 5
# print("Action", action)

# follower.send_action(action)
# move_smooth(startingobs)

##new starting obs for pick up

startingobs["wrist_flex.pos"] += 80
startingobs["elbow_flex.pos"] +=-60
startingobs["shoulder_lift.pos"]+=60

move_smooth(startingobs)


# sys.exit()

k = 3  # start small
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
# cam2 = cv2.VideoCapture(1)
timestep = 0.3
climit=2

start_time = time.time()
lastprinttime = start_time

olddist = 1000

newdistlist = []
dxlist, dylist = [], []
cnt=0
ind = 0

import numpy as np
import time

def measure_distance():
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        return None

    _, cube1, grip1, _ = detect_and_annotate(frame1)
    _, cube2, grip2, _ = detect_and_annotate_cam2(frame2)

    if cube1 is None or grip1 is None or cube2 is None or grip2 is None:
        return None

    dx1 = cube1[0] - grip1[0]
    dy1 = cube1[1] - grip1[1]

    dx2 = cube2[0] - grip2[0]
    dy2 = cube2[1] - grip2[1]

    return dx1**2 + dy1**2 + dx2**2 + dy2**2


step_size = 10.0   # joint step size (tune this)
# step_size = 7.5   # joint step size (tune this)
wstep = 5
# sleep_time = 0.15
sleep_time = 0.05
targetdist = 35000

while True:

    obs = follower.get_observation()
    base_dist = measure_distance()

    if base_dist is None:
        continue

    # candidate moves (pan, lift)
    candidates = [(step_size,0), (-step_size,0), (0,step_size), (0,-step_size)]
    # candidates = sample_candidates(step_size, 8)
    wcandidates = [-wstep, 0, wstep]
    best_dist = base_dist
    best_move = (0, 0, 0)

    if best_dist/10 < targetdist: 
        for dw in wcandidates:
            for dp, dl in candidates:
                test_action = obs.copy()
                test_action["shoulder_pan.pos"]  += dp
                # test_action["shoulder_lift.pos"] += dl
                test_action["elbow_flex.pos"] += dl
                test_action["wrist_flex.pos"] += dw
                test_action = safevals(test_action)
                move_smooth(test_action)
                time.sleep(sleep_time)

                new_dist = measure_distance()

                if new_dist is None:
                    continue

                if new_dist < best_dist:
                    best_dist = new_dist
                    best_move = (dp, dl, dw)

            # apply best move
            action = obs.copy()
            action["shoulder_pan.pos"]  += best_move[0]
            # action["elbow_flex.pos"] += best_move[1]
            action["shoulder_lift.pos"] += best_move[1]
            action["wrist_flex.pos"] += best_move[2]

            move_smooth(action)

            print(f"Best move: {best_move}, dist: {best_dist}")


    else:
        for dp, dl in candidates:
            test_action = obs.copy()
            test_action["shoulder_pan.pos"]  += dp
            # test_action["shoulder_lift.pos"] += dl
            test_action["elbow_flex.pos"] += dl
            test_action = safevals(test_action)
            move_smooth(test_action)
            time.sleep(sleep_time)

            new_dist = measure_distance()

            if new_dist is None:
                continue

            if new_dist < best_dist:
                best_dist = new_dist
                best_move = (dp, dl)

        # apply best move
        action = obs.copy()
        action["shoulder_pan.pos"]  += best_move[0]
        # action["elbow_flex.pos"] += best_move[1]
        action["shoulder_lift.pos"] += best_move[1]

        move_smooth(action)

        print(f"Best move: {best_move}, dist: {best_dist}")


    if best_dist<targetdist:
        print("Breaking!")
        break

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        continue

    output1, cube1, gripper1, dist1 = detect_and_annotate(frame1)
    output2, cube2, gripper2, dist2 = detect_and_annotate(frame2)

    combined = np.hstack((output1, output2))
    cv2.imshow("Debug View", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

orig=False

if orig:
    while True:

        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1:
            break

        output1, cube1, gripper1, newdist1 = detect_and_annotate(frame1)
        output2, cube2, gripper2, newdist2 = detect_and_annotate_cam2(frame2)

        if newdist1 is None or newdist2 is None:
            continue

        newdist = newdist1+newdist2
        if newdist<40000:
            print("Breaking!")
            break
        if cube1 and gripper1:

            ###
            dx1 = cube1[0] - gripper1[0]
            dy1 = cube1[1] - gripper1[1]

            dx2 = cube2[0] - gripper2[0]
            dy2 = cube2[1] - gripper2[1]

            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

            k = 0.01

            delta_pan  = -k * dx
            delta_lift = -k * dy

            obs = follower.get_observation()
            action = obs.copy()
            # Clamp (VERY important)
            def clamp(x, limit=climit):
                return max(min(x, limit), -limit)
                    
            obs = follower.get_observation()
            action = obs.copy()

            # action["shoulder_pan.pos"]  += clamp(delta_pan)
            # action["elbow_flex.pos"] += clamp(delta_lift)
            action["shoulder_pan.pos"]  += delta_pan
            action["elbow_flex.pos"] += delta_lift
            ###

            # delta_pan  =  -k * (newdist - olddist) #+ random.uniform(0,1)
            # delta_flex  =  -k * (newdist - olddist) #+ random.uniform(0,1)

            # delta_lift = -k * dy
            # delta_lift = 0






            
            # action["shoulder_pan.pos"]+= clamp(delta_pan)
            # # action["shoulder_lift.pos"]+= clamp(delta_lift)

            # # action["shoulder_pan.pos"] = -70 + 14*ind1
            # #70 -70
            # action["elbow_flex.pos"] += clamp(delta_flex)



            if time.time() - lastprinttime>timestep:
                cnt=0
                print("Action", action)
                # follower.send_action(action)
                move_smooth(action)

                # print(f"dx: {dx:.3f}, dy: {dy:.3f}, dist: {dist:.1f}")
                print("Delta Pan: ", delta_pan, "Delta Lift:", delta_lift)
                lastprinttime = time.time()


            if time.time() - lastprinttime>timestep*0.5:
                if cnt==0:
                    ind+=1
                    olddist = newdist.copy()

                    if ind%20==0:
                        # plt.plot(np.arange(len(newdistlist)), newdistlist)
                        plt.plot(np.arange(len(dxlist)), dxlist, label="dx")
                        plt.plot(np.arange(len(dylist)), dylist, label="dy")
                        plt.xlabel("Time")
                        plt.ylabel("Distance")
                        plt.legend()
                        plt.savefig(f"distances{ind}.png", dpi=500)
                        plt.close()

                    cnt+=1

        newdistlist.append(newdist)
        dxlist.append(dx)
        dylist.append(dy)



        # cv2.imshow("Detection", output1)
        # Combine views side-by-side
        combined = np.hstack((output1, output2))

        cv2.imshow("Dual Camera Detection", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


