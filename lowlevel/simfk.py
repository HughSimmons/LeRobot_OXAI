import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import numpy as np
import meshcat.geometry as g
from testkinematics import relativexyz, vectodic


urdf_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

mesh_dir = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_path,
    mesh_dir
)

viz = MeshcatVisualizer(model, collision_model, visual_model)

viz.initViewer(open=True)

viz.loadViewerModel()

q = pin.neutral(model)

print("nq:", model.nq)
print("q:", q)

# viz.display(q)

corner1 = [97.32,0.40,28.40,66.55,177.80,4.95]
corner2 = [38.59,60.88,-58.55,100.48,178.15,4.95]
corner3 = [-52.48,57.98,-56.26,96.26,172.70,4.95]
corner4 = [-108.75, 26.33, 0.79, 81.85, 172.88, 1.4]

home = np.array([96.92307692307692,  -107.86813186813187,  97.36263736263736, 65.18681318681318,  -29.846153846153847,  4.62962962962963])



# cartcorner1 = relativexyz(vectodic(np.array(corner1)), np.array([0,0.01,0]))
def simxyz(currentjnt, direcxyz):
    cartcorner1 = relativexyz(np.array(currentjnt), np.array(direcxyz))
    return(cartcorner1)


mv1 = simxyz(corner1,[0.05,0,0])
mv2 = simxyz(mv1,[0.05,0,0])
mv3 = simxyz(mv2,[0.05,0,0])
mv4 = simxyz(mv3,[0.05,0,0])

corner1 = np.array(corner1)
# corner1, corner2, corner3, corner4 = np.deg2rad(corner1), np.deg2rad(corner2), np.deg2rad(corner3), np.deg2rad(corner4)
# viz.display(np.deg2rad(corner1))

viz.display(corner1)
time.sleep(2)
def smoothmove(pos1, pos2):
    for alpha in np.linspace(0,1,50):

        q = (1-alpha)*np.array(pos1) + alpha*np.array(pos2)

        viz.display(np.deg2rad(q))
        time.sleep(0.01)


# smoothmove(corner1, corner2)

# time.sleep(3)

start = simxyz(corner1, [0,0,0.025])
start = simxyz(start, [0.05,0,0])
start = simxyz(start, [0.05,0,0])



FILES = "abcdefgh"

SQUARE_SIZE = 0.04   # 4 cm example


def chess_to_xy(square):
    """
    Convert chess square like 'e4'
    into board coordinates.
    """

    file = FILES.index(square[0].lower())
    rank = int(square[1]) - 1

    return np.array([
        file * SQUARE_SIZE +0.1,
        rank * SQUARE_SIZE +0.1,
        0
    ])



board_origin_joints = corner1.copy()
# board_origin_xyz = chess_to_xy("a1") + np.array([0.2,0.2,0])


def move_to_squareold(current_joints, from_square, to_square):

    from_xyz = chess_to_xy(from_square)
    to_xyz = chess_to_xy(to_square)

    delta = to_xyz - from_xyz

    current = current_joints.copy()

    steps = 10

    for i in range(steps):

        step_delta = delta / steps

        new = simxyz(current, step_delta)

        smoothmove(current, new)

        current = new

    return current

def move_to_square_v1(current_joints, from_square, to_square):

    from_xyz = chess_to_xy(from_square)
    to_xyz = chess_to_xy(to_square)

    delta = to_xyz - from_xyz

    current = current_joints.copy()

    # ---------------------------
    # 0. Close gripper
    # ---------------------------

    grip_closed = current.copy()
    grip_closed[5] = 0.0   # adjust as needed

    smoothmove(current, grip_closed)

    current = grip_closed.copy()

    # ---------------------------
    # 1. Move vertically 
    # ---------------------------

    lift = np.array([0, 0, 0.10])

    new = simxyz(current, lift)

    smoothmove(current, new)

    current = new.copy()

    # ---------------------------
    # 2. Move across board
    # ---------------------------

    steps = 10

    for _ in range(steps):

        step_delta = delta / steps

        new = simxyz(current, step_delta)

        smoothmove(current, new)

        current = new.copy()

    # ---------------------------
    # 3. Move vertically down
    # ---------------------------

    lower = np.array([0, 0, -0.10])

    new = simxyz(current, lower)

    smoothmove(current, new)

    current = new.copy()

    # ---------------------------
    # 4. Open gripper
    # ---------------------------

    grip_open = current.copy()
    grip_open[5] = 5.0   # adjust as needed

    smoothmove(current, grip_open)

    current = grip_open.copy()


    lift = np.array([0, 0, 0.10])

    new = simxyz(current, lift)

    smoothmove(current, new)

    return current



def move_to_square_v2(current_joints, from_square, to_square):
    """
    Move from current_joints to home, then from home to from_square, pick up piece,
    move to to_square, place piece, and return to home. Returns the final joint position (home).
    """
    global home
    # 1. Move to home
    smoothmove(current_joints, home)
    current = home.copy()

    # 2. Move to from_square (above)
    from_xyz = chess_to_xy(from_square)
    above_from = simxyz(current, from_xyz + np.array([0, 0, 0.10]))
    smoothmove(current, above_from)
    current = above_from.copy()
    # Lower to from_square
    at_from = simxyz(current, np.array([0, 0, -0.10]))
    smoothmove(current, at_from)
    current = at_from.copy()
    # Close gripper
    grip_closed = current.copy()
    grip_closed[5] = 0.0  # adjust as needed
    smoothmove(current, grip_closed)
    current = grip_closed.copy()
    # Lift piece
    lifted = simxyz(current, np.array([0, 0, 0.10]))
    smoothmove(current, lifted)
    current = lifted.copy()

    # 3. Move to to_square (above)
    to_xyz = chess_to_xy(to_square)
    move_delta = to_xyz - from_xyz
    above_to = simxyz(current, move_delta)
    smoothmove(current, above_to)
    current = above_to.copy()
    # Lower to to_square
    at_to = simxyz(current, np.array([0, 0, -0.10]))
    smoothmove(current, at_to)
    current = at_to.copy()
    # Open gripper
    grip_open = current.copy()
    grip_open[5] = 5.0  # adjust as needed
    smoothmove(current, grip_open)
    current = grip_open.copy()
    # Lift up
    lifted = simxyz(current, np.array([0, 0, 0.10]))
    smoothmove(current, lifted)
    current = lifted.copy()

    # 4. Return to home
    smoothmove(current, home)
    current = home.copy()

    return current



# move_to_square_v2(corner2, "e1", "c5")

# smoothmove(corner2, corner2+np.array([0,0,0,0,0,30]))

# time.sleep(5)

# move_to_square_v2(home, "e1", "c5")



from chess_traj import pickupmove_traj


movelist, idx = pickupmove_traj("c5", "c5", board_origin=(0.25, 0, 0), GRASP_OFFSET=np.array([0,0,0]))


rlref =  np.array([-6.021978021978022, 13.714285714285714, -93.67032967032966, 7.956043956043956, -30.10989010989011, 4.761904761904762])
rlref2 = np.array([1.8021978021978022, -0.21978021978021978, -80.48351648351648, -3.340659340659341, -0.04395604395604396, 4.587765957446808])
rlref2correct = rlref2 + np.array([0,0,-10,0,0,0])
start = home.copy()


reach_pose = np.array([
    0.0,     # shoulder_pan
    70,   # shoulder_lift
    -90,     # elbow_flex
    40,    # wrist_flex
    90.0,     # wrist_roll
    5.0      # gripper
])

# smoothmove(start, rlref2correct)







smoothmove(start, reach_pose)

###
from testkinematics import kinematics
joints = reach_pose.copy()

# GRASP_OFFSET = np.array([
#     -0.025,
#     -0.0,
#     0.005
# ])

GRASP_OFFSET = np.array([
    -0.0,
    -0.0,
    0.00
])

fk = kinematics.forward_kinematics(joints)

R = fk[:3,:3]
t = fk[:3,3]

grasp_xyz = t + R @ GRASP_OFFSET

T = np.eye(4)
T[:3,3] = grasp_xyz

# viz.viewer["grasp_point"].set_transform(T)



# Create a large sphere
viz.viewer["test_sphere"].set_object(
    g.Sphere(0.01)   # 10 cm radius
)

# # Place it above the robot
# T = np.eye(4)
# T[:3, 3] = [0.2, 0.0, 0.4]

viz.viewer["test_sphere"].set_transform(T)




# viz.viewer["test_sphere"].set_object(
#     g.Sphere(0.1),
#     g.MeshLambertMaterial(color=0xff0000)
# )

# T = np.eye(4)
# T[:3, 3] = [0.2, 0.0, 0.4]

# viz.viewer["test_sphere"].set_transform(T)




# viz.initViewer(open=True)
# viz.loadViewerModel()

###


for move in movelist:
    
    smoothmove(start, move)
    print(move)
    start = move.copy()
    time.sleep(0.1)

import sys
sys.exit()


savestart = start.copy()
tdelay = 0.01

for _ in range(3):
    new = simxyz(start,[0.05,0,0])
    smoothmove(start, new)
    start = new
    time.sleep(tdelay)


# start = corner2

for _ in range(7):
    new = simxyz(start,[0,0.05,0])
    smoothmove(start, new)
    start = new
    time.sleep(tdelay)


for _ in range(3):
    new = simxyz(start,[-0.05,0,0])
    smoothmove(start, new)
    start = new
    time.sleep(tdelay)

for _ in range(7):
    new = simxyz(start,[0,-0.05,0])
    smoothmove(start, new)
    start = new
    time.sleep(tdelay)

smoothmove(start, savestart)

# smoothmove(mv1, mv2)
# time.sleep(2)
# smoothmove(mv2, mv3)
# time.sleep(2)
# smoothmove(mv3, mv4)
if False:
    for alpha in np.linspace(0,1,50):

        q = (1-alpha)*np.array(corner2) + alpha*np.array(corner3)

        viz.display(q)
        time.sleep(0.1)

    for alpha in np.linspace(0,1,50):

        q = (1-alpha)*np.array(corner3) + alpha*np.array(corner4)

        viz.display(q)
        time.sleep(0.1)

# while True:
#     time.sleep(1)