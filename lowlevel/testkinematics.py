import inspect
from lerobot.model.kinematics import RobotKinematics
import numpy as np
import sys

# Path to your downloaded URDF
# URDF_PATH = "./SO101/so101_new_calib.urdf"
# URDF_PATH = "/Users/zhg603/Documents/OXAI/so101_new_calib.urdf"
URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# Joint names in the SAME order as the robot motors
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Create kinematics solver
kinematics = RobotKinematics(
    urdf_path=URDF_PATH,
    # target_frame_name="gripper_frame_link",
    target_frame_name="gripper_link",
    # target_frame_name="gripper",
    joint_names=JOINT_NAMES,
)





# print(kinematics.robot.frame_names())
# print(list(kinematics.robot.frame_names()))


# import sys
# sys.exit()



# print("\nURDF joint order:")
# print(kinematics.joint_names)

# print("\nYour joint order:")
# print(JOINT_NAMES)

# import sys
# sys.exit()

# print(inspect.signature(kinematics.inverse_kinematics))

# import sys
# sys.exit()


# Example current joint state
current_joints = np.array([
    0.0,   # shoulder_pan
    0.0,   # shoulder_lift
    0.0,   # elbow_flex
    0.0,   # wrist_flex
    0.0,   # wrist_roll
    0.0,   # gripper
])


####
current_fk = kinematics.forward_kinematics(current_joints)

####

# Target end effector transform
# target_pose = np.eye(4)

# target_pose[0, 3] = 0.15
# target_pose[1, 3] = 0.40
# target_pose[2, 3] = 0.10

target_pose = current_fk.copy()

target_pose[0,3] += 0.04
target_pose[2,3] += 0.04

print("Target EE pose:")
print(target_pose)


print("Target EE pose:")
print(target_pose)

# Solve IK
# joint_solution = kinematics.inverse_kinematics(
#     target_pose,
#     initial_joint_positions=current_joints,
# )

# print("\nJoint solution:")
# for name, angle in zip(JOINT_NAMES, joint_solution):
#     print(f"{name}: {angle:.3f}")



# Solve IK
# joint_solution = kinematics.inverse_kinematics(
#     current_joints,
#     target_pose,
# )


joint_solution = current_joints

for _ in range(4):
    joint_solution = kinematics.inverse_kinematics(
        joint_solution,
        target_pose,
        position_weight=100.0,
        orientation_weight=0.0,
    )



print("\nJoint solution:")
for name, angle in zip(JOINT_NAMES, joint_solution):
    print(f"{name}: {angle:.3f}")



# ---------------------------
# Forward Kinematics Check
# ---------------------------

fk_pose = kinematics.forward_kinematics(joint_solution)

print("\nFK reconstructed pose:")
print(fk_pose)

# ---------------------------
# Position error
# ---------------------------

target_position = target_pose[:3, 3]
fk_position = fk_pose[:3, 3]

position_error = np.linalg.norm(
    target_position - fk_position
)

print(f"\nPosition error: {position_error:.6f} m")


