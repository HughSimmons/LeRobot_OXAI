# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# dataset = LeRobotDataset("lerobot/svla_so101_pickplace")

# print(f"Total episodes: {len(dataset.meta.episodes)}")
# print(f"Features: {dataset.features}")

# frame = dict(dataset[0])
# print({k: v.shape if hasattr(v, 'shape') else v for k, v in frame.items()})
#0 is webcam 
#1 is laptop

import cv2

for i in range(2):  # try a few indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
    cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()



import sys
sys.exit()
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = LeRobotDataset("lerobot/svla_so101_pickplace")
print("Dataset keys:", list(dataset.features.keys()))

# Load policy using its own preprocessor
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    "lerobot/smolvla_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Get a frame and inspect what keys it has
frame = dict(dataset[0])
print("Frame keys:", list(frame.keys()))
print("Image keys in frame:", [k for k in frame.keys() if 'image' in k])

# Remap dataset camera keys to what the model expects
# Dataset has: observation.images.up, observation.images.side
# Model expects: observation.images.camera1, observation.images.camera2
frame["observation.images.camera1"] = frame.pop("observation.images.up")
frame["observation.images.camera2"] = frame.pop("observation.images.side")

# The model expects 3 cameras but we only have 2
# Create a dummy third camera by copying one of the existing ones
frame["observation.images.camera3"] = frame["observation.images.camera2"].clone()

print("Remapped frame keys:", [k for k in frame.keys() if 'image' in k])

batch = preprocess(frame)

with torch.inference_mode():
    pred_action = policy.select_action(batch)
    pred_action = postprocess(pred_action)

print("Action shape:", pred_action.shape)
print("Action values:", pred_action)