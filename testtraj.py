from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
import torch 

# Load model
# policy = PI0FastPolicy.from_pretrained("gpudad/pi0fast-so101-pick-cube")

policy = PI0FastPolicy.from_pretrained(
    "gpudad/pi0fast-so101-pick-cube",
    use_fast=False
)

# policy.to("cuda")
policy.eval()



import sys
sys.exit()

# Load processors
preprocessor = PolicyProcessorPipeline.from_pretrained(
    "gpudad/pi0fast-so101-pick-cube", 
    "policy_preprocessor.json"
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    "gpudad/pi0fast-so101-pick-cube", 
    "policy_postprocessor.json"
)

# Run inference
observation = {
    "observation.state": state_tensor,
    "observation.images.front": front_image,
    "observation.images.wrist": wrist_image,
    "observation.images.overhead": overhead_image,
    "task": "pick up the object and place it in the target location",
}

batch = preprocessor(observation)
batch['observation.language.attention_mask'] = batch['observation.language.attention_mask'].bool()

policy.reset()
with torch.no_grad():
    action = policy.select_action(batch)

result = postprocessor({"action": action})
final_action = result["action"]
