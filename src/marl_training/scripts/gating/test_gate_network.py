import math
import torch

from gate_network import GateNet
from features import AgentState, gate_features

gate = GateNet()

ego = AgentState(x=0.0, y=0.0, yaw=0.0, v=0.10)
other = AgentState(x=1.2, y=0.3, yaw=math.pi, v=0.05)

# --- empty slot (no message received this step) ---
empty_feats = torch.tensor(gate_features(ego, None, msg_valid=False), dtype=torch.float32)
print("Empty-slot features [dist, rel_v, rel_heading, valid]:", empty_feats.tolist())

# --- populated slot ---
feats = torch.tensor(gate_features(ego, other, msg_valid=True), dtype=torch.float32)
print("Populated features   [dist, rel_v, rel_heading, valid]:", feats.tolist())

logits = gate(feats)
print("Gate logits [silent, talk]:", logits.tolist())

probs = torch.softmax(logits, dim=-1)
print("Gate probabilities:", probs.tolist())

decision = gate.decide(feats)
print("Hard decision (0=silent, 1=talk):", decision.item())

print("\n--- Testing with a batch dimension (matching how Actor is called) ---")
batch_feats = torch.stack([empty_feats, feats])
batch_logits = gate(batch_feats)
print("Batch logits shape:", batch_logits.shape)   # expect [2, 2]
print("Batch logits:", batch_logits.tolist())
