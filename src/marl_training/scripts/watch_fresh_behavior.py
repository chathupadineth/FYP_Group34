import torch
import random

from jetbot_env import MultiJetBotEnv
from networks import Actor

# Fresh, untrained networks -- NOT loaded from any checkpoint
actor = Actor()

env = MultiJetBotEnv()

print("Resetting with FRESH random weights (no checkpoint loaded)...")
obs = env.reset()

for step in range(200):
    actions = {}
    for name in ['jb_0', 'jb_1']:
        obs_tensor = torch.tensor(obs[name], dtype=torch.float32).view(1, 1, 16)
        hidden = actor.init_hidden(1)
        logits, _ = actor(obs_tensor, hidden)
        dist = torch.distributions.Categorical(logits=logits.squeeze())
        action = dist.sample()
        actions[name] = action.item()

    obs, rewards, dones = env.step(actions)

    if step % 10 == 0:
        print(f"Step {step}: actions={actions}, rewards={rewards}")

    if all(dones.values()):
        print("Episode ended, resetting...")
        obs = env.reset()

env.close()
print("Done -- this was a temporary, untrained-weights observation only.")