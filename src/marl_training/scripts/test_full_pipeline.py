import torch
import torch.optim as optim
import random

from jetbot_env import MultiJetBotEnv
from networks import Actor, Critic
from buffer import RolloutBuffer
from gae import compute_gae
from ppo_update import ppo_update

actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=5e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=5e-4)

env = MultiJetBotEnv()
buffer = RolloutBuffer()

print("Resetting environment...")
obs = env.reset()

NUM_STEPS = 20
for step in range(NUM_STEPS):
    joint_obs = obs['jb_0'] + obs['jb_1']

    actions = {}
    log_probs = {}
    for name in ['jb_0', 'jb_1']:
        obs_tensor = torch.tensor(obs[name], dtype=torch.float32).view(1, 1, 16)
        hidden = actor.init_hidden(1)
        logits, _ = actor(obs_tensor, hidden)
        dist = torch.distributions.Categorical(logits=logits.squeeze())
        action = dist.sample()
        actions[name] = action.item()
        log_probs[name] = dist.log_prob(action).item()

    joint_obs_tensor = torch.tensor(joint_obs, dtype=torch.float32).view(1, 1, 32)
    critic_hidden = critic.init_hidden(1)
    values, _ = critic(joint_obs_tensor, critic_hidden)
    values = values.squeeze().tolist()  # [value_jb0, value_jb1]

    next_obs, rewards, dones = env.step(actions)

    buffer.add(obs, joint_obs, actions, log_probs, rewards, values, any(dones.values()))
    obs = next_obs

    print(f"Step {step}: actions={actions}, rewards={rewards}, dones={dones}")

    if any(dones.values()):
        print("Episode ended, resetting...")
        obs = env.reset()

print(f"\nCollected {len(buffer)} steps. Computing GAE...")

advantages = {}
returns = {}
for i, name in enumerate(['jb_0', 'jb_1']):
    agent_values = [v[i] for v in buffer.values]
    adv, ret = compute_gae(buffer.rewards[name], agent_values, buffer.dones)
    advantages[name] = adv
    returns[name] = ret

print("jb_0 advantages:", advantages['jb_0'])
print("jb_1 advantages:", advantages['jb_1'])

print("\nRunning PPO update...")
critic_loss, actor_loss = ppo_update(
    actor, critic, actor_optimizer, critic_optimizer,
    buffer, advantages, returns
)
print(f"Critic loss: {critic_loss}, Actor loss: {actor_loss}")

env.close()
print("\nDone.")