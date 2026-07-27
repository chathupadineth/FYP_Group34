import torch
from networks import Actor, Critic

actor = Actor()
batch_size = 1
seq_len = 1

fake_obs = torch.randn(batch_size, seq_len, 16)
hidden = actor.init_hidden(batch_size)

logits, new_hidden = actor(fake_obs, hidden)
print("Logits shape:", logits.shape)
probs = torch.softmax(logits, dim=-1)
print("Probabilities:", probs)

print("\n--- Testing Critic (now outputs 2 values) ---")
critic = Critic()
fake_joint_obs = torch.randn(batch_size, seq_len, 32)
critic_hidden = critic.init_hidden(batch_size)

values, new_critic_hidden = critic(fake_joint_obs, critic_hidden)
print("Values shape:", values.shape)   # expect [1, 1, 2] now
print("Values:", values)