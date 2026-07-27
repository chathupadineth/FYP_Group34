import torch
from networks import Actor

actor = Actor()
batch_size = 1
seq_len = 1

fake_obs = torch.randn(batch_size, seq_len, 16)
hidden = actor.init_hidden(batch_size)

logits, new_hidden = actor(fake_obs, hidden)
print("Logits shape:", logits.shape)
print("Logits:", logits)

probs = torch.softmax(logits, dim=-1)
print("Probabilities:", probs)