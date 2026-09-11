import torch
import torch.nn as nn

# 16 own values + 4 from the communication slot. The slot holds what the
# OTHER robot told us this step, already relative and normalised
# (gating/features.py gate_features); all four are zero when it stayed silent.
OWN_OBS_DIM = 16
MSG_DIM = 4
OBS_DIM = OWN_OBS_DIM + MSG_DIM        # 20
ACTION_DIM = 4
HIDDEN_DIM = 64


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN_DIM)
        self.rnn = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.fc_out = nn.Linear(HIDDEN_DIM, ACTION_DIM)

    def forward(self, obs, hidden_state):
        x = torch.relu(self.fc1(obs))
        x, new_hidden = self.rnn(x, hidden_state)
        action_logits = self.fc_out(x)
        return action_logits, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, HIDDEN_DIM)

CENTRAL_OBS_DIM = 2 * OBS_DIM   # 40 -- both agents' observations concatenated


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CENTRAL_OBS_DIM, HIDDEN_DIM)
        self.rnn = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.fc_out = nn.Linear(HIDDEN_DIM, 2)   # <-- changed from 1 to 2

    def forward(self, joint_obs, hidden_state):
        x = torch.relu(self.fc1(joint_obs))
        x, new_hidden = self.rnn(x, hidden_state)
        values = self.fc_out(x)   # shape: (batch, seq_len, 2)
        return values, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, HIDDEN_DIM)