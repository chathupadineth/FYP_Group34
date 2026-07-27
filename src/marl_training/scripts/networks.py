import torch
import torch.nn as nn

OBS_DIM = 16
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