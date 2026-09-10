"""
gate_network.py

The communication gate: a small, separate network that decides whether an
agent shares its state with the other robot this step -- "talk" or
"silent". This is Objective 2 (see overview.md): a hard-decision gate,
kept deliberately separate from the actor/critic in networks.py, matching
the design decision already on file -- attention and soft/fading gates
were rejected in favour of exactly this: a small binary gate network.

STATUS: standalone, untrained (random init), and NOT wired into
jetbot_env.py / networks.py / ppo_update.py yet. See README.md in this
folder for why, and for the exact integration plan for once Objective 1
finishes across the full curriculum.

Input features (see features.py for how these are computed -- 4 values,
matching your note):
    0: relative distance      -- metres, between the two robots
    1: relative velocity      -- other's forward speed minus own (m/s)
    2: relative heading       -- bearing to the other robot, in ego's own
                                  frame, normalised to [-1, 1] by pi
    3: message-valid flag     -- 1.0 if a message actually arrived this
                                  slot, else 0.0 (the "zero-value trap" fix)

Output: TWO logits, [silent, talk] -- deliberately shaped like Actor's
action_logits in networks.py, so ppo_update.py's existing
torch.distributions.Categorical(logits=...) pattern (used for the 4 nav
actions) can be reused unchanged when this is integrated later.
"""

import torch
import torch.nn as nn

GATE_INPUT_DIM = 4      # rel_distance, rel_velocity, rel_heading, msg_valid
GATE_HIDDEN_DIM = 16
GATE_OUTPUT_DIM = 2     # [silent, talk]

SILENT, TALK = 0, 1


class GateNet(nn.Module):
    """Per-agent gate. Feedforward, no GRU: the design note this follows
    treats the decision as a function of the *current* relative state, not
    a sequence -- unlike Actor/Critic, which need recurrence for partial
    observability of the navigation task itself. If chattering between
    talk/silent turns out to be a problem once this trains on real data,
    a GRU can be added here the same way Actor does it; nothing else in
    this module assumes feedforward.
    """

    def __init__(self, input_dim=GATE_INPUT_DIM, hidden_dim=GATE_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, GATE_OUTPUT_DIM)

    def forward(self, features):
        """features: (..., 4) -> logits: (..., 2)."""
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)

    def decide(self, features):
        """Deterministic hard decision for execution/evaluation (no
        sampling, no gradient). Returns a LongTensor of 0 (silent) /
        1 (talk). Meaningless until this network is actually trained --
        right now it is random-init, this method just fixes the calling
        convention that later training/integration code will rely on.
        """
        with torch.no_grad():
            logits = self.forward(features)
            return torch.argmax(logits, dim=-1)
