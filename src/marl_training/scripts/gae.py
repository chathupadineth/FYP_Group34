import torch

GAMMA = 0.99
LAMBDA = 0.95


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    """
    rewards: list of rewards, one per timestep
    values: list of critic value estimates, one per timestep (same length as rewards)
    dones: list of booleans, one per timestep

    Returns: advantages (list, same length), returns (list, same length)
    """
    advantages = [0] * len(rewards)
    last_advantage = 0
    last_value = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
            next_advantage = 0
        else:
            next_value = last_value
            next_advantage = last_advantage

        td_error = rewards[t] + gamma * next_value - values[t]
        advantages[t] = td_error + gamma * lam * next_advantage

        last_value = values[t]
        last_advantage = advantages[t]

    returns = [advantages[t] + values[t] for t in range(len(rewards))]
    return advantages, returns