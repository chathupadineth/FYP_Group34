import torch
import torch.optim as optim

CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.03
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 10
LEARNING_RATE = 5e-4


def _forward_actor_sequence(actor, obs_seq, dones):
    """Runs the actor's GRU one step at a time in order, carrying hidden
    state forward across steps and resetting after each episode boundary --
    matching how hidden state is handled during rollout collection."""
    hidden = actor.init_hidden(1)
    logits_list = []
    for t in range(obs_seq.shape[0]):
        obs_t = obs_seq[t].view(1, 1, -1)
        logit_t, hidden = actor(obs_t, hidden)
        logits_list.append(logit_t.view(-1))
        if dones[t]:
            hidden = actor.init_hidden(1)
    return torch.stack(logits_list, dim=0)


def _forward_critic_sequence(critic, joint_obs_seq, dones):
    """Same idea as _forward_actor_sequence, for the centralized critic."""
    hidden = critic.init_hidden(1)
    values_list = []
    for t in range(joint_obs_seq.shape[0]):
        obs_t = joint_obs_seq[t].view(1, 1, -1)
        value_t, hidden = critic(obs_t, hidden)
        values_list.append(value_t.view(-1))
        if dones[t]:
            hidden = critic.init_hidden(1)
    return torch.stack(values_list, dim=0)


def _agent_masks(buffer, agent_names, length):
    """1.0 where the robot was still running, 0.0 after it finished.

    Steps after a robot finishes are padding: it was stopped, its sampled
    action did nothing, and its 0.0 reward describes the environment being
    over, not the action. Including them trains the actor on meaningless
    samples and hands the critic contradictory targets for the same
    observation, which is why critic loss used to stall.
    """
    masks = {}
    stored = getattr(buffer, 'active', None)
    for name in agent_names:
        flags = stored.get(name) if isinstance(stored, dict) else None
        if not flags:
            flags = [True] * length         # old buffers: treat everything as active
        masks[name] = torch.tensor([1.0 if f else 0.0 for f in flags],
                                    dtype=torch.float32)
    return masks


def _masked_normalise(x, mask):
    n = mask.sum().clamp(min=1.0)
    mean = (x * mask).sum() / n
    var = (((x - mean) ** 2) * mask).sum() / n
    return (x - mean) / (var.sqrt() + 1e-8)


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer,
               advantages, returns, agent_names=('jb_0', 'jb_1')):
    """
    advantages, returns: dicts keyed by agent name, each a list matching buffer length
    """
    joint_obs = torch.tensor(buffer.joint_obs, dtype=torch.float32)  # (batch, 32)
    dones = buffer.dones  # episode boundaries -> where the GRU state resets
    masks = _agent_masks(buffer, agent_names, len(dones))

    critic_loss = None
    actor_loss_total = None

    for epoch in range(PPO_EPOCHS):
        # ----- Critic update -----
        values = _forward_critic_sequence(critic, joint_obs, dones)  # (batch, 2)
        critic_loss = 0
        for i, name in enumerate(agent_names):
            target_returns = torch.tensor(returns[name], dtype=torch.float32)
            m = masks[name]
            sq_err = (values[:, i] - target_returns) ** 2
            critic_loss += (sq_err * m).sum() / m.sum().clamp(min=1.0)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        critic_optimizer.step()

        # ----- Actor update (per agent) -----
        actor_loss_total = 0
        for name in agent_names:
            obs = torch.tensor(buffer.obs[name], dtype=torch.float32)  # (batch, 16)
            old_log_probs = torch.tensor(buffer.log_probs[name], dtype=torch.float32)
            actions = torch.tensor(buffer.actions[name], dtype=torch.long)
            m = masks[name]
            n_active = m.sum().clamp(min=1.0)

            agent_advantages = torch.tensor(advantages[name], dtype=torch.float32)
            agent_advantages = _masked_normalise(agent_advantages, m)

            # The GRU is still rolled over EVERY step, including padding, so the
            # hidden-state trajectory matches the one used during collection.
            # Only the loss is masked.
            logits = _forward_actor_sequence(actor, obs, dones)  # (batch, 4)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = (dist.entropy() * m).sum() / n_active
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * agent_advantages
            policy_loss = -(torch.min(surr1, surr2) * m).sum() / n_active
            actor_loss_total += policy_loss - ENTROPY_COEF * entropy

        actor_optimizer.zero_grad()
        actor_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        actor_optimizer.step()

    return critic_loss.item(), actor_loss_total.item()
