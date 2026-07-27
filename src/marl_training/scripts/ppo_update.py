import torch
import torch.optim as optim

CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 10
LEARNING_RATE = 5e-4


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer,
               advantages, returns, agent_names=('jb_0', 'jb_1')):
    """
    advantages, returns: dicts keyed by agent name, each a list matching buffer length
    """
    batch_size = len(buffer)

    # Prepare tensors (treating each step independently: seq_len=1 per item)
    joint_obs = torch.tensor(buffer.joint_obs, dtype=torch.float32).unsqueeze(1)  # (batch, 1, 32)

    for epoch in range(PPO_EPOCHS):
        actor_hidden = actor.init_hidden(batch_size)
        critic_hidden = critic.init_hidden(batch_size)

        # ----- Critic update -----
        values, _ = critic(joint_obs, critic_hidden)  # (batch, 1, 2)
        values = values.squeeze(1)  # (batch, 2)

        critic_loss = 0
        for i, name in enumerate(agent_names):
            target_returns = torch.tensor(returns[name], dtype=torch.float32)
            critic_loss += torch.nn.functional.mse_loss(values[:, i], target_returns)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        critic_optimizer.step()

        # ----- Actor update (per agent, since each has its own observation/action) -----
        actor_loss_total = 0
        for name in agent_names:
            obs = torch.tensor(buffer.obs[name], dtype=torch.float32).unsqueeze(1)  # (batch, 1, 16)
            old_log_probs = torch.tensor(buffer.log_probs[name], dtype=torch.float32)
            actions = torch.tensor(buffer.actions[name], dtype=torch.long)
            agent_advantages = torch.tensor(advantages[name], dtype=torch.float32)

            # Normalize advantages (standard PPO trick -- stabilizes training)
            agent_advantages = (agent_advantages - agent_advantages.mean()) / (agent_advantages.std() + 1e-8)

            logits, _ = actor(obs, actor_hidden)
            logits = logits.squeeze(1)  # (batch, 4)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * agent_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            actor_loss_total += policy_loss - ENTROPY_COEF * entropy

        actor_optimizer.zero_grad()
        actor_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        actor_optimizer.step()

    return critic_loss.item(), actor_loss_total.item()