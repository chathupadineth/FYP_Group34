import os
import csv
import time
import torch
import torch.optim as optim

from jetbot_env import MultiJetBotEnv
from networks import Actor, Critic
from buffer import RolloutBuffer
from gae import compute_gae
from ppo_update import ppo_update

ROLLOUT_LENGTH = 200      # steps collected per update (~1 episode's worth)
NUM_UPDATES = 50          # short test run
CHECKPOINT_EVERY = 10
LEARNING_RATE = 5e-4

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'training_log.csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(actor, critic, update_num, tag='latest'):
    torch.save(actor.state_dict(), os.path.join(CHECKPOINT_DIR, f'actor_{tag}.pt'))
    torch.save(critic.state_dict(), os.path.join(CHECKPOINT_DIR, f'critic_{tag}.pt'))
    print(f"Saved checkpoint: {tag} (update {update_num})")


def main():
    actor = Actor()
    critic = Critic()
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    env = MultiJetBotEnv()

    log_file = open(LOG_PATH, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['update', 'avg_reward_jb0', 'avg_reward_jb1',
                          'episodes_completed', 'goals_reached', 'collisions',
                          'critic_loss', 'actor_loss', 'elapsed_sec'])

    print("Initial reset...")
    obs = env.reset()

    start_time = time.time()

    for update in range(1, NUM_UPDATES + 1):
        buffer = RolloutBuffer()

        episode_rewards = {'jb_0': [], 'jb_1': []}
        current_ep_reward = {'jb_0': 0.0, 'jb_1': 0.0}
        goals_reached = 0
        collisions = 0

        for step in range(ROLLOUT_LENGTH):
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
            values = values.squeeze().tolist()

            next_obs, rewards, dones = env.step(actions)

            episode_done = any(dones.values())
            buffer.add(obs, joint_obs, actions, log_probs, rewards, values, episode_done)

            for name in ['jb_0', 'jb_1']:
                current_ep_reward[name] += rewards[name]
                if rewards[name] >= 10.0:
                    goals_reached += 1
                elif rewards[name] <= -10.0:
                    collisions += 1

            obs = next_obs

            if episode_done:
                for name in ['jb_0', 'jb_1']:
                    episode_rewards[name].append(current_ep_reward[name])
                    current_ep_reward[name] = 0.0
                obs = env.reset()

        # ----- Compute GAE per agent -----
        advantages = {}
        returns = {}
        for i, name in enumerate(['jb_0', 'jb_1']):
            agent_values = [v[i] for v in buffer.values]
            adv, ret = compute_gae(buffer.rewards[name], agent_values, buffer.dones)
            advantages[name] = adv
            returns[name] = ret

        critic_loss, actor_loss = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer,
            buffer, advantages, returns
        )

        avg_r0 = sum(episode_rewards['jb_0']) / len(episode_rewards['jb_0']) if episode_rewards['jb_0'] else 0.0
        avg_r1 = sum(episode_rewards['jb_1']) / len(episode_rewards['jb_1']) if episode_rewards['jb_1'] else 0.0
        elapsed = time.time() - start_time

        print(f"[Update {update}/{NUM_UPDATES}] "
              f"avg_reward jb_0={avg_r0:.2f} jb_1={avg_r1:.2f} | "
              f"episodes={len(episode_rewards['jb_0'])} goals={goals_reached} collisions={collisions} | "
              f"critic_loss={critic_loss:.4f} actor_loss={actor_loss:.4f} | "
              f"elapsed={elapsed/60:.1f}min")

        log_writer.writerow([update, avg_r0, avg_r1, len(episode_rewards['jb_0']),
                              goals_reached, collisions, critic_loss, actor_loss, elapsed])
        log_file.flush()

        save_checkpoint(actor, critic, update, tag='latest')
        if update % CHECKPOINT_EVERY == 0:
            save_checkpoint(actor, critic, update, tag=f'update{update}')

    for name in ['jb_0', 'jb_1']:
        env.agents[name].publish_action(3)  # 3 = stop

    log_file.close()
    env.close()
    print("\nTraining complete.")


if __name__ == '__main__':
    main()