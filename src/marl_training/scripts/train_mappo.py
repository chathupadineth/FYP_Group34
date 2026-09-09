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

def get_curriculum_max_goal_distance(update_num):
    return 1.0   # close goals for this entire 50-update test run

ROLLOUT_LENGTH = 200      # steps collected per update (~1 episode's worth)
NUM_UPDATES = 110         # short test run
CHECKPOINT_EVERY = 10
LEARNING_RATE = 5e-4

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'training_log.csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(actor, critic, update_num, tag='latest'):
    torch.save(actor.state_dict(), os.path.join(CHECKPOINT_DIR, f'actor_{tag}.pt'))
    torch.save(critic.state_dict(), os.path.join(CHECKPOINT_DIR, f'critic_{tag}.pt'))
    if tag == 'latest':
        with open(os.path.join(CHECKPOINT_DIR, 'last_update.txt'), 'w') as f:
            f.write(str(update_num))
    print(f"Saved checkpoint: {tag} (update {update_num})")

def load_checkpoint(actor, critic):
    actor_path = os.path.join(CHECKPOINT_DIR, 'actor_latest.pt')
    critic_path = os.path.join(CHECKPOINT_DIR, 'critic_latest.pt')
    last_update_path = os.path.join(CHECKPOINT_DIR, 'last_update.txt')

    if os.path.exists(actor_path) and os.path.exists(critic_path) and os.path.exists(last_update_path):
        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))
        with open(last_update_path, 'r') as f:
            last_update = int(f.read().strip())
        print(f"Resumed from checkpoint at update {last_update}")
        return last_update
    else:
        print("No checkpoint found, starting fresh")
        return 0

def main():
    actor = Actor()
    critic = Critic()
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    start_update = load_checkpoint(actor, critic)

    env = MultiJetBotEnv()

    log_file_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, 'a' if log_file_exists else 'w', newline='')
    log_writer = csv.writer(log_file)
    if not log_file_exists:
        log_writer.writerow(['update', 'avg_reward_jb0', 'avg_reward_jb1',
                              'episodes_completed', 'goals_reached', 'collisions',
                              'critic_loss', 'actor_loss', 'elapsed_sec',
                              'active_steps_jb0', 'active_steps_jb1'])

    print("Initial reset...")
    obs = env.reset(max_goal_distance=get_curriculum_max_goal_distance(start_update + 1))

    actor_hidden = {name: actor.init_hidden(1) for name in ['jb_0', 'jb_1']}
    critic_hidden_rollout = critic.init_hidden(1)

    start_time = time.time()

    for update in range(start_update + 1, NUM_UPDATES + 1):
        buffer = RolloutBuffer()

        episode_rewards = {'jb_0': [], 'jb_1': []}
        current_ep_reward = {'jb_0': 0.0, 'jb_1': 0.0}
        goals_reached = 0
        collisions = 0
        action_counts = {'jb_0': [0, 0, 0, 0], 'jb_1': [0, 0, 0, 0]}

        for step in range(ROLLOUT_LENGTH):
            joint_obs = obs['jb_0'] + obs['jb_1']

            # Which robots are actually still running THIS step? A robot that
            # already reached its goal or crashed is stopped by the env, so its
            # sampled action does nothing and its 0.0 reward says nothing about
            # that action. Those steps get masked out of the loss later.
            active = {name: not env.agent_done[name] for name in ['jb_0', 'jb_1']}

            actions = {}
            log_probs = {}
            with torch.no_grad():
                for name in ['jb_0', 'jb_1']:
                    obs_tensor = torch.tensor(obs[name], dtype=torch.float32).view(1, 1, 16)
                    logits, actor_hidden[name] = actor(obs_tensor, actor_hidden[name])
                    dist = torch.distributions.Categorical(logits=logits.squeeze())
                    action = dist.sample()
                    actions[name] = action.item()
                    log_probs[name] = dist.log_prob(action).item()
                    action_counts[name][action.item()] += 1

                joint_obs_tensor = torch.tensor(joint_obs, dtype=torch.float32).view(1, 1, 32)
                values, critic_hidden_rollout = critic(joint_obs_tensor, critic_hidden_rollout)
                values = values.squeeze().tolist()

            next_obs, rewards, dones = env.step(actions)   

            episode_done = all(dones.values())
            buffer.add(obs, joint_obs, actions, log_probs, rewards, values, episode_done,
                       active=active, agent_dones=dones)

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
                obs = env.reset(max_goal_distance=get_curriculum_max_goal_distance(update))
                actor_hidden = {name: actor.init_hidden(1) for name in ['jb_0', 'jb_1']}
                critic_hidden_rollout = critic.init_hidden(1)

        # ----- Compute GAE per agent -----
        advantages = {}
        returns = {}
        for i, name in enumerate(['jb_0', 'jb_1']):
            agent_values = [v[i] for v in buffer.values]
            # Use THIS robot's own terminal flag, not the shared episode flag.
            # Otherwise value bootstrapping runs past the point where the robot
            # actually finished and keeps crediting it for the other robot's
            # remaining steps.
            adv, ret = compute_gae(buffer.rewards[name], agent_values,
                                    buffer.agent_dones[name])
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
        print(f"    action counts (fwd,left,right,back) — jb_0: {action_counts['jb_0']} | jb_1: {action_counts['jb_1']}")

        # How much of this rollout was real experience vs padding after a robot
        # had already finished. Low numbers here mean most of the buffer is
        # useless for that robot.
        act = buffer.active_counts()
        n = len(buffer)
        print(f"    active steps — jb_0: {act['jb_0']}/{n} ({100*act['jb_0']/max(1,n):.0f}%) | "
              f"jb_1: {act['jb_1']}/{n} ({100*act['jb_1']/max(1,n):.0f}%)")

        log_writer.writerow([update, avg_r0, avg_r1, len(episode_rewards['jb_0']),
                              goals_reached, collisions, critic_loss, actor_loss, elapsed,
                              act['jb_0'], act['jb_1']])
        log_file.flush()
        os.fsync(log_file.fileno())
        save_checkpoint(actor, critic, update, tag='latest')
        if update % CHECKPOINT_EVERY == 0:
            save_checkpoint(actor, critic, update, tag=f'update{update}')

    for name in ['jb_0', 'jb_1']:
        # publish_action(3) is BACKWARD, not stop -- it used to drive both
        # robots backwards when training finished.
        env.agents[name].publish_stop()

    log_file.close()
    env.close()
    print("\nTraining complete.")


if __name__ == '__main__':
    main()