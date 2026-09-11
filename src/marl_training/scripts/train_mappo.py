import os
import csv
import time
from collections import deque

import torch
import torch.optim as optim

from jetbot_env import MultiJetBotEnv
from networks import Actor, Critic
from buffer import RolloutBuffer
from gae import compute_gae
from ppo_update import ppo_update

# ---------------------------------------------------------------------------
# CURRICULUM
# ---------------------------------------------------------------------------
# How far a goal may be from its robot. It used to be pinned at 1.0 for a short
# test run and never moved.
#
# The rungs are chosen from how often a STRAIGHT LINE from start to goal is
# collision-free on this map -- i.e. the best a policy can do while ignoring
# its LIDAR entirely:
#
#     max dist    mean dist    straight line clear
#       1.0 m       0.76 m           71%
#       1.5 m       1.06 m           49%     <- obstacle avoidance becomes
#       2.0 m       1.31 m           40%        mandatory from here on
#       2.5 m       1.58 m           31%
#       3.0 m       1.87 m           24%
#       none        1.97 m           21%
#
# A policy that only learned "turn to the goal and drive" tops out at those
# numbers, so each rung forces it to actually use the range sensor.
CURRICULUM = [1.0, 1.5, 2.0, 2.5, 3.0, None]     # None = no limit

# Set False to freeze the curriculum wherever it currently stands. Useful for
# giving one rung more training time without risking a jump to the next one
# mid-run: the rolling success rate is still computed and logged, it just
# never triggers a promotion.
PROMOTE_ENABLED = True

# Promotion is driven by RESULTS, not by update number. Advancing on a fixed
# schedule is how a curriculum kills a run: if the policy is still weak at 1.0
# when update 150 arrives, moving it to 1.5 leaves it failing at both.
PROMOTE_SR = 0.55           # rolling success rate needed to move up a rung
PROMOTE_WINDOW = 10         # updates in the rolling window
PROMOTE_MIN_UPDATES = 15    # minimum time on a rung before promotion

ROLLOUT_LENGTH = 200      # steps collected per update (~1 episode's worth)
NUM_UPDATES = 400
CHECKPOINT_EVERY = 10
LEARNING_RATE = 5e-4

# Critic-only updates before the actor is touched. Needed when the actor came
# from behavioural cloning and the critic is random. Continuing run 2 resumes a
# critic that is already trained, so this is 0 -- warming up here would only
# throw away 10 updates.
CRITIC_WARMUP_UPDATES = 0

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'training_log.csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


STAGE_PATH = os.path.join(CHECKPOINT_DIR, 'curriculum_stage.txt')


def save_stage(stage, update_num):
    """The curriculum rung must survive a restart. Without this, resuming would
    silently drop the policy back to 1.0 m goals and undo the progress that
    earned the promotion."""
    with open(STAGE_PATH, 'w') as f:
        f.write(f"{stage}\n{update_num}\n")


def load_stage():
    try:
        with open(STAGE_PATH) as f:
            stage = int(f.readline().strip())
        return max(0, min(stage, len(CURRICULUM) - 1))
    except (OSError, ValueError):
        return 0


def stage_label(stage):
    d = CURRICULUM[stage]
    return 'unrestricted' if d is None else f'{d:.1f} m'


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
                              'active_steps_jb0', 'active_steps_jb1',
                              'max_goal_distance', 'rolling_success_rate'])

    stage = load_stage()
    updates_on_stage = 0
    # Success rate is per ROBOT-RUN, not per update. Every episode ends both
    # robots, so runs = 2 x episodes. Counting per update instead makes a rung
    # look easier simply because its episodes got longer.
    window = deque(maxlen=PROMOTE_WINDOW)      # (goals, runs) per update
    print(f"Curriculum stage {stage}/{len(CURRICULUM)-1}: goals within {stage_label(stage)}")

    print("Initial reset...")
    obs = env.reset(max_goal_distance=CURRICULUM[stage])

    actor_hidden = {name: actor.init_hidden(1) for name in ['jb_0', 'jb_1']}
    critic_hidden_rollout = critic.init_hidden(1)

    start_time = time.time()

    for update in range(start_update + 1, NUM_UPDATES + 1):
        buffer = RolloutBuffer()

        episode_rewards = {'jb_0': [], 'jb_1': []}
        current_ep_reward = {'jb_0': 0.0, 'jb_1': 0.0}
        goals_reached = 0
        collisions = 0
        prev_collision_events = dict(env.collision_events)
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

            # Collisions are no longer terminal and no longer produce a unique
            # -10.0, so they are read from the environment's own counter. Taken
            # as a delta because the counter resets with the episode, not with
            # the rollout.
            for name in ['jb_0', 'jb_1']:
                ev = env.collision_events[name]
                if ev >= prev_collision_events[name]:
                    collisions += ev - prev_collision_events[name]
                else:
                    collisions += ev          # env.reset() zeroed the counter
                prev_collision_events[name] = ev

            obs = next_obs

            if episode_done:
                for name in ['jb_0', 'jb_1']:
                    episode_rewards[name].append(current_ep_reward[name])
                    current_ep_reward[name] = 0.0
                obs = env.reset(max_goal_distance=CURRICULUM[stage])
                # reset() zeroed the counter; re-baseline so the next delta is
                # measured from 0 rather than from the finished episode's total.
                prev_collision_events = {'jb_0': 0, 'jb_1': 0}
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

        # A behaviour-cloned actor arrives with a random critic. Applying PPO
        # immediately would compute advantages from a meaningless value
        # function and undo the cloning in a handful of updates, so the critic
        # is given a head start on its own. Starting from scratch (no BC), the
        # actor is random anyway and the warmup costs nothing.
        warming_up = update <= start_update + CRITIC_WARMUP_UPDATES
        critic_loss, actor_loss = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer,
            buffer, advantages, returns, update_actor=not warming_up
        )

        avg_r0 = sum(episode_rewards['jb_0']) / len(episode_rewards['jb_0']) if episode_rewards['jb_0'] else 0.0
        avg_r1 = sum(episode_rewards['jb_1']) / len(episode_rewards['jb_1']) if episode_rewards['jb_1'] else 0.0
        elapsed = time.time() - start_time

        if warming_up:
            print(f"    [critic warmup {update - start_update}/{CRITIC_WARMUP_UPDATES}"
                  f" — actor frozen]")

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

        # ----- Curriculum: promote on results, never on a fixed schedule -----
        episodes = len(episode_rewards['jb_0'])
        window.append((goals_reached, 2 * episodes))
        win_goals = sum(g for g, _ in window)
        win_runs = sum(r for _, r in window)
        rolling_sr = win_goals / win_runs if win_runs else 0.0
        updates_on_stage += 1

        print(f"    curriculum — goals within {stage_label(stage)} | "
              f"rolling SR (last {len(window)} updates) = {100*rolling_sr:.1f}% "
              f"({win_goals}/{win_runs} runs) | {updates_on_stage} updates on this rung")

        if (PROMOTE_ENABLED
                and stage < len(CURRICULUM) - 1
                and updates_on_stage >= PROMOTE_MIN_UPDATES
                and len(window) == PROMOTE_WINDOW
                and rolling_sr >= PROMOTE_SR):
            stage += 1
            updates_on_stage = 0
            window.clear()          # the old rung's scores say nothing about the new one
            print(f"    *** PROMOTED to stage {stage}: goals within {stage_label(stage)} ***")
            print(f"    Expect success rate to DROP now. That is the curriculum working,")
            print(f"    not the policy breaking.")
        save_stage(stage, update)

        log_writer.writerow([update, avg_r0, avg_r1, episodes,
                              goals_reached, collisions, critic_loss, actor_loss, elapsed,
                              act['jb_0'], act['jb_1'],
                              '' if CURRICULUM[stage] is None else CURRICULUM[stage],
                              round(rolling_sr, 4)])
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