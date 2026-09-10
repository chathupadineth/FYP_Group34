"""
evaluate_policy.py

Runs a trained policy over many random scenarios and reports the metrics your
FYP actually needs: success rate, collision rate, and SPL.

    python3 evaluate_policy.py --episodes 20
    python3 evaluate_policy.py --episodes 20 --policy random      # the floor
    python3 evaluate_policy.py --episodes 20 --checkpoint checkpoints/actor_update50.pt

WHY NOT eval_run.py
-------------------
That script runs ONE fixed scenario, so it cannot give a success rate. It also
teleports the robots by hand instead of going through env.reset(), which skips
the odometry anchor calibration -- fine while Gazebo ground truth is available,
broken the moment it is not.

This script uses env.reset() for every episode, so everything the environment
normally does still happens.

REPEATABILITY
-------------
Scenarios come from the global `random` module, seeded with --seed. Action
selection uses its own separate generator, so the SAME seed gives the SAME
scenarios whichever policy you evaluate. That makes

    --policy trained    vs    --policy random

a fair, like-for-like comparison on identical scenarios.

METRICS
-------
  SR   success rate -- fraction of robot-runs that reached the goal
  CR   collision rate
  SPL  Success weighted by Path Length, the standard navigation metric:

           SPL = mean over runs of  success * optimal / max(actual, optimal)

       1.0 means every run succeeded by the shortest possible route.
       The optimal route comes from A* over the known map, so it accounts for
       having to go around buildings -- not just straight-line distance.
"""

import argparse
import math
import random
import time

import torch

from jetbot_env import MultiJetBotEnv, MAX_EPISODE_STEPS
from networks import Actor
import scripted_nav_eval as snav

AGENTS = ('jb_0', 'jb_1')
DEVICE = torch.device('cpu')


def load_actor(path):
    actor = Actor().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'actor_state_dict' in state:
        state = state['actor_state_dict']
    actor.load_state_dict(state)
    actor.eval()
    return actor


def optimal_length(start, goal):
    """Shortest route around the buildings, from A*. Falls back to the straight
    line if no route exists (then SPL is optimistic for that run, so we flag it)."""
    try:
        waypoints = snav.plan_path(start, goal)
    except RuntimeError:
        return math.hypot(goal[0] - start[0], goal[1] - start[1]), False
    total, prev = 0.0, start
    for wp in waypoints:
        total += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        prev = wp
    return total, True


def run_episode(env, actor, rng, stochastic, verbose, max_goal_distance):
    obs = env.reset(max_goal_distance=max_goal_distance)

    start_pose, goals = {}, {}
    for name in AGENTS:
        p = env.pose_source.world_pose(name, env.agents[name].latest_odom)
        start_pose[name] = (p[0], p[1])
        goals[name] = env.goals[name]

    hidden = {n: actor.init_hidden(1).to(DEVICE) for n in AGENTS} if actor else None
    last_pos = dict(start_pose)
    path_len = {n: 0.0 for n in AGENTS}
    outcome = {n: 'TIMEOUT' for n in AGENTS}
    steps_taken = {n: 0 for n in AGENTS}
    finished = {n: False for n in AGENTS}
    hits = {n: 0 for n in AGENTS}

    for t in range(MAX_EPISODE_STEPS + 5):
        actions = {}
        for name in AGENTS:
            if env.agent_done[name]:
                actions[name] = 0
                continue
            if actor is None:
                actions[name] = rng.randrange(4)
            else:
                obs_t = torch.tensor(obs[name], dtype=torch.float32,
                                      device=DEVICE).view(1, 1, -1)
                with torch.no_grad():
                    logits, hidden[name] = actor(obs_t, hidden[name])
                logits = logits[0, -1]
                if stochastic:
                    actions[name] = torch.distributions.Categorical(
                        logits=logits).sample().item()
                else:
                    actions[name] = torch.argmax(logits).item()

        obs, rewards, dones = env.step(actions)

        for name in AGENTS:
            if finished[name]:
                continue
            p = env.pose_source.world_pose(name, env.agents[name].latest_odom)
            if p is not None:
                path_len[name] += math.hypot(p[0] - last_pos[name][0],
                                              p[1] - last_pos[name][1])
                last_pos[name] = (p[0], p[1])
            steps_taken[name] = t + 1
            if dones[name]:
                finished[name] = True
                # Collisions are no longer terminal, so an episode ends only at
                # the goal or at the step cap. Contacts are counted separately
                # and reported as a rate, which is what the thesis needs anyway:
                # "did it arrive" and "how much did it hit" are now independent.
                outcome[name] = 'GOAL' if rewards[name] >= 10.0 else 'TIMEOUT'
                hits[name] = env.collision_events[name]

        if all(dones.values()):
            break

    results = []
    for name in AGENTS:
        opt, exact = optimal_length(start_pose[name], goals[name])
        success = outcome[name] == 'GOAL'
        spl = (opt / max(path_len[name], opt)) if (success and opt > 1e-6) else 0.0
        results.append({
            'robot': name, 'outcome': outcome[name], 'steps': steps_taken[name],
            'path': path_len[name], 'optimal': opt, 'spl': spl,
            'exact_optimal': exact, 'hits': max(hits[name], env.collision_events[name]),
            'start': start_pose[name], 'goal': goals[name],
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--checkpoint', default='checkpoints/actor_latest.pt')
    ap.add_argument('--policy', choices=['trained', 'random'], default='trained')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions instead of taking the best one')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--max-goal-distance', type=float, default=1.0,
                    help='Cap on how far a goal can be from its robot. MUST match '
                         'what training used (train_mappo.py -> '
                         'get_curriculum_max_goal_distance, currently 1.0). Pass 0 '
                         'for unrestricted goals, which tests generalisation to '
                         'distances the policy was never trained on.')
    args = ap.parse_args()
    max_goal_distance = args.max_goal_distance if args.max_goal_distance > 0 else None

    random.seed(args.seed)              # scenarios
    action_rng = random.Random(args.seed + 9999)   # actions, kept separate

    actor = None
    if args.policy == 'trained':
        actor = load_actor(args.checkpoint)
        print(f"Loaded {args.checkpoint}")
    else:
        print("Using a RANDOM policy (this is the floor to beat)")

    mode = 'stochastic' if args.stochastic else 'deterministic (argmax)'
    limit = ('unrestricted (tests generalisation)' if max_goal_distance is None
             else f'{max_goal_distance:.2f} m')
    print(f"policy: {args.policy} / {mode}   seed: {args.seed}   "
          f"episodes: {args.episodes}")
    print(f"goal distance limit: {limit}\n")

    env = MultiJetBotEnv()
    print("Waiting for sensor data...")
    t0 = time.time()
    import rclpy
    while not all(a.has_fresh_data() for a in env.agents.values()):
        rclpy.spin_once(env.node, timeout_sec=0.1)
        if time.time() - t0 > 8.0:
            raise TimeoutError("No sensor data - is the sim running?")

    all_runs = []
    if not args.quiet:
        print(f"{'ep':>3} {'robot':<6} {'outcome':<10} {'steps':>6} "
              f"{'path':>7} {'optimal':>8} {'SPL':>6} {'hits':>5}")

    started = time.time()
    for ep in range(1, args.episodes + 1):
        runs = run_episode(env, actor, action_rng, args.stochastic,
                           not args.quiet, max_goal_distance)
        all_runs.extend(runs)
        if not args.quiet:
            for r in runs:
                flag = '' if r['exact_optimal'] else '  (no A* route; SPL optimistic)'
                print(f"{ep:>3} {r['robot']:<6} {r['outcome']:<10} {r['steps']:>6} "
                      f"{r['path']:>7.2f} {r['optimal']:>8.2f} {r['spl']:>6.2f} "
                      f"{r['hits']:>5}{flag}")

    n = len(all_runs)
    goals = sum(1 for r in all_runs if r['outcome'] == 'GOAL')
    crashes = sum(1 for r in all_runs if r['hits'] > 0)      # runs that touched anything
    total_hits = sum(r['hits'] for r in all_runs)
    timeouts = sum(1 for r in all_runs if r['outcome'] == 'TIMEOUT')
    spl = sum(r['spl'] for r in all_runs) / max(1, n)
    succ_steps = [r['steps'] for r in all_runs if r['outcome'] == 'GOAL']
    succ_path = [r['path'] / r['optimal'] for r in all_runs
                 if r['outcome'] == 'GOAL' and r['optimal'] > 1e-6]

    print("\n" + "=" * 60)
    print(f"RESULTS  --  {args.policy} policy, {n} robot-runs "
          f"({args.episodes} episodes)")
    print("=" * 60)
    print(f"  Success rate  (SR) : {100*goals/n:5.1f}%   ({goals}/{n})")
    print(f"  Collision rate(CR) : {100*crashes/n:5.1f}%   ({crashes}/{n} runs "
          f"touched something at least once)")
    print(f"  Contacts per run   : {total_hits/n:5.2f}   ({total_hits} total)")
    print(f"  Timeout rate       : {100*timeouts/n:5.1f}%   ({timeouts}/{n})")
    print(f"  SPL                : {spl:5.3f}   (1.000 = always the shortest route)")
    if succ_steps:
        print(f"  Avg steps to goal  : {sum(succ_steps)/len(succ_steps):5.1f}")
    if succ_path:
        print(f"  Route efficiency   : {sum(succ_path)/len(succ_path):5.2f}x optimal "
              f"(1.00 = perfect)")
    print(f"  Wall clock         : {(time.time()-started)/60:5.1f} min")

    # Success against how far the goal was. Training only ever showed the
    # policy goals within get_curriculum_max_goal_distance (1.0 m), so a sharp
    # drop past that band is the policy failing to generalise -- NOT one robot
    # being worse than the other.
    bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]
    print("\n  Success by optimal route length:")
    for lo, hi in bands:
        sel = [r for r in all_runs if lo <= r['optimal'] < hi]
        if not sel:
            continue
        g = sum(1 for r in sel if r['outcome'] == 'GOAL')
        label = f"{lo:.0f}-{hi:.0f} m" if hi < 90 else f"{lo:.0f} m+"
        print(f"    {label:<8} {100*g/len(sel):5.1f}%   ({g}/{len(sel)})")

    print("\n  Per robot (the shared actor is identical for both -- a gap here "
          "is\n  scenario luck, not a per-robot policy):")
    for name in AGENTS:
        sel = [r for r in all_runs if r['robot'] == name]
        g = sum(1 for r in sel if r['outcome'] == 'GOAL')
        mean_opt = sum(r['optimal'] for r in sel) / max(1, len(sel))
        print(f"    {name}   SR {100*g/max(1,len(sel)):5.1f}%   "
              f"avg goal distance {mean_opt:.2f} m")
    print()
    print("  Compare against the floor with:")
    print(f"      python3 evaluate_policy.py --episodes {args.episodes} "
          f"--policy random --seed {args.seed} "
          f"--max-goal-distance {args.max_goal_distance}")
    print("  (same seed = same scenarios, so the numbers are directly comparable)")

    for a in env.agents.values():
        a.publish_stop()
    env.close()


if __name__ == '__main__':
    main()
