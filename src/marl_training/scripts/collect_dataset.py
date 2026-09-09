"""
collect_dataset.py

Step 1 + 2 of the instructor's plan: drive a hand-written (non-RL) controller
from many different start positions to many different goals across the whole
map, and record the 16-value observation at EVERY step.

Produces a dataset you can later use for:
  * Pattern A -- diverse reset states: sample (start, goal) pairs by difficulty
                 instead of pure random spawns.
  * Behavioural cloning: train the Actor to copy the expert's action first,
                 then fine-tune with PPO.

--------------------------------------------------------------------------
IMPORTANT -- WHY THIS DOES NOT CALL jetbot_env._build_observation()
--------------------------------------------------------------------------
That function reads the robot's position from /model/<name>/odometry, which is
in the `odom` frame (it starts at 0,0 wherever the robot is) and compares it
against a WORLD-frame goal. So observation values 13 (distance to goal) and
14 (angle to goal) come out wrong.

This collector rebuilds the SAME 16 values in the SAME order, but takes the
position from Gazebo's ground-truth pose stream. Everything else is identical,
so the dataset stays compatible with networks.py (OBS_DIM = 16).

Velocities (values 15, 16) are taken from odometry as before -- those are body
-frame speeds and are correct regardless of where the odom origin sits.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Start the simulator first:
    ros2 launch jetbot_description spawn_two_jetbots.launch.py

Then:
    python3 collect_dataset.py --episodes 100
    python3 collect_dataset.py --episodes 100 --append      # add more later
    python3 collect_dataset.py --episodes 5 --dry-run       # quick test

Output: nav_dataset.npz  (+ nav_dataset_episodes.csv for eyeballing)

Data is written to disk every few episodes, so a crash or Ctrl+C does not
lose the hours already collected.
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from jetbot_env import (
    JetBotAgent,
    NUM_LIDAR_SECTORS,
    MAX_LIDAR_RANGE,
    DISCRETE_ACTIONS,
    WORLD_NAME,
)
import scripted_nav_eval as snav

# ---------------------------------------------------------------------------
# These MUST match jetbot_env.py, or the dataset will not be compatible with
# what the RL environment produces during training.
# ---------------------------------------------------------------------------
LIDAR_NORM = MAX_LIDAR_RANGE      # jetbot_env divides by 12.0
GOAL_DIST_NORM = 3.2              # jetbot_env uses min(dist / 3.2, 1.0)

# One env step in jetbot_env = ACTION_REPEAT(3) LIDAR cycles at 5.5 Hz ~ 0.55 s.
# We drive smoothly at 20 Hz but only RECORD at that slower rate, so the
# dataset has the same time resolution the RL agent will see.
SAMPLE_PERIOD = 0.55
TICKS_PER_SAMPLE = max(1, int(round(SAMPLE_PERIOD / snav.CONTROL_DT)))

MAX_EPISODE_SECONDS = 75.0
AGENTS = ('jb_0', 'jb_1')

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nav_dataset.npz')


# ---------------------------------------------------------------------------
# Observation -- same 16 values as jetbot_env, but with a correct position
# ---------------------------------------------------------------------------
def build_observation(scan, odom, world_pose, goal):
    """Returns the 16-value observation, in exactly jetbot_env's order."""
    ranges = scan.ranges
    n = len(ranges)
    sector_size = max(1, n // NUM_LIDAR_SECTORS)

    sectors = []
    for i in range(NUM_LIDAR_SECTORS):
        chunk = [r for r in ranges[i * sector_size:(i + 1) * sector_size]
                 if not math.isinf(r) and not math.isnan(r)]
        min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
        min_r = min(min_r, MAX_LIDAR_RANGE)
        sectors.append(min_r / LIDAR_NORM)

    px, py, yaw = world_pose
    gx, gy = goal
    dx, dy = gx - px, gy - py
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) - yaw
    angle = math.atan2(math.sin(angle), math.cos(angle))

    vx = odom.twist.twist.linear.x     # body-frame speeds: correct as-is
    vz = odom.twist.twist.angular.z

    obs = sectors + [
        min(dist / GOAL_DIST_NORM, 1.0),
        angle / math.pi,
        vx,
        vz,
    ]
    return obs, dist


# ---------------------------------------------------------------------------
# Expert action: map the controller's (linear, angular) to one of the 4 actions
# ---------------------------------------------------------------------------
_ACTION_PROTOTYPES = {
    a: (lin / snav.MAX_LIN, ang / snav.MAX_ANG)
    for a, (lin, ang) in DISCRETE_ACTIONS.items()
}


def expert_action(linear, angular):
    """Nearest discrete action, comparing in normalised (linear, angular) space
    so the two axes carry comparable weight."""
    ln, an = linear / snav.MAX_LIN, angular / snav.MAX_ANG
    best, best_d = 0, float('inf')
    for a, (pl, pa) in _ACTION_PROTOTYPES.items():
        d = (ln - pl) ** 2 + (an - pa) ** 2
        if d < best_d:
            best, best_d = a, d
    return best


# ---------------------------------------------------------------------------
# Scenario sampling -- spread across the whole map, across many distances
# ---------------------------------------------------------------------------
DIFFICULTY_BANDS = [
    ('easy',   0.4, 1.0),
    ('medium', 1.0, 2.0),
    ('hard',   2.0, 3.2),
    ('very_hard', 3.2, 99.0),
]


def sample_free_point():
    for _ in range(2000):
        x = random.uniform(snav.PLATFORM_X_MIN, snav.PLATFORM_X_MAX)
        y = random.uniform(snav.PLATFORM_Y_MIN, snav.PLATFORM_Y_MAX)
        if snav.is_free(x, y):
            return (x, y)
    raise RuntimeError("Could not find a free point -- is the map/clearance sane?")


def sample_scenario(min_d, max_d, tries=200):
    """A (start, goal) pair whose straight-line distance is in [min_d, max_d]
    AND for which A* can actually find a route."""
    for _ in range(tries):
        start = sample_free_point()
        goal = sample_free_point()
        d = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if not (min_d <= d <= max_d):
            continue
        try:
            path = snav.plan_path(start, goal)
        except RuntimeError:
            continue
        return start, goal, path
    return None


def sample_two_scenarios(band, min_separation=0.5):
    """One scenario per robot, with starts far enough apart to spawn safely."""
    _, lo, hi = band
    for _ in range(60):
        s0 = sample_scenario(lo, hi)
        s1 = sample_scenario(lo, hi)
        if s0 is None or s1 is None:
            continue
        if math.hypot(s0[0][0] - s1[0][0], s0[0][1] - s1[0][1]) < min_separation:
            continue
        return {'jb_0': s0, 'jb_1': s1}
    return None


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(node, agents, gt, scenarios, episode_id, random_yaw_prob):
    starts = {n: scenarios[n][0] for n in AGENTS}
    goals = {n: scenarios[n][1] for n in AGENTS}
    paths = {n: scenarios[n][2] for n in AGENTS}

    # Face the first waypoint, or sometimes a random direction so the dataset
    # also contains "the goal is behind me" states -- otherwise the policy
    # never learns to turn around.
    yaws = {}
    for n in AGENTS:
        if random.random() < random_yaw_prob:
            yaws[n] = random.uniform(-math.pi, math.pi)
        else:
            sx, sy = starts[n]
            wx, wy = paths[n][0]
            yaws[n] = math.atan2(wy - sy, wx - sx)
        snav.teleport(n, starts[n][0], starts[n][1], yaw=yaws[n])

    time.sleep(0.4)
    snav.spin_for(node, 0.4)

    wp_idx = {n: 0 for n in AGENTS}
    cmd = {n: [0.0, 0.0] for n in AGENTS}
    done = {n: False for n in AGENTS}
    samples = []
    step_in_ep = {n: 0 for n in AGENTS}

    t0 = time.time()
    tick = 0
    while time.time() - t0 < MAX_EPISODE_SECONDS and not all(done.values()):
        snav.spin_for(node, snav.CONTROL_DT)
        tick += 1
        record = (tick % TICKS_PER_SAMPLE == 0)

        poses = {}
        for n in AGENTS:
            p = gt.get(n)
            if p is None and agents[n].latest_odom is not None:
                p = None       # ground truth is the only trustworthy source here
            poses[n] = p

        for name in AGENTS:
            agent = agents[name]
            if done[name]:
                agent.publish_stop()
                continue
            if poses[name] is None or agent.latest_scan is None or agent.latest_odom is None:
                continue

            px, py, yaw = poses[name]
            dist_to_goal = math.hypot(goals[name][0] - px, goals[name][1] - py)
            if dist_to_goal <= snav.GOAL_TOLERANCE:
                agent.publish_stop()
                done[name] = True
                continue

            path = paths[name]
            target = path[wp_idx[name]]
            ang_err, dist_wp = snav.angle_and_dist_to(px, py, yaw, target)
            if dist_wp <= snav.WAYPOINT_TOL and wp_idx[name] < len(path) - 1:
                wp_idx[name] += 1
                target = path[wp_idx[name]]
                ang_err, dist_wp = snav.angle_and_dist_to(px, py, yaw, target)

            lin_t, ang_t = snav.desired_command(ang_err, dist_to_goal)

            other = 'jb_1' if name == 'jb_0' else 'jb_0'
            if poses[other] is not None:
                is_pri = (name == snav.PRIORITY[0])
                strength = (snav.K_AGENT_REP_PRIORITY if is_pri
                            else snav.K_AGENT_REP_YIELD)
                rep, gap = snav.agent_repulsion(px, py, yaw,
                                                 poses[other][0], poses[other][1], strength)
                ang_t = max(-snav.MAX_ANG, min(snav.MAX_ANG, ang_t + rep))
                if not is_pri and gap < snav.AGENT_EMERGENCY_RADIUS:
                    lin_t = 0.0
                elif not is_pri and gap < snav.AGENT_AVOID_RADIUS:
                    span = snav.AGENT_AVOID_RADIUS - snav.AGENT_EMERGENCY_RADIUS
                    slow = snav.YIELD_MIN_SPEED_FRACTION + (1 - snav.YIELD_MIN_SPEED_FRACTION) * (
                        (gap - snav.AGENT_EMERGENCY_RADIUS) / span)
                    lin_t *= max(snav.YIELD_MIN_SPEED_FRACTION, min(1.0, slow))

            lin = snav.rate_limit(cmd[name][0], lin_t, snav.ACCEL_LIN * snav.CONTROL_DT)
            ang = snav.rate_limit(cmd[name][1], ang_t, snav.ACCEL_ANG * snav.CONTROL_DT)
            cmd[name] = [lin, ang]

            twist = Twist()
            twist.linear.x = lin
            twist.angular.z = ang
            agent.cmd_pub.publish(twist)

            if record:
                obs, d = build_observation(agent.latest_scan, agent.latest_odom,
                                            poses[name], goals[name])
                samples.append({
                    'obs': obs,
                    'action': expert_action(lin, ang),
                    'x': px, 'y': py, 'yaw': yaw,
                    'gx': goals[name][0], 'gy': goals[name][1],
                    'dist': d,
                    'agent': 0 if name == 'jb_0' else 1,
                    'episode': episode_id,
                    'step': step_in_ep[name],
                })
                step_in_ep[name] += 1

    for agent in agents.values():
        agent.publish_stop()

    return samples, done, starts, goals


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save(out_path, samples, episodes):
    if not samples:
        print("  (nothing to save yet)")
        return
    np.savez_compressed(
        out_path,
        obs=np.asarray([s['obs'] for s in samples], dtype=np.float32),
        action=np.asarray([s['action'] for s in samples], dtype=np.int8),
        x=np.asarray([s['x'] for s in samples], dtype=np.float32),
        y=np.asarray([s['y'] for s in samples], dtype=np.float32),
        yaw=np.asarray([s['yaw'] for s in samples], dtype=np.float32),
        goal_x=np.asarray([s['gx'] for s in samples], dtype=np.float32),
        goal_y=np.asarray([s['gy'] for s in samples], dtype=np.float32),
        dist_to_goal=np.asarray([s['dist'] for s in samples], dtype=np.float32),
        agent=np.asarray([s['agent'] for s in samples], dtype=np.int8),
        episode=np.asarray([s['episode'] for s in samples], dtype=np.int32),
        step=np.asarray([s['step'] for s in samples], dtype=np.int32),
        ep_start_x=np.asarray([e['start'][0] for e in episodes], dtype=np.float32),
        ep_start_y=np.asarray([e['start'][1] for e in episodes], dtype=np.float32),
        ep_goal_x=np.asarray([e['goal'][0] for e in episodes], dtype=np.float32),
        ep_goal_y=np.asarray([e['goal'][1] for e in episodes], dtype=np.float32),
        ep_agent=np.asarray([e['agent'] for e in episodes], dtype=np.int8),
        ep_success=np.asarray([e['success'] for e in episodes], dtype=np.int8),
        ep_distance=np.asarray([e['distance'] for e in episodes], dtype=np.float32),
        ep_band=np.asarray([e['band'] for e in episodes]),
    )
    csv_path = out_path.replace('.npz', '_episodes.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agent', 'start_x', 'start_y', 'goal_x', 'goal_y',
                    'distance', 'band', 'success'])
        for e in episodes:
            w.writerow([e['agent'], round(e['start'][0], 3), round(e['start'][1], 3),
                        round(e['goal'][0], 3), round(e['goal'][1], 3),
                        round(e['distance'], 3), e['band'], e['success']])
    print(f"  saved {len(samples)} samples / {len(episodes)} robot-runs -> {out_path}")


def load_existing(out_path):
    if not os.path.exists(out_path):
        return [], []
    d = np.load(out_path, allow_pickle=True)
    samples = [{
        'obs': d['obs'][i].tolist(), 'action': int(d['action'][i]),
        'x': float(d['x'][i]), 'y': float(d['y'][i]), 'yaw': float(d['yaw'][i]),
        'gx': float(d['goal_x'][i]), 'gy': float(d['goal_y'][i]),
        'dist': float(d['dist_to_goal'][i]), 'agent': int(d['agent'][i]),
        'episode': int(d['episode'][i]), 'step': int(d['step'][i]),
    } for i in range(len(d['obs']))]
    episodes = [{
        'start': (float(d['ep_start_x'][i]), float(d['ep_start_y'][i])),
        'goal': (float(d['ep_goal_x'][i]), float(d['ep_goal_y'][i])),
        'agent': int(d['ep_agent'][i]), 'success': int(d['ep_success'][i]),
        'distance': float(d['ep_distance'][i]), 'band': str(d['ep_band'][i]),
    } for i in range(len(d['ep_start_x']))]
    return samples, episodes


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100,
                    help='number of episodes (each gives data for BOTH robots)')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--append', action='store_true',
                    help='add to an existing dataset instead of overwriting')
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--random-yaw-prob', type=float, default=0.3,
                    help='fraction of spawns facing a random direction')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='run a few episodes without writing the dataset')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    samples, episodes = ([], [])
    ep_offset = 0
    if args.append:
        samples, episodes = load_existing(args.out)
        ep_offset = (max(s['episode'] for s in samples) + 1) if samples else 0
        print(f"Appending to existing dataset: {len(samples)} samples already there.")

    rclpy.init()
    node = Node('collect_dataset')
    agents = {n: JetBotAgent(node, n) for n in AGENTS}

    print("Waiting for sensor data...")
    t0 = time.time()
    while not all(a.has_fresh_data() for a in agents.values()):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 8.0:
            raise TimeoutError("No LIDAR/odometry after 8s - is the sim running?")

    gt = snav.GroundTruthTracker(WORLD_NAME, AGENTS)
    if not (gt.start() and gt.wait_for(timeout=4.0)):
        raise RuntimeError(
            "Ground-truth pose stream unavailable.\n"
            "This collector needs it -- odometry alone gives a wrong goal distance\n"
            "and would produce a corrupt dataset. Check:\n"
            "    ign topic -e -n 1 -t /world/empty/dynamic_pose/info")
    print("Ground truth OK. Position values will be correct.\n")

    est = args.episodes * MAX_EPISODE_SECONDS / 60.0
    print(f"Collecting {args.episodes} episodes. Worst case ~{est:.0f} min "
          f"(usually much less -- episodes end early on success).\n")

    try:
        for ep in range(args.episodes):
            band = DIFFICULTY_BANDS[ep % len(DIFFICULTY_BANDS)]
            scen = sample_two_scenarios(band)
            if scen is None:
                print(f"[{ep+1}/{args.episodes}] could not sample a '{band[0]}' pair, skipping")
                continue

            new, done, starts, goals = run_episode(
                node, agents, gt, scen, ep_offset + ep, args.random_yaw_prob)
            samples.extend(new)

            for i, n in enumerate(AGENTS):
                episodes.append({
                    'start': starts[n], 'goal': goals[n], 'agent': i,
                    'success': int(done[n]), 'band': band[0],
                    'distance': math.hypot(goals[n][0] - starts[n][0],
                                            goals[n][1] - starts[n][1]),
                })

            ok = sum(1 for n in AGENTS if done[n])
            print(f"[{ep+1}/{args.episodes}] band={band[0]:<9} "
                  f"samples+={len(new):<4} reached={ok}/2  total={len(samples)}")

            if not args.dry_run and (ep + 1) % args.save_every == 0:
                save(args.out, samples, episodes)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        for agent in agents.values():
            agent.publish_stop()
        if not args.dry_run:
            save(args.out, samples, episodes)
        gt.stop()
        rclpy.shutdown()

    print("\nDone. Inspect it with:")
    print(f"    python3 dataset_tools.py summary {args.out}")


if __name__ == '__main__':
    main()
