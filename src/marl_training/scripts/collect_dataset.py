"""
collect_dataset.py

Step 1 + 2 of the instructor's plan: drive a hand-written (non-RL) controller
from many different start positions to many different goals across the whole
map, and record the 20-value observation at EVERY step.

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

This collector rebuilds the SAME values in the SAME order, but takes the
position from Gazebo's ground-truth pose stream. Everything else is identical,
so the dataset stays compatible with networks.py (OBS_DIM = 20: 16 own values
plus the 4-value communication slot, which is all zeros here because the
scripted expert never transmits).

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
    LIDAR_NORM,
    DISCRETE_ACTIONS,
    WORLD_NAME,
    GOAL_DIST_NORM,
    wall_clearance,
    ROBOT_RADIUS,
    CONTACT_DIST,
)
from networks import MSG_DIM
import scripted_nav_eval as snav

# ---------------------------------------------------------------------------
# These MUST match jetbot_env.py, or the dataset will not be compatible with
# what the RL environment produces during training.
# ---------------------------------------------------------------------------
# LIDAR_NORM is IMPORTED from jetbot_env above -- do not redefine it here.
# It used to be reassigned to MAX_LIDAR_RANGE on this line, which silently
# overwrote the import and would have produced a dataset normalised by 12.0
# while training normalised by 3.5.
# GOAL_DIST_NORM is IMPORTED from jetbot_env, like LIDAR_NORM -- never redefine
# it here, or the dataset and the environment will normalise goal distance
# differently and the cloned policy will misread every distance.

# One env step in jetbot_env = ACTION_REPEAT(3) LIDAR cycles at 5.5 Hz ~ 0.55 s.
# We drive smoothly at 20 Hz but only RECORD at that slower rate, so the
# dataset has the same time resolution the RL agent will see.
SAMPLE_PERIOD = 0.55
TICKS_PER_SAMPLE = max(1, int(round(SAMPLE_PERIOD / snav.CONTROL_DT)))

MAX_EPISODE_SECONDS = 75.0
AGENTS = ('jb_0', 'jb_1')

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nav_dataset.npz')


# ---------------------------------------------------------------------------
# Observation -- same values as jetbot_env, but with a correct position
# ---------------------------------------------------------------------------
def build_observation(scan, odom, world_pose, goal):
    """Returns the 20-value observation, in exactly jetbot_env's order:
    12 LIDAR sectors, goal distance, goal angle, vx, vz, then the 4-value
    communication slot (always silent here -- see the note at the end)."""
    ranges = scan.ranges
    n = len(ranges)
    sector_size = max(1, n // NUM_LIDAR_SECTORS)

    sectors = []
    for i in range(NUM_LIDAR_SECTORS):
        chunk = [r for r in ranges[i * sector_size:(i + 1) * sector_size]
                 if not math.isinf(r) and not math.isnan(r)]
        min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
        min_r = min(min_r, MAX_LIDAR_RANGE)
        sectors.append(min(min_r / LIDAR_NORM, 1.0))

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
    # The communication slot, appended LAST exactly as jetbot_env does. The
    # scripted expert never transmits, so every slot is the "silent" encoding:
    # all zeros with the valid flag down. That is not padding -- it is the
    # literal observation the agent would receive with nobody talking, so the
    # cloned policy is a correct no-communication policy and can serve as the
    # No-Comm arm of the No-Comm / Always-Comm / Learned-Gate comparison.
    obs = obs + [0.0] * MSG_DIM
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
# EXPERT MODE -- why the demonstrations are now driven with discrete actions
# ---------------------------------------------------------------------------
# The first behavioural-cloning attempt reached 94.9% validation accuracy and
# produced a robot that spun in place. The labels were right; the trajectory
# behind them was not reproducible.
#
# The scripted controller commands continuous velocities: up to MAX_ANG = 1.0
# rad/s while still driving at up to MAX_LIN = 0.15 m/s. The agent has four
# fixed choices, and its turns are (0.05, +/-0.6) -- it can never turn faster
# than 0.6 rad/s, and turning drops it to a third of cruise speed. So a
# recorded state labelled "turn left" was reached by a robot turning at 0.8
# rad/s while moving at 0.13 m/s; the agent executing that same label turns at
# 0.6 and crawls at 0.05. Within a few steps it is somewhere the expert never
# went, the cloned policy has never seen that state, and it degenerates.
#
# The fix is to stop demonstrating actions the learner cannot perform. In the
# discrete modes the controller still plans and steers exactly as before, but
# what actually reaches the wheels is DISCRETE_ACTIONS[a] -- so every recorded
# trajectory is one the agent could have produced itself.
#
#   'hold'       pick the action at each recording boundary and hold it for the
#                whole sample window. This matches the agent exactly: one env
#                step = ACTION_REPEAT LIDAR cycles with a single action held
#                throughout, which is TICKS_PER_SAMPLE ticks here. The default.
#   'tick'       re-pick every control tick. Velocities are in-set, but the
#                expert still changes action TICKS_PER_SAMPLE times per
#                recorded sample, which the agent cannot do. Fallback if 'hold'
#                turns out too coarse to complete episodes.
#   'continuous' the original behaviour, kept so the failure is reproducible
#                and so the two can be compared in the write-up.
#
# The expert gets slower and jerkier under 'hold'. That is the intended trade:
# a mediocre expert the agent can copy beats a perfect one it cannot.
EXPERT_MODES = ('hold', 'tick', 'continuous')


# ---------------------------------------------------------------------------
# Scenario sampling -- spread across the whole map, across many distances
# ---------------------------------------------------------------------------
# Kept only so an OLD nav_dataset.npz still loads in dataset_tools.py. New
# episodes are labelled by curriculum_band() below, not from this list.
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


# ---------------------------------------------------------------------------
# Curriculum over goal distance, ending in full-map corner crossings
# ---------------------------------------------------------------------------
# Early episodes use short goals (~1 m) and the reachable distance grows with
# every accepted episode, up to the widest separation this map allows. The last
# stretch is corner-to-corner: each robot crosses the entire floor on a diagonal
# while the other crosses the opposite diagonal, so their routes intersect in
# the middle. That gives long-range navigation AND a genuine crossing, without
# forcing the head-on situations that this map is too narrow to solve.
#
# Measured on this map:
#     corner anchors SW/SE/NW/NE are all free space
#     corner-to-corner diagonal        3.82 m
#     widest free-point separation     3.76 m
GOAL_DIST_START = 1.0        # first episodes
GOAL_DIST_MAX = 3.6          # just inside what the map allows
CORNER_PHASE_FRACTION = 0.25 # last quarter of the run is corner crossings

_CORNER_INSET = 0.21
CORNERS = {
    'SW': (snav.PLATFORM_X_MIN + _CORNER_INSET, snav.PLATFORM_Y_MIN + _CORNER_INSET),
    'SE': (snav.PLATFORM_X_MAX - _CORNER_INSET, snav.PLATFORM_Y_MIN + _CORNER_INSET),
    'NW': (snav.PLATFORM_X_MIN + _CORNER_INSET, snav.PLATFORM_Y_MAX - _CORNER_INSET),
    'NE': (snav.PLATFORM_X_MAX - _CORNER_INSET, snav.PLATFORM_Y_MAX - _CORNER_INSET),
}
# The two diagonals. Robot 0 takes one, robot 1 takes the other, so the routes
# cross near the middle of the map instead of meeting head-on.
CORNER_PAIRS = [
    (('SW', 'NE'), ('NW', 'SE')),
    (('NE', 'SW'), ('SE', 'NW')),
    (('NW', 'SE'), ('NE', 'SW')),
    (('SE', 'NW'), ('SW', 'NE')),
]


def curriculum_band(k, n_total):
    """(label, min_distance, max_distance) for the k-th ACCEPTED episode."""
    frac = k / max(1, n_total)
    if frac >= 1.0 - CORNER_PHASE_FRACTION:
        return ('corner', GOAL_DIST_MAX, 99.0)
    ramp = frac / max(1e-9, 1.0 - CORNER_PHASE_FRACTION)
    hi = GOAL_DIST_START + ramp * (GOAL_DIST_MAX - GOAL_DIST_START)
    lo = max(0.4, 0.55 * hi)
    if hi <= 1.2:
        label = 'short'
    elif hi <= 2.2:
        label = 'medium'
    elif hi <= 3.0:
        label = 'long'
    else:
        label = 'very_long'
    return (label, lo, hi)


def _routed(start, goal):
    try:
        return start, goal, snav.plan_path(start, goal)
    except RuntimeError:
        return None


def _near_corner(name, radius=0.30):
    cx, cy = CORNERS[name]
    for _ in range(300):
        x = cx + random.uniform(-radius, radius)
        y = cy + random.uniform(-radius, radius)
        if snav.is_free(x, y):
            return (x, y)
    return (cx, cy) if snav.is_free(cx, cy) else None


def sample_corner_scenarios():
    """Both robots cross the whole map, on opposite diagonals."""
    (a0, b0), (a1, b1) = random.choice(CORNER_PAIRS)
    for _ in range(40):
        s0, g0 = _near_corner(a0), _near_corner(b0)
        s1, g1 = _near_corner(a1), _near_corner(b1)
        if None in (s0, g0, s1, g1):
            continue
        r0, r1 = _routed(s0, g0), _routed(s1, g1)
        if r0 and r1:
            return {'jb_0': r0, 'jb_1': r1}
    return None


def sample_two_scenarios(band, min_separation=0.5):
    """One scenario per robot, drawn from the current curriculum band."""
    label, lo, hi = band
    if label == 'corner':
        return sample_corner_scenarios()
    for _ in range(80):
        s0 = sample_scenario(lo, hi)
        s1 = sample_scenario(lo, hi)
        if s0 is None or s1 is None:
            continue
        if math.hypot(s0[0][0] - s1[0][0], s0[0][1] - s1[0][1]) < min_separation:
            continue
        return {'jb_0': s0, 'jb_1': s1}
    return None



# ---------------------------------------------------------------------------
# Goal markers -- purely so a human watching the simulator can see where each
# robot is actually heading this episode
# ---------------------------------------------------------------------------
# The discs are <visual> only, with no <collision> element, so the LIDAR --
# which raycasts collision geometry -- cannot see them and they cannot leak
# into the dataset. Without this the only markers on screen are whatever
# run_demo.sh left behind, which sit at the demo's goals and never move, so
# every episode looks like the robots are ignoring them.
_markers_spawned = False


def place_goal_markers(goals):
    """Put the two discs on this episode's goals.

    Tries to create them on the first call, but run_demo.sh may already have
    spawned markers under the same names -- in which case the create simply
    fails and the move below adopts the existing ones, which is what we want.
    Either way this is cosmetic: if it does not work the data is unaffected.
    """
    global _markers_spawned
    if not _markers_spawned:
        for n in AGENTS:
            try:
                snav.spawn_goal_marker(f'goal_{n}', goals[n][0], goals[n][1],
                                        snav.GOAL_MARKER_COLOR[n])
            except Exception:
                pass
        _markers_spawned = True
    for n in AGENTS:
        # Moving beats delete-and-respawn: one service call instead of two,
        # and no window where the marker is missing.
        try:
            snav.teleport(f'goal_{n}', goals[n][0], goals[n][1], z=snav.MARKER_Z)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(node, agents, gt, scenarios, episode_id, random_yaw_prob,
                expert_mode='hold'):
    starts = {n: scenarios[n][0] for n in AGENTS}
    goals = {n: scenarios[n][1] for n in AGENTS}
    paths = {n: scenarios[n][2] for n in AGENTS}

    # Face the first waypoint, or sometimes a random direction so the dataset
    # also contains "the goal is behind me" states -- otherwise the policy
    # never learns to turn around.
    place_goal_markers(goals)

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
    # The discrete action currently being executed, in 'hold' mode. Re-decided
    # at each recording boundary and held for the whole sample window, which is
    # how the agent experiences one env step.
    held_action = {n: None for n in AGENTS}
    done = {n: False for n in AGENTS}
    samples = []
    step_in_ep = {n: 0 for n in AGENTS}
    # Contacts are checked on EVERY control tick, not only on recording ticks,
    # so a brief scrape between two samples cannot slip through unnoticed.
    contacts = {n: 0 for n in AGENTS}
    min_clearance = {n: float('inf') for n in AGENTS}
    touching = {n: False for n in AGENTS}

    t0 = time.time()
    tick = 0
    while time.time() - t0 < MAX_EPISODE_SECONDS and not all(done.values()):
        snav.spin_for(node, snav.CONTROL_DT)
        tick += 1
        # Record on tick 1, not tick TICKS_PER_SAMPLE. Sampling only from tick
        # 11 onwards meant the robot had already spent 0.55 s turning toward
        # its goal before anything was written down, so the dataset never
        # contained a "the goal is behind me" state -- which is exactly the
        # state the policy most needs to learn to turn around from.
        record = ((tick - 1) % TICKS_PER_SAMPLE == 0)

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

            # --- contact check (same geometry jetbot_env uses) --------------
            clear = wall_clearance(px, py)
            other_pose = poses['jb_1' if name == 'jb_0' else 'jb_0']
            if other_pose is not None:
                gap_c = math.hypot(other_pose[0] - px, other_pose[1] - py)
                clear = min(clear, gap_c - ROBOT_RADIUS)
            min_clearance[name] = min(min_clearance[name], clear)
            if clear <= CONTACT_DIST:
                if not touching[name]:
                    contacts[name] += 1
                touching[name] = True
            else:
                touching[name] = False

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
            # cmd[] stays the CONTINUOUS controller state. The rate limiter
            # models actuator acceleration and needs a smooth signal to work
            # against; discretising it here would make it chatter. The
            # discretisation happens at the output only, below.
            cmd[name] = [lin, ang]

            # Which action the expert is executing this tick.
            #   hold  -> decided at the recording boundary, held until the next
            #   tick  -> re-decided every tick
            # In both cases the wheels receive that action's velocities, so the
            # demonstrated motion is inside the agent's action space.
            if expert_mode == 'hold':
                if record or held_action[name] is None:
                    held_action[name] = expert_action(lin, ang)
                action_id = held_action[name]
            else:
                action_id = expert_action(lin, ang)

            if expert_mode == 'continuous':
                out_lin, out_ang = lin, ang
            else:
                out_lin, out_ang = DISCRETE_ACTIONS[action_id]

            twist = Twist()
            twist.linear.x = out_lin
            twist.angular.z = out_ang
            agent.cmd_pub.publish(twist)

            if record:
                obs, d = build_observation(agent.latest_scan, agent.latest_odom,
                                            poses[name], goals[name])
                samples.append({
                    'obs': obs,
                    # The action actually executed, not the continuous command
                    # it was derived from -- otherwise the label would once
                    # again describe motion the agent cannot reproduce.
                    'action': action_id,
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

    return samples, done, starts, goals, contacts, min_clearance


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
        ep_contacts=np.asarray([e.get('contacts', 0) for e in episodes], dtype=np.int16),
        ep_min_clearance=np.asarray([e.get('min_clearance', 0.0) for e in episodes], dtype=np.float32),
    )
    csv_path = out_path.replace('.npz', '_episodes.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agent', 'start_x', 'start_y', 'goal_x', 'goal_y',
                    'distance', 'band', 'contacts', 'min_clearance', 'success'])
        for e in episodes:
            w.writerow([e['agent'], round(e['start'][0], 3), round(e['start'][1], 3),
                        round(e['goal'][0], 3), round(e['goal'][1], 3),
                        round(e['distance'], 3), e['band'],
                        e.get('contacts', 0),
                        round(e.get('min_clearance', 0.0), 3), e['success']])
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
        'contacts': int(d['ep_contacts'][i]) if 'ep_contacts' in d else 0,
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
    ap.add_argument('--random-yaw-prob', type=float, default=0.5,
                    help='fraction of spawns facing a random direction')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--max-attempt-factor', type=float, default=3.0,
                    help='give up after episodes x this many attempts '
                         '(rejected runs still cost time)')
    ap.add_argument('--expert-mode', choices=EXPERT_MODES, default='hold',
                    help="what actually reaches the wheels. 'hold' (default) "
                         "drives the agent's own 4 discrete actions, one held "
                         "per sample window -- demonstrations the agent can "
                         "reproduce. 'tick' re-picks every control tick. "
                         "'continuous' is the original behaviour, which is "
                         "what made the first BC attempt spin.")
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
    print(f"Target: {args.episodes} ACCEPTED episodes "
          f"(both robots reach their goal, neither touches anything).")
    print(f"Goal distance ramps {GOAL_DIST_START:.1f} m -> {GOAL_DIST_MAX:.1f} m, "
          f"then the last {100*CORNER_PHASE_FRACTION:.0f}% are corner-to-corner "
          f"crossings.")
    print(f"Rejected episodes are DISCARDED, so the dataset contains only clean "
          f"runs.\n")

    accepted = 0
    attempts = 0
    rejected = {'not_reached': 0, 'contact': 0, 'no_scenario': 0}
    max_attempts = args.episodes * args.max_attempt_factor

    try:
        while accepted < args.episodes and attempts < max_attempts:
            attempts += 1
            band = curriculum_band(accepted, args.episodes)
            scen = sample_two_scenarios(band)
            if scen is None:
                rejected['no_scenario'] += 1
                print(f"[{accepted}/{args.episodes}] band={band[0]:<10} "
                      f"REJECT  no route found")
                continue

            new, done, starts, goals, contacts, min_clear = run_episode(
                node, agents, gt, scen, ep_offset + accepted, args.random_yaw_prob,
                expert_mode=args.expert_mode)

            both_reached = all(done[n] for n in AGENTS)
            total_contacts = sum(contacts[n] for n in AGENTS)
            worst = min(min_clear[n] for n in AGENTS)

            if not both_reached or total_contacts > 0:
                # Throw the whole episode away -- including the robot that DID
                # succeed. A trajectory recorded next to a robot that crashed is
                # still a trajectory the policy should not copy.
                reason = 'contact' if total_contacts else 'not_reached'
                rejected[reason] += 1
                detail = (f"{total_contacts} contact(s), worst clearance {worst:.3f}m"
                          if total_contacts
                          else f"reached {sum(done[n] for n in AGENTS)}/2")
                print(f"[{accepted}/{args.episodes}] band={band[0]:<10} "
                      f"REJECT  {detail}")
                continue

            samples.extend(new)
            for i, n in enumerate(AGENTS):
                episodes.append({
                    'start': starts[n], 'goal': goals[n], 'agent': i,
                    'success': 1, 'band': band[0],
                    'contacts': 0,
                    'min_clearance': float(min_clear[n]),
                    'distance': math.hypot(goals[n][0] - starts[n][0],
                                            goals[n][1] - starts[n][1]),
                })
            accepted += 1
            d0 = math.hypot(goals['jb_0'][0] - starts['jb_0'][0],
                            goals['jb_0'][1] - starts['jb_0'][1])
            d1 = math.hypot(goals['jb_1'][0] - starts['jb_1'][0],
                            goals['jb_1'][1] - starts['jb_1'][1])
            print(f"[{accepted}/{args.episodes}] band={band[0]:<10} ACCEPT  "
                  f"dist {d0:.2f}/{d1:.2f}m  clearance {worst:.3f}m  "
                  f"samples+={len(new):<4} total={len(samples)}")

            if not args.dry_run and accepted % args.save_every == 0:
                save(args.out, samples, episodes)

        if accepted < args.episodes:
            print(f"\nStopped after {attempts} attempts with {accepted} accepted. "
                  f"Raise --max-attempt-factor if you want more.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # rclpy may already be torn down after Ctrl+C, in which case publishing
        # raises RCLError and hides whatever actually happened. Saving the data
        # matters more than the stop command.
        for agent in agents.values():
            try:
                agent.publish_stop()
            except Exception:
                pass
        print(f"\nAccepted {accepted} / attempted {attempts}")
        print(f"  rejected -- did not reach goal : {rejected['not_reached']}")
        print(f"  rejected -- made contact       : {rejected['contact']}")
        print(f"  rejected -- no route sampled   : {rejected['no_scenario']}")
        if not args.dry_run:
            save(args.out, samples, episodes)
        gt.stop()
        rclpy.shutdown()

    print("\nDone. Inspect it with:")
    print(f"    python3 dataset_tools.py summary {args.out}")


if __name__ == '__main__':
    main()
