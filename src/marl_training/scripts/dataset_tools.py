"""
dataset_tools.py

Inspect the dataset made by collect_dataset.py, and turn it into a reset pool
for training (the instructor's "Pattern A -- diverse reset states").

USAGE
    python3 dataset_tools.py summary nav_dataset.npz
    python3 dataset_tools.py pool    nav_dataset.npz --n 20 --band medium

USE INSIDE TRAINING
    from dataset_tools import ResetPool
    pool = ResetPool('nav_dataset.npz', successful_only=True)
    start, goal = pool.sample(band='easy')          # or band=None for any
"""

import argparse
import math
import os
import sys

import numpy as np

# Old datasets used easy/medium/hard/very_hard; new ones use the curriculum
# labels written by collect_dataset.curriculum_band().
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'nav_dataset.npz')

BANDS = ['easy', 'medium', 'hard', 'very_hard',
         'short', 'long', 'very_long', 'corner']

OBS_NAMES = ([f'lidar_{i+1:02d}' for i in range(12)]
             + ['dist_to_goal', 'angle_to_goal', 'vx', 'vz'])


def load(path):
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
def summary(path):
    d = load(path)
    obs = d['obs']
    act = d['action']

    print(f"=== {path} ===")
    print(f"samples      : {len(obs)}")
    print(f"robot-runs   : {len(d['ep_start_x'])}")
    print(f"observation  : {obs.shape[1]} values per sample")
    success = d['ep_success']
    print(f"success rate : {100.0 * success.mean():.1f}%  "
          f"({int(success.sum())}/{len(success)} runs reached the goal)")

    print("\n--- action balance (what the expert did) ---")
    names = ['forward', 'left', 'right', 'backward']
    for a in range(4):
        c = int((act == a).sum())
        pct = 100.0 * c / max(1, len(act))
        bar = '#' * int(pct / 2)
        print(f"  {a} {names[a]:<9} {c:>7}  {pct:5.1f}%  {bar}")
    missing = [names[a] for a in range(4) if (act == a).sum() == 0]
    if missing:
        print(f"  WARNING: expert never used: {', '.join(missing)}")
        print("           the policy cannot learn an action it never sees.")

    print("\n--- runs per difficulty band ---")
    bands = d['ep_band']
    for b in BANDS:
        sel = bands == b
        n = int(sel.sum())
        if n:
            sr = 100.0 * d['ep_success'][sel].mean()
            print(f"  {b:<10} {n:>5} runs   success {sr:5.1f}%")

    print("\n--- observation ranges (this is the important check) ---")
    print(f"  {'value':<15} {'min':>8} {'max':>8} {'mean':>8}   note")
    for i, nm in enumerate(OBS_NAMES):
        col = obs[:, i]
        note = ''
        if nm.startswith('lidar') and col.std() < 0.05:
            note = '<-- squashed; normaliser too big'
        if nm == 'dist_to_goal' and col.max() >= 0.999:
            note = '<-- saturating at 1.0 (clipped)'
        if nm == 'angle_to_goal' and abs(col).max() < 0.4:
            note = '<-- goal almost always ahead; add random spawn yaw'
        print(f"  {nm:<15} {col.min():8.3f} {col.max():8.3f} {col.mean():8.3f}   {note}")

    lidar = obs[:, :12]
    print(f"\n  all 12 lidar values together: min {lidar.min():.3f}  "
          f"max {lidar.max():.3f}  std {lidar.std():.3f}")
    # std is the number that matters, not max: the goal-angle input has a std
    # around 0.3-0.5, so anything much below that is drowned out.
    if lidar.std() < 0.05:
        print("  >> SQUASHED. These barely vary, so the network cannot learn from")
        print("     them. Check LIDAR_NORM in jetbot_env.py (should be ~3.5).")
    else:
        print("  >> Spread looks healthy for a ~3 m room.")

    ang = obs[:, 13]
    if abs(ang).max() < 0.7:
        print(f"\n  angle_to_goal only reaches {abs(ang).max():.2f} of 1.0.")
        print("     The dataset has no 'goal is behind me' states, so the clone")
        print("     never learns to turn around. Raise --random-yaw-prob.")

    print("\n--- map coverage ---")
    print(f"  start x: {d['ep_start_x'].min():.2f} .. {d['ep_start_x'].max():.2f}")
    print(f"  start y: {d['ep_start_y'].min():.2f} .. {d['ep_start_y'].max():.2f}")
    print(f"  goal  x: {d['ep_goal_x'].min():.2f} .. {d['ep_goal_x'].max():.2f}")
    print(f"  goal  y: {d['ep_goal_y'].min():.2f} .. {d['ep_goal_y'].max():.2f}")
    print(f"  distance: {d['ep_distance'].min():.2f} .. {d['ep_distance'].max():.2f} m")

    _coverage_map(d)


def _coverage_map(d, w=46, h=18):
    """Rough picture of which parts of the map the starts actually covered."""
    xs, ys = d['ep_start_x'], d['ep_start_y']
    x0, x1 = 0.175, 3.161
    y0, y1 = -1.232, 2.019
    grid = [[0] * w for _ in range(h)]
    for x, y in zip(xs, ys):
        c = min(w - 1, max(0, int((x - x0) / (x1 - x0) * (w - 1))))
        r = min(h - 1, max(0, int((y1 - y) / (y1 - y0) * (h - 1))))
        grid[r][c] += 1
    print("\n  start-position coverage (. none, o few, # many):")
    for row in grid:
        print('    ' + ''.join('.' if v == 0 else ('o' if v < 3 else '#') for v in row))


# ---------------------------------------------------------------------------
class ResetPool:
    """Diverse (start, goal) pairs harvested from the scripted runs.

    Use it in place of random spawning so the agent gets a spread of easy and
    hard scenarios instead of whatever chance produces.
    """

    def __init__(self, path, successful_only=True):
        d = load(path)
        keep = d['ep_success'] == 1 if successful_only else np.ones(len(d['ep_success']), bool)
        self.starts = list(zip(d['ep_start_x'][keep], d['ep_start_y'][keep]))
        self.goals = list(zip(d['ep_goal_x'][keep], d['ep_goal_y'][keep]))
        self.bands = list(d['ep_band'][keep])
        self.distances = list(d['ep_distance'][keep])
        if not self.starts:
            raise RuntimeError(f"No usable runs in {path}")

    def __len__(self):
        return len(self.starts)

    def sample(self, band=None, max_distance=None, rng=None):
        """Returns ((start_x, start_y), (goal_x, goal_y))."""
        import random as _r
        rng = rng or _r
        idx = range(len(self.starts))
        if band is not None:
            idx = [i for i in idx if self.bands[i] == band]
        if max_distance is not None:
            idx = [i for i in idx if self.distances[i] <= max_distance]
        if not idx:
            idx = range(len(self.starts))       # fall back rather than crash
        i = rng.choice(list(idx))
        return ((float(self.starts[i][0]), float(self.starts[i][1])),
                (float(self.goals[i][0]), float(self.goals[i][1])))

    def counts(self):
        return {b: sum(1 for x in self.bands if x == b) for b in BANDS}


# ---------------------------------------------------------------------------
def show_pool(path, n, band):
    pool = ResetPool(path)
    print(f"pool size: {len(pool)}   per band: {pool.counts()}\n")
    for _ in range(n):
        s, g = pool.sample(band=band)
        print(f"  start ({s[0]:6.2f},{s[1]:6.2f})  ->  goal ({g[0]:6.2f},{g[1]:6.2f})"
              f"   d={math.hypot(g[0]-s[0], g[1]-s[1]):.2f}m")


# ---------------------------------------------------------------------------
# Looking at the actual state vectors
# ---------------------------------------------------------------------------
# The dataset stores 16 values PER ROBOT PER STEP -- that is what the Actor
# sees. The 32-value vector is not stored anywhere: it only exists inside
# train_mappo.py, where it is built for the Critic as
#
#     joint_obs = obs['jb_0'] + obs['jb_1']
#
# so it is rebuilt here the same way, by pairing the two robots' rows from the
# same (episode, step). Rows stop being paired once a robot reaches its goal,
# because the collector stops recording it from that point on.
ACTION_NAMES = ['forward', 'left', 'right', 'backward']


def _joint_rows(d):
    """Yield (episode, step, obs0, obs1, act0, act1) for steps where BOTH
    robots were still being recorded."""
    obs, act = d['obs'], d['action']
    ep, ag, st = d['episode'], d['agent'], d['step']
    by_key = {}
    for i in range(len(obs)):
        by_key[(int(ep[i]), int(st[i]), int(ag[i]))] = i
    seen = sorted({(int(ep[i]), int(st[i])) for i in range(len(obs))})
    for e, s in seen:
        i0 = by_key.get((e, s, 0))
        i1 = by_key.get((e, s, 1))
        if i0 is None or i1 is None:
            continue
        yield e, s, obs[i0], obs[i1], int(act[i0]), int(act[i1])


def dump_all_rows(path, csv_out):
    """Every stored row, one per robot per step -- the raw 16 values the Actor
    sees. Unlike the paired view this includes steps where only one robot was
    still being recorded, so it is the complete dataset."""
    import csv as _csv
    d = load(path)
    obs, act = d['obs'], d['action']
    ep, ag, st = d['episode'], d['agent'], d['step']
    extra = [k for k in ('x', 'y', 'yaw', 'goal_x', 'goal_y', 'dist_to_goal')
             if k in d.files]
    with open(csv_out, 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['episode', 'agent', 'step', 'time_s'] + OBS_NAMES
                   + ['expert_action'] + extra)
        for i in range(len(obs)):
            w.writerow([int(ep[i]), f"jb_{int(ag[i])}", int(st[i]),
                        round(float(st[i]) * 0.55, 2)]
                       + [f'{v:.4f}' for v in obs[i]]
                       + [ACTION_NAMES[int(act[i])]]
                       + [f'{float(d[k][i]):.4f}' for k in extra])
    print(f"Wrote {csv_out}")
    print(f"  {len(obs)} rows x {4 + len(OBS_NAMES) + 1 + len(extra)} columns")
    print(f"  every stored sample: {int(ep.max())+1} episodes, both robots, all steps")


def states(path, episode=None, limit=6, csv_out=None):
    d = load(path)
    rows = list(_joint_rows(d))
    if episode is not None:
        rows = [r for r in rows if r[0] == episode]
        if not rows:
            eps = sorted({r[0] for r in _joint_rows(d)})
            raise SystemExit(f"No paired rows for episode {episode}. "
                             f"Available: {eps[:20]}{' ...' if len(eps) > 20 else ''}")

    total16 = len(d['obs'])
    print(f"=== {path} ===")
    print(f"stored rows        : {total16}  (16 values each, one robot per row)")
    print(f"paired steps       : {len(rows)}  (both robots recorded -> a 32-value joint state)")
    print(f"unpaired rows      : {total16 - 2*len(rows)}  (one robot had already "
          f"reached its goal)\n")

    if csv_out:
        import csv as _csv
        with open(csv_out, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['episode', 'step']
                       + [f'jb0_{n}' for n in OBS_NAMES]
                       + [f'jb1_{n}' for n in OBS_NAMES]
                       + ['jb0_action', 'jb1_action'])
            for e, s, o0, o1, a0, a1 in rows:
                w.writerow([e, s] + [f'{v:.4f}' for v in o0]
                           + [f'{v:.4f}' for v in o1]
                           + [ACTION_NAMES[a0], ACTION_NAMES[a1]])
        print(f"Wrote {csv_out}  ({len(rows)} rows x 34 columns) "
              f"-- open it in LibreOffice/Excel\n")

    for e, s, o0, o1, a0, a1 in rows[:limit]:
        print(f"--- episode {e}, step {s} "
              f"(t = {s*0.55:.2f}s) ---------------------------------")
        print(f"{'':<16}{'jb_0':>10}{'jb_1':>10}")
        for j, name in enumerate(OBS_NAMES):
            mark = ''
            if name.startswith('lidar') and min(o0[j], o1[j]) < 0.15:
                mark = '  <- something close'
            print(f"  [{j:>2}] {name:<12}{o0[j]:>10.3f}{o1[j]:>10.3f}{mark}")
        print(f"  expert action  {ACTION_NAMES[a0]:>10}{ACTION_NAMES[a1]:>10}")
        print(f"  joint vector for the Critic: 32 values "
              f"= jb_0[0:16] ++ jb_1[0:16]\n")

    if len(rows) > limit:
        print(f"({len(rows) - limit} more paired steps -- use --limit, "
              f"--episode, or --csv to see them all)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['summary', 'pool', 'states', 'dump'])
    ap.add_argument('path', nargs='?', default=DEFAULT_PATH,
                    help='dataset file (default: nav_dataset.npz beside this script)')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--band', default=None, choices=BANDS)
    ap.add_argument('--episode', type=int, default=None,
                    help='states: show only this episode')
    ap.add_argument('--limit', type=int, default=6,
                    help='states: how many paired steps to print')
    ap.add_argument('--csv', default=None,
                    help='states: also write every paired step to this CSV file')
    args = ap.parse_args()

    if args.command == 'summary':
        summary(args.path)
    elif args.command == 'states':
        states(args.path, args.episode, args.limit, args.csv)
    elif args.command == 'dump':
        dump_all_rows(args.path, args.csv or 'nav_dataset_all_rows.csv')
    else:
        show_pool(args.path, args.n, args.band)


if __name__ == '__main__':
    main()
