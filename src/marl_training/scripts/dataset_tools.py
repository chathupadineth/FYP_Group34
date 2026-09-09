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
import sys

import numpy as np

BANDS = ['easy', 'medium', 'hard', 'very_hard']

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
        if nm.startswith('lidar') and col.max() < 0.5:
            note = '<-- never uses upper range; normaliser too big'
        if nm == 'dist_to_goal' and col.max() >= 0.999:
            note = '<-- saturating at 1.0 (clipped)'
        if nm == 'angle_to_goal' and abs(col).max() < 0.4:
            note = '<-- goal almost always ahead; add random spawn yaw'
        print(f"  {nm:<15} {col.min():8.3f} {col.max():8.3f} {col.mean():8.3f}   {note}")

    lidar = obs[:, :12]
    print(f"\n  all 12 lidar values together: min {lidar.min():.3f}  max {lidar.max():.3f}")
    if lidar.max() < 0.5:
        print("  >> The lidar inputs only use a small part of 0..1.")
        print("     jetbot_env divides by 12.0 but the room is ~3 m.")
        print("     Dividing by ~3.5 (with min(x,1.0)) would spread them properly.")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['summary', 'pool'])
    ap.add_argument('path')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--band', default=None, choices=BANDS)
    args = ap.parse_args()

    if args.command == 'summary':
        summary(args.path)
    else:
        show_pool(args.path, args.n, args.band)


if __name__ == '__main__':
    main()
