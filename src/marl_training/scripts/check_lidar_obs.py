"""
check_lidar_obs.py  --  is the LIDAR normalisation fix actually live?

READ-ONLY. Subscribes to /jb_0/scan and /jb_1/scan and nothing else. It never
teleports, never publishes cmd_vel, never calls env.reset(). Safe to run while
train_mappo.py is training.

    python3 check_lidar_obs.py
    python3 check_lidar_obs.py --samples 60      # watch for longer

WHAT IT PROVES
--------------
It rebuilds the 12 LIDAR sector values exactly the way jetbot_env does, using
the SAME constants imported from jetbot_env, and reports the spread.

  BROKEN (/12.0) : values sit in roughly 0.02 - 0.24,  std ~0.035
  FIXED  (/3.5)  : values sit in roughly 0.07 - 0.82,  std ~0.120

The std is the number that matters. The goal-angle input spans -1..+1 with a
std around 0.5, so if the LIDAR std is 0.035 the network hears the obstacle
inputs about 15x quieter than the goal inputs and learns to ignore walls.
"""

import argparse
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from jetbot_env import NUM_LIDAR_SECTORS, MAX_LIDAR_RANGE, LIDAR_NORM

AGENTS = ('jb_0', 'jb_1')


def sectors_from_scan(scan):
    """Byte-for-byte the same computation jetbot_env._build_observation does."""
    ranges = scan.ranges
    n = len(ranges)
    sector_size = max(1, n // NUM_LIDAR_SECTORS)
    out = []
    for i in range(NUM_LIDAR_SECTORS):
        chunk = [r for r in ranges[i * sector_size:(i + 1) * sector_size]
                 if not math.isinf(r) and not math.isnan(r)]
        min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
        min_r = min(min_r, MAX_LIDAR_RANGE)
        out.append((min_r, min(min_r / LIDAR_NORM, 1.0)))
    return out


class Listener(Node):
    def __init__(self):
        super().__init__('check_lidar_obs')
        self.latest = {n: None for n in AGENTS}
        for n in AGENTS:
            self.create_subscription(
                LaserScan, f'/{n}/scan',
                lambda msg, name=n: self.latest.__setitem__(name, msg), 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=25)
    args = ap.parse_args()

    print(f"jetbot_env constants:  NUM_LIDAR_SECTORS={NUM_LIDAR_SECTORS}  "
          f"MAX_LIDAR_RANGE={MAX_LIDAR_RANGE}  LIDAR_NORM={LIDAR_NORM}")
    if abs(LIDAR_NORM - MAX_LIDAR_RANGE) < 1e-9:
        print("  -> LIDAR_NORM still equals MAX_LIDAR_RANGE. The fix is NOT applied.")
    print()

    rclpy.init()
    node = Listener()

    t0 = time.time()
    while not all(node.latest[n] is not None for n in AGENTS):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 10.0:
            print("No scan data. Is the simulation running?")
            node.destroy_node()
            rclpy.shutdown()
            return

    raw_all, norm_all = [], []
    shown = 0
    print(f"{'#':>3}  {'raw metres (min..max)':<24}  {'normalised (min..max)':<24}  n")
    for k in range(args.samples):
        rclpy.spin_once(node, timeout_sec=0.5)
        for name in AGENTS:
            scan = node.latest[name]
            if scan is None:
                continue
            pairs = sectors_from_scan(scan)
            raw = [p[0] for p in pairs]
            norm = [p[1] for p in pairs]
            raw_all.extend(raw)
            norm_all.extend(norm)
            if shown < 6:
                print(f"{shown+1:>3}  {min(raw):>6.2f} .. {max(raw):<15.2f}  "
                      f"{min(norm):>6.3f} .. {max(norm):<15.3f}  {len(norm)}")
                shown += 1
        time.sleep(0.1)

    print()
    print("=" * 64)
    print(f"{len(norm_all)} sector readings collected")
    print("=" * 64)
    print(f"  raw metres        min={min(raw_all):.2f}  "
          f"median={statistics.median(raw_all):.2f}  max={max(raw_all):.2f}")
    print(f"  normalised        min={min(norm_all):.3f}  "
          f"median={statistics.median(norm_all):.3f}  max={max(norm_all):.3f}")
    std = statistics.pstdev(norm_all)
    print(f"  normalised std    {std:.3f}")
    print(f"  input range used  {100*(max(norm_all)-min(norm_all)):.0f}% of 0..1")
    print()

    if std >= 0.08:
        print("  VERDICT: FIXED. The obstacle inputs now vary enough for the")
        print("           network to actually learn from them.")
    elif std >= 0.05:
        print("  VERDICT: better than /12.0, but weaker than expected.")
        print("           Likely the robots are parked in open space right now.")
        print("           Re-run while they are moving near walls.")
    else:
        print("  VERDICT: STILL SQUASHED. Check that jetbot_env.py line ~201 reads")
        print("           sectors.append(min(min_r / LIDAR_NORM, 1.0))")
        print("           and that nothing reassigns LIDAR_NORM further down.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
