"""
diagnose_collision.py

Finds out WHY the environment reports a collision when the robot has not hit
anything, e.g.

    [jb_1] COLLISION triggered (forward-facing) - dist=0.185m

right after a reset.

METHOD
------
Park the robot in the most open spot on the map, with the other robot moved
far away, so we know for certain that nothing is close to it. Then rotate it
through 360 degrees in steps and, at each angle, compare:

    what the LIDAR reports   vs   what the map says it SHOULD report

The map distance is computed by ray-casting against the real wall and building
rectangles, so it is exact.

Then the answer is simple:

  * If the LIDAR reads far shorter than the map says, in EVERY direction the
    robot turns, then it is seeing something attached to itself -- its own
    chassis. The short reading sits at a FIXED bearing in the robot's frame.

  * If the short readings stay at fixed WORLD bearings instead, the robot is
    genuinely near something and the map or the position is wrong.

Run it with the simulator up:
    python3 diagnose_collision.py
"""

import math
import time

import rclpy
from rclpy.node import Node

from jetbot_env import (
    JetBotAgent,
    NUM_LIDAR_SECTORS,
    MAX_LIDAR_RANGE,
    COLLISION_DIST,
    FORWARD_HALF_ANGLE,
    WORLD_NAME,
)
from spawn_utils import (
    OBSTACLES,
    PLATFORM_X_MIN, PLATFORM_X_MAX,
    PLATFORM_Y_MIN, PLATFORM_Y_MAX,
    ROBOT_CLEARANCE,
)
import scripted_nav_eval as snav
from pose_source import GroundTruthTracker

ROBOT_HALF_DIAGONAL = math.hypot(0.075, 0.065)     # chassis is 0.15 x 0.13 m
YAW_STEPS = 12                                      # every 30 degrees
PARK_OTHER_AT = (0.45, 1.75)                        # far corner for the other robot


# ---------------------------------------------------------------------------
# Ray-casting against the true map (no clearance inflation -- real geometry)
# ---------------------------------------------------------------------------
def _ray_box(px, py, dx, dy, xmin, xmax, ymin, ymax):
    tmin, tmax = 0.0, float('inf')
    for p, d, lo, hi in ((px, dx, xmin, xmax), (py, dy, ymin, ymax)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None
        else:
            t1, t2 = (lo - p) / d, (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    if tmin > 1e-9:
        return tmin
    return tmax if tmax > 1e-9 else None


def map_ray_distance(px, py, world_bearing):
    """How far the laser SHOULD travel, according to the known map."""
    dx, dy = math.cos(world_bearing), math.sin(world_bearing)
    best = MAX_LIDAR_RANGE

    # the surrounding walls (robot is inside the room, so the ray exits)
    ts = []
    if dx > 1e-12:
        ts.append((PLATFORM_X_MAX - px) / dx)
    elif dx < -1e-12:
        ts.append((PLATFORM_X_MIN - px) / dx)
    if dy > 1e-12:
        ts.append((PLATFORM_Y_MAX - py) / dy)
    elif dy < -1e-12:
        ts.append((PLATFORM_Y_MIN - py) / dy)
    for t in ts:
        if t > 0:
            best = min(best, t)

    for (ox, oy, hw, hh) in OBSTACLES:
        t = _ray_box(px, py, dx, dy, ox - hw, ox + hw, oy - hh, oy + hh)
        if t is not None:
            best = min(best, t)

    return best


def clearance_at(x, y):
    """Distance from (x, y) to the nearest wall or building face."""
    d = min(x - PLATFORM_X_MIN, PLATFORM_X_MAX - x,
            y - PLATFORM_Y_MIN, PLATFORM_Y_MAX - y)
    for (ox, oy, hw, hh) in OBSTACLES:
        ddx = max(ox - hw - x, x - (ox + hw), 0.0)
        ddy = max(oy - hh - y, y - (oy + hh), 0.0)
        d = min(d, math.hypot(ddx, ddy) if (ddx > 0 or ddy > 0) else -1.0)
    return d


def most_open_point():
    best, best_c = None, -1.0
    x = PLATFORM_X_MIN
    while x <= PLATFORM_X_MAX:
        y = PLATFORM_Y_MIN
        while y <= PLATFORM_Y_MAX:
            c = clearance_at(x, y)
            if c > best_c:
                best, best_c = (x, y), c
            y += 0.02
        x += 0.02
    return best, best_c


# ---------------------------------------------------------------------------
def sector_bearings(scan):
    """Centre bearing of each of the 12 sectors, exactly as jetbot_env cuts them."""
    n = len(scan.ranges)
    size = max(1, n // NUM_LIDAR_SECTORS)
    out = []
    for i in range(NUM_LIDAR_SECTORS):
        b = scan.angle_min + (i * size + size / 2.0) * scan.angle_increment
        out.append(math.atan2(math.sin(b), math.cos(b)))
    return out


def sector_minima(scan):
    n = len(scan.ranges)
    size = max(1, n // NUM_LIDAR_SECTORS)
    out = []
    for i in range(NUM_LIDAR_SECTORS):
        chunk = [r for r in scan.ranges[i * size:(i + 1) * size]
                 if not math.isinf(r) and not math.isnan(r)]
        out.append(min(min(chunk), MAX_LIDAR_RANGE) if chunk else MAX_LIDAR_RANGE)
    return out


def forward_min(scan):
    """Exactly what jetbot_env calls relevant_min_dist when driving forward."""
    fm = MAX_LIDAR_RANGE
    for b, m in zip(sector_bearings(scan), sector_minima(scan)):
        if abs(b) <= FORWARD_HALF_ANGLE:
            fm = min(fm, m)
    return fm


def short_ray_bearings(scan, limit=0.35):
    """Bearings (robot frame, degrees) of every raw ray closer than `limit`."""
    out = []
    for i, r in enumerate(scan.ranges):
        if math.isinf(r) or math.isnan(r) or r >= limit:
            continue
        b = scan.angle_min + i * scan.angle_increment
        out.append((math.degrees(math.atan2(math.sin(b), math.cos(b))), r))
    return out


# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("STEP 1  --  find the most open spot on the map")
    print("=" * 72)
    spot, clear = most_open_point()
    print(f"  position ({spot[0]:.2f}, {spot[1]:.2f})")
    print(f"  nearest wall/building is {clear:.3f} m away")
    print(f"  the robot's own half-size is {ROBOT_HALF_DIAGONAL:.3f} m, so there is")
    print(f"  {clear - ROBOT_HALF_DIAGONAL:.3f} m of genuinely empty space around it.")
    if clear < 0.6:
        print("  WARNING: this map has no really open area; results will be less clear.")

    rclpy.init()
    node = Node('diagnose_collision')
    agents = {n: JetBotAgent(node, n) for n in ('jb_0', 'jb_1')}

    print("\nWaiting for sensor data...")
    t0 = time.time()
    while not all(a.has_fresh_data() for a in agents.values()):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 8.0:
            raise TimeoutError("No LIDAR/odometry after 8s - is the sim running?")

    truth = GroundTruthTracker(WORLD_NAME, ['jb_0', 'jb_1'])
    have_truth = truth.start() and truth.wait_for(timeout=4.0)

    print("\n" + "=" * 72)
    print("STEP 2  --  park jb_0 there, move jb_1 far away")
    print("=" * 72)
    snav.teleport('jb_1', PARK_OTHER_AT[0], PARK_OTHER_AT[1], yaw=0.0)
    snav.teleport('jb_0', spot[0], spot[1], yaw=0.0)
    time.sleep(0.8)
    snav.spin_for(node, 0.5)
    print(f"  jb_1 parked at {PARK_OTHER_AT}, "
          f"{math.hypot(PARK_OTHER_AT[0]-spot[0], PARK_OTHER_AT[1]-spot[1]):.2f} m away")

    print("\n" + "=" * 72)
    print("STEP 3  --  rotate 360 degrees; LIDAR vs what the map says")
    print("=" * 72)
    print(f"  (jetbot_env calls it a collision when the forward minimum <= "
          f"{COLLISION_DIST:.2f} m)\n")
    print(f"  {'yaw':>5}  {'LIDAR fwd':>10} {'map says':>10} {'shortfall':>10}   collision?")

    false_hits = 0
    lidar_mins, robot_frame_min_bearings = [], []

    for k in range(YAW_STEPS):
        yaw = 2 * math.pi * k / YAW_STEPS
        snav.teleport('jb_0', spot[0], spot[1], yaw=yaw)
        time.sleep(0.45)
        snav.spin_for(node, 0.45)

        scan = agents['jb_0'].latest_scan
        if scan is None:
            continue

        p = truth.get('jb_0') if have_truth else (spot[0], spot[1], yaw)
        px, py, real_yaw = p if p else (spot[0], spot[1], yaw)

        lid = forward_min(scan)
        mapped = min(map_ray_distance(px, py, real_yaw + b)
                     for b in sector_bearings(scan) if abs(b) <= FORWARD_HALF_ANGLE)

        hit = lid <= COLLISION_DIST
        false_hits += int(hit)
        lidar_mins.append(lid)

        mins = sector_minima(scan)
        bs = sector_bearings(scan)
        robot_frame_min_bearings.append(math.degrees(bs[mins.index(min(mins))]))

        print(f"  {math.degrees(yaw):5.0f}  {lid:10.3f} {mapped:10.3f} "
              f"{mapped - lid:10.3f}   {'*** YES ***' if hit else 'no'}")

    print("\n" + "=" * 72)
    print("STEP 4  --  where are the short readings pointing?")
    print("=" * 72)
    print("  Bearing of the closest sector, in the ROBOT's own frame, at each yaw:")
    print("   ", [f"{b:+.0f}" for b in robot_frame_min_bearings])
    spread = (max(robot_frame_min_bearings) - min(robot_frame_min_bearings)
              if robot_frame_min_bearings else 0.0)
    print(f"  spread = {spread:.0f} degrees")

    scan = agents['jb_0'].latest_scan
    shorts = short_ray_bearings(scan, limit=0.35)
    print(f"\n  Raw rays closer than 0.35 m at this orientation: {len(shorts)} of {len(scan.ranges)}")
    if shorts:
        print("    bearing(deg) : range(m)")
        for b, r in shorts[:24]:
            print(f"      {b:+7.1f}   : {r:.3f}")
        if len(shorts) > 24:
            print(f"      ... and {len(shorts) - 24} more")

    # ------------------------------------------------------------- verdict --
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    worst = min(lidar_mins) if lidar_mins else float('nan')
    print(f"  In a spot with {clear:.2f} m of clear space all around:")
    print(f"    smallest forward reading seen : {worst:.3f} m")
    print(f"    collision threshold            : {COLLISION_DIST:.3f} m")
    print(f"    false collisions               : {false_hits} of {len(lidar_mins)} orientations")
    print()

    if worst < clear - 0.15:
        print("  >> The LIDAR reports something FAR closer than anything that")
        print("     actually exists. It is seeing the robot itself.")
        if spread < 90:
            print(f"     The closest reading also stays near a fixed bearing in the")
            print(f"     robot's own frame (spread only {spread:.0f} deg) -- that")
            print("     confirms it: the obstacle turns with the robot.")
    else:
        print("  >> The LIDAR roughly agrees with the map. Self-detection is NOT")
        print("     the problem; the collision threshold vs spawn clearance is.")

    print()
    print(f"  Either way, note the margins:")
    print(f"    real contact happens at about   {ROBOT_HALF_DIAGONAL:.3f} m")
    print(f"    COLLISION_DIST is set to        {COLLISION_DIST:.3f} m")
    print(f"    spawns are placed at least      {ROBOT_CLEARANCE:.3f} m from obstacles")
    print(f"    -> only {ROBOT_CLEARANCE - COLLISION_DIST:.3f} m of margin between a legal")
    print(f"       spawn and being declared a collision.")
    print()

    truth.stop()
    for a in agents.values():
        a.publish_stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
