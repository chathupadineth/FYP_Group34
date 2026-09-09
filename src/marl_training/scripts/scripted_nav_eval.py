"""
scripted_nav_eval.py

Two JetBots drive to two goals in the existing Gazebo world, using ONLY
hand-written rules. No RL, no neural networks, no training, no learning of
any kind -- A* runs once over the known map, then a fixed control law is
evaluated each tick.

  1. PLAN ONCE: A* over the known static map (spawn_utils.py's OBSTACLES +
     platform bounds) -> a fixed waypoint route per robot.
  2. FOLLOW: each tick, a pure function maps (world pose, current waypoint)
     -> (linear, angular) velocity, with smooth acceleration limits.
  3. AVOID EACH OTHER: using both robots' true world positions.

--------------------------------------------------------------------------
IMPORTANT -- WHY THIS DOES NOT USE /model/<name>/odometry FOR POSITION
--------------------------------------------------------------------------
The DiffDrive plugin (gz-sim-diff-drive-system) publishes WHEEL ODOMETRY in
the `odom` frame. That frame starts at (0, 0) wherever the robot happens to
be -- it is NOT the world frame. Treating it as a world position makes a
robot at (0.5, -1.0) believe it is standing at the origin, so it drives off
toward a goal that is in world coordinates and ploughs into a wall. Worse,
wheel odometry keeps integrating while the wheels spin against that wall, so
the robot "reaches" its goal on paper while physically stuck.

So position comes from Gazebo's ground-truth pose stream instead
(/world/<world>/dynamic_pose/info), with odometry used only as a fallback,
dead-reckoned from the known teleport pose.

NOTE FOR jetbot_env.py: _build_observation() reads odom.pose.pose.position
as if it were a world position and compares it against world-frame goals.
That is the same bug, and it means the RL agent's distance/angle-to-goal
inputs have been wrong the whole time. Worth checking before more training.

Topics used (same as jetbot_env.py):
    /jb_0/scan, /jb_1/scan                       (LaserScan) -- logging only
    /model/jb_0/odometry, /model/jb_1/odometry   (Odometry)  -- fallback only
    /jb_0/cmd_vel, /jb_1/cmd_vel                  (Twist)

Prerequisite: the sim must already be running:
    ros2 launch jetbot_description spawn_two_jetbots.launch.py

Run (from this scripts/ folder, same as eval_run.py):
    python3 scripted_nav_eval.py
"""

import heapq
import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from jetbot_env import (
    JetBotAgent,
    NUM_LIDAR_SECTORS,
    MAX_LIDAR_RANGE,
    FORWARD_HALF_ANGLE,
    WORLD_NAME,
)
from spawn_utils import (
    OBSTACLES,
    PLATFORM_X_MIN, PLATFORM_X_MAX,
    PLATFORM_Y_MIN, PLATFORM_Y_MAX,
)

# ---------------------------------------------------------------------------
# Scenario -- edit these to move the two goals / two start positions.
# ---------------------------------------------------------------------------
# The world is a street grid: a ring road around the outside, one central
# north-south street, and two east-west cross-streets forming junctions.
# These two routes are full diagonal traverses in opposite directions, so each
# robot has to pick its way around the buildings, turn at several junctions,
# and pass the other robot on the way.
#
#   jb_0: bottom-left  -> top-right   (left street, left cross, central, top)
#   jb_1: bottom-right -> top-left    (right street, right cross, central, left)
JB0_START, JB0_GOAL = (0.44, -0.95), (2.89, 1.75)
JB1_START, JB1_GOAL = (2.89, -0.95), (0.44, 1.75)

# Previous simple straight-line scenario, kept for reference:
#   JB0_START, JB0_GOAL = (0.5, -1.0), (1.6, -1.0)
#   JB1_START, JB1_GOAL = (2.9, -1.0), (2.9, 0.3)

GOAL_MARKER_COLOR = {
    'jb_0': (0.1, 0.9, 0.2),   # green disc marks jb_0's goal
    'jb_1': (0.2, 0.45, 1.0),  # blue disc marks jb_1's goal
}

# How close the robot's centre must get before the goal counts as reached.
# Deliberately NOT jetbot_env.GOAL_REACHED_DIST -- that value belongs to the RL
# environment and shouldn't be changed just to retune this demo.
#
# IMPORTANT, because it is counter-intuitive: the robot stops the instant it
# crosses this distance, so it always halts on the RIM of this circle, never at
# its centre. A BIGGER value therefore parks the robot FURTHER from the marker,
# not closer. Small value = drives right onto the goal point (looks parked).
GOAL_TOLERANCE = 0.08

# Disc slightly wider than the robot's own footprint (0.15 x 0.13 m), so a robot
# stopped within GOAL_TOLERANCE is sitting squarely on top of it.
GOAL_MARKER_RADIUS = 0.16

# Inside this distance the robot commits to driving straight in and stops
# re-steering. Without it, tiny lateral offsets produce huge heading errors when
# the goal is centimetres away, and the robot circles it instead of parking.
FINAL_APPROACH_LOCK = 0.20

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
# base_link spawns at z=0.815; wheel joint sits 0.015 below it and the wheel
# radius is 0.033 -> the platform surface is ~0.767. Goal markers MUST sit on
# that surface, not at z=0 (which is ~0.77m below the floor, i.e. invisible).
SPAWN_Z = 0.815
FLOOR_Z = SPAWN_Z - 0.015 - 0.033      # ~0.767
MARKER_Z = FLOOR_Z + 0.02              # disc centre, sits on the floor

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
GRID_RES = 0.05          # meters per grid cell
LINE_CHECK_STEP = 0.03   # meters, resolution used when straightening the path

# How far the robot's CENTRE must stay from any wall or building.
#
# spawn_utils.ROBOT_CLEARANCE is 0.22, but the chassis is only 0.15 x 0.13 m,
# i.e. 0.10 m from centre to its furthest corner. Inflating by 0.22 shrinks the
# streets in this world to 3-9 cm slots and closes the central junction almost
# completely, so most routes come back as "no path". 0.16 keeps a 6 cm margin
# beyond the robot's true footprint and leaves the street grid intact.
PLANNER_CLEARANCE = 0.16


def is_free(x, y, clearance=PLANNER_CLEARANCE):
    """Free space test for the robot's centre, using the known map."""
    if not (PLATFORM_X_MIN + clearance <= x <= PLATFORM_X_MAX - clearance):
        return False
    if not (PLATFORM_Y_MIN + clearance <= y <= PLATFORM_Y_MAX - clearance):
        return False
    for (ox, oy, half_w, half_h) in OBSTACLES:
        if (ox - half_w - clearance <= x <= ox + half_w + clearance and
                oy - half_h - clearance <= y <= oy + half_h + clearance):
            return False
    return True


# The chassis is 0.15 x 0.13 m, so its centre is this far from its own corner.
# Physical contact happens when wall_clearance() drops below it.
ROBOT_HALF_DIAGONAL = math.hypot(0.075, 0.065)   # ~0.099 m
CONTACT_WARN_DIST = ROBOT_HALF_DIAGONAL + 0.02


def wall_clearance(x, y):
    """True distance from the robot's centre to the nearest wall/building face.
    Used for warnings, so cutting slightly inside the planner's safety margin
    while cornering doesn't get reported as if it were a crash."""
    d = min(x - PLATFORM_X_MIN, PLATFORM_X_MAX - x,
            y - PLATFORM_Y_MIN, PLATFORM_Y_MAX - y)
    for (ox, oy, half_w, half_h) in OBSTACLES:
        dx = max(ox - half_w - x, x - (ox + half_w), 0.0)
        dy = max(oy - half_h - y, y - (oy + half_h), 0.0)
        d = min(d, math.hypot(dx, dy) if (dx > 0.0 or dy > 0.0) else -1.0)
    return d

# ---------------------------------------------------------------------------
# Controller (meters, radians, seconds)
# ---------------------------------------------------------------------------
CONTROL_HZ = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ
MAX_STEPS = int(120 * CONTROL_HZ)      # 120 s -- the street-grid routes are long

MAX_LIN = 0.15                 # cruise speed
MAX_ANG = 1.0                  # max turn rate
KP_ANG = 1.6                   # heading gain
ACCEL_LIN = 0.35               # m/s^2  -- ramps make the motion smooth
ACCEL_ANG = 3.0                # rad/s^2
TURN_IN_PLACE_ANGLE = math.radians(70)   # beyond this heading error, turn before driving
APPROACH_SLOWDOWN_DIST = GOAL_TOLERANCE + 0.25   # easing runway outside the accept zone
MIN_APPROACH_SPEED = 0.045     # but never crawl slower than this while still moving

WAYPOINT_TOL = 0.12            # switch to the next waypoint within this distance

# The chassis is only 0.15 x 0.13 m, so these are generous for robot-vs-robot.
AGENT_AVOID_RADIUS = 0.60       # start steering away from the other robot
AGENT_EMERGENCY_RADIUS = 0.25   # yielding robot stops translating inside this
K_AGENT_REP_PRIORITY = 0.5      # priority robot reacts mildly
K_AGENT_REP_YIELD = 2.0         # yielding robot steers hard away
YIELD_MIN_SPEED_FRACTION = 0.3
PRIORITY = ['jb_0', 'jb_1']     # jb_0 has right of way and never fully freezes

GT_STALE_AFTER = 1.0            # seconds; older ground truth is treated as lost


# ---------------------------------------------------------------------------
# Grid + A*  (map knowledge, evaluated once at startup)
# ---------------------------------------------------------------------------
NX = int(round((PLATFORM_X_MAX - PLATFORM_X_MIN) / GRID_RES)) + 1
NY = int(round((PLATFORM_Y_MAX - PLATFORM_Y_MIN) / GRID_RES)) + 1

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def to_grid(x, y):
    i = int(round((x - PLATFORM_X_MIN) / GRID_RES))
    j = int(round((y - PLATFORM_Y_MIN) / GRID_RES))
    return max(0, min(NX - 1, i)), max(0, min(NY - 1, j))


def to_world(i, j):
    return PLATFORM_X_MIN + i * GRID_RES, PLATFORM_Y_MIN + j * GRID_RES


def cell_free(i, j):
    return is_free(*to_world(i, j))


def astar(start_cell, goal_cell):
    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def passable(cell):
        # The requested start/goal coordinates are validated separately on their
        # exact float values, so always let the grid use those two cells even if
        # rounding snaps them onto an inflated obstacle boundary.
        return cell in (start_cell, goal_cell) or cell_free(*cell)

    open_heap = [(heuristic(start_cell, goal_cell), 0.0, start_cell)]
    came_from = {start_cell: None}
    gscore = {start_cell: 0.0}
    visited = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        ci, cj = current
        for di, dj in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < NX and 0 <= nj < NY):
                continue
            neighbor = (ni, nj)
            if not passable(neighbor):
                continue
            if di != 0 and dj != 0:
                # No diagonal corner-cutting: the straight segment used later
                # can clip a blocked corner even when both endpoints are free.
                if not (passable((ci + di, cj)) and passable((ci, cj + dj))):
                    continue
            ng = g + math.hypot(di, dj)
            if ng < gscore.get(neighbor, float('inf')):
                gscore[neighbor] = ng
                came_from[neighbor] = current
                heapq.heappush(open_heap, (ng + heuristic(neighbor, goal_cell), ng, neighbor))

    return None


def line_clear(p1, p2):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    n = max(1, int(dist / LINE_CHECK_STEP))
    for k in range(n + 1):
        t = k / n
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        if not is_free(x, y):
            return False
    return True


def simplify_path(points):
    """String-pulling: keep only the waypoints where the route has to turn."""
    if len(points) <= 2:
        return points
    result = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        farthest = None
        for j in range(len(points) - 1, anchor, -1):
            if line_clear(points[anchor], points[j]):
                farthest = j
                break
        if farthest is None:
            farthest = anchor + 1
        result.append(points[farthest])
        anchor = farthest
    return result


def plan_path(start, goal):
    """One-time A* over the known static map. Returns waypoints to follow
    (excludes the start point, always ends exactly at `goal`)."""
    start_cell, goal_cell = to_grid(*start), to_grid(*goal)
    cells = astar(start_cell, goal_cell)
    if cells is None:
        raise RuntimeError(f"No path found from {start} to {goal} on the known map")

    world_pts = [to_world(i, j) for (i, j) in cells]
    world_pts[0] = start
    world_pts[-1] = goal
    waypoints = simplify_path(world_pts)

    prev = start
    for wp in waypoints:
        if not line_clear(prev, wp):
            raise RuntimeError(
                f"Internal planning error: segment {prev} -> {wp} is not clear. Refusing to drive it.")
        prev = wp

    return waypoints[1:]


# ---------------------------------------------------------------------------
# Gazebo services
# ---------------------------------------------------------------------------
def _ign_service(service, reqtype, req, reptype='ignition.msgs.Boolean', timeout='3000'):
    try:
        return subprocess.run(
            ['ign', 'service', '-s', service, '--reqtype', reqtype,
             '--reptype', reptype, '--timeout', timeout, '--req', req],
            capture_output=True, text=True, timeout=10)
    except Exception as exc:                       # noqa: BLE001
        print(f"    ign service call failed: {exc}")
        return None


def teleport(name, x, y, yaw=0.0, z=SPAWN_Z):
    qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    req = (f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}')
    _ign_service(f'/world/{WORLD_NAME}/set_pose', 'ignition.msgs.Pose', req)


def remove_entity(name):
    _ign_service(f'/world/{WORLD_NAME}/remove', 'ignition.msgs.Entity',
                 f'name: "{name}", type: MODEL')


def spawn_goal_marker(name, x, y, color):
    """Flat disc sitting on the platform surface, marking a goal.

    No <collision> element, so the LIDAR (which raycasts collision geometry)
    cannot mistake it for an obstacle. Height matters: the platform floor is
    at ~0.767, so a marker at z=0 would be buried far below it and invisible.
    """
    r, g, b = color
    remove_entity(name)
    sdf = (
        f"<sdf version='1.6'><model name='{name}'><static>true</static>"
        f"<link name='link'><visual name='visual'>"
        f"<geometry><cylinder><radius>{GOAL_MARKER_RADIUS}</radius><length>0.04</length></cylinder></geometry>"
        f"<material>"
        f"<ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>"
        f"<specular>0.2 0.2 0.2 1</specular><emissive>{r*0.5} {g*0.5} {b*0.5} 1</emissive>"
        f"</material></visual></link></model></sdf>"
    )
    req = (f'sdf: "{sdf}", name: "{name}", '
           f'pose: {{position: {{x: {x}, y: {y}, z: {MARKER_Z}}}}}')
    res = _ign_service(f'/world/{WORLD_NAME}/create', 'ignition.msgs.EntityFactory', req)
    ok = res is not None and 'true' in (res.stdout or '').lower()
    if ok:
        print(f"    marker '{name}' placed at ({x:.2f}, {y:.2f}, z={MARKER_Z:.2f}) "
              f"radius={GOAL_MARKER_RADIUS:.2f}m (goal tolerance {GOAL_TOLERANCE:.2f}m)")
    else:
        detail = ((res.stdout or '') + (res.stderr or '')).strip() if res else 'no response'
        print(f"    WARNING: marker '{name}' may not have spawned -> {detail[:200]}")
    return ok


# ---------------------------------------------------------------------------
# Ground-truth world poses (the fix for the odom-frame problem)
# ---------------------------------------------------------------------------
class GroundTruthTracker:
    """Streams Gazebo's true model poses from /world/<world>/dynamic_pose/info.

    Runs `ign topic -e` once as a background process and parses its text output
    in a thread, so the control loop never blocks waiting on a subprocess.
    """

    def __init__(self, world, names):
        self.names = set(names)
        self._poses = {}          # name -> (x, y, yaw, timestamp)
        self._lock = threading.Lock()
        self._proc = None
        self._topic = f'/world/{world}/dynamic_pose/info'

    def start(self):
        base = ['ign', 'topic', '-e', '-t', self._topic]
        for cmd in (['stdbuf', '-oL'] + base, base):   # line-buffered if available
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1)
                break
            except FileNotFoundError:
                continue
        if self._proc is None:
            return False
        threading.Thread(target=self._read_loop, daemon=True).start()
        return True

    def _read_loop(self):
        name, section, buf = None, None, {}
        try:
            for line in self._proc.stdout:
                s = line.strip()
                if s.startswith('name:'):
                    name = s.split('"')[1] if '"' in s else None
                    section, buf = None, {}
                elif s.startswith('position'):
                    section = 'position'
                elif s.startswith('orientation'):
                    section = 'orientation'
                elif section == 'position' and s.startswith('x:'):
                    buf['x'] = float(s.split(':', 1)[1])
                elif section == 'position' and s.startswith('y:'):
                    buf['y'] = float(s.split(':', 1)[1])
                elif section == 'orientation' and s.startswith('z:'):
                    buf['qz'] = float(s.split(':', 1)[1])
                elif section == 'orientation' and s.startswith('w:'):
                    buf['qw'] = float(s.split(':', 1)[1])
                    # 'w' closes an orientation block -> the pose is complete
                    if name in self.names and {'x', 'y', 'qz'} <= buf.keys():
                        yaw = 2 * math.atan2(buf['qz'], buf['qw'])
                        with self._lock:
                            self._poses[name] = (buf['x'], buf['y'], yaw, time.time())
                    section, buf = None, {}
        except Exception:                              # noqa: BLE001
            pass  # stream ended or unparsable -> caller falls back to odometry

    def get(self, name):
        with self._lock:
            entry = self._poses.get(name)
        if entry is None or (time.time() - entry[3]) > GT_STALE_AFTER:
            return None
        return entry[0], entry[1], entry[2]

    def wait_for(self, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(self.get(n) is not None for n in self.names):
                return True
            time.sleep(0.05)
        return False

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:                          # noqa: BLE001
                pass


class OdomDeadReckoner:
    """Fallback pose source: odometry deltas anchored to the known teleport pose.

    Odometry is in the `odom` frame (starts at zero, arbitrary origin), so it is
    only usable as a DISPLACEMENT from a reference reading captured right after
    the teleport, rotated into world coordinates.
    """

    def __init__(self, start_world):
        self.start = start_world      # (x, y, yaw) commanded via teleport
        self.ref = None               # odom reading at that moment

    def calibrate(self, odom_pose):
        self.ref = odom_pose

    def world_pose(self, odom_pose):
        if self.ref is None:
            return None
        xr, yr, tr = self.ref
        xo, yo, to = odom_pose
        Xs, Ys, Ts = self.start
        dtheta = Ts - tr
        dx, dy = xo - xr, yo - yr
        wx = Xs + dx * math.cos(dtheta) - dy * math.sin(dtheta)
        wy = Ys + dx * math.sin(dtheta) + dy * math.cos(dtheta)
        wyaw = math.atan2(math.sin(Ts + (to - tr)), math.cos(Ts + (to - tr)))
        return wx, wy, wyaw


# ---------------------------------------------------------------------------
# Per-tick rules (pure functions; nothing learned)
# ---------------------------------------------------------------------------
def pose_from_odom(odom):
    p = odom.pose.pose
    return p.position.x, p.position.y, 2 * math.atan2(p.orientation.z, p.orientation.w)


def angle_and_dist_to(px, py, yaw, target):
    dx, dy = target[0] - px, target[1] - py
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) - yaw
    return math.atan2(math.sin(angle), math.cos(angle)), dist


def desired_command(angle_err, dist_to_goal):
    """Smooth goal-tracking law: turn toward the target, drive at a speed that
    falls off with heading error and eases down on the final approach."""
    angular = max(-MAX_ANG, min(MAX_ANG, KP_ANG * angle_err))

    if dist_to_goal <= FINAL_APPROACH_LOCK:
        # Committed: creep straight in and park. Steering is damped and
        # turn-in-place is disabled, otherwise a centimetre of lateral offset
        # becomes a large heading error and the robot orbits the goal.
        return MIN_APPROACH_SPEED, 0.3 * angular

    if abs(angle_err) >= TURN_IN_PLACE_ANGLE:
        return 0.0, angular                       # square up first, then drive

    linear = MAX_LIN * math.cos(angle_err)        # smooth, no hard cutoff
    if dist_to_goal < APPROACH_SLOWDOWN_DIST:
        eased = MAX_LIN * (dist_to_goal / APPROACH_SLOWDOWN_DIST)
        linear = min(linear, max(MIN_APPROACH_SPEED, eased))
    return max(0.0, linear), angular


def rate_limit(current, target, max_delta):
    return current + max(-max_delta, min(max_delta, target - current))


def agent_repulsion(px, py, yaw, other_px, other_py, strength):
    """Steer-away term computed from both robots' true world positions."""
    ddist = math.hypot(other_px - px, other_py - py)
    if ddist >= AGENT_AVOID_RADIUS:
        return 0.0, ddist
    bearing = math.atan2(other_py - py, other_px - px) - yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    weight = (AGENT_AVOID_RADIUS - ddist) / AGENT_AVOID_RADIUS
    side = -1.0 if bearing >= 0 else 1.0
    return side * weight * strength, ddist


def forward_lidar_clearance(scan):
    """Logging only. This robot's LIDAR sees its own chassis at ~0.15-0.25m in
    the forward cone no matter what, so it is NOT used for any decision."""
    ranges = scan.ranges
    sector_size = max(1, len(ranges) // NUM_LIDAR_SECTORS)
    forward_min = MAX_LIDAR_RANGE
    for i in range(NUM_LIDAR_SECTORS):
        chunk = [r for r in ranges[i * sector_size:(i + 1) * sector_size]
                 if not math.isinf(r) and not math.isnan(r)]
        min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
        bearing = scan.angle_min + (i * sector_size + sector_size / 2.0) * scan.angle_increment
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        if abs(bearing) <= FORWARD_HALF_ANGLE:
            forward_min = min(forward_min, min(min_r, MAX_LIDAR_RANGE))
    return forward_min


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def spin_for(node, duration):
    end = time.time() + duration
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        rclpy.spin_once(node, timeout_sec=remaining)


def check_positions():
    for label, pos in [("jb_0 start", JB0_START), ("jb_0 goal", JB0_GOAL),
                       ("jb_1 start", JB1_START), ("jb_1 goal", JB1_GOAL)]:
        if not is_free(*pos):
            raise ValueError(f"{label} {pos} is invalid (wall/obstacle/out of bounds)")
    print("Positions validated OK.")


def run():
    check_positions()

    starts = {'jb_0': JB0_START, 'jb_1': JB1_START}
    goals = {'jb_0': JB0_GOAL, 'jb_1': JB1_GOAL}

    print("Planning paths on the known map (A*, one-time)...")
    paths = {n: plan_path(starts[n], goals[n]) for n in starts}
    for n in starts:
        print(f"  {n}: {[(round(x, 2), round(y, 2)) for x, y in paths[n]]}")

    # Face each robot at its first waypoint so it starts driving immediately
    # instead of spinning on the spot -- this is most of what "smooth" means.
    start_yaw = {}
    for n in starts:
        sx, sy = starts[n]
        wx, wy = paths[n][0]
        start_yaw[n] = math.atan2(wy - sy, wx - sx)

    rclpy.init()
    node = Node('scripted_nav_eval')
    agents = {n: JetBotAgent(node, n) for n in ('jb_0', 'jb_1')}

    # Confirm the simulator is actually alive BEFORE touching it. Orphaned
    # ros_gz_bridge / robot_state_publisher nodes from a previous run keep
    # advertising these topics with no Gazebo behind them, so the topics can
    # exist while nothing publishes. Checking for real data first means the
    # error names the actual problem instead of a pile of service timeouts.
    print("Waiting for sensor data...")
    t0 = time.time()
    while not all(a.has_fresh_data() for a in agents.values()):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 8.0:
            missing = [n for n, a in agents.items() if not a.has_fresh_data()]
            raise TimeoutError(
                f"No LIDAR/odometry data after 8s for: {', '.join(missing)}.\n"
                f"    The topics may exist while Gazebo is NOT running (stale bridge\n"
                f"    nodes from a previous run do that). Check with:\n"
                f"        ros2 topic echo /model/jb_0/odometry --once\n"
                f"    If that hangs, restart the sim cleanly:\n"
                f"        bash {__file__.rsplit('/', 1)[0]}/run_demo.sh --fresh")

    print("Spawning goal markers...")
    for n, goal in goals.items():
        spawn_goal_marker(f'goal_{n}', goal[0], goal[1], GOAL_MARKER_COLOR[n])

    print("Respawning agents at start positions (facing their first waypoint)...")
    for n, pos in starts.items():
        teleport(n, pos[0], pos[1], yaw=start_yaw[n])
    time.sleep(0.5)

    # flush any pre-teleport sensor readings so the first control tick is fresh
    t0 = time.time()
    while time.time() - t0 < 0.5:
        rclpy.spin_once(node, timeout_sec=0.05)

    # --- position source -----------------------------------------------------
    gt = GroundTruthTracker(WORLD_NAME, agents.keys())
    using_gt = gt.start() and gt.wait_for(timeout=3.0)
    reckoners = {n: OdomDeadReckoner((starts[n][0], starts[n][1], start_yaw[n])) for n in agents}
    for n, agent in agents.items():
        reckoners[n].calibrate(pose_from_odom(agent.latest_odom))

    if using_gt:
        print("Position source: Gazebo ground truth (/dynamic_pose/info).")
        for n in agents:
            gx, gy, gyaw = gt.get(n)
            print(f"    {n} is really at ({gx:.2f}, {gy:.2f}) facing {math.degrees(gyaw):.0f}deg "
                  f"| commanded ({starts[n][0]:.2f}, {starts[n][1]:.2f})")
    else:
        print("Position source: odometry dead-reckoned from the teleport pose "
              "(ground-truth stream unavailable).")

    def world_pose(name):
        if using_gt:
            p = gt.get(name)
            if p is not None:
                return p
        odom = agents[name].latest_odom
        return reckoners[name].world_pose(pose_from_odom(odom)) if odom else None

    done = {n: False for n in agents}
    waypoint_idx = {n: 0 for n in agents}
    cmd = {n: [0.0, 0.0] for n in agents}          # last published (lin, ang)
    closest_pair = float('inf')
    off_route_warned = {n: False for n in agents}

    print(f"\njb_0: {JB0_START} -> {JB0_GOAL} | jb_1: {JB1_START} -> {JB1_GOAL}\n")

    for t in range(MAX_STEPS):
        spin_for(node, CONTROL_DT)

        poses = {n: world_pose(n) for n in agents}
        if all(p is not None for p in poses.values()):
            closest_pair = min(closest_pair, math.hypot(
                poses['jb_0'][0] - poses['jb_1'][0], poses['jb_0'][1] - poses['jb_1'][1]))

        for name, agent in agents.items():
            if done[name]:
                agent.publish_stop()
                continue
            if poses[name] is None:
                continue

            px, py, yaw = poses[name]
            path = paths[name]

            dist_to_goal = math.hypot(goals[name][0] - px, goals[name][1] - py)
            if dist_to_goal <= GOAL_TOLERANCE:
                agent.publish_stop()
                cmd[name] = [0.0, 0.0]
                done[name] = True
                print(f"  [{name}] reached goal at t={t / CONTROL_HZ:.1f}s "
                      f"(pos {px:.2f},{py:.2f}, dist={dist_to_goal:.2f}m)")
                continue

            # Warn on genuine proximity to a wall, not merely on shaving inside
            # the planner's safety margin while cornering (which is harmless).
            clr = wall_clearance(px, py)
            if clr < CONTACT_WARN_DIST and not off_route_warned[name]:
                off_route_warned[name] = True
                print(f"  [{name}] WARNING: only {clr:.3f}m from a wall/building "
                      f"at ({px:.2f}, {py:.2f}) -- contact at {ROBOT_HALF_DIAGONAL:.3f}m")

            target = path[waypoint_idx[name]]
            angle_err, dist_to_wp = angle_and_dist_to(px, py, yaw, target)
            if dist_to_wp <= WAYPOINT_TOL and waypoint_idx[name] < len(path) - 1:
                waypoint_idx[name] += 1
                target = path[waypoint_idx[name]]
                angle_err, dist_to_wp = angle_and_dist_to(px, py, yaw, target)

            lin_target, ang_target = desired_command(angle_err, dist_to_goal)

            other = 'jb_1' if name == 'jb_0' else 'jb_0'
            if poses[other] is not None:
                is_priority = (name == PRIORITY[0])
                strength = K_AGENT_REP_PRIORITY if is_priority else K_AGENT_REP_YIELD
                rep, gap = agent_repulsion(px, py, yaw, poses[other][0], poses[other][1], strength)
                ang_target = max(-MAX_ANG, min(MAX_ANG, ang_target + rep))
                if not is_priority and gap < AGENT_EMERGENCY_RADIUS:
                    lin_target = 0.0      # never freeze the priority robot too,
                                          # or both stop and neither moves again
                elif not is_priority and gap < AGENT_AVOID_RADIUS:
                    span = AGENT_AVOID_RADIUS - AGENT_EMERGENCY_RADIUS
                    slow = YIELD_MIN_SPEED_FRACTION + (1 - YIELD_MIN_SPEED_FRACTION) * (
                        (gap - AGENT_EMERGENCY_RADIUS) / span)
                    lin_target *= max(YIELD_MIN_SPEED_FRACTION, min(1.0, slow))

            # smooth ramps instead of step changes
            lin = rate_limit(cmd[name][0], lin_target, ACCEL_LIN * CONTROL_DT)
            ang = rate_limit(cmd[name][1], ang_target, ACCEL_ANG * CONTROL_DT)
            cmd[name] = [lin, ang]

            twist = Twist()
            twist.linear.x = lin
            twist.angular.z = ang
            agent.cmd_pub.publish(twist)

        if t % int(CONTROL_HZ) == 0:               # once per second
            for name, agent in agents.items():
                if poses[name] is None:
                    continue
                px, py, _ = poses[name]
                dg = math.hypot(goals[name][0] - px, goals[name][1] - py)
                lidar = (f" lidar_fwd={forward_lidar_clearance(agent.latest_scan):.2f}m"
                         if agent.latest_scan is not None else "")
                status = "DONE" if done[name] else f"wp {waypoint_idx[name] + 1}/{len(paths[name])}"
                print(f"  t={t / CONTROL_HZ:4.1f}s | {name}: pos=({px:5.2f},{py:5.2f}) "
                      f"dist_to_goal={dg:.2f}m v={cmd[name][0]:.2f}{lidar} [{status}]")

        if all(done.values()):
            print(f"\nBoth agents reached their goals at t={t / CONTROL_HZ:.1f}s.")
            break
    else:
        print(f"\nHit the {MAX_STEPS / CONTROL_HZ:.0f}s time limit without both finishing.")

    for agent in agents.values():
        agent.publish_stop()

    print("\n--- Summary ---")
    for name in agents:
        p = world_pose(name)
        where = f"at ({p[0]:.2f}, {p[1]:.2f})" if p else "position unknown"
        print(f"  {name}: {'REACHED' if done[name] else 'NOT REACHED'} -- {where}, "
              f"goal was {goals[name]}")
    print(f"  Closest the two robots came to each other: {closest_pair:.2f}m")
    print(f"  Position source: {'Gazebo ground truth' if using_gt else 'odometry dead reckoning'}")

    gt.stop()
    rclpy.shutdown()


if __name__ == "__main__":
    run()
