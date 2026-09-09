"""
pose_source.py

Gives a robot's TRUE position in the Gazebo WORLD frame.

--------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------
The DiffDrive plugin publishes wheel odometry on /model/<name>/odometry, and
that pose is in the `odom` frame -- its origin is wherever the robot happened
to be when odometry started, NOT the Gazebo world origin.

Goals in this project are in WORLD coordinates. Subtracting an odom-frame
position from a world-frame goal mixes two different coordinate systems, so
distance-to-goal and angle-to-goal come out wrong. Proof from a real run:

    jb_0 at (0.5, -1.0), goal (1.6, -1.0)   true distance 1.10 m
    but the env reported                                  1.89 m
    and sqrt(1.6^2 + 1.0^2) = 1.887  <-- matches "robot thinks it is at (0,0)"

This module fixes that, with two independent sources:

  GROUND TRUTH  -- read Gazebo's real model poses from
                   /world/<world>/dynamic_pose/info. Exact, no drift.
                   Only exists in simulation.

  ODOM ANCHOR   -- treat odometry as a DISPLACEMENT from the pose we
                   teleported the robot to, and rotate it into world
                   coordinates. Pure ROS, so it also works on the real
                   JetBot. Drifts a little if the wheels slip.

--------------------------------------------------------------------------
CHOOSING A SOURCE
--------------------------------------------------------------------------
    PoseSource(world, names, mode='auto')     # ground truth, else odom
    PoseSource(world, names, mode='odom')     # force odom (real-robot case)

Or without touching the code, from the shell:

    JETBOT_POSE_SOURCE=odom python3 train_mappo.py

That switch is worth keeping: training with ground truth and then evaluating
with 'odom' measures how much the policy relies on perfect localisation --
a real sim-to-real number for the thesis.
"""

import math
import os
import subprocess
import threading
import time

GT_STALE_AFTER = 1.0     # seconds; older ground truth is treated as lost


def pose_from_odom(odom_msg):
    """(x, y, yaw) straight out of an Odometry message -- in the ODOM frame."""
    p = odom_msg.pose.pose
    return (p.position.x,
            p.position.y,
            2 * math.atan2(p.orientation.z, p.orientation.w))


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
class GroundTruthTracker:
    """Streams Gazebo's true model poses in a background thread.

    Runs `ign topic -e` once and parses its text output, so the caller never
    blocks waiting on a subprocess.
    """

    def __init__(self, world, names):
        self.names = set(names)
        self._poses = {}          # name -> (x, y, yaw, timestamp)
        self._lock = threading.Lock()
        self._proc = None
        self._topic = f'/world/{world}/dynamic_pose/info'

    def start(self):
        base = ['ign', 'topic', '-e', '-t', self._topic]
        for cmd in (['stdbuf', '-oL'] + base, base):
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
                    if name in self.names and {'x', 'y', 'qz'} <= buf.keys():
                        yaw = 2 * math.atan2(buf['qz'], buf['qw'])
                        with self._lock:
                            self._poses[name] = (buf['x'], buf['y'], yaw, time.time())
                    section, buf = None, {}
        except Exception:                              # noqa: BLE001
            pass   # stream ended or unparsable -> caller falls back to odometry

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
            self._proc = None


# ---------------------------------------------------------------------------
class OdomAnchor:
    """Turns odometry into a world pose, using the teleport pose as the anchor.

        world = teleport_pose  +  R(theta) * (odom_now - odom_at_teleport)

    Valid as long as the wheels do not slip. A collision ends the episode and
    reset() re-anchors, so any slip error cannot accumulate across episodes.
    """

    def __init__(self):
        self.start = None      # (x, y, yaw) we teleported to, in world coords
        self.ref = None        # odom reading captured at that moment

    def note_teleport(self, x, y, yaw=0.0):
        self.start = (x, y, yaw)
        self.ref = None        # stale until calibrate() is called again

    def calibrate(self, odom_msg):
        if self.start is None:
            return False
        self.ref = pose_from_odom(odom_msg)
        return True

    def ready(self):
        return self.start is not None and self.ref is not None

    def world_pose(self, odom_msg):
        if not self.ready():
            return None
        xr, yr, tr = self.ref
        xo, yo, to = pose_from_odom(odom_msg)
        sx, sy, st = self.start

        dtheta = st - tr
        dx, dy = xo - xr, yo - yr
        wx = sx + dx * math.cos(dtheta) - dy * math.sin(dtheta)
        wy = sy + dx * math.sin(dtheta) + dy * math.cos(dtheta)
        wyaw = _wrap(st + (to - tr))
        return wx, wy, wyaw


# ---------------------------------------------------------------------------
class PoseSource:
    """One place to ask "where is this robot, in world coordinates?"."""

    MODES = ('auto', 'ground_truth', 'odom')

    def __init__(self, world, names, mode='auto', verbose=True):
        env_mode = os.environ.get('JETBOT_POSE_SOURCE')
        if env_mode:
            mode = env_mode.strip().lower()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")

        self.mode = mode
        self.names = list(names)
        self.anchors = {n: OdomAnchor() for n in self.names}
        self.gt = None
        self._warned = set()
        self._verbose = verbose

        if mode in ('auto', 'ground_truth'):
            tracker = GroundTruthTracker(world, self.names)
            if tracker.start() and tracker.wait_for(timeout=3.0):
                self.gt = tracker
            else:
                tracker.stop()
                if mode == 'ground_truth':
                    raise RuntimeError(
                        "mode='ground_truth' but Gazebo's pose stream is unavailable.\n"
                        f"Check:  ign topic -e -n 1 -t /world/{world}/dynamic_pose/info")

        if self._verbose:
            print(f"[PoseSource] mode={mode} -> using {self.active_source()}")

    def active_source(self):
        return 'Gazebo ground truth' if self.gt is not None else 'odometry (anchored)'

    # -- called by the environment -----------------------------------------
    def note_teleport(self, name, x, y, yaw=0.0):
        self.anchors[name].note_teleport(x, y, yaw)

    def calibrate(self, name, odom_msg):
        if odom_msg is None:
            return False
        return self.anchors[name].calibrate(odom_msg)

    def world_pose(self, name, odom_msg=None):
        """Best available world pose. Never raises -- returns None only if
        there is nothing at all to work from."""
        if self.gt is not None:
            p = self.gt.get(name)
            if p is not None:
                return p
            if 'gt_gap' not in self._warned:
                self._warned.add('gt_gap')
                print("[PoseSource] ground truth went stale, falling back to odometry")

        if odom_msg is not None:
            p = self.anchors[name].world_pose(odom_msg)
            if p is not None:
                return p
            # Not calibrated yet. Raw odom is WRONG in world terms, but
            # returning it beats crashing -- say so once, loudly.
            if 'no_anchor' not in self._warned:
                self._warned.add('no_anchor')
                print("[PoseSource] WARNING: no anchor set for "
                      f"'{name}' -- call calibrate() after reset(). "
                      "Falling back to RAW odom, which is not a world pose.")
            return pose_from_odom(odom_msg)

        return None

    def close(self):
        if self.gt is not None:
            self.gt.stop()
            self.gt = None
