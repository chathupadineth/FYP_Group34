"""
features.py

Turns two robots' raw state into the 4 numbers GateNet consumes. Pure
Python/math, no ROS2 / Gazebo / torch import -- keeps this package
runnable and testable completely offline, in line with the "don't touch
the live training system until Objective 1 is fully done" plan (see
README.md).

FRAME NOTE -- ties directly into claude/odometry-frame-bug.md:
Every (x, y, yaw) passed in here MUST already be a WORLD-frame pose, i.e.
what pose_source.PoseSource.world_pose() returns -- never raw
odom.pose.pose.position, which is in the `odom` frame (an arbitrary
per-robot origin). Two robots' odom-frame coordinates cannot be compared
directly; only world-frame ones can. jetbot_env.py already fixed this for
goal-distance; the gate's relative-state features need the same fix, and
pose_source.py already provides it -- nothing new to build there.

Your handwritten Message format was [x, y, theta, v]. That maps directly
onto the AgentState tuple below (x, y, yaw, v), provided x/y/yaw came from
PoseSource and not raw odometry.
"""

import math
from collections import namedtuple

AgentState = namedtuple('AgentState', ['x', 'y', 'yaw', 'v'])
# x, y : world-frame position (metres)          -- from PoseSource.world_pose()
# yaw  : world-frame heading (radians)           -- from PoseSource.world_pose()
# v    : forward speed (m/s)                     -- odom.twist.twist.linear.x,
#        i.e. obs[-2] in the existing 16-value observation
#        (jetbot_env.py _build_observation: obs = sectors + [dist, angle_to_goal, vx, vz])


def _wrap(angle):
    """Wrap to (-pi, pi]. Same convention jetbot_env.py uses for angle_to_goal."""
    return math.atan2(math.sin(angle), math.cos(angle))


def relative_distance(ego: AgentState, other: AgentState) -> float:
    return math.hypot(other.x - ego.x, other.y - ego.y)


def relative_heading(ego: AgentState, other: AgentState) -> float:
    """Bearing to the other robot, in ego's own frame, normalised to
    [-1, 1] by pi -- the same scale jetbot_env.py already uses for
    angle_to_goal, so this feature lines up if the gate's inputs are ever
    concatenated alongside the actor's observation."""
    bearing = math.atan2(other.y - ego.y, other.x - ego.x)
    return _wrap(bearing - ego.yaw) / math.pi


def relative_velocity(ego: AgentState, other: AgentState) -> float:
    """Signed difference in forward speed (m/s): other.v - ego.v. This is
    the simplest reading of "relative velocity" that only needs a single
    message pair, no history. If closing speed (rate of change of
    relative_distance across two samples) turns out to separate
    "dangerous" from "safe" better once real curriculum logs exist, swap
    this one function -- nothing else here needs to change."""
    return other.v - ego.v


# NORMALISATION -- measured, not guessed.
#
# LIDAR_NORM's history in jetbot_env.py is the cautionary tale: two inputs
# 15x apart in magnitude meant the quieter one was ignored outright. So these
# divisors were taken from 4,163 paired samples in nav_dataset.npz -- the same
# scripted runs the dataset was collected from -- not picked by eye.
#
#   relative_distance   min 0.23   p50 1.59   p90 2.54   p99 3.09   max 3.41 m
#   relative_velocity   min -0.150  p50 0.000  max 0.150 m/s
#
#   divisor      p50     std    clipped
#   dist / 2.0   0.795   0.260   31.2%   <- throws away a third of the range
#   dist / 3.0   0.530   0.228    1.5%   <- here
#   dist / 4.0   0.398   0.173    0.0%   <- weaker signal for nothing gained
#   relv / 0.15    -     0.290    0.0%   <- here; 0.15 is MAX_LIN, so this is
#                                           exactly [-1, 1] by construction
#
# Result: distance std 0.228, velocity std 0.290, heading already in [-1, 1].
# All three within ~1.3x of each other, against the 15x that caused the
# original failure.
REL_DIST_NORM = 3.0        # metres
REL_VEL_NORM = 0.15        # m/s -- scripted_nav_eval.MAX_LIN


def gate_features(ego: AgentState, other_or_none, msg_valid: bool):
    """Builds the 4-value input GateNet expects for one ego/other pair.

    other_or_none is None exactly when that slot is empty (per your note:
    "if received no messages, slots 1/2/3 are empty") -- in that case
    every relative feature is 0.0 AND msg_valid is forced to 0.0, so the
    network sees a distinct, unambiguous "nothing here" input rather than
    a real zero-distance/zero-velocity reading. This is the "zero-value
    trap" already logged as a design decision.

    Call this once per neighbour slot. The current sim only has 2 robots
    (jb_0, jb_1), so there is only ever one "other" today -- see the open
    question in README.md about your notes' slots 1/2/3 implying up to 3
    neighbours. Nothing here assumes a fixed number of neighbours.
    """
    if other_or_none is None or not msg_valid:
        return [0.0, 0.0, 0.0, 0.0]
    d = min(relative_distance(ego, other_or_none) / REL_DIST_NORM, 1.0)
    v = max(-1.0, min(1.0, relative_velocity(ego, other_or_none) / REL_VEL_NORM))
    return [
        d,
        v,
        relative_heading(ego, other_or_none),
        1.0,
    ]
