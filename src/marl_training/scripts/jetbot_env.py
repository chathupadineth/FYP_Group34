import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from spawn_utils import (
    sample_two_agents_and_goals,
    OBSTACLES,
    PLATFORM_X_MIN, PLATFORM_X_MAX,
    PLATFORM_Y_MIN, PLATFORM_Y_MAX,
)
from pose_source import PoseSource
from gating.features import AgentState, gate_features

NUM_LIDAR_SECTORS = 12
MAX_LIDAR_RANGE = 12.0     # the sensor's own maximum -- clamp and "nothing in range" fallback

# What LIDAR readings are DIVIDED BY before they reach the network.
# This is deliberately NOT the sensor maximum. The room is only ~3 x 3.25 m, so
# dividing by 12.0 squashed every reading into 0.02-0.24 (std 0.035) while the
# goal-angle input spans -1..+1 (std ~0.5). The obstacle inputs were ~15x
# quieter than the goal inputs, so the policy learned to ignore walls entirely:
# 0% timeouts and 0.95x-optimal routes, but a 30% collision rate.
# ---------------------------------------------------------------------------
# RESTORED TO RUN 2's SETTINGS so checkpoints_run2_lidarfix_terminal can be
# resumed. A checkpoint's weights only mean anything under the observation
# scaling they were trained with -- load them under different constants and
# every input arrives at the wrong magnitude, which is exactly how the BC
# attempt failed.
#
# Run 2 used LIDAR_NORM = 3.5. Do NOT change it back to 2.0 while continuing
# that run; start a fresh run if you want 2.0.
#     12.0  run 1 (blind baseline)
#      3.5  run 2  <- here
#      2.0  the value measured data later showed is better, for a FUTURE run
LIDAR_NORM = 3.5

MAX_EPISODE_STEPS = 200        # run 2's value
GOAL_REACHED_DIST = 0.15

# Run 2 ended an episode on first contact, with -10.0. Restored so the resumed
# policy meets the same rules it was trained under. (The non-terminal -2.0
# variant is still worth trying, but as a FRESH run, not a continuation.)
COLLISION_TERMINAL = True
COLLISION_PENALTY = -10.0

# Run 2's value. 4.0 is better for long goals (the map reaches 3.82 m) but the
# resumed network was trained against 3.2 -- changing it would rescale every
# distance the policy has ever seen.
GOAL_DIST_NORM = 3.2
COLLISION_DIST = 0.20      # legacy LIDAR threshold -- no longer used for collisions

# ---------------------------------------------------------------------------
# GATE INPUT SCALING
# ---------------------------------------------------------------------------
# How many silent steps count as "my information is completely out of date".
# Beyond this the staleness input saturates at 1.0 -- past ~20 control steps
# the last message is useless whether it is 20 or 200 steps old, so there is
# nothing left for the gate to discriminate.
#
# Why 20 and not MAX_EPISODE_STEPS (200): at the current ~50% talk rate a robot
# hears something every 1-2 steps. Dividing by 200 would squash every real
# value into 0.005-0.01 and the gate would never see this input move -- exactly
# the failure LIDAR_NORM = 12.0 caused for the obstacle inputs. 20 keeps the
# decision-relevant range spread across 0.05-1.0.
STALENESS_NORM = 20.0

# The nearest obstacle, in metres, that still counts as "tight". Contact is at
# ~0.114 m and the room is ~3 m across, so 1.0 m puts the range that actually
# matters (0.15-1.0 m) across most of 0..1. LIDAR_NORM = 3.5 would have
# compressed it into 0.04-0.29 instead.
GATE_CLEARANCE_NORM = 1.0

# ---------------------------------------------------------------------------
# COLLISION DETECTION -- from the map, not the LIDAR
# ---------------------------------------------------------------------------
# The LIDAR cannot decide this. Its <min> range is 0.15 m, but the chassis
# (0.15 x 0.13 m) only actually touches something at ~0.099 m from its centre.
# So any LIDAR-based threshold has to sit far too high, and 0.20 m was firing
# while the robot was still 10 cm clear.
#
# Measured on this map (diagnose_collision.py):
#     most open point anywhere      0.320 m of clearance
#     median legal spawn            0.244 m
#     old threshold                 0.200 m   <- only 4 cm below a normal spawn
#
# That marked roughly half of the physically usable space as "collision", so
# the robot was punished for manoeuvring near a wall it never touched.
#
# Now that the true world position is available (pose_source.py), the distance
# to the nearest wall can be computed exactly from the known map instead.
# This is privileged information used ONLY to compute the reward -- the agent's
# observation is unchanged and still contains nothing but its own LIDAR.
ROBOT_RADIUS = math.hypot(0.075, 0.065)      # 0.0993 m, chassis centre to corner
COLLISION_MARGIN = 0.015
CONTACT_DIST = ROBOT_RADIUS + COLLISION_MARGIN     # ~0.114 m


def wall_clearance(x, y):
    """Distance from a robot centre to the nearest wall or building face."""
    d = min(x - PLATFORM_X_MIN, PLATFORM_X_MAX - x,
            y - PLATFORM_Y_MIN, PLATFORM_Y_MAX - y)
    for (ox, oy, half_w, half_h) in OBSTACLES:
        dx = max(ox - half_w - x, x - (ox + half_w), 0.0)
        dy = max(oy - half_h - y, y - (oy + half_h), 0.0)
        d = min(d, math.hypot(dx, dy) if (dx > 0.0 or dy > 0.0) else -1.0)
    return d
FORWARD_HALF_ANGLE = math.radians(75)  # ±75° cone in front counts as "forward-facing" for collisions
ACTION_REPEAT = 3  
WORLD_NAME = "empty"

DISCRETE_ACTIONS = {
    0: (0.15, 0.0),    # forward
    1: (0.05, 0.6),    # turn left
    2: (0.05, -0.6),   # turn right
    3: (-0.15, 0.0),   # backward
}


class JetBotAgent:
    def __init__(self, node: Node, name: str):
        self.name = name
        self.latest_scan = None
        self.latest_odom = None

        self.scan_sub = node.create_subscription(
            LaserScan, f'/{name}/scan', self._scan_cb, 10)
        self.odom_sub = node.create_subscription(
            Odometry, f'/model/{name}/odometry', self._odom_cb, 10)
        self.cmd_pub = node.create_publisher(
            Twist, f'/{name}/cmd_vel', 10)

    def _scan_cb(self, msg):
        self.latest_scan = msg

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def has_fresh_data(self):
        return self.latest_scan is not None and self.latest_odom is not None

    def publish_action(self, action_id):
        linear, angular = DISCRETE_ACTIONS[action_id]
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


class MultiJetBotEnv:
    def __init__(self):
        rclpy.init()
        self.node = Node('multi_jetbot_env')
        self.agents = {
            'jb_0': JetBotAgent(self.node, 'jb_0'),
            'jb_1': JetBotAgent(self.node, 'jb_1'),
        }
        # Where the robots really are, in WORLD coordinates.
        # NOT odom.pose.pose.position -- that is in the `odom` frame, whose
        # origin is wherever the robot started, so comparing it against a
        # world-frame goal gives a wrong distance and angle. See pose_source.py.
        self.pose_source = PoseSource(WORLD_NAME, ['jb_0', 'jb_1'], mode='auto')

        self.goals = {'jb_0': None, 'jb_1': None}
        self.step_count = 0
        self.agent_done = {'jb_0': False, 'jb_1': False}
        self.agent_colliding = {'jb_0': False, 'jb_1': False}
        self.prev_distance = {'jb_0': None, 'jb_1': None}
        self.collision_events = {'jb_0': 0, 'jb_1': 0}

        # COMMUNICATION SLOTS
        # msg_slot[name] is the 4-value block appended to THAT robot's
        # observation: what the OTHER robot told it, already relative and
        # normalised. All zeros means the other robot stayed silent, which is
        # a distinct input from "it spoke and the distance happens to be 0"
        # -- the fourth value is the valid flag that separates the two.
        self.msg_slot = {'jb_0': [0.0] * 4, 'jb_1': [0.0] * 4}
        # What each gate looks at when deciding whether to transmit. Read by the
        # training loop, which owns the gate network so PPO can collect its
        # log-probs the same way it does for the actor. Assembled by
        # _update_gate_input() -- see the comment there for why this is NOT
        # msg_slot any more.
        self.gate_input = {'jb_0': [0.0] * 4, 'jb_1': [0.0] * 4}
        self.comm_sent = {'jb_0': False, 'jb_1': False}

        # BELIEF MEMORY -- the state the staleness gate input is built from.
        # last_rel_dist: the normalised relative distance from the last message
        #   that actually arrived. Kept after the message stops arriving.
        # steps_since_msg: how many steps ago that was. Climbs during silence.
        # ever_heard: has anything arrived at all this episode.
        # own_min_lidar: nearest obstacle in any direction, from this robot's
        #   OWN LIDAR -- no ground truth, so the gate stays deployable.
        self.last_rel_dist = {'jb_0': 0.0, 'jb_1': 0.0}
        self.steps_since_msg = {'jb_0': 0, 'jb_1': 0}
        self.ever_heard = {'jb_0': False, 'jb_1': False}
        self.own_min_lidar = {'jb_0': 1.0, 'jb_1': 1.0}

    def agent_state(self, name):
        """World-frame (x, y, yaw, v) -- the message payload. World frame is
        not optional here: two robots' odom frames have different origins and
        cannot be compared. See pose_source.py."""
        odom = self.agents[name].latest_odom
        if odom is None:
            return None
        p = self.pose_source.world_pose(name, odom)
        if p is None:
            return None
        return AgentState(p[0], p[1], p[2], odom.twist.twist.linear.x)

    def _exchange_messages(self, comm):
        """comm[name] is True if THAT robot transmitted this step.

        A robot's slot is filled from the other robot's message, so jb_0's
        slot depends on jb_1's gate decision, not its own. Silence leaves the
        slot at zeros with the valid flag down.
        """
        states = {n: self.agent_state(n) for n in self.agents}
        for name in self.agents:
            other = 'jb_1' if name == 'jb_0' else 'jb_0'
            ego = states[name]
            spoke = bool(comm.get(other, False)) if comm else False
            if ego is None or states[other] is None or not spoke:
                self.msg_slot[name] = [0.0, 0.0, 0.0, 0.0]
            else:
                self.msg_slot[name] = gate_features(ego, states[other], True)
        # ---- Belief bookkeeping: silence AGES information, it does not erase it
        #
        # The gate used to read msg_slot directly. msg_slot drops to all zeros
        # the instant the other robot goes quiet, and GateNet is feedforward --
        # fed [0,0,0,0] it returns the SAME logits every single time, because a
        # constant input can only produce a constant output. At a ~50% talk rate
        # that meant half of every gate decision was made blind, and two silent
        # robots sat in a deadlock where neither input could ever change enough
        # to justify speaking up.
        #
        # So the environment now remembers the last thing each robot heard and
        # how long ago it heard it. steps_since_msg climbs on every silent step,
        # which means the gate's input keeps moving even when nobody is talking.
        # That is what gives it a gradient to learn against.
        for name in self.agents:
            if self.msg_slot[name][3] > 0.0:        # valid flag up -> it landed
                self.last_rel_dist[name] = self.msg_slot[name][0]
                self.steps_since_msg[name] = 0
                self.ever_heard[name] = True
            else:
                # Bounded so the counter cannot run away over a long episode;
                # anything past 2x the norm is already saturated at 1.0 anyway.
                self.steps_since_msg[name] = min(self.steps_since_msg[name] + 1,
                                                 int(STALENESS_NORM * 2))
        self.comm_sent = {n: bool(comm.get(n, False)) if comm else False
                          for n in self.agents}

    def _update_gate_input(self):
        """Assemble the 4 numbers GateNet decides from.

        Must be called AFTER observations are built -- own_min_lidar comes from
        the LIDAR sweep inside _build_observation.

        Every value here is something a real robot could compute from its own
        sensors and its own memory. Nothing about the other robot's true pose
        leaks in. That constraint is not optional: unlike the critic, the gate
        runs at execution time, so anything it reads has to survive deployment.
        """
        for name in self.agents:
            self.gate_input[name] = [
                # 0: what it last heard -- kept, not erased, when silence falls
                self.last_rel_dist[name],
                # 1: how out-of-date that is. The one input guaranteed to keep
                #    changing when both robots are quiet.
                min(self.steps_since_msg[name] / STALENESS_NORM, 1.0),
                # 2: own nearest obstacle. Keeps moving as the robot moves, so
                #    even a saturated staleness leaves something to react to.
                self.own_min_lidar[name],
                # 3: has anything arrived at all this episode. Separates "last
                #    heard 1.2 m away, long ago" from "never heard anything",
                #    which index 0 alone cannot do.
                1.0 if self.ever_heard[name] else 0.0,
            ]

    def _spin_until_fresh(self, timeout_sec=2.0):
        start = time.time()
        for agent in self.agents.values():
            agent.latest_scan = None
            agent.latest_odom = None
        while not all(a.has_fresh_data() for a in self.agents.values()):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if time.time() - start > timeout_sec:
                raise TimeoutError("Timed out waiting for fresh sensor data")

    def _teleport(self, name, x, y, z=0.815, yaw=0.0):
        """Teleport an entity using ign service call (Gazebo Fortress)."""
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        req = (
            f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}'
        )
        subprocess.run([
            'ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '2000',
            '--req', req
        ], capture_output=True)

        # Remember where we put it: this is the anchor that lets odometry be
        # converted into a world pose if ground truth is unavailable.
        self.pose_source.note_teleport(name, x, y, yaw)

    def reset(self, max_goal_distance=None):
        self.step_count = 0
        self.agent_done = {'jb_0': False, 'jb_1': False}
        self.agent_colliding = {'jb_0': False, 'jb_1': False}
        self.prev_distance = {'jb_0': None, 'jb_1': None}
        # Collisions are no longer terminal, so they can't be counted from the
        # reward any more (there is no unique -10.0 to look for). Callers read
        # this instead: how many times each robot has made contact this episode.
        self.collision_events = {'jb_0': 0, 'jb_1': 0}
        # A new episode starts with nobody having said anything yet.
        self.msg_slot = {'jb_0': [0.0] * 4, 'jb_1': [0.0] * 4}
        self.gate_input = {'jb_0': [0.0] * 4, 'jb_1': [0.0] * 4}
        self.comm_sent = {'jb_0': False, 'jb_1': False}
        # Nobody has heard anything yet, so there is no belief to age.
        self.last_rel_dist = {'jb_0': 0.0, 'jb_1': 0.0}
        self.steps_since_msg = {'jb_0': 0, 'jb_1': 0}
        self.ever_heard = {'jb_0': False, 'jb_1': False}
        self.own_min_lidar = {'jb_0': 1.0, 'jb_1': 1.0}
        pos0, pos1, goal0, goal1 = sample_two_agents_and_goals(max_goal_distance=max_goal_distance)
        
        self._teleport('jb_0', pos0[0], pos0[1])
        self._teleport('jb_1', pos1[0], pos1[1])
        self.goals['jb_0'] = goal0
        self.goals['jb_1'] = goal1

        time.sleep(0.2)
        self._spin_until_fresh()

        # Lock the odometry anchor to the pose we just teleported to. Must come
        # AFTER fresh sensor data, so the odom reading matches the new position.
        for agent_name, agent in self.agents.items():
            self.pose_source.calibrate(agent_name, agent.latest_odom)

        obs = {name: self._build_observation(name)[0] for name in self.agents}
        # After the observations, because own_min_lidar is filled in there.
        self._update_gate_input()
        return obs

    def _build_observation(self, name):
        agent = self.agents[name]
        scan = agent.latest_scan
        odom = agent.latest_odom

        ranges = scan.ranges
        n = len(ranges)
        sector_size = n // NUM_LIDAR_SECTORS
        angle_increment = scan.angle_increment
        angle_min = scan.angle_min

        vx = odom.twist.twist.linear.x
        vz = odom.twist.twist.angular.z
        moving_backward = vx < -0.01  # small deadband to avoid noise near zero velocity

        sectors = []
        relevant_min_dist = MAX_LIDAR_RANGE
        nearest_obstacle = MAX_LIDAR_RANGE   # any direction -- for the gate input
        for i in range(NUM_LIDAR_SECTORS):
            chunk = ranges[i*sector_size:(i+1)*sector_size]
            chunk = [r for r in chunk if not math.isinf(r) and not math.isnan(r)]
            min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
            min_r = min(min_r, MAX_LIDAR_RANGE)
            sectors.append(min(min_r / LIDAR_NORM, 1.0))
            nearest_obstacle = min(nearest_obstacle, min_r)

            sector_center = angle_min + (i * sector_size + sector_size / 2.0) * angle_increment
            sector_center = math.atan2(math.sin(sector_center), math.cos(sector_center))

            if moving_backward:
                rear_relative = math.atan2(math.sin(sector_center - math.pi), math.cos(sector_center - math.pi))
                if abs(rear_relative) <= FORWARD_HALF_ANGLE:
                    relevant_min_dist = min(relevant_min_dist, min_r)
            else:
                if abs(sector_center) <= FORWARD_HALF_ANGLE:
                    relevant_min_dist = min(relevant_min_dist, min_r)

        # THE FIX: the goal is in world coordinates, so the robot's position
        # must be too. odom.pose.pose.position is in the `odom` frame (origin
        # = wherever the robot started), and mixing the two made every
        # distance-to-goal and angle-to-goal wrong. See pose_source.py.
        # Scaled on its own terms, not by LIDAR_NORM -- see GATE_CLEARANCE_NORM.
        # This is the ONLY thing the gate learns about the world around it, so
        # it has to arrive at a magnitude the network can actually see.
        self.own_min_lidar[name] = min(nearest_obstacle / GATE_CLEARANCE_NORM, 1.0)

        gx, gy = self.goals[name]
        px, py, yaw = self.pose_source.world_pose(name, odom)

        dx, dy = gx - px, gy - py
        dist = math.sqrt(dx**2 + dy**2)
        goal_angle = math.atan2(dy, dx)
        angle_to_goal = math.atan2(math.sin(goal_angle - yaw), math.cos(goal_angle - yaw))

        # How close this robot really is to touching anything, in metres.
        # Walls and buildings come from the map; the other robot is checked
        # separately because the map does not know about it. (The LIDAR used
        # to cover both, but it cannot see closer than 0.15 m -- see the note
        # on CONTACT_DIST above.)
        clearance = wall_clearance(px, py)
        for other_name in self.agents:
            if other_name == name:
                continue
            other_odom = self.agents[other_name].latest_odom
            if other_odom is None:
                continue
            op = self.pose_source.world_pose(other_name, other_odom)
            if op is None:
                continue
            centre_gap = math.hypot(op[0] - px, op[1] - py)
            clearance = min(clearance, centre_gap - ROBOT_RADIUS)

        # Only announce a collision for a robot that is still running. Once it
        # is done it stays parked against whatever it hit, so this would
        # otherwise reprint the same line every step for the rest of the
        # episode and bury the real output.
        if clearance <= CONTACT_DIST and not self.agent_done[name]:
            print(f"[{name}] COLLISION — clearance={clearance:.3f}m "
                  f"(contact at {ROBOT_RADIUS:.3f}m)")

        obs = sectors + [
            min(dist / GOAL_DIST_NORM, 1.0),
            angle_to_goal / math.pi,
            vx,
            vz,
        ]
        # The communication slot is appended LAST so the first 16 values keep
        # exactly the meaning and order the pre-gating checkpoints were trained
        # on. migrate_checkpoint.py relies on that: it copies the old input
        # weights into the first 16 columns and zeroes the new four.
        obs = obs + list(self.msg_slot[name])

        # Third value is now the TRUE clearance in metres (was a normalised
        # LIDAR reading). Compare it against CONTACT_DIST, not COLLISION_DIST.
        return obs, dist, clearance

    def step(self, actions: dict, comm: dict = None):
        """comm[name] = True if that robot transmits this step. Pass None to
        run with communication switched off entirely, which reproduces the
        pre-gating environment exactly (every slot stays zero)."""
        for name, action_id in actions.items():
            if self.agent_done[name]:
                self.agents[name].publish_stop()
            else:
                self.agents[name].publish_action(action_id)

        for _ in range(ACTION_REPEAT):
            self._spin_until_fresh()

        self.step_count += 1

        # Messages are exchanged BEFORE observations are built, so the slot a
        # robot acts on this step is the one it just received.
        self._exchange_messages(comm)

        observations, rewards, dones = {}, {}, {}
        for name in self.agents:
            if self.agent_done[name]:
                obs, _, _ = self._build_observation(name)
                observations[name] = obs
                rewards[name] = 0.0
                dones[name] = True
                continue

            obs, dist, clearance = self._build_observation(name)

            if self.prev_distance[name] is None:
                shaping_reward = 0.0
            else:
                shaping_reward = 5.0 * (self.prev_distance[name] - dist)
            self.prev_distance[name] = dist

            reward = shaping_reward - 0.01
            done = False
            is_colliding_now = clearance <= CONTACT_DIST
            was_colliding = self.agent_colliding[name]

            if dist <= GOAL_REACHED_DIST:
                reward = 10.0
                done = True
            elif is_colliding_now:
                reward = COLLISION_PENALTY
                if not was_colliding:
                    self.collision_events[name] += 1
                if COLLISION_TERMINAL:
                    done = True

            self.agent_colliding[name] = is_colliding_now

            if not done and self.step_count >= MAX_EPISODE_STEPS:
                done = True

            observations[name] = obs
            rewards[name] = reward
            dones[name] = done
            if done:
                self.agent_done[name] = True

        # Last, because it reads own_min_lidar, which _build_observation fills.
        # What the training loop reads on the NEXT step to make its gate call.
        self._update_gate_input()

        return observations, rewards, dones

    def close(self):
        self.pose_source.close()
        rclpy.shutdown()