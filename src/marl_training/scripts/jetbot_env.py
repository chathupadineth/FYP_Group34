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

NUM_LIDAR_SECTORS = 12
MAX_LIDAR_RANGE = 12.0
MAX_EPISODE_STEPS = 200
GOAL_REACHED_DIST = 0.15
COLLISION_DIST = 0.20      # legacy LIDAR threshold -- no longer used for collisions

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

        return {name: self._build_observation(name)[0] for name in self.agents}

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
        for i in range(NUM_LIDAR_SECTORS):
            chunk = ranges[i*sector_size:(i+1)*sector_size]
            chunk = [r for r in chunk if not math.isinf(r) and not math.isnan(r)]
            min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
            min_r = min(min_r, MAX_LIDAR_RANGE)
            sectors.append(min_r / MAX_LIDAR_RANGE)

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

        if clearance <= CONTACT_DIST:
            print(f"[{name}] COLLISION — clearance={clearance:.3f}m "
                  f"(contact at {ROBOT_RADIUS:.3f}m)")

        obs = sectors + [
            min(dist / 3.2, 1.0),
            angle_to_goal / math.pi,
            vx,
            vz,
        ]
        # Third value is now the TRUE clearance in metres (was a normalised
        # LIDAR reading). Compare it against CONTACT_DIST, not COLLISION_DIST.
        return obs, dist, clearance

    def step(self, actions: dict):
        for name, action_id in actions.items():
            if self.agent_done[name]:
                self.agents[name].publish_stop()
            else:
                self.agents[name].publish_action(action_id)

        for _ in range(ACTION_REPEAT):
            self._spin_until_fresh()

        self.step_count += 1

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

            if dist <= GOAL_REACHED_DIST:
                reward = 10.0
                done = True
            elif is_colliding_now:
                reward = -10.0
                done = True

            self.agent_colliding[name] = is_colliding_now

            if not done and self.step_count >= MAX_EPISODE_STEPS:
                done = True

            observations[name] = obs
            rewards[name] = reward
            dones[name] = done
            if done:
                self.agent_done[name] = True

        return observations, rewards, dones

    def close(self):
        self.pose_source.close()
        rclpy.shutdown()