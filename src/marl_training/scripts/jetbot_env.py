import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from spawn_utils import sample_two_agents_and_goals

NUM_LIDAR_SECTORS = 12
MAX_LIDAR_RANGE = 12.0
MAX_EPISODE_STEPS = 200
GOAL_REACHED_DIST = 0.15
COLLISION_DIST = 0.12
WORLD_NAME = "empty"

DISCRETE_ACTIONS = {
    0: (0.15, 0.0),
    1: (0.05, 0.6),
    2: (0.05, -0.6),
    3: (0.0, 0.0),
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


class MultiJetBotEnv:
    def __init__(self):
        rclpy.init()
        self.node = Node('multi_jetbot_env')
        self.agents = {
            'jb_0': JetBotAgent(self.node, 'jb_0'),
            'jb_1': JetBotAgent(self.node, 'jb_1'),
        }
        self.goals = {'jb_0': None, 'jb_1': None}
        self.step_count = 0

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

    def reset(self):
        self.step_count = 0
        pos0, pos1, goal0, goal1 = sample_two_agents_and_goals()

        self._teleport('jb_0', pos0[0], pos0[1])
        self._teleport('jb_1', pos1[0], pos1[1])
        self.goals['jb_0'] = goal0
        self.goals['jb_1'] = goal1

        time.sleep(0.2)
        self._spin_until_fresh()

        return {name: self._build_observation(name)[0] for name in self.agents}

    def _build_observation(self, name):
        agent = self.agents[name]
        scan = agent.latest_scan
        odom = agent.latest_odom

        ranges = scan.ranges
        n = len(ranges)
        sector_size = n // NUM_LIDAR_SECTORS
        sectors = []
        for i in range(NUM_LIDAR_SECTORS):
            chunk = ranges[i*sector_size:(i+1)*sector_size]
            chunk = [r for r in chunk if not math.isinf(r) and not math.isnan(r)]
            min_r = min(chunk) if chunk else MAX_LIDAR_RANGE
            sectors.append(min(min_r, MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE)

        gx, gy = self.goals[name]
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        dx, dy = gx - px, gy - py
        dist = math.sqrt(dx**2 + dy**2)
        goal_angle = math.atan2(dy, dx)

        qz = odom.pose.pose.orientation.z
        qw = odom.pose.pose.orientation.w
        yaw = 2 * math.atan2(qz, qw)
        angle_to_goal = math.atan2(math.sin(goal_angle - yaw), math.cos(goal_angle - yaw))

        vx = odom.twist.twist.linear.x
        vz = odom.twist.twist.angular.z

        obs = sectors + [
            min(dist / 3.2, 1.0),
            angle_to_goal / math.pi,
            vx,
            vz,
        ]
        return obs, dist, min(sectors)

    def step(self, actions: dict):
        for name, action_id in actions.items():
            self.agents[name].publish_action(action_id)

        self._spin_until_fresh()
        self.step_count += 1

        observations, rewards, dones = {}, {}, {}
        for name in self.agents:
            obs, dist, min_lidar = self._build_observation(name)
            reward = -0.01
            done = False

            if dist <= GOAL_REACHED_DIST:
                reward = 10.0
                done = True
            elif min_lidar * MAX_LIDAR_RANGE <= COLLISION_DIST:
                reward = -10.0
                done = True
            elif self.step_count >= MAX_EPISODE_STEPS:
                done = True

            observations[name] = obs
            rewards[name] = reward
            dones[name] = done

        return observations, rewards, dones

    def close(self):
        rclpy.shutdown()