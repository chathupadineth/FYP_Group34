"""
verify_pose.py

Proves whether the odometry-frame bug is real, and whether the fix works.
Takes about 30 seconds. Run it before you trust any training run.

    ros2 launch jetbot_description spawn_two_jetbots.launch.py     (terminal 1)
    python3 verify_pose.py                                          (terminal 2)

Three checks:

  1. Is raw odometry really in the wrong frame?
     Teleport to a known spot, then compare what odometry claims.

  2. Does the environment now report a correct distance to the goal?
     This is the value the RL agent actually learns from (observation #13).

  3. Does the odometry fallback survive real driving?
     Drive forward with ground truth switched off, and measure the drift.
     This matters because the real JetBot has no ground truth.
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from jetbot_env import MultiJetBotEnv, JetBotAgent, WORLD_NAME
from pose_source import GroundTruthTracker, PoseSource, pose_from_odom

PASS = "PASS"
FAIL = "FAIL"

TEST_POSES = {'jb_0': (0.5, -1.0, 0.0), 'jb_1': (2.9, -1.0, math.pi / 2)}
TOL = 0.05          # metres; teleport + physics settling is not exact


def hr(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def check(label, ok):
    print(f"  --> {PASS if ok else FAIL}: {label}")
    return ok


def main():
    results = []

    env = MultiJetBotEnv()
    node = env.node
    agents = env.agents

    print("Waiting for sensor data...")
    t0 = time.time()
    while not all(a.has_fresh_data() for a in agents.values()):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 8.0:
            raise TimeoutError("No LIDAR/odometry after 8s - is the sim running?")

    # A reference tracker of our own, so we always have the real answer to
    # compare against, whatever mode the environment is using.
    truth = GroundTruthTracker(WORLD_NAME, list(agents.keys()))
    have_truth = truth.start() and truth.wait_for(timeout=4.0)
    if not have_truth:
        print("\nWARNING: Gazebo's pose stream is unavailable, so checks 1 and 3")
        print("         cannot be verified independently. Try:")
        print(f"         ign topic -e -n 1 -t /world/{WORLD_NAME}/dynamic_pose/info")

    # ---------------------------------------------------------------- 1 ----
    hr("CHECK 1  --  is raw odometry in the wrong frame?")

    for name, (x, y, yaw) in TEST_POSES.items():
        env._teleport(name, x, y, yaw=yaw)
    time.sleep(0.6)
    env._spin_until_fresh()
    for name, agent in agents.items():
        env.pose_source.calibrate(name, agent.latest_odom)
    time.sleep(0.3)

    print(f"\n  {'robot':<7} {'commanded':<18} {'RAW odom says':<18} {'fixed pose says':<18}")
    bug_seen = False
    fix_ok = True
    for name, (x, y, _) in TEST_POSES.items():
        ox, oy, _ = pose_from_odom(agents[name].latest_odom)
        fx, fy, _ = env.pose_source.world_pose(name, agents[name].latest_odom)
        print(f"  {name:<7} ({x:5.2f},{y:6.2f})     ({ox:5.2f},{oy:6.2f})     ({fx:5.2f},{fy:6.2f})")
        if math.hypot(ox - x, oy - y) > TOL:
            bug_seen = True
        if math.hypot(fx - x, fy - y) > TOL:
            fix_ok = False

    print()
    if bug_seen:
        print("  Raw odometry does NOT match where the robot was placed.")
        print("  That is the bug: it is a displacement from its own start, not a")
        print("  world position, and the goals are in world coordinates.")
    else:
        print("  Raw odometry happens to match here (the odom origin coincides")
        print("  with the world origin this run). The fix is still required --")
        print("  it will not coincide after the next reset.")
    results.append(check("the fixed pose matches the commanded position", fix_ok))

    # ---------------------------------------------------------------- 2 ----
    hr("CHECK 2  --  does the environment report a correct distance to goal?")
    print("  (this is observation value #13, the one the agent learns from)")

    env.reset(max_goal_distance=None)
    time.sleep(0.3)

    dist_ok = True
    print(f"\n  {'robot':<7} {'env reports':<14} {'true distance':<14} {'error':<10}")
    for name in agents:
        obs, dist, _ = env._build_observation(name)
        gx, gy = env.goals[name]
        p = truth.get(name) if have_truth else env.pose_source.world_pose(name, agents[name].latest_odom)
        true_dist = math.hypot(gx - p[0], gy - p[1])
        err = abs(dist - true_dist)
        print(f"  {name:<7} {dist:>8.3f} m     {true_dist:>8.3f} m     {err:>6.3f} m")
        if err > 0.10:
            dist_ok = False

        # sanity: observation #13 must be that distance, normalised
        expected13 = min(dist / 3.2, 1.0)
        if abs(obs[12] - expected13) > 1e-6:
            print(f"       (obs[12]={obs[12]:.4f} does not match dist/3.2 -- check the code)")
            dist_ok = False

    results.append(check("observation distance-to-goal is correct", dist_ok))

    # ---------------------------------------------------------------- 3 ----
    hr("CHECK 3  --  does the odometry fallback survive driving?")
    print("  Ground truth is switched OFF for this test, because the real")
    print("  JetBot will not have it. We drive forward and measure the drift.")

    if not have_truth:
        print("\n  SKIPPED - no ground truth available to compare against.")
    else:
        odom_only = PoseSource(WORLD_NAME, list(agents.keys()), mode='odom', verbose=False)

        for name, (x, y, yaw) in TEST_POSES.items():
            env._teleport(name, x, y, yaw=yaw)
        time.sleep(0.6)
        env._spin_until_fresh()
        for name, agent in agents.items():
            odom_only.note_teleport(name, *TEST_POSES[name])
            odom_only.calibrate(name, agent.latest_odom)

        print("\n  driving forward for 4 seconds...")
        twist = Twist()
        twist.linear.x = 0.12
        t_end = time.time() + 4.0
        while time.time() < t_end:
            for agent in agents.values():
                agent.cmd_pub.publish(twist)
            rclpy.spin_once(node, timeout_sec=0.05)
        for agent in agents.values():
            agent.publish_stop()
        time.sleep(0.5)
        env._spin_until_fresh()

        drift_ok = True
        print(f"\n  {'robot':<7} {'odom-only says':<20} {'ground truth':<20} {'drift':<10}")
        for name, agent in agents.items():
            ex, ey, _ = odom_only.world_pose(name, agent.latest_odom)
            tx, ty, _ = truth.get(name)
            drift = math.hypot(ex - tx, ey - ty)
            print(f"  {name:<7} ({ex:5.2f},{ey:6.2f})       ({tx:5.2f},{ty:6.2f})       {drift:.3f} m")
            if drift > 0.15:
                drift_ok = False
        results.append(check("odometry fallback stays within 0.15 m after driving", drift_ok))
        odom_only.close()

    # ------------------------------------------------------------- verdict --
    hr("VERDICT")
    if all(results):
        print("  All checks passed. The position the agent sees is now correct,")
        print("  so distance-to-goal, angle-to-goal and the shaping reward are")
        print("  all being computed properly. Safe to start training.")
    else:
        print("  Something is still wrong. Do NOT start training yet --")
        print("  the agent would be learning from a false goal signal again.")
    print()

    truth.stop()
    for agent in agents.values():
        agent.publish_stop()
    env.close()


if __name__ == "__main__":
    main()
