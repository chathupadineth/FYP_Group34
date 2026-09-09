#!/usr/bin/env bash
# One-command launcher for the scripted (non-RL) two-JetBot navigation demo.
#
#   bash ~/fyp_ws/src/marl_training/scripts/run_demo.sh            # run it
#   bash ~/fyp_ws/src/marl_training/scripts/run_demo.sh --fresh    # force a clean restart
#   bash ~/fyp_ws/src/marl_training/scripts/run_demo.sh --stop     # shut the sim down
#
# Reuses a running simulation if one is genuinely alive, otherwise starts a
# clean one. Gazebo is left running at the end so you can inspect the result.
#
# NOTE: deliberately NO `set -u`. ROS 2's setup.bash references unbound
# variables (AMENT_TRACE_SETUP_FILES, COLCON_TRACE, ...), and under nounset the
# shell aborts the moment it is sourced.

WS="${HOME}/fyp_ws"
SCRIPTS="${WS}/src/marl_training/scripts"
LAUNCH_LOG="/tmp/jetbot_gazebo_launch.log"

MODE="run"
case "${1:-}" in
  --fresh) MODE="fresh" ;;
  --stop)  MODE="stop" ;;
  "")      ;;
  *)       echo "Unknown option '$1' (use --fresh or --stop)"; exit 2 ;;
esac

export ROS_LOCALHOST_ONLY=1

# Kill everything this demo starts. `pkill -f spawn_two_jetbots` alone is NOT
# enough: it ends the launch process but leaves ros_gz_bridge and
# robot_state_publisher orphaned, and those keep advertising /jb_0/scan etc.
# with no simulator behind them -- which makes a later run think the sim is up.
stop_sim() {
  echo "=== Stopping simulation and any orphaned nodes ==="
  pkill -f 'spawn_two_jetbots'   2>/dev/null
  pkill -f 'jet_world.launch'    2>/dev/null
  pkill -f 'parameter_bridge'    2>/dev/null
  pkill -f 'robot_state_publisher' 2>/dev/null
  pkill -f 'ign gazebo'          2>/dev/null
  pkill -f 'ign-gazebo'          2>/dev/null
  pkill -f 'ruby.*gazebo'        2>/dev/null
  sleep 2
  echo "    done."
}

if [ "${MODE}" = "stop" ]; then
  stop_sim
  exit 0
fi

echo "=== Sourcing ROS 2 Humble + workspace ==="
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash || { echo "ERROR: could not source ROS 2 Humble"; exit 1; }
if [ -f "${WS}/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${WS}/install/setup.bash"
else
  echo "ERROR: ${WS}/install/setup.bash not found -- run 'colcon build' in ${WS} first."
  exit 1
fi

# A topic EXISTING proves nothing (see stop_sim comment above). The only real
# test is whether a message actually arrives on it.
sim_alive() {
  timeout 6 ros2 topic echo /model/jb_0/odometry --once >/dev/null 2>&1 || return 1
  timeout 6 ros2 topic echo /model/jb_1/odometry --once >/dev/null 2>&1 || return 1
  return 0
}

if [ "${MODE}" = "fresh" ]; then
  stop_sim
  echo "=== Starting Gazebo fresh (log: ${LAUNCH_LOG}) ==="
  NEED_LAUNCH=1
elif sim_alive; then
  echo "=== Simulation is alive and publishing -- reusing it ==="
  NEED_LAUNCH=0
else
  echo "=== No live simulation (topics may exist but nothing is publishing) ==="
  stop_sim
  echo "=== Starting Gazebo (log: ${LAUNCH_LOG}) ==="
  NEED_LAUNCH=1
fi

if [ "${NEED_LAUNCH}" = "1" ]; then
  ros2 launch jetbot_description spawn_two_jetbots.launch.py > "${LAUNCH_LOG}" 2>&1 &
  LAUNCH_PID=$!

  echo -n "    waiting for the robots to start publishing "
  READY=0
  for _ in $(seq 1 20); do          # sim_alive costs up to ~12s, so ~20 tries
    if sim_alive; then
      READY=1
      echo " ready."
      break
    fi
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      echo ""
      echo "ERROR: the launch process exited. Last lines of ${LAUNCH_LOG}:"
      tail -n 30 "${LAUNCH_LOG}"
      exit 1
    fi
    echo -n "."
  done

  if [ "${READY}" != "1" ]; then
    echo ""
    echo "ERROR: robots never started publishing. Last lines of ${LAUNCH_LOG}:"
    tail -n 30 "${LAUNCH_LOG}"
    echo ""
    echo "Shut everything down with:  bash $0 --stop"
    exit 1
  fi

  echo "    letting physics settle..."
  sleep 3
fi

echo ""
echo "=== Running scripted_nav_eval.py ==="
cd "${SCRIPTS}" || { echo "ERROR: ${SCRIPTS} not found"; exit 1; }
python3 scripted_nav_eval.py
STATUS=$?

echo ""
echo "=== Done (exit code ${STATUS}) ==="
echo "Gazebo is still running so you can inspect the result."
echo "Run again:      bash $0"
echo "Clean restart:  bash $0 --fresh"
echo "Shut it down:   bash $0 --stop"
exit "${STATUS}"
