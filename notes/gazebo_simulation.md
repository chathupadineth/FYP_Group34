## Building & Relaunching the Gazebo Simulation


### Standard rebuild + relaunch sequence
```bash
cd ~/fyp_ws
colcon build
source install/setup.bash
ros2 launch jet_auto_simulation jet_world.launch.py



### Quick launch-only command (if already sourced this terminal)
```bash
ros2 launch jet_auto_simulation jet_world.launch.py
```

### Manual launch without ros2 launch (for quick testing)
```bash
ign gazebo ~/fyp_ws/src/jet_auto_simulation/worlds/jet_world.sdf
```
cd ~/fyp_ws
colcon build
source install/setup.bash
ros2 launch jetbot_description spawn_two_jetbots.launch.py

Terminal 1: Launch Gazebo + robots + bridge
bash
cd ~/fyp_ws
source install/setup.bash
ros2 launch jetbot_description spawn_two_jetbots.launch.py

Leave this running — check the Gazebo window opens, shows both robots, and isn't paused/flickering.

Terminal 2: Run training (after the 3 fixes are saved)
bash
cd ~/fyp_ws/src/marl_training/scripts
python3 train_mappo.py
