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