## JetBot URDF — Design Decisions & Real Hardware Data

### Assumptions made (estimated, not from real hardware data)
- **Chassis box**: 0.15m × 0.13m × 0.08m (length × width × height)
  — width kept slightly under the 0.1592m wheel separation, so wheels sit
  just outside the body
- **Wheel radius**: 0.033m, thickness 0.026m
  — reasonable small-robot wheel size; no exact wheel radius was available
  in the real hardware CSV (only mass was given)
- **Casters**: simple frictionless spheres, positioned at the real x-offsets
  from the CSV (±0.072m)

### Real data actually used (from jetbot_real.csv + RPLIDAR A1 datasheet)
- **Wheel separation**: 0.1592m
  (derived from wheel_right_joint y=-0.0796 and wheel_left_joint y=+0.0796)
- **LIDAR mount height**: 0.092m above base_link, ~centered (x=0.005, y=0.0003)
  (from base_scan / scan_joint origin in CSV)
- **LIDAR sensor specs** (real RPLIDAR A1, via Slamtec datasheet):
  - Range: 0.15m – 12m
  - Angular range: 360°
  - Angular resolution: ~1°
  - Scan rate: 5.5Hz (typical), configurable 1–10Hz
  - Sample rate: ~8000 samples/sec

### Source of real hardware data
Extracted from `jetbot_real.csv` (SolidWorks URDF export table) found in
instructor-provided reference files (`jetbot_world-main` package, ROS1/Catkin,
not directly used — only the dimension data was extracted).