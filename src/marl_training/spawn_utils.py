import random

# Platform bounds (confirmed via wall positions in jet_world.sdf)
PLATFORM_X_MIN, PLATFORM_X_MAX = 0.175, 3.161   # matches wall_west / wall_east
PLATFORM_Y_MIN, PLATFORM_Y_MAX = -1.232, 2.019  # matches wall_south / wall_north

# Obstacle bounding boxes: (center_x, center_y, half_width_x, half_width_y)
# All values confirmed from the final jet_world.sdf
OBSTACLES = [
    (2.195,    0.902,    0.425,  0.585),   # Building_A  (footprint: 0.850 x 1.170)
    (1.001,    1.170,    0.300,  0.300),   # Building_B  (footprint: 0.600 x 0.600)
    (0.999021, -0.144534, 0.300, 0.550),   # obstacle_box_1 (size 0.6 x 1.1)
    (2.21225,  -0.424662, 0.425, 0.275),   # obstacle_box_2 (size 0.85 x 0.55)
]

ROBOT_CLEARANCE = 0.15  # half of robot's chassis length, adds safety margin


def is_position_valid(x, y):
    """Check that (x, y) is inside the platform and doesn't collide with any obstacle."""
    if not (PLATFORM_X_MIN + ROBOT_CLEARANCE <= x <= PLATFORM_X_MAX - ROBOT_CLEARANCE):
        return False
    if not (PLATFORM_Y_MIN + ROBOT_CLEARANCE <= y <= PLATFORM_Y_MAX - ROBOT_CLEARANCE):
        return False

    for (ox, oy, half_w, half_h) in OBSTACLES:
        if (ox - half_w - ROBOT_CLEARANCE <= x <= ox + half_w + ROBOT_CLEARANCE and
                oy - half_h - ROBOT_CLEARANCE <= y <= oy + half_h + ROBOT_CLEARANCE):
            return False
    return True


def sample_valid_position(max_attempts=100):
    """Randomly sample a collision-free (x, y) position on the platform."""
    for _ in range(max_attempts):
        x = random.uniform(PLATFORM_X_MIN, PLATFORM_X_MAX)
        y = random.uniform(PLATFORM_Y_MIN, PLATFORM_Y_MAX)
        if is_position_valid(x, y):
            return (x, y)
    raise RuntimeError("Could not find a valid spawn position after max_attempts")


def sample_robot_and_goal(min_separation=0.5):
    """Sample a robot start position and a goal position, ensuring they're not too close together."""
    robot_pos = sample_valid_position()
    for _ in range(100):
        goal_pos = sample_valid_position()
        dist = ((goal_pos[0] - robot_pos[0]) ** 2 + (goal_pos[1] - robot_pos[1]) ** 2) ** 0.5
        if dist >= min_separation:
            return robot_pos, goal_pos
    raise RuntimeError("Could not find a valid goal position with sufficient separation")