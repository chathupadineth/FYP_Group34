import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    pkg_share = get_package_share_directory('jet_auto_simulation')
    world_path = os.path.join(pkg_share, 'worlds', 'jet_world.sdf')

    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', world_path, '-r'],
        output='screen'
    )

    return LaunchDescription([
        gazebo
    ])