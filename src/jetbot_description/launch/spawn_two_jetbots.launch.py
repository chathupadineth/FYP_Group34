import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    jetbot_desc_share = get_package_share_directory('jetbot_description')
    jet_sim_share = get_package_share_directory('jet_auto_simulation')

    xacro_file = os.path.join(jetbot_desc_share, 'urdf', 'jetbot.urdf.xacro')

    # Include your existing Gazebo world launch
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(jet_sim_share, 'launch', 'jet_world.launch.py')
        )
    )

    def make_robot_group(name, x, y, z, yaw):
        robot_description = ParameterValue(
    Command(['xacro ', xacro_file, ' botname:=', name]),
    value_type=str
)

        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'frame_prefix': name + '/'
            }]
        )

        spawn_entity = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', 'robot_description',
                '-name', name,
                '-x', str(x), '-y', str(y), '-z', str(z),
                '-Y', str(yaw)
            ],
            output='screen'
        )

        return GroupAction([
            PushRosNamespace(name),
            robot_state_publisher,
            spawn_entity
        ])

    # Two robots, distinct namespaces and spawn positions
    robot_1 = make_robot_group('jb_0', x=1.5, y=0.0, z=0.815, yaw=0.0)
    robot_2 = make_robot_group('jb_1', x=0.5, y=0.0, z=0.815, yaw=0.0)

    return LaunchDescription([
        world_launch,
        robot_1,
        robot_2
    ])