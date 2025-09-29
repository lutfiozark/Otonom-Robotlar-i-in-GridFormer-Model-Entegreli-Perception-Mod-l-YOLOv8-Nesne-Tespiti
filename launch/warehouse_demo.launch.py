#!/usr/bin/env python3
"""
GridFormer Robot Warehouse Demo Launch File
Launches complete perception and navigation pipeline with Nav2 integration
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for warehouse demo"""

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for nodes'
    )

    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Start RViz visualization'
    )

    enable_gridformer_arg = DeclareLaunchArgument(
        'enable_gridformer',
        default_value='true',
        description='Enable GridFormer image restoration'
    )

    enable_yolo_arg = DeclareLaunchArgument(
        'enable_yolo',
        default_value='true',
        description='Enable YOLOv8 object detection'
    )

    enable_nav2_arg = DeclareLaunchArgument(
        'enable_nav2',
        default_value='true',
        description='Enable Nav2 navigation stack'
    )

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    enable_rviz = LaunchConfiguration('enable_rviz')
    enable_gridformer = LaunchConfiguration('enable_gridformer')
    enable_yolo = LaunchConfiguration('enable_yolo')
    enable_nav2 = LaunchConfiguration('enable_nav2')

    # Camera simulation node (using env.py)
    camera_sim_node = Node(
        package='perception',
        executable='camera_simulator',
        name='camera_simulator',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'camera_frame': 'camera_link',
            'image_topic': '/camera/image_raw',
            'publish_rate': 30.0,
            'add_weather_noise': True
        }],
        arguments=['--ros-args', '--log-level', log_level]
    )

    # GridFormer restoration node
    gridformer_node = Node(
        package='perception',
        executable='gridformer_node',
        name='gridformer_node',
        output='screen',
        condition=IfCondition(enable_gridformer),
        parameters=[{
            'use_sim_time': use_sim_time,
            'model_path': '/workspace/models/gridformer_optimized_384.onnx',
            'input_topic': '/camera/image_raw',
            'output_topic': '/camera/image_restored',
            'enable_benchmarking': True
        }],
        arguments=['--ros-args', '--log-level', log_level]
    )

    # YOLOv8 detection node
    yolo_node = Node(
        package='perception',
        executable='yolov8_node',
        name='yolov8_node',
        output='screen',
        condition=IfCondition(enable_yolo),
        parameters=[{
            'use_sim_time': use_sim_time,
            'model_path': '/workspace/models/yolov8s_optimized_416.onnx',
            'input_topic': '/camera/image_restored',
            'output_topic': '/camera/image_detections',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.5,
            'target_classes': ['person', 'car', 'truck', 'box', 'pallet']
        }],
        arguments=['--ros-args', '--log-level', log_level]
    )

    # BBox to costmap converter node
    bbox_costmap_node = Node(
        package='navigation',
        executable='bbox2costmap_node',
        name='bbox2costmap_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'detection_topic': '/camera/image_detections',
            'cloud_topic': '/bbox_cloud',
            'assumed_depth': 1.0,
            'camera_fov': 60.0,
            'image_width': 640.0,
            'image_height': 480.0,
            'camera_frame': 'camera_link',
            'target_frame': 'base_link',
            'min_confidence': 0.5
        }],
        arguments=['--ros-args', '--log-level', log_level]
    )

    # Nav2 bringup
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'bringup_launch.py'
            ])
        ]),
        condition=IfCondition(enable_nav2),
        launch_arguments={
            'params_file': PathJoinSubstitution([
                FindPackageShare('navigation'),
                'nav2_params.yaml'
            ]),
            'use_sim_time': use_sim_time,
            'autostart': 'true',
            'use_composition': 'true',
            'use_respawn': 'false'
        }.items()
    )

    # TF2 static transforms
    tf_camera_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_camera_to_base',
        arguments=[
            '0.3', '0', '0.8',  # x, y, z
            '0', '0.3', '0', '0.95',  # qx, qy, qz, qw (looking slightly down)
            'base_link',
            'camera_link'
        ]
    )

    tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_map_to_odom',
        arguments=[
            '0', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'map',
            'odom'
        ]
    )

    tf_odom_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_odom_to_base',
        arguments=[
            '0', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'odom',
            'base_link'
        ]
    )

    tf_base_to_footprint = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_footprint',
        arguments=[
            '0', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'base_footprint',
            'base_link'
        ]
    )

    # Robot State Publisher (for robot description)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': """
                <robot name="gridformer_robot">
                    <link name="base_footprint"/>
                    <link name="base_link">
                        <visual>
                            <geometry>
                                <box size="0.4 0.3 0.1"/>
                            </geometry>
                            <material name="blue">
                                <color rgba="0.0 0.0 1.0 1.0"/>
                            </material>
                        </visual>
                        <collision>
                            <geometry>
                                <box size="0.4 0.3 0.1"/>
                            </geometry>
                        </collision>
                    </link>
                    <link name="camera_link">
                        <visual>
                            <geometry>
                                <box size="0.05 0.1 0.05"/>
                            </geometry>
                            <material name="black">
                                <color rgba="0.0 0.0 0.0 1.0"/>
                            </material>
                        </visual>
                    </link>
                    <joint name="base_joint" type="fixed">
                        <parent link="base_footprint"/>
                        <child link="base_link"/>
                        <origin xyz="0 0 0.05"/>
                    </joint>
                    <joint name="camera_joint" type="fixed">
                        <parent link="base_link"/>
                        <child link="camera_link"/>
                        <origin xyz="0.3 0 0.8" rpy="0 0.3 0"/>
                    </joint>
                </robot>
            """
        }]
    )

    # RViz visualization with Nav2 config
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('nav2_bringup'),
        'rviz',
        'nav2_default_view.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(enable_rviz),
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Performance monitor
    performance_monitor = Node(
        package='perception',
        executable='performance_monitor',
        name='performance_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'monitor_topics': [
                '/camera/image_raw',
                '/camera/image_restored',
                '/camera/image_detections',
                '/bbox_cloud'
            ],
            'report_interval': 5.0
        }]
    )

    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        log_level_arg,
        enable_rviz_arg,
        enable_gridformer_arg,
        enable_yolo_arg,
        enable_nav2_arg,

        # Robot description
        robot_state_publisher,

        # TF transforms
        tf_camera_to_base,
        tf_map_to_odom,
        tf_odom_to_base,
        tf_base_to_footprint,

        # Perception pipeline
        camera_sim_node,
        gridformer_node,
        yolo_node,

        # Navigation
        bbox_costmap_node,
        nav2_bringup,

        # Visualization and monitoring
        rviz_node,
        performance_monitor,
    ])
