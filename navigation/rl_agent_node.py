#!/usr/bin/env python3
"""
PPO RL Agent Node for GridFormer Robot
Reinforcement learning agent for autonomous navigation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import cv2
from typing import Dict, Any, Tuple, Optional
import time
import os
import argparse


class RobotNavigationEnv(gym.Env):
    """Custom Gym environment for robot navigation"""

    def __init__(self, ros_node):
        super().__init__()

        self.ros_node = ros_node

        # Action space: [linear_x, angular_z]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -1.0]),
            high=np.array([0.5, 1.0]),
            dtype=np.float32
        )

        # Observation space: [camera_image, laser_scan, odometry]
        self.observation_space = gym.spaces.Dict({
            'camera': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'laser': gym.spaces.Box(low=0, high=10.0, shape=(360,), dtype=np.float32),
            'odometry': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })

        # Environment state
        self.current_image = None
        self.current_laser = None
        self.current_odom = None
        self.goal_position = np.array([5.0, 5.0])  # Target position
        self.start_position = np.array([0.0, 0.0])
        self.max_episode_steps = 1000
        self.current_step = 0

        # Rewards
        self.previous_distance_to_goal = None
        self.collision_penalty = -100.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        self.current_step = 0
        self.previous_distance_to_goal = None

        # Reset robot position (would interface with simulation)
        self._reset_robot_position()

        # Wait for initial observations
        self._wait_for_observations()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Execute action and return next state"""
        self.current_step += 1

        # Send action to robot
        self._send_action(action)

        # Wait for environment response
        time.sleep(0.1)

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward, done, info = self._calculate_reward()

        # Check termination conditions
        truncated = self.current_step >= self.max_episode_steps

        return observation, reward, done, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        # Process camera image
        if self.current_image is not None:
            camera_obs = cv2.resize(self.current_image, (64, 64))
        else:
            camera_obs = np.zeros((64, 64, 3), dtype=np.uint8)

        # Process laser scan
        if self.current_laser is not None:
            laser_obs = self.current_laser[:360]  # Ensure 360 points
            laser_obs = np.clip(laser_obs, 0, 10.0)
        else:
            laser_obs = np.full(360, 10.0, dtype=np.float32)

        # Process odometry
        if self.current_odom is not None:
            odom_obs = self.current_odom
        else:
            odom_obs = np.zeros(6, dtype=np.float32)

        return {
            'camera': camera_obs,
            'laser': laser_obs,
            'odometry': odom_obs
        }

    def _calculate_reward(self) -> Tuple[float, bool, Dict]:
        """Calculate reward based on current state"""
        reward = self.step_penalty
        done = False
        info = {}

        if self.current_odom is None:
            return reward, done, info

        # Current position
        current_pos = np.array([self.current_odom[0], self.current_odom[1]])
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)

        # Goal reached
        if distance_to_goal < 0.5:
            reward += self.goal_reward
            done = True
            info['success'] = True

        # Progress reward
        if self.previous_distance_to_goal is not None:
            progress = self.previous_distance_to_goal - distance_to_goal
            reward += progress * 10.0  # Scale progress reward

        self.previous_distance_to_goal = distance_to_goal

        # Collision detection (simple laser-based)
        if self.current_laser is not None:
            min_distance = np.min(self.current_laser)
            if min_distance < 0.3:  # Too close to obstacle
                reward += self.collision_penalty
                done = True
                info['collision'] = True

        info['distance_to_goal'] = distance_to_goal

        return reward, done, info

    def _send_action(self, action):
        """Send action to robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.angular.z = float(action[1])
        self.ros_node.cmd_vel_publisher.publish(cmd_vel)

    def _reset_robot_position(self):
        """Reset robot to starting position"""
        # This would interface with simulation to reset robot
        pass

    def _wait_for_observations(self):
        """Wait for initial observations"""
        timeout = 5.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            if (self.current_image is not None and
                self.current_laser is not None and
                    self.current_odom is not None):
                break
            time.sleep(0.1)


class RLTrainingCallback(BaseCallback):
    """Custom callback for RL training monitoring"""

    def __init__(self, ros_node, verbose=0):
        super().__init__(verbose)
        self.ros_node = ros_node
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at each step"""
        # Log training progress
        if len(self.model.ep_info_buffer) > 0:
            recent_episode = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(recent_episode['r'])
            self.episode_lengths.append(recent_episode['l'])

            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])

                self.ros_node.get_logger().info(
                    f"Training - Episodes: {len(self.episode_rewards)}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Avg Length: {avg_length:.1f}"
                )

        return True


class RLAgentNode(Node):
    """ROS 2 node for RL agent training and inference"""

    def __init__(self):
        super().__init__('rl_agent_node')

        # Parameters
        # 'training' or 'inference'
        self.declare_parameter('mode', 'inference')
        self.declare_parameter(
            'model_path', '/workspace/models/ppo_navigation.zip')
        self.declare_parameter('total_timesteps', 100000)
        self.declare_parameter('learning_rate', 3e-4)

        # Get parameters
        self.mode = self.get_parameter(
            'mode').get_parameter_value().string_value
        self.model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self.total_timesteps = self.get_parameter(
            'total_timesteps').get_parameter_value().integer_value
        self.learning_rate = self.get_parameter(
            'learning_rate').get_parameter_value().double_value

        # Initialize components
        self.bridge = CvBridge()
        self.model = None
        self.env = None

        # Current observations
        self.current_image = None
        self.current_laser = None
        self.current_odom = None

        # ROS 2 setup
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_detections',
            self.image_callback,
            10
        )

        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize based on mode
        if self.mode == 'training':
            self.setup_training()
        else:
            self.setup_inference()

        self.get_logger().info(
            f"RL Agent node initialized in {self.mode} mode")

    def setup_training(self):
        """Setup for training mode"""
        self.get_logger().info("Setting up RL training environment...")

        # Create environment
        self.env = RobotNavigationEnv(self)

        # Create PPO model
        self.model = PPO(
            'MultiInputPolicy',
            self.env,
            learning_rate=self.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="/workspace/logs/tensorboard/",
            verbose=1
        )

        self.get_logger().info("✅ Training setup complete")

    def setup_inference(self):
        """Setup for inference mode"""
        self.get_logger().info("Setting up RL inference...")

        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
            self.get_logger().info(f"✅ Model loaded from {self.model_path}")
        else:
            self.get_logger().error(f"Model not found: {self.model_path}")
            return

        # Setup inference timer
        self.inference_timer = self.create_timer(0.1, self.inference_step)

    def start_training(self):
        """Start training process"""
        if self.model is None or self.env is None:
            self.get_logger().error("Training not properly setup")
            return

        self.get_logger().info(
            f"Starting training for {self.total_timesteps} timesteps...")

        # Setup callback
        callback = RLTrainingCallback(self)

        # Train the model
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback,
            progress_bar=True
        )

        # Save trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        self.get_logger().info(
            f"✅ Training completed! Model saved to {self.model_path}")

    def inference_step(self):
        """Single inference step"""
        if self.model is None:
            return

        # Get current observation
        observation = self._get_current_observation()
        if observation is None:
            return

        # Predict action
        action, _ = self.model.predict(observation, deterministic=True)

        # Send action
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.angular.z = float(action[1])
        self.cmd_vel_publisher.publish(cmd_vel)

    def _get_current_observation(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current observation for inference"""
        if (self.current_image is None or
            self.current_laser is None or
                self.current_odom is None):
            return None

        # Process observations similar to environment
        camera_obs = cv2.resize(self.current_image, (64, 64))
        laser_obs = self.current_laser[:360]
        laser_obs = np.clip(laser_obs, 0, 10.0)
        odom_obs = self.current_odom

        return {
            'camera': camera_obs,
            'laser': laser_obs,
            'odometry': odom_obs
        }

    def image_callback(self, msg: Image):
        """Process incoming images"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def laser_callback(self, msg: LaserScan):
        """Process laser scan data"""
        self.current_laser = np.array(msg.ranges)
        # Replace inf values with max range
        self.current_laser[np.isinf(self.current_laser)] = msg.range_max

        # Update environment if training
        if hasattr(self, 'env') and self.env is not None:
            self.env.current_laser = self.current_laser

    def odom_callback(self, msg: Odometry):
        """Process odometry data"""
        pose = msg.pose.pose
        twist = msg.twist.twist

        self.current_odom = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,  # Simplified orientation
            twist.linear.x,
            twist.angular.z
        ])

        # Update environment if training
        if hasattr(self, 'env') and self.env is not None:
            self.env.current_odom = self.current_odom


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RL Agent Node')
    parser.add_argument('--train', action='store_true', help='Start training')
    parser.add_argument('--timesteps', type=int,
                        default=100000, help='Training timesteps')
    args = parser.parse_args()

    rclpy.init()

    try:
        if args.train:
            # Training mode
            node = RLAgentNode()
            node.set_parameters([
                rclpy.parameter.Parameter(
                    'mode', rclpy.Parameter.Type.STRING, 'training'),
                rclpy.parameter.Parameter(
                    'total_timesteps', rclpy.Parameter.Type.INTEGER, args.timesteps)
            ])
            node.start_training()
        else:
            # Inference mode
            node = RLAgentNode()
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
