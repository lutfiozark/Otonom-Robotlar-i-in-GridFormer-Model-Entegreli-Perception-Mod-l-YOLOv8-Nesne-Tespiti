#!/usr/bin/env bash
set -euo pipefail

# ROS2 + Demo launcher (WSL/Linux)

if ! command -v ros2 >/dev/null 2>&1; then
  echo "❌ ROS 2 not found. Please install/switch to WSL2 and source ROS 2 Humble." >&2
  exit 1
fi

source /opt/ros/humble/setup.bash

if [ -d "install" ]; then
  source install/setup.bash || true
fi

echo "🚀 Launching warehouse demo..."
ros2 launch launch/warehouse_demo.launch.py enable_gridformer:=true enable_yolo:=true enable_nav2:=true enable_rviz:=true


