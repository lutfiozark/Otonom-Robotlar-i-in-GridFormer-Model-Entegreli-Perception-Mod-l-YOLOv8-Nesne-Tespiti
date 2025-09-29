# WSL üzerinden ROS2 demo başlatıcı (Windows)
# Gereksinim: WSL2 + Ubuntu + ROS 2 Humble kurulu ve kaynak yapılmış olmalı

Write-Host "GridFormer Robot - ROS2 Demo (WSL)" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

$script = @"
#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
cd /mnt/$(wslpath -a "${PWD}" | sed 's#^/mnt/##')
colcon build --symlink-install || true
source install/setup.bash || true
ros2 launch launch/warehouse_demo.launch.py enable_gridformer:=true enable_yolo:=true enable_nav2:=true enable_rviz:=true
"@

wsl bash -lc $script


