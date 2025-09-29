cmake_minimum_required(VERSION 3.8)
project(navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_msgs REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})

# Create bbox2costmap_node executable
add_executable(bbox2costmap_node src/bbox2costmap_node.cpp)

target_link_libraries(bbox2costmap_node ${PCL_LIBRARIES})

ament_target_dependencies(bbox2costmap_node 
  rclcpp 
  sensor_msgs 
  geometry_msgs
  vision_msgs
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  nav2_core
  nav2_costmap_2d
  nav2_msgs
)

# RL agent node (Python)
# install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install C++ executables
install(TARGETS
  bbox2costmap_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  rl_agent_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 