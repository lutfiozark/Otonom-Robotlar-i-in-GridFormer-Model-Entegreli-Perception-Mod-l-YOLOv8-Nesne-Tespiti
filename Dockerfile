# GridFormer Robot Perception Pipeline Docker
# ROS 2 Humble + TensorRT + OpenCV + PyTorch

FROM osrf/ros:humble-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy CUDA keyring (pre-downloaded to avoid network issues)
COPY deps/cuda-keyring_1.1-1_all.deb /tmp/

# Install NVIDIA TensorRT (for GPU support)
RUN dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
    rm /tmp/cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y \
    tensorrt \
    python3-libnvinfer-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional ROS 2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-nav2-bringup \
    ros-humble-navigation2 \
    ros-humble-nav2-costmap-2d \
    ros-humble-nav2-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create ROS 2 workspace structure
RUN mkdir -p /workspace/ros2_ws/src
RUN cd /workspace/ros2_ws && \
    ln -sf /workspace/perception ./src/perception && \
    ln -sf /workspace/navigation ./src/navigation && \
    ln -sf /workspace/launch ./src/launch

# Build ROS 2 workspace
RUN cd /workspace/ros2_ws && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set up ROS 2 environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc

# Create models directory
RUN mkdir -p /workspace/models

# Set permissions
RUN chmod +x scripts/*.sh

# Expose ports for ROS 2, RViz, MLflow
EXPOSE 11311 8080 5000

# Default command
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && exec bash"] 