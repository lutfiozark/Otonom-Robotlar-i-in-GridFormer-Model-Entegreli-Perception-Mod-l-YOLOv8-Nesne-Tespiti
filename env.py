#!/usr/bin/env python3
"""
GridFormer Robot PyBullet Environment
Warehouse simulation with degraded weather conditions for testing
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import cv2
from typing import Tuple, Optional


class WarehouseEnv:
    """PyBullet warehouse environment with camera and objects"""

    def __init__(self, render_mode: str = "GUI"):
        """Initialize the simulation environment

        Args:
            render_mode: "GUI" for visual rendering, "DIRECT" for headless
        """
        self.render_mode = render_mode
        self.physics_client = None
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 60
        self.camera_near = 0.1
        self.camera_far = 10.0

        # Object IDs for tracking
        self.plane_id = None
        self.table_id = None
        self.cube_ids = []

    def connect(self):
        """Connect to PyBullet physics server"""
        if self.render_mode == "GUI":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return self.physics_client

    def setup_scene(self):
        """Setup the warehouse scene with table and cubes"""
        # Physics setup
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create table
        table_position = [0, 0, 0.5]
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # Table collision shape
        table_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[1.0, 0.6, 0.05]
        )
        table_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1.0, 0.6, 0.05],
            rgbaColor=[0.6, 0.4, 0.2, 1.0]
        )

        self.table_id = p.createMultiBody(
            baseMass=0,  # Static table
            baseCollisionShapeIndex=table_collision_shape,
            baseVisualShapeIndex=table_visual_shape,
            basePosition=table_position,
            baseOrientation=table_orientation
        )

        # Create 3 cubes on the table
        cube_positions = [
            [-0.3, -0.2, 0.65],  # Left cube
            [0.0, 0.1, 0.65],    # Middle cube
            [0.4, -0.1, 0.65]    # Right cube
        ]

        cube_colors = [
            [1.0, 0.2, 0.2, 1.0],  # Red
            [0.2, 1.0, 0.2, 1.0],  # Green
            [0.2, 0.2, 1.0, 1.0]   # Blue
        ]

        for i, (pos, color) in enumerate(zip(cube_positions, cube_colors)):
            cube_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05]
            )
            cube_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=color
            )

            cube_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=pos
            )
            self.cube_ids.append(cube_id)

    def get_camera_image(self,
                         camera_pos: Tuple[float, float,
                                           float] = (-1.5, 0, 1.2),
                         target_pos: Tuple[float, float, float] = (0, 0, 0.6),
                         add_noise: bool = False) -> np.ndarray:
        """Capture camera image from specified viewpoint

        Args:
            camera_pos: Camera position (x, y, z)
            target_pos: Camera target position 
            add_noise: Add weather degradation noise

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        # Camera setup
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )

        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to numpy array
        rgb_array = np.array(rgb_img).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

        # Add weather degradation if requested
        if add_noise:
            rgb_array = self._add_weather_degradation(rgb_array)

        return rgb_array

    def _add_weather_degradation(self, image: np.ndarray) -> np.ndarray:
        """Add rain/fog degradation to simulate bad weather conditions

        Args:
            image: Input RGB image

        Returns:
            Degraded image
        """
        degraded = image.copy().astype(np.float32)

        # Add fog (reduce contrast and brightness)
        fog_intensity = 0.3
        degraded = degraded * (1 - fog_intensity) + 255 * fog_intensity

        # Add noise
        noise = np.random.normal(0, 10, degraded.shape)
        degraded = degraded + noise

        # Add rain streaks (simple vertical lines)
        if np.random.random() > 0.5:
            rain_mask = np.random.random(degraded.shape[:2]) > 0.98
            rain_mask = np.stack([rain_mask] * 3, axis=2)
            degraded[rain_mask] = 255

        # Clip values
        degraded = np.clip(degraded, 0, 255)

        return degraded.astype(np.uint8)

    def step_simulation(self, steps: int = 1):
        """Step the physics simulation"""
        for _ in range(steps):
            p.stepSimulation()

    def get_fps_benchmark(self, duration: float = 5.0) -> float:
        """Benchmark rendering FPS

        Args:
            duration: Test duration in seconds

        Returns:
            Average FPS
        """
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            self.get_camera_image()
            self.step_simulation()
            frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        return fps

    def disconnect(self):
        """Disconnect from PyBullet"""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None


class WeatherTransformationEnv:
    """Minimal RL-style environment wrapper for weather transformation.

    Provides `reset(seed)` and `step(action)` APIs expected by tests. Internally
    uses `WarehouseEnv` to render an image, optionally with weather degradation.
    """

    def __init__(self, render_mode: str = "DIRECT"):
        self.render_mode = render_mode
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()
        self._warehouse = WarehouseEnv(render_mode=render_mode)

        # Lazy connect/setup to keep initialization light for tests
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            self._warehouse.connect()
            self._warehouse.setup_scene()
            self._connected = True

    def reset(self, seed: Optional[int] = None):
        """Reset environment and return initial observation (RGB image)."""
        # Seed control for reproducibility in tests
        if seed is not None:
            self._seed = int(seed)
            # Seed both numpy global and local RNG to be safe
            np.random.seed(self._seed)
            self._rng = np.random.default_rng(self._seed)

        # Reinitialize the scene
        if self._connected:
            self._warehouse.disconnect()
            self._connected = False

        self._ensure_connected()

        # Return a degraded image to simulate weather effects
        observation = self._warehouse.get_camera_image(add_noise=True)
        return observation

    def step(self, action=None):
        """Single-step simulation. Returns (obs, reward, done, info)."""
        # Simple simulation advance
        self._warehouse.step_simulation(steps=1)
        observation = self._warehouse.get_camera_image(add_noise=True)

        # Minimal placeholders for RL API
        reward = 0.0
        done = False
        info = {}
        return observation, reward, done, info

    def close(self):
        if self._connected:
            self._warehouse.disconnect()
            self._connected = False


def main():
    """Main function for testing the environment"""
    parser = argparse.ArgumentParser(
        description="GridFormer Robot Environment Test")
    parser.add_argument("--render", action="store_true",
                        help="Enable GUI rendering")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run FPS benchmark")
    parser.add_argument("--save-images", action="store_true",
                        help="Save test images")
    args = parser.parse_args()

    # Initialize environment
    render_mode = "GUI" if args.render else "DIRECT"
    env = WarehouseEnv(render_mode=render_mode)

    try:
        # Connect and setup
        env.connect()
        env.setup_scene()

        print("✅ Environment initialized successfully!")
        print(f"Render mode: {render_mode}")

        if args.benchmark:
            print("\n🔄 Running FPS benchmark...")
            fps = env.get_fps_benchmark(duration=5.0)
            print(f"Average FPS: {fps:.1f}")

        if args.save_images:
            print("\n📸 Capturing test images...")

            # Clean image
            clean_img = env.get_camera_image(add_noise=False)
            cv2.imwrite("docs/figures/clean_scene.jpg",
                        cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))

            # Degraded image
            degraded_img = env.get_camera_image(add_noise=True)
            cv2.imwrite("docs/figures/degraded_scene.jpg",
                        cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR))

            print("Images saved to docs/figures/")

        if args.render:
            print("\n🎮 Interactive mode - Press ESC to exit")
            try:
                while True:
                    env.step_simulation()
                    time.sleep(1./60.)  # 60 FPS
            except KeyboardInterrupt:
                print("\nExiting...")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
