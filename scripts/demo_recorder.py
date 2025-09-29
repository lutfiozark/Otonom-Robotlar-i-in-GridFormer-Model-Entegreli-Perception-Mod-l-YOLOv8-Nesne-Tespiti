#!/usr/bin/env python3
"""Demo recording and GIF generation script."""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

try:
    import mlflow
    from mlops.mlflow_utils import log_metrics
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class DemoRecorder(Node):
    """ROS 2 node for recording demo videos."""

    def __init__(self, output_path: str, record_topics: list):
        super().__init__('demo_recorder')
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.recording = False
        self.frame_count = 0
        self.video_writers = {}

        # Setup subscribers for each topic
        self.subscribers = {}
        for topic in record_topics:
            self.subscribers[topic] = self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self.image_callback(msg, t),
                10
            )

        self.get_logger().info(
            f"Demo recorder initialized for topics: {record_topics}")

    def start_recording(self):
        """Start recording."""
        self.recording = True
        self.frame_count = 0
        self.get_logger().info("🎬 Recording started")

    def stop_recording(self):
        """Stop recording."""
        self.recording = False
        for writer in self.video_writers.values():
            if writer:
                writer.release()
        self.video_writers.clear()
        self.get_logger().info(
            f"🛑 Recording stopped. Frames recorded: {self.frame_count}")

    def image_callback(self, msg, topic_name):
        """Handle incoming image messages."""
        if not self.recording:
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Initialize video writer if needed
            if topic_name not in self.video_writers:
                height, width = cv_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename = self.output_path / \
                    f"{topic_name.replace('/', '_')}_demo.mp4"

                self.video_writers[topic_name] = cv2.VideoWriter(
                    str(filename), fourcc, 10.0, (width, height)
                )
                self.get_logger().info(
                    f"📹 Started recording {topic_name} to {filename}")

            # Write frame
            if self.video_writers[topic_name]:
                self.video_writers[topic_name].write(cv_image)
                self.frame_count += 1

                # Add frame info overlay
                if self.frame_count % 30 == 0:  # Every 3 seconds at 10fps
                    self.get_logger().info(
                        f"📊 Frames recorded: {self.frame_count}")

        except Exception as e:
            self.get_logger().error(f"Error recording frame: {e}")


def create_navigation_demo_gif(video_path: str, output_gif: str, duration: int = 10):
    """Create GIF from navigation demo video."""
    print(f"🎬 Creating demo GIF from {video_path}")

    video_path = Path(video_path)
    output_gif = Path(output_gif)

    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return None

    # Use ffmpeg to create GIF
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f'fps=8,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-t', str(duration),
        str(output_gif)
    ]

    try:
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)

        if output_gif.exists():
            file_size_mb = output_gif.stat().st_size / (1024 * 1024)
            print(f"✅ GIF created: {output_gif} ({file_size_mb:.1f}MB)")
            return str(output_gif)
        else:
            print(f"❌ GIF creation failed")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        print(f"   stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"❌ ffmpeg not found. Please install ffmpeg.")
        return None


def create_comparison_gif(before_video: str, after_video: str, output_gif: str):
    """Create side-by-side comparison GIF."""
    print(f"🎬 Creating comparison GIF...")

    before_path = Path(before_video)
    after_path = Path(after_video)
    output_gif = Path(output_gif)

    if not before_path.exists() or not after_path.exists():
        print(f"❌ Video files not found")
        return None

    # Use ffmpeg to create side-by-side comparison
    cmd = [
        'ffmpeg', '-y',
        '-i', str(before_path),
        '-i', str(after_path),
        '-filter_complex',
        '[0:v]scale=360:240[left];[1:v]scale=360:240[right];[left][right]hstack=inputs=2,fps=8,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-t', '10',
        str(output_gif)
    ]

    try:
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)

        if output_gif.exists():
            file_size_mb = output_gif.stat().st_size / (1024 * 1024)
            print(
                f"✅ Comparison GIF created: {output_gif} ({file_size_mb:.1f}MB)")
            return str(output_gif)
        else:
            print(f"❌ Comparison GIF creation failed")
            return None

    except Exception as e:
        print(f"❌ Comparison GIF creation failed: {e}")
        return None


def record_ros_demo(topics: list, duration: int, output_dir: str):
    """Record ROS demo with specified topics."""
    if not ROS2_AVAILABLE:
        print(f"❌ ROS 2 not available")
        return None

    print(f"🎬 Recording ROS demo for {duration} seconds...")
    print(f"   Topics: {topics}")
    print(f"   Output: {output_dir}")

    rclpy.init()

    try:
        # Create recorder node
        recorder = DemoRecorder(output_dir, topics)

        # Start recording
        recorder.start_recording()

        # Record for specified duration
        start_time = time.time()
        while (time.time() - start_time) < duration:
            rclpy.spin_once(recorder, timeout_sec=0.1)

        # Stop recording
        recorder.stop_recording()

        # Clean up
        recorder.destroy_node()

        print(f"✅ ROS demo recording completed")
        return output_dir

    except Exception as e:
        print(f"❌ ROS demo recording failed: {e}")
        return None
    finally:
        rclpy.shutdown()


def capture_rviz_screenshots(duration: int = 30):
    """Capture RViz screenshots for documentation."""
    print(f"📸 Capturing RViz screenshots...")

    output_dir = Path("docs/figures/rviz_screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshots = []

    # Different views to capture
    views = [
        "navigation_view",
        "costmap_view",
        "detection_view",
        "full_pipeline_view"
    ]

    for i, view in enumerate(views):
        time.sleep(duration / len(views))  # Space out screenshots

        screenshot_path = output_dir / f"{view}_{int(time.time())}.png"

        # Use gnome-screenshot or similar tool
        try:
            cmd = ['gnome-screenshot', '-w', '-f', str(screenshot_path)]
            subprocess.run(cmd, check=True, timeout=10)

            if screenshot_path.exists():
                screenshots.append(str(screenshot_path))
                print(f"📸 Screenshot saved: {screenshot_path}")

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"⚠️  Screenshot failed for {view}")

    return screenshots


def main():
    parser = argparse.ArgumentParser(
        description='Record and create demo materials')
    parser.add_argument('--mode', choices=['record', 'gif', 'comparison', 'screenshot'],
                        default='record', help='Demo mode')
    parser.add_argument('--topics', nargs='+',
                        default=['/camera/image_raw',
                                 '/enhanced_image', '/detection_image'],
                        help='ROS topics to record')
    parser.add_argument('--duration', type=int, default=30,
                        help='Recording duration (seconds)')
    parser.add_argument('--output-dir', default='demo_output',
                        help='Output directory')
    parser.add_argument('--input-video', help='Input video for GIF creation')
    parser.add_argument('--before-video', help='Before video for comparison')
    parser.add_argument('--after-video', help='After video for comparison')
    parser.add_argument('--output-gif', help='Output GIF path')

    args = parser.parse_args()

    if args.mode == 'record':
        # Record ROS demo
        result = record_ros_demo(args.topics, args.duration, args.output_dir)
        if result:
            print(f"\n🎉 Demo recording completed!")
            print(f"📁 Output directory: {result}")
            print(f"\nNext steps:")
            print(
                f"1. Create GIF: python scripts/demo_recorder.py --mode gif --input-video {result}/your_video.mp4")
            print(f"2. Upload to MLflow: mlflow log-artifact {result}")

    elif args.mode == 'gif':
        # Create GIF from video
        if not args.input_video:
            print(f"❌ --input-video required for GIF mode")
            sys.exit(1)

        if not args.output_gif:
            video_path = Path(args.input_video)
            args.output_gif = video_path.parent / f"{video_path.stem}_demo.gif"

        result = create_navigation_demo_gif(
            args.input_video, args.output_gif, args.duration)
        if result:
            print(f"\n🎉 Demo GIF created!")
            print(f"📄 GIF file: {result}")

    elif args.mode == 'comparison':
        # Create comparison GIF
        if not args.before_video or not args.after_video:
            print(f"❌ --before-video and --after-video required for comparison mode")
            sys.exit(1)

        if not args.output_gif:
            args.output_gif = "comparison_demo.gif"

        result = create_comparison_gif(
            args.before_video, args.after_video, args.output_gif)
        if result:
            print(f"\n🎉 Comparison GIF created!")
            print(f"📄 GIF file: {result}")

    elif args.mode == 'screenshot':
        # Capture RViz screenshots
        screenshots = capture_rviz_screenshots(args.duration)
        if screenshots:
            print(f"\n🎉 Screenshots captured!")
            for screenshot in screenshots:
                print(f"📸 {screenshot}")

    # Log to MLflow if available
    if MLFLOW_AVAILABLE and args.mode in ['gif', 'comparison']:
        try:
            mlflow.start_run()
            if args.mode == 'gif' and result:
                mlflow.log_artifact(result)
            elif args.mode == 'comparison' and result:
                mlflow.log_artifact(result)
            mlflow.end_run()
            print(f"📊 Demo artifacts logged to MLflow")
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
