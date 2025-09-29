#!/usr/bin/env python3
"""Test the complete warehouse AGV pipeline."""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
from pathlib import Path
import yaml
import json
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge


class PipelineTestNode(Node):
    """Test node for complete warehouse AGV pipeline."""

    def __init__(self):
        super().__init__('pipeline_test')

        self.bridge = CvBridge()

        # Test parameters
        self.test_images_dir = Path('data/synthetic/clear')
        self.results = {
            'test_start_time': time.time(),
            'tests': [],
            'metrics': {}
        }

        # Publishers (simulate camera input)
        self.image_pub = self.create_publisher(
            Image, '/camera/image_raw', 10)

        # Subscribers (pipeline outputs)
        self.enhanced_sub = self.create_subscription(
            Image, '/enhanced_image', self.enhanced_callback, 10)
        self.detection_sub = self.create_subscription(
            Image, '/detection_image', self.detection_callback, 10)
        self.bbox_cloud_sub = self.create_subscription(
            PointCloud2, '/bbox_cloud', self.bbox_callback, 10)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 10)

        # Test state
        self.current_test = None
        self.test_results = {}

        # Create timer for test sequence
        self.test_timer = self.create_timer(2.0, self.run_test_sequence)
        self.test_index = 0

        self.get_logger().info("🧪 Pipeline test node initialized")

    def load_test_images(self):
        """Load test images from synthetic dataset."""
        if not self.test_images_dir.exists():
            self.get_logger().error(
                f"Test images directory not found: {self.test_images_dir}")
            return []

        image_files = list(self.test_images_dir.glob(
            '*.jpg'))[:10]  # Test 10 images
        self.get_logger().info(f"Found {len(image_files)} test images")
        return image_files

    def run_test_sequence(self):
        """Run pipeline test sequence."""
        test_images = self.load_test_images()

        if self.test_index >= len(test_images):
            self.get_logger().info("🎉 All tests completed")
            self.generate_test_report()
            self.test_timer.cancel()
            return

        # Load and publish test image
        image_path = test_images[self.test_index]
        cv_image = cv2.imread(str(image_path))

        if cv_image is not None:
            self.get_logger().info(
                f"📸 Testing image {self.test_index + 1}/{len(test_images)}: {image_path.name}")

            # Start new test
            self.current_test = {
                'image_name': image_path.name,
                'start_time': time.time(),
                'enhanced_received': False,
                'detection_received': False,
                'bbox_received': False,
                'costmap_received': False,
                'metrics': {}
            }

            # Publish test image
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_link'
            self.image_pub.publish(ros_image)

            self.test_index += 1

    def enhanced_callback(self, msg):
        """Callback for enhanced image (GridFormer output)."""
        if self.current_test:
            self.current_test['enhanced_received'] = True
            self.current_test['enhanced_time'] = time.time(
            ) - self.current_test['start_time']
            self.get_logger().info(
                f"   ✅ Enhanced image received ({self.current_test['enhanced_time']:.3f}s)")
            self.check_test_completion()

    def detection_callback(self, msg):
        """Callback for detection image (YOLO output)."""
        if self.current_test:
            self.current_test['detection_received'] = True
            self.current_test['detection_time'] = time.time(
            ) - self.current_test['start_time']
            self.get_logger().info(
                f"   ✅ Detection image received ({self.current_test['detection_time']:.3f}s)")
            self.check_test_completion()

    def bbox_callback(self, msg):
        """Callback for bounding box point cloud."""
        if self.current_test:
            self.current_test['bbox_received'] = True
            self.current_test['bbox_time'] = time.time() - \
                self.current_test['start_time']

            # Count detected objects
            point_count = len(msg.data) // (msg.point_step or 1)
            self.current_test['detected_objects'] = point_count

            self.get_logger().info(
                f"   ✅ BBox cloud received ({self.current_test['bbox_time']:.3f}s, {point_count} points)")
            self.check_test_completion()

    def costmap_callback(self, msg):
        """Callback for local costmap."""
        if self.current_test:
            self.current_test['costmap_received'] = True
            self.current_test['costmap_time'] = time.time(
            ) - self.current_test['start_time']

            # Analyze costmap for obstacles
            costmap_data = np.array(msg.data).reshape(
                (msg.info.height, msg.info.width))
            # Threshold for obstacles
            obstacle_cells = np.sum(costmap_data > 50)
            self.current_test['obstacle_cells'] = obstacle_cells

            self.get_logger().info(
                f"   ✅ Costmap received ({self.current_test['costmap_time']:.3f}s, {obstacle_cells} obstacle cells)")
            self.check_test_completion()

    def check_test_completion(self):
        """Check if current test is complete."""
        if not self.current_test:
            return

        # Check if all components responded
        all_received = (
            self.current_test['enhanced_received'] and
            self.current_test['detection_received'] and
            self.current_test['bbox_received'] and
            self.current_test['costmap_received']
        )

        if all_received:
            # Calculate total time
            total_time = time.time() - self.current_test['start_time']
            self.current_test['total_time'] = total_time

            # Calculate metrics
            self.current_test['metrics'] = {
                'fps': 1.0 / total_time if total_time > 0 else 0,
                'latency_ms': total_time * 1000,
                'pipeline_success': True
            }

            self.get_logger().info(
                f"   🎯 Test completed: {total_time:.3f}s total, {self.current_test['metrics']['fps']:.1f} FPS")

            # Store results
            self.results['tests'].append(self.current_test.copy())
            self.current_test = None

    def generate_test_report(self):
        """Generate comprehensive test report."""
        if not self.results['tests']:
            self.get_logger().error("No test results to report")
            return

        # Calculate aggregate metrics
        total_tests = len(self.results['tests'])
        successful_tests = sum(
            1 for test in self.results['tests'] if test['metrics']['pipeline_success'])

        latencies = [test['metrics']['latency_ms']
                     for test in self.results['tests']]
        fps_values = [test['metrics']['fps'] for test in self.results['tests']]

        self.results['metrics'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests * 100,
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'avg_fps': np.mean(fps_values),
            'test_duration': time.time() - self.results['test_start_time']
        }

        # Print report
        print("\n" + "="*60)
        print("🧪 WAREHOUSE AGV PIPELINE TEST REPORT")
        print("="*60)

        metrics = self.results['metrics']
        print(f"📊 Summary:")
        print(f"   Tests run: {metrics['total_tests']}")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        print(f"   Average latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"   Average FPS: {metrics['avg_fps']:.1f}")
        print(f"   Total test time: {metrics['test_duration']:.1f}s")

        print(f"\n🔍 Component Performance:")

        # Component-wise analysis
        enhanced_times = [test.get('enhanced_time', 0)
                          for test in self.results['tests']]
        detection_times = [test.get('detection_time', 0)
                           for test in self.results['tests']]
        bbox_times = [test.get('bbox_time', 0)
                      for test in self.results['tests']]
        costmap_times = [test.get('costmap_time', 0)
                         for test in self.results['tests']]

        print(f"   GridFormer (avg): {np.mean(enhanced_times)*1000:.1f}ms")
        print(f"   YOLO (avg): {np.mean(detection_times)*1000:.1f}ms")
        print(f"   BBox Cloud (avg): {np.mean(bbox_times)*1000:.1f}ms")
        print(f"   Costmap (avg): {np.mean(costmap_times)*1000:.1f}ms")

        # Detection analysis
        detected_objects = [test.get('detected_objects', 0)
                            for test in self.results['tests']]
        obstacle_cells = [test.get('obstacle_cells', 0)
                          for test in self.results['tests']]

        print(f"\n🎯 Detection Results:")
        print(f"   Avg objects detected: {np.mean(detected_objects):.1f}")
        print(f"   Avg obstacle cells: {np.mean(obstacle_cells):.1f}")

        # Performance assessment
        print(f"\n🏆 Performance Assessment:")
        if metrics['avg_fps'] >= 5.0:
            print(
                f"   ✅ Real-time performance achieved ({metrics['avg_fps']:.1f} FPS)")
        elif metrics['avg_fps'] >= 2.0:
            print(
                f"   ⚠️  Moderate performance ({metrics['avg_fps']:.1f} FPS)")
        else:
            print(f"   ❌ Performance too slow ({metrics['avg_fps']:.1f} FPS)")

        if metrics['success_rate'] >= 90:
            print(
                f"   ✅ High reliability ({metrics['success_rate']:.1f}% success)")
        elif metrics['success_rate'] >= 70:
            print(
                f"   ⚠️  Moderate reliability ({metrics['success_rate']:.1f}% success)")
        else:
            print(
                f"   ❌ Low reliability ({metrics['success_rate']:.1f}% success)")

        # Save results
        report_file = Path('pipeline_test_results.json')
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Detailed results saved: {report_file}")
        print("="*60)


def main():
    rclpy.init()

    # Check if required topics are available (basic connectivity test)
    print("🔍 Starting Pipeline Test...")
    print("Make sure the following are running:")
    print("  - GridFormer node (perception/gridformer_node.py)")
    print("  - YOLO node (perception/yolov8_node.py)")
    print("  - Costmap node (navigation/bbox2costmap_node)")
    print("  - RViz (optional, for visualization)")

    node = PipelineTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
