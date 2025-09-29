#!/usr/bin/env python3
"""
YOLOv8 Standalone Test
Test YOLOv8 object detection without ROS dependencies
"""

import cv2
import numpy as np
import time
import os
from typing import List, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("❌ Ultralytics YOLO not available")
    YOLO_AVAILABLE = False
    exit(1)


class YOLOv8Tester:
    """Simple YOLOv8 model tester"""

    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load YOLOv8 model"""
        try:
            print(f"🔄 Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print(f"✅ YOLOv8 model loaded successfully")

            # Print model info
            print(f"📊 Model info:")
            print(f"   Model type: {self.model_path}")
            print(f"   Classes: {len(self.model.names)} total")
            print(f"   Sample classes: {list(self.model.names.values())[:10]}")

            return True

        except Exception as e:
            print(f"❌ Failed to load YOLOv8 model: {e}")
            return False

    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """Run object detection on image"""
        if not self.model:
            print("❌ Model not loaded")
            return image, 0.0

        start_time = time.time()

        # Run detection
        results = self.model(image, conf=conf_threshold, verbose=False)

        inference_time = time.time() - start_time

        # Draw results on image
        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = self.model.names[class_id]

                    # Draw bounding box
                    cv2.rectangle(annotated_image, (int(x1), int(
                        y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_image, inference_time

    def create_test_image(self) -> np.ndarray:
        """Create a test image with objects"""
        # Create a simple scene
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Draw some basic shapes (cars, people simulation)
        # Car-like rectangle
        cv2.rectangle(image, (100, 200), (200, 280), (0, 0, 255), -1)
        cv2.rectangle(image, (120, 220), (180, 240), (255, 255, 255), -1)

        # Person-like ellipse
        cv2.ellipse(image, (350, 300), (20, 40), 0, 0, 360, (255, 128, 0), -1)
        cv2.circle(image, (350, 260), 15, (255, 200, 150), -1)

        # Add some noise to make it realistic
        noise = np.random.normal(0, 20, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) +
                        noise, 0, 255).astype(np.uint8)

        return image

    def test_with_sample_image(self):
        """Test with a sample image"""
        print("\n🧪 Testing with sample image...")

        # Create test image
        test_image = self.create_test_image()

        # Run detection
        detected_image, inference_time = self.detect_objects(test_image)

        # Create comparison
        comparison = np.hstack([test_image, detected_image])

        # Add labels
        h, w = test_image.shape[:2]
        cv2.putText(comparison, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Detected ({inference_time:.3f}s)", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save results
        cv2.imwrite("yolo_test_original.jpg", test_image)
        cv2.imwrite("yolo_test_detected.jpg", detected_image)
        cv2.imwrite("yolo_test_comparison.jpg", comparison)

        print(f"✅ Detection completed in {inference_time:.3f}s")
        print(f"💾 Results saved: yolo_test_*.jpg")

        # Show window
        cv2.imshow("YOLOv8 Test Results", comparison)
        print("📺 Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True

    def benchmark(self, num_iterations: int = 50):
        """Benchmark YOLOv8 model"""
        if not self.model:
            print("❌ Model not loaded")
            return

        print(
            f"\n🏃 Running YOLOv8 benchmark with {num_iterations} iterations...")

        # Create dummy input
        dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Warmup
        print("🔥 Warming up...")
        for _ in range(5):
            _ = self.model(dummy_image, verbose=False)

        # Benchmark
        print("📏 Benchmarking...")
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.model(dummy_image, verbose=False)
            times.append(time.time() - start_time)

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{num_iterations}")

        # Statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1 / avg_time

        print(f"\n📊 YOLOv8 Benchmark Results:")
        print(f"   Average time: {avg_time*1000:.2f}ms")
        print(f"   Min time:     {min_time*1000:.2f}ms")
        print(f"   Max time:     {max_time*1000:.2f}ms")
        print(f"   Average FPS:  {fps:.2f}")

        return avg_time, fps


def main():
    """Main function"""
    print("🚀 YOLOv8 Standalone Test")
    print("=" * 50)

    # Initialize tester
    tester = YOLOv8Tester("yolov8s.pt")

    if not tester.model:
        print("❌ Failed to initialize YOLOv8")
        return

    # Test with sample image
    tester.test_with_sample_image()

    # Benchmark
    tester.benchmark(30)

    print("\n✅ All YOLOv8 tests completed!")
    print("🎯 Ready for GridFormer + YOLO integration")


if __name__ == "__main__":
    main()
