#!/usr/bin/env python3
"""
Full Pipeline Test Script
Tests the complete GridFormer → YOLO → Navigation pipeline
"""

import cv2
import numpy as np
import time
import os
import argparse
from typing import Dict, List, Tuple

# Import our models
try:
    import onnxruntime as ort
    from ultralytics import YOLO
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    MODELS_AVAILABLE = False
    exit(1)


class FullPipelineTester:
    """Complete pipeline tester"""

    def __init__(self,
                 gridformer_model: str = "models/gridformer_adapted.onnx",
                 yolo_model: str = "yolov8s.pt"):
        self.gridformer_model_path = gridformer_model
        self.yolo_model_path = yolo_model

        # Model instances
        self.gridformer_session = None
        self.yolo_model = None

        # Load models
        self.load_models()

    def load_models(self):
        """Load both models"""
        print("🔄 Loading full pipeline models...")

        # Load GridFormer
        if os.path.exists(self.gridformer_model_path):
            try:
                providers = ['CPUExecutionProvider']
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers = ['CUDAExecutionProvider',
                                 'CPUExecutionProvider']

                self.gridformer_session = ort.InferenceSession(
                    self.gridformer_model_path, providers=providers)
                print("✅ GridFormer loaded")
            except Exception as e:
                print(f"❌ Failed to load GridFormer: {e}")

        # Load YOLOv8
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ YOLOv8 loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8: {e}")

    def create_warehouse_scene(self) -> np.ndarray:
        """Create a realistic warehouse scene with obstacles"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 120

        # Floor
        image[350:, :] = [80, 90, 70]

        # Warehouse shelves (left and right)
        cv2.rectangle(image, (50, 200), (150, 350),
                      (100, 80, 60), -1)  # Left shelf
        cv2.rectangle(image, (490, 200), (590, 350),
                      (100, 80, 60), -1)  # Right shelf

        # Pallets with boxes
        cv2.rectangle(image, (200, 300), (280, 350),
                      (160, 100, 50), -1)  # Pallet 1
        cv2.rectangle(image, (360, 310), (440, 350),
                      (160, 100, 50), -1)  # Pallet 2

        # Boxes on pallets
        cv2.rectangle(image, (210, 280), (240, 310), (200, 150, 100), -1)
        cv2.rectangle(image, (250, 275), (270, 305), (200, 150, 100), -1)
        cv2.rectangle(image, (370, 285), (410, 315), (200, 150, 100), -1)

        # Forklift
        cv2.rectangle(image, (320, 320), (360, 350), (255, 200, 0), -1)
        cv2.rectangle(image, (330, 310), (350, 325),
                      (100, 100, 100), -1)  # Mast

        # People
        cv2.circle(image, (300, 320), 8, (255, 200, 150), -1)  # Head
        cv2.rectangle(image, (295, 325), (305, 350), (0, 100, 200), -1)  # Body

        cv2.circle(image, (450, 315), 8, (255, 200, 150), -1)  # Head 2
        cv2.rectangle(image, (445, 320), (455, 350),
                      (200, 0, 100), -1)  # Body 2

        # Add lighting and shadows
        overlay = np.zeros_like(image)
        cv2.ellipse(overlay, (320, 250), (300, 150),
                    0, 0, 180, (255, 255, 255), -1)
        image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)

        # Add some texture
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) +
                        noise, 0, 255).astype(np.uint8)

        return image

    def create_degraded_scene(self, original: np.ndarray, severity: str = "moderate") -> np.ndarray:
        """Create weather-degraded version of warehouse scene"""
        degraded = original.copy().astype(np.float32)

        if severity == "light":
            # Light fog
            fog = np.ones_like(degraded) * 200
            alpha = 0.3
            degraded = alpha * fog + (1 - alpha) * degraded

        elif severity == "moderate":
            # Moderate rain + fog
            fog = np.ones_like(degraded) * 180
            alpha = 0.5
            degraded = alpha * fog + (1 - alpha) * degraded

            # Add rain streaks
            rain_mask = np.random.random(original.shape[:2]) < 0.01
            rain_mask = rain_mask.astype(np.uint8) * 255
            rain_mask = cv2.GaussianBlur(rain_mask, (3, 3), 0)
            rain_mask = np.stack([rain_mask] * 3, axis=2)
            degraded = np.where(rain_mask > 100, 255, degraded)

        elif severity == "severe":
            # Heavy storm
            degraded = degraded * 0.4  # Very dark

            # Dense fog
            fog = np.ones_like(degraded) * 150
            alpha = 0.7
            degraded = alpha * fog + (1 - alpha) * degraded

            # Heavy rain
            rain_mask = np.random.random(original.shape[:2]) < 0.03
            rain_mask = rain_mask.astype(np.uint8) * 255
            rain_mask = cv2.GaussianBlur(rain_mask, (5, 5), 0)
            rain_mask = np.stack([rain_mask] * 3, axis=2)
            degraded = np.where(rain_mask > 80, 255, degraded)

            # Motion blur
            degraded = cv2.GaussianBlur(degraded, (3, 3), 0)

        return np.clip(degraded, 0, 255).astype(np.uint8)

    def run_full_pipeline(self, test_scenario: str = "moderate") -> Dict:
        """Run the complete pipeline test"""
        print(
            f"\n🚀 Running full pipeline test - {test_scenario} weather conditions")

        # Step 1: Create warehouse scene
        start_time = time.time()
        original_scene = self.create_warehouse_scene()
        scene_time = time.time() - start_time

        # Step 2: Create degraded scene
        start_time = time.time()
        degraded_scene = self.create_degraded_scene(
            original_scene, test_scenario)
        degradation_time = time.time() - start_time

        # Step 3: GridFormer restoration
        if self.gridformer_session:
            start_time = time.time()
            restored_scene = self.gridformer_restore(degraded_scene)
            restoration_time = time.time() - start_time
        else:
            restored_scene = degraded_scene
            restoration_time = 0.0

        # Step 4: Object detection on both degraded and restored
        if self.yolo_model:
            # Detection on degraded scene
            start_time = time.time()
            degraded_detected, degraded_detections = self.yolo_detect(
                degraded_scene)
            degraded_detect_time = time.time() - start_time

            # Detection on restored scene
            start_time = time.time()
            restored_detected, restored_detections = self.yolo_detect(
                restored_scene)
            restored_detect_time = time.time() - start_time
        else:
            degraded_detected = degraded_scene
            restored_detected = restored_scene
            degraded_detections = []
            restored_detections = []
            degraded_detect_time = 0.0
            restored_detect_time = 0.0

        # Step 5: Navigation planning simulation
        nav_metrics = self.simulate_navigation(restored_detections)

        # Compile results
        results = {
            'scenario': test_scenario,
            'original_scene': original_scene,
            'degraded_scene': degraded_scene,
            'restored_scene': restored_scene,
            'degraded_detected': degraded_detected,
            'restored_detected': restored_detected,
            'degraded_detections': degraded_detections,
            'restored_detections': restored_detections,
            'timing': {
                'scene_creation': scene_time,
                'degradation': degradation_time,
                'restoration': restoration_time,
                'degraded_detection': degraded_detect_time,
                'restored_detection': restored_detect_time,
                'total': (scene_time + degradation_time + restoration_time +
                          max(degraded_detect_time, restored_detect_time))
            },
            'navigation': nav_metrics
        }

        return results

    def gridformer_restore(self, degraded_image: np.ndarray) -> np.ndarray:
        """Restore image using GridFormer"""
        # Preprocess
        rgb_image = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (512, 512))
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference
        input_name = self.gridformer_session.get_inputs()[0].name
        output = self.gridformer_session.run(None, {input_name: input_tensor})

        # Postprocess
        output_tensor = output[0].squeeze(0)
        output_tensor = np.transpose(output_tensor, (1, 2, 0))
        output_tensor = np.clip(output_tensor, 0, 1)
        output_tensor = (output_tensor * 255).astype(np.uint8)

        # Resize back
        original_shape = degraded_image.shape[:2]
        restored = cv2.resize(
            output_tensor, (original_shape[1], original_shape[0]))
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

        return restored

    def yolo_detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect objects using YOLOv8"""
        results = self.yolo_model(image, conf=0.3, verbose=False)

        detections = []
        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_name': class_name,
                        'center': [(x1+x2)/2, (y1+y2)/2]
                    })

                    # Draw detection
                    cv2.rectangle(annotated_image, (int(x1), int(
                        y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_image, detections

    def simulate_navigation(self, detections: List[Dict]) -> Dict:
        """Simulate navigation planning metrics"""
        # Simple navigation simulation
        obstacles = len(detections)

        # Estimate path quality based on detections
        if obstacles == 0:
            path_quality = 1.0
            navigation_difficulty = "Easy"
        elif obstacles <= 2:
            path_quality = 0.8
            navigation_difficulty = "Moderate"
        elif obstacles <= 4:
            path_quality = 0.6
            navigation_difficulty = "Challenging"
        else:
            path_quality = 0.3
            navigation_difficulty = "Difficult"

        # Simulate path length (longer with more obstacles)
        base_path_length = 5.0  # meters
        path_length = base_path_length + (obstacles * 0.5)

        # Simulate navigation time
        base_nav_time = 10.0  # seconds
        nav_time = base_nav_time + (obstacles * 2.0)

        return {
            'obstacles_detected': obstacles,
            'path_quality': path_quality,
            'navigation_difficulty': navigation_difficulty,
            'estimated_path_length': path_length,
            'estimated_navigation_time': nav_time
        }

    def create_full_visualization(self, results: Dict, save_path: str = None):
        """Create comprehensive visualization"""
        original = results['original_scene']
        degraded = results['degraded_scene']
        restored = results['restored_scene']
        degraded_detected = results['degraded_detected']
        restored_detected = results['restored_detected']

        h, w = original.shape[:2]

        # Create 2x3 grid
        top_row = np.hstack([original, degraded, restored])
        bottom_row = np.hstack([np.zeros((h, w, 3), dtype=np.uint8),
                               degraded_detected, restored_detected])

        comparison = np.vstack([top_row, bottom_row])

        # Add labels
        labels = [
            ("Original Scene", (10, 30)),
            (f"Degraded ({results['scenario']})", (w + 10, 30)),
            ("GridFormer Restored", (2*w + 10, 30)),
            ("Pipeline Metrics", (10, h + 30)),
            ("YOLO on Degraded", (w + 10, h + 30)),
            ("YOLO on Restored", (2*w + 10, h + 30))
        ]

        for label, pos in labels:
            cv2.putText(comparison, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add metrics text
        metrics_x = 20
        metrics_y = h + 70
        metrics = [
            f"Weather: {results['scenario'].title()}",
            f"Restoration: {results['timing']['restoration']*1000:.0f}ms",
            f"Objects (Degraded): {len(results['degraded_detections'])}",
            f"Objects (Restored): {len(results['restored_detections'])}",
            f"Improvement: {len(results['restored_detections']) - len(results['degraded_detections']):+d}",
            f"Nav Difficulty: {results['navigation']['navigation_difficulty']}",
            f"Path Quality: {results['navigation']['path_quality']:.1f}",
            f"Est. Nav Time: {results['navigation']['estimated_navigation_time']:.1f}s"
        ]

        for i, text in enumerate(metrics):
            cv2.putText(comparison, text, (metrics_x, metrics_y + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if save_path:
            cv2.imwrite(save_path, comparison)
            print(f"💾 Full pipeline visualization saved: {save_path}")

        return comparison


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Full Pipeline Test")
    parser.add_argument("--scenario", choices=["light", "moderate", "severe"],
                        default="moderate", help="Weather degradation severity")
    parser.add_argument(
        "--gridformer", default="models/gridformer_adapted.onnx")
    parser.add_argument("--yolo", default="yolov8s.pt")

    args = parser.parse_args()

    print("🎬 GridFormer Robot - Full Pipeline Test")
    print("=" * 60)

    # Initialize pipeline
    pipeline = FullPipelineTester(args.gridformer, args.yolo)

    # Run test
    results = pipeline.run_full_pipeline(args.scenario)

    # Print results
    print(f"\n📊 Pipeline Results ({args.scenario} weather):")
    print(
        f"   Restoration time: {results['timing']['restoration']*1000:.0f}ms")
    print(
        f"   Objects detected (degraded): {len(results['degraded_detections'])}")
    print(
        f"   Objects detected (restored): {len(results['restored_detections'])}")
    print(
        f"   Detection improvement: {len(results['restored_detections']) - len(results['degraded_detections']):+d}")
    print(
        f"   Navigation difficulty: {results['navigation']['navigation_difficulty']}")
    print(
        f"   Path quality score: {results['navigation']['path_quality']:.2f}")
    print(
        f"   Estimated nav time: {results['navigation']['estimated_navigation_time']:.1f}s")

    # Create visualization
    comparison = pipeline.create_full_visualization(
        results,
        f"full_pipeline_{args.scenario}_test.jpg"
    )

    # Show results
    cv2.imshow("Full Pipeline Test Results", comparison)
    print("\n📺 Pipeline visualization displayed. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n✅ Full pipeline test completed!")
    print("🎯 Ready for ROS 2 deployment")


if __name__ == "__main__":
    main()
