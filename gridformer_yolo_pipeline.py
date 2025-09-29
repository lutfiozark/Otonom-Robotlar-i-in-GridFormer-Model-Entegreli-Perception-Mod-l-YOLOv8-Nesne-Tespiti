#!/usr/bin/env python3
"""
GridFormer + YOLOv8 End-to-End Pipeline
Weather degraded → Image restoration → Object detection
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Dict
import argparse

# Import our models
try:
    import onnxruntime as ort
    from ultralytics import YOLO
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    MODELS_AVAILABLE = False
    exit(1)


class GridFormerYOLOPipeline:
    """Complete pipeline: Weather restoration + Object detection"""

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
        """Load both GridFormer and YOLOv8 models"""
        print("🔄 Loading pipeline models...")

        # Load GridFormer (ONNX)
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
        else:
            print(
                f"⚠️  GridFormer model not found: {self.gridformer_model_path}")

        # Load YOLOv8
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ YOLOv8 loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8: {e}")

    def create_degraded_image(self, original: np.ndarray, degradation_type: str = "rain") -> np.ndarray:
        """Create weather degraded image"""
        degraded = original.copy().astype(np.float32)

        if degradation_type == "rain":
            # Heavy rain effect
            rain_mask = np.random.random(original.shape[:2]) < 0.02
            rain_mask = rain_mask.astype(np.uint8) * 255
            rain_mask = cv2.GaussianBlur(rain_mask, (5, 5), 0)
            rain_mask = np.stack([rain_mask] * 3, axis=2)
            degraded = np.where(rain_mask > 100, 255, degraded)

            # Add overall darkness
            degraded = degraded * 0.7

        elif degradation_type == "fog":
            # Dense fog effect
            fog = np.ones_like(degraded) * 220
            alpha = 0.7
            degraded = alpha * fog + (1 - alpha) * degraded

        elif degradation_type == "snow":
            # Snow effect
            snow_mask = np.random.random(original.shape[:2]) < 0.005
            snow_mask = snow_mask.astype(np.uint8) * 255
            snow_mask = cv2.dilate(snow_mask, np.ones((3, 3), np.uint8))
            snow_mask = np.stack([snow_mask] * 3, axis=2)
            degraded = np.where(snow_mask > 0, 255, degraded)

            # Add overall brightness reduction
            degraded = degraded * 0.8

        elif degradation_type == "storm":
            # Storm effect (combination)
            # Heavy rain
            rain_mask = np.random.random(original.shape[:2]) < 0.03
            rain_mask = rain_mask.astype(np.uint8) * 255
            rain_mask = cv2.GaussianBlur(rain_mask, (7, 7), 0)
            rain_mask = np.stack([rain_mask] * 3, axis=2)
            degraded = np.where(rain_mask > 80, 255, degraded)

            # Dark atmosphere
            degraded = degraded * 0.5

            # Add blur
            degraded = cv2.GaussianBlur(degraded, (5, 5), 0)

        return np.clip(degraded, 0, 255).astype(np.uint8)

    def gridformer_restore(self, degraded_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Restore image using GridFormer"""
        if not self.gridformer_session:
            print("⚠️  GridFormer not available, returning original")
            return degraded_image, 0.0

        start_time = time.time()

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

        # Resize back to original size
        original_shape = degraded_image.shape[:2]
        restored = cv2.resize(
            output_tensor, (original_shape[1], original_shape[0]))
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

        inference_time = time.time() - start_time
        return restored, inference_time

    def yolo_detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict], float]:
        """Detect objects using YOLOv8"""
        if not self.yolo_model:
            print("⚠️  YOLOv8 not available")
            return image, [], 0.0

        start_time = time.time()

        # Run detection
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)

        # Parse results
        detections = []
        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]

                    # Store detection
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })

                    # Draw on image
                    cv2.rectangle(annotated_image, (int(x1), int(
                        y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        inference_time = time.time() - start_time
        return annotated_image, detections, inference_time

    def run_pipeline(self, input_image: np.ndarray, degradation_type: str = "rain") -> Dict:
        """Run complete pipeline: degrade → restore → detect"""
        print(f"\n🚀 Running pipeline with {degradation_type} degradation...")

        results = {
            'original_image': input_image,
            'degradation_type': degradation_type
        }

        # Step 1: Create degraded image
        start_time = time.time()
        degraded_image = self.create_degraded_image(
            input_image, degradation_type)
        results['degraded_image'] = degraded_image
        results['degradation_time'] = time.time() - start_time

        # Step 2: Restore using GridFormer
        restored_image, restore_time = self.gridformer_restore(degraded_image)
        results['restored_image'] = restored_image
        results['restoration_time'] = restore_time

        # Step 3a: Detect objects in degraded image
        degraded_detected, degraded_detections, degraded_detect_time = self.yolo_detect(
            degraded_image)
        results['degraded_detected'] = degraded_detected
        results['degraded_detections'] = degraded_detections
        results['degraded_detection_time'] = degraded_detect_time

        # Step 3b: Detect objects in restored image
        restored_detected, restored_detections, restored_detect_time = self.yolo_detect(
            restored_image)
        results['restored_detected'] = restored_detected
        results['restored_detections'] = restored_detections
        results['restored_detection_time'] = restored_detect_time

        # Calculate metrics
        results['total_time'] = (results['degradation_time'] +
                                 results['restoration_time'] +
                                 results['restored_detection_time'])

        results['detection_improvement'] = len(
            restored_detections) - len(degraded_detections)

        print(f"📊 Pipeline Results:")
        print(f"   Degradation time: {results['degradation_time']*1000:.1f}ms")
        print(f"   Restoration time: {results['restoration_time']*1000:.1f}ms")
        print(
            f"   Detection time (degraded): {degraded_detect_time*1000:.1f}ms")
        print(
            f"   Detection time (restored): {restored_detect_time*1000:.1f}ms")
        print(f"   Total time: {results['total_time']*1000:.1f}ms")
        print(f"   Objects detected (degraded): {len(degraded_detections)}")
        print(f"   Objects detected (restored): {len(restored_detections)}")
        print(
            f"   Detection improvement: {results['detection_improvement']:+d}")

        return results

    def create_comparison_visualization(self, results: Dict, save_path: str = None):
        """Create visual comparison of pipeline results"""
        original = results['original_image']
        degraded = results['degraded_image']
        restored = results['restored_image']
        degraded_detected = results['degraded_detected']
        restored_detected = results['restored_detected']

        # Create 2x3 grid
        h, w = original.shape[:2]

        # Top row: Original | Degraded | Restored
        top_row = np.hstack([original, degraded, restored])

        # Bottom row: Detection on degraded | Detection on restored | Metrics
        bottom_row = np.hstack(
            [degraded_detected, restored_detected, np.ones((h, w, 3), dtype=np.uint8) * 50])

        # Combine
        comparison = np.vstack([top_row, bottom_row])

        # Add labels
        labels = [
            ("Original", (10, 30)),
            ("Degraded", (w + 10, 30)),
            ("Restored", (2*w + 10, 30)),
            ("Detected (Degraded)", (10, h + 30)),
            ("Detected (Restored)", (w + 10, h + 30)),
            ("Metrics", (2*w + 10, h + 30))
        ]

        for label, pos in labels:
            cv2.putText(comparison, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add metrics text
        metrics_x = 2*w + 20
        metrics_y = h + 70
        metrics_text = [
            f"Degradation: {results['degradation_type']}",
            f"Restoration: {results['restoration_time']*1000:.0f}ms",
            f"Detection Time:",
            f"  Degraded: {results['degraded_detection_time']*1000:.0f}ms",
            f"  Restored: {results['restored_detection_time']*1000:.0f}ms",
            f"Objects Found:",
            f"  Degraded: {len(results['degraded_detections'])}",
            f"  Restored: {len(results['restored_detections'])}",
            f"Improvement: {results['detection_improvement']:+d}"
        ]

        for i, text in enumerate(metrics_text):
            cv2.putText(comparison, text, (metrics_x, metrics_y + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save
        if save_path:
            cv2.imwrite(save_path, comparison)
            print(f"💾 Comparison saved: {save_path}")

        return comparison


def create_test_scene():
    """Create a test scene with various objects"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 120

    # Sky gradient
    for y in range(200):
        color = int(120 + (y/200) * 100)
        image[y, :] = [color, color, min(255, color + 20)]

    # Ground
    image[350:, :] = [80, 100, 60]

    # Buildings
    cv2.rectangle(image, (50, 200), (150, 350), (100, 100, 100), -1)
    cv2.rectangle(image, (200, 150), (300, 350), (120, 120, 120), -1)
    cv2.rectangle(image, (450, 180), (550, 350), (90, 90, 90), -1)

    # Cars
    cv2.rectangle(image, (100, 320), (180, 350), (0, 0, 200), -1)  # Red car
    cv2.rectangle(image, (300, 310), (380, 350), (200, 200, 0), -1)  # Blue car

    # People
    cv2.circle(image, (250, 320), 8, (255, 200, 150), -1)  # Head
    cv2.rectangle(image, (245, 325), (255, 350), (0, 100, 200), -1)  # Body

    cv2.circle(image, (400, 315), 8, (255, 200, 150), -1)  # Head
    cv2.rectangle(image, (395, 320), (405, 350), (200, 0, 100), -1)  # Body

    # Add some texture
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GridFormer + YOLOv8 Pipeline Demo")
    parser.add_argument("--gridformer", default="models/gridformer_adapted.onnx",
                        help="GridFormer model path")
    parser.add_argument("--yolo", default="yolov8s.pt",
                        help="YOLOv8 model path")
    parser.add_argument("--degradation", choices=["rain", "fog", "snow", "storm"],
                        default="rain", help="Weather degradation type")

    args = parser.parse_args()

    print("🎬 GridFormer + YOLOv8 Pipeline Demo")
    print("=" * 60)

    # Initialize pipeline
    pipeline = GridFormerYOLOPipeline(args.gridformer, args.yolo)

    # Create test scene
    test_scene = create_test_scene()

    # Run pipeline
    results = pipeline.run_pipeline(test_scene, args.degradation)

    # Create visualization
    comparison = pipeline.create_comparison_visualization(
        results,
        f"pipeline_{args.degradation}_comparison.jpg"
    )

    # Show results
    cv2.imshow("GridFormer + YOLOv8 Pipeline", comparison)
    print("\n📺 Pipeline visualization displayed. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n✅ Pipeline demo completed!")
    print("🎯 Ready for real-world deployment")


if __name__ == "__main__":
    main()
