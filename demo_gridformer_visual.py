#!/usr/bin/env python3
"""
GridFormer Visual Demo
Interactive demo to see before/after image restoration results
"""

import cv2
import numpy as np
import os
import time
from typing import Tuple
import argparse

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ ONNX Runtime not available")
    ONNX_AVAILABLE = False
    exit(1)


class GridFormerVisualDemo:
    """Visual demo for GridFormer image restoration"""

    def __init__(self, model_path: str = "models/gridformer_adapted.onnx"):
        self.model_path = model_path
        self.session = None
        self.load_model()

    def load_model(self):
        """Load ONNX model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False

        try:
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("🚀 Using CUDA acceleration")
            else:
                print("💻 Using CPU for inference")

            self.session = ort.InferenceSession(
                self.model_path, providers=providers)
            print(f"✅ Model loaded: {self.model_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (512, 512))
        normalized = resized.astype(np.float32) / 255.0
        preprocessed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        return preprocessed

    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output"""
        output = output.squeeze(0)
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        output = cv2.resize(output, (original_shape[1], original_shape[0]))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process image and return result with timing"""
        original_shape = image.shape[:2]

        start_time = time.time()
        input_tensor = self.preprocess_image(image)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        restored_image = self.postprocess_output(output[0], original_shape)
        inference_time = time.time() - start_time

        return restored_image, inference_time

    def create_degraded_image(self, original: np.ndarray, degradation_type: str = "rain") -> np.ndarray:
        """Create artificially degraded image"""
        degraded = original.copy().astype(np.float32)

        if degradation_type == "rain":
            # Add rain effect
            rain_mask = np.random.random(original.shape[:2]) < 0.01
            rain_mask = rain_mask.astype(np.uint8) * 255
            rain_mask = cv2.GaussianBlur(rain_mask, (3, 3), 0)
            rain_mask = np.stack([rain_mask] * 3, axis=2)
            degraded = np.where(rain_mask > 128, 255, degraded)

        elif degradation_type == "fog":
            # Add fog effect
            fog = np.ones_like(degraded) * 200
            alpha = 0.6
            degraded = alpha * fog + (1 - alpha) * degraded

        elif degradation_type == "noise":
            # Add noise
            noise = np.random.normal(0, 25, degraded.shape)
            degraded = degraded + noise

        elif degradation_type == "blur":
            # Add motion blur
            degraded = cv2.GaussianBlur(degraded, (15, 15), 0)

        return np.clip(degraded, 0, 255).astype(np.uint8)

    def show_existing_results(self):
        """Show existing test results"""
        print("\n📸 Showing existing test results...")

        original_path = "dummy_test_original.jpg"
        restored_path = "dummy_test_restored.jpg"

        if os.path.exists(original_path) and os.path.exists(restored_path):
            original = cv2.imread(original_path)
            restored = cv2.imread(restored_path)

            # Create side-by-side comparison
            comparison = np.hstack([original, restored])

            # Add labels
            cv2.putText(comparison, "Original (Degraded)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, "Restored", (original.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("GridFormer Results - Existing Test", comparison)
            print("📺 Image window opened. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Test result images not found")

    def live_demo_with_webcam(self):
        """Live demo using webcam"""
        print("\n📹 Starting webcam demo...")
        print("Press 'q' to quit, 'space' to process frame")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Show original frame
            cv2.imshow("Webcam - Press SPACE to process", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space key
                print("🔄 Processing frame...")

                # Create degraded version
                degraded = self.create_degraded_image(frame, "rain")

                # Process with GridFormer
                restored, inference_time = self.process_image(degraded)

                # Create comparison
                comparison = np.hstack([frame, degraded, restored])

                # Add labels
                h, w = frame.shape[:2]
                cv2.putText(comparison, "Original", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, "Degraded", (w + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(comparison, f"Restored ({inference_time:.1f}s)", (2*w + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("GridFormer Comparison", comparison)
                print(f"✅ Processing completed in {inference_time:.2f}s")

        cap.release()
        cv2.destroyAllWindows()

    def demo_with_sample_images(self):
        """Demo with sample degraded images"""
        print("\n🖼️  Creating demo with sample images...")

        # Create sample image
        sample_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Add some realistic patterns
        cv2.rectangle(sample_image, (100, 100), (200, 200), (255, 0, 0), -1)
        cv2.circle(sample_image, (400, 300), 50, (0, 255, 0), -1)
        cv2.line(sample_image, (0, 400), (640, 400), (0, 0, 255), 5)

        degradation_types = ["rain", "fog", "noise", "blur"]

        for deg_type in degradation_types:
            print(f"\n🔄 Testing {deg_type} degradation...")

            # Create degraded version
            degraded = self.create_degraded_image(sample_image, deg_type)

            # Process with GridFormer
            restored, inference_time = self.process_image(degraded)

            # Create comparison
            comparison = np.hstack([sample_image, degraded, restored])

            # Add labels
            h, w = sample_image.shape[:2]
            cv2.putText(comparison, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, f"Degraded ({deg_type})", (w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, f"Restored ({inference_time:.1f}s)", (2*w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save comparison
            output_path = f"demo_{deg_type}_comparison.jpg"
            cv2.imwrite(output_path, comparison)
            print(f"💾 Saved: {output_path}")

            # Show window
            cv2.imshow(f"GridFormer Demo - {deg_type.title()}", comparison)
            print(f"📺 Press any key to continue to next degradation type...")
            cv2.waitKey(0)

        cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GridFormer Visual Demo")
    parser.add_argument("--mode", choices=["existing", "webcam", "samples"],
                        default="existing", help="Demo mode")
    parser.add_argument("--model", default="models/gridformer_adapted.onnx",
                        help="Path to ONNX model")

    args = parser.parse_args()

    print("🎬 GridFormer Visual Demo")
    print("=" * 50)

    demo = GridFormerVisualDemo(args.model)

    if not demo.session:
        print("❌ Cannot initialize demo without model")
        return

    if args.mode == "existing":
        demo.show_existing_results()
    elif args.mode == "webcam":
        demo.live_demo_with_webcam()
    elif args.mode == "samples":
        demo.demo_with_sample_images()

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
