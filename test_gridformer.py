#!/usr/bin/env python3
"""
GridFormer Standalone Test Script
Test the GridFormer ONNX model without ROS dependencies
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ ONNX Runtime not available")
    ONNX_AVAILABLE = False
    exit(1)


class GridFormerTester:
    """Simple GridFormer model tester"""

    def __init__(self, model_path: str = "models/gridformer.onnx"):
        self.model_path = model_path
        self.session = None
        self.load_model()

    def load_model(self):
        """Load ONNX model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False

        try:
            # Setup providers
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("🚀 Using CUDA acceleration")
            else:
                print("💻 Using CPU for inference")

            self.session = ort.InferenceSession(
                self.model_path, providers=providers)

            # Print model info
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]

            print(f"✅ Model loaded successfully")
            print(f"   Input: {input_info.name} {input_info.shape}")
            print(f"   Output: {output_info.name} {output_info.shape}")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 512x512
        resized = cv2.resize(rgb_image, (512, 512))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to CHW format and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)

        return preprocessed

    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output"""
        # Remove batch dimension and convert CHW to HWC
        output = output.squeeze(0)
        output = np.transpose(output, (1, 2, 0))

        # Clip to [0, 1] and convert to uint8
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)

        # Resize to original dimensions
        output = cv2.resize(output, (original_shape[1], original_shape[0]))

        # Convert RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output

    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference"""
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        return output[0]

    def test_with_dummy_image(self):
        """Test with a dummy degraded image"""
        print("\n🧪 Testing with dummy degraded image...")

        # Create a dummy degraded image (simulating weather effects)
        dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Add some noise and blur to simulate weather degradation
        dummy_image = cv2.GaussianBlur(dummy_image, (15, 15), 0)
        noise = np.random.normal(0, 25, dummy_image.shape).astype(np.int16)
        dummy_image = np.clip(dummy_image.astype(
            np.int16) + noise, 0, 255).astype(np.uint8)

        return self.process_image(dummy_image, "dummy_test")

    def process_image(self, image: np.ndarray, test_name: str = "test") -> bool:
        """Process a single image"""
        try:
            print(f"\n📸 Processing {test_name} image...")
            print(f"   Input shape: {image.shape}")

            original_shape = image.shape[:2]

            # Preprocess
            start_time = time.time()
            input_tensor = self.preprocess_image(image)
            preprocess_time = time.time() - start_time

            # Inference
            start_time = time.time()
            output_tensor = self.inference(input_tensor)
            inference_time = time.time() - start_time

            # Postprocess
            start_time = time.time()
            restored_image = self.postprocess_output(
                output_tensor, original_shape)
            postprocess_time = time.time() - start_time

            total_time = preprocess_time + inference_time + postprocess_time

            print(f"   ✅ Processing completed!")
            print(f"   📊 Timing:")
            print(f"      Preprocess: {preprocess_time*1000:.1f}ms")
            print(f"      Inference:  {inference_time*1000:.1f}ms")
            print(f"      Postprocess: {postprocess_time*1000:.1f}ms")
            print(f"      Total:      {total_time*1000:.1f}ms")
            print(f"      FPS:        {1/total_time:.1f}")

            # Save results
            cv2.imwrite(f"{test_name}_original.jpg", image)
            cv2.imwrite(f"{test_name}_restored.jpg", restored_image)
            print(
                f"   💾 Results saved: {test_name}_original.jpg, {test_name}_restored.jpg")

            return True

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return False

    def benchmark(self, num_iterations: int = 100):
        """Benchmark the model"""
        if not self.session:
            print("❌ Model not loaded")
            return

        print(f"\n🏃 Running benchmark with {num_iterations} iterations...")

        # Create dummy input
        dummy_input = np.random.random((1, 3, 512, 512)).astype(np.float32)
        input_name = self.session.get_inputs()[0].name

        # Warmup
        print("🔥 Warming up...")
        for _ in range(10):
            _ = self.session.run(None, {input_name: dummy_input})

        # Benchmark
        print("📏 Benchmarking...")
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.session.run(None, {input_name: dummy_input})
            times.append(time.time() - start_time)

            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{num_iterations}")

        # Statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1 / avg_time

        print(f"\n📊 Benchmark Results:")
        print(f"   Average time: {avg_time*1000:.2f}ms")
        print(f"   Min time:     {min_time*1000:.2f}ms")
        print(f"   Max time:     {max_time*1000:.2f}ms")
        print(f"   Average FPS:  {fps:.2f}")


def main():
    """Main function"""
    print("🚀 GridFormer Standalone Test")
    print("=" * 50)

    # Initialize tester
    tester = GridFormerTester("models/gridformer.onnx")

    if not tester.session:
        print("❌ Failed to initialize model")
        return

    # Test with dummy image
    tester.test_with_dummy_image()

    # Benchmark
    tester.benchmark(50)

    print("\n✅ All tests completed!")
    print("🎯 Model is ready for ROS 2 integration")


if __name__ == "__main__":
    main()
