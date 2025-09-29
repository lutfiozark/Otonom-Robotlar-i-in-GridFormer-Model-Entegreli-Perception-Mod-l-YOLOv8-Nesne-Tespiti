#!/usr/bin/env python3
"""
Optimized Models Testing Script
Test optimized GridFormer (384x384) and YOLO (416x416) models
"""

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time
from pathlib import Path

def test_optimized_models():
    """Test optimized models performance"""
    print("🚀 Testing OPTIMIZED Models (384x384 GridFormer + 416x416 YOLO)")
    print("=" * 60)
    
    # Model paths
    gridformer_path = "models/gridformer_optimized_384.onnx"
    yolo_path = "models/yolov8s_optimized_416.onnx"
    
    # Check if files exist
    if not Path(gridformer_path).exists():
        print(f"❌ GridFormer not found: {gridformer_path}")
        return False
    if not Path(yolo_path).exists():
        print(f"❌ YOLO not found: {yolo_path}")
        return False
    
    try:
        # Load models
        print("📥 Loading optimized models...")
        gf_session = ort.InferenceSession(gridformer_path)
        yolo_session = ort.InferenceSession(yolo_path)
        
        # Get model info
        gf_input = gf_session.get_inputs()[0].name
        gf_output = gf_session.get_outputs()[0].name
        yolo_input = yolo_session.get_inputs()[0].name
        yolo_output = yolo_session.get_outputs()[0].name
        
        print(f"✅ Models loaded successfully!")
        print(f"   GridFormer: {gf_session.get_inputs()[0].shape}")
        print(f"   YOLO: {yolo_session.get_inputs()[0].shape}")
        
        # Test individual models
        print(f"\n🔧 Testing individual models...")
        
        # GridFormer test (384x384)
        gf_input_tensor = np.random.rand(1, 3, 384, 384).astype(np.float32)
        start_time = time.time()
        gf_output_tensor = gf_session.run([gf_output], {gf_input: gf_input_tensor})
        gf_time = (time.time() - start_time) * 1000
        
        print(f"   GridFormer (384x384): {gf_time:.1f}ms")
        
        # YOLO test (416x416)  
        yolo_input_tensor = np.random.rand(1, 3, 416, 416).astype(np.float32)
        start_time = time.time()
        yolo_output_tensor = yolo_session.run([yolo_output], {yolo_input: yolo_input_tensor})
        yolo_time = (time.time() - start_time) * 1000
        
        print(f"   YOLO (416x416): {yolo_time:.1f}ms")
        
        # Test complete pipeline (multiple iterations)
        print(f"\n🚀 Testing optimized pipeline (10 iterations)...")
        pipeline_times = []
        
        for i in range(10):
            # Use smaller compatible sizes
            test_image_gf = np.random.rand(1, 3, 384, 384).astype(np.float32)
            test_image_yolo = np.random.rand(1, 3, 416, 416).astype(np.float32)
            
            start_time = time.time()
            
            # GridFormer inference (384x384)
            restored = gf_session.run([gf_output], {gf_input: test_image_gf})
            
            # YOLO inference (416x416)
            detections = yolo_session.run([yolo_output], {yolo_input: test_image_yolo})
            
            total_time = (time.time() - start_time) * 1000
            pipeline_times.append(total_time)
        
        # Calculate statistics
        avg_latency = np.mean(pipeline_times)
        min_latency = np.min(pipeline_times)
        max_latency = np.max(pipeline_times)
        
        print(f"\n📊 OPTIMIZED Performance Results:")
        print(f"   GridFormer (384x384): {gf_time:.1f}ms")
        print(f"   YOLO (416x416): {yolo_time:.1f}ms")
        print(f"   Pipeline Average: {avg_latency:.1f}ms")
        print(f"   Pipeline Min: {min_latency:.1f}ms")
        print(f"   Pipeline Max: {max_latency:.1f}ms")
        print(f"   Target: <350ms")
        
        success = avg_latency < 350
        print(f"   Status: {'✅ SUCCESS' if success else '⚠️ STILL NEEDS WORK'}")
        
        # Compare with original
        print(f"\n📈 Performance Comparison:")
        print(f"   Original (448x448): ~1666ms")
        print(f"   Optimized: {avg_latency:.1f}ms")
        improvement = ((1666 - avg_latency) / 1666) * 100
        print(f"   Improvement: {improvement:.1f}%")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_real_image_sizes():
    """Test with realistic image processing (resizing from common camera sizes)"""
    print(f"\n🎥 Testing with realistic image pipeline...")
    
    try:
        gridformer_path = "models/gridformer_optimized_384.onnx"
        yolo_path = "models/yolov8s_optimized_416.onnx"
        
        gf_session = ort.InferenceSession(gridformer_path)
        yolo_session = ort.InferenceSession(yolo_path)
        
        gf_input = gf_session.get_inputs()[0].name
        gf_output = gf_session.get_outputs()[0].name
        yolo_input = yolo_session.get_inputs()[0].name
        yolo_output = yolo_session.get_outputs()[0].name
        
        # Simulate common camera resolution (640x480) processing
        camera_image = np.random.rand(480, 640, 3).astype(np.uint8)
        
        start_time = time.time()
        
        # Preprocess for GridFormer (384x384)
        gf_resized = cv2.resize(camera_image, (384, 384))
        gf_normalized = gf_resized.astype(np.float32) / 255.0
        gf_input_tensor = np.transpose(gf_normalized, (2, 0, 1))
        gf_input_tensor = np.expand_dims(gf_input_tensor, axis=0)
        
        # GridFormer inference
        gf_output_tensor = gf_session.run([gf_output], {gf_input: gf_input_tensor})
        
        # Preprocess for YOLO (416x416)
        yolo_resized = cv2.resize(camera_image, (416, 416))
        yolo_normalized = yolo_resized.astype(np.float32) / 255.0
        yolo_input_tensor = np.transpose(yolo_normalized, (2, 0, 1))
        yolo_input_tensor = np.expand_dims(yolo_input_tensor, axis=0)
        
        # YOLO inference
        yolo_output_tensor = yolo_session.run([yolo_output], {yolo_input: yolo_input_tensor})
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"   Real image pipeline (640x480 → 384x384 & 416x416): {total_time:.1f}ms")
        print(f"   Includes: Resize + Normalize + Inference + Preprocessing")
        
        return total_time < 350
        
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        return False

def main():
    """Main test function"""
    success1 = test_optimized_models()
    success2 = test_real_image_sizes()
    
    print(f"\n🎯 FINAL OPTIMIZATION RESULTS:")
    print(f"   Optimized Models: {'✅' if success1 else '❌'}")
    print(f"   Real Image Pipeline: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print(f"\n🎉 OPTIMIZATION SUCCESSFUL! Ready for ROS2 deployment.")
        print(f"💡 Next steps: Update node parameters and test in Docker environment.")
    else:
        print(f"\n⚠️  Further optimization needed.")
        print(f"💡 Consider: Smaller models, TensorRT, or GPU acceleration.")

if __name__ == "__main__":
    main()