#!/usr/bin/env python3
"""
Standalone Model Testing Script
Test GridFormer and YOLO models without ROS2
"""

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time
from pathlib import Path

def test_gridformer_onnx():
    """Test GridFormer ONNX model"""
    print("🔧 Testing GridFormer ONNX model...")
    
    model_path = "models/gridformer_trained.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load model
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create test input
        test_input = np.random.rand(1, 3, 448, 448).astype(np.float32)
        
        # Run inference
        start_time = time.time()
        outputs = session.run([output_name], {input_name: test_input})
        inference_time = (time.time() - start_time) * 1000
        
        print(f"✅ GridFormer ONNX inference successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Inference time: {inference_time:.1f}ms")
        return True
        
    except Exception as e:
        print(f"❌ GridFormer test failed: {e}")
        return False

def test_yolo_onnx():
    """Test YOLO ONNX model"""
    print("\n🔧 Testing YOLO ONNX model...")
    
    model_path = "models/yolov8s.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load model
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create test input
        test_input = np.random.rand(1, 3, 448, 448).astype(np.float32)
        
        # Run inference
        start_time = time.time()
        outputs = session.run([output_name], {input_name: test_input})
        inference_time = (time.time() - start_time) * 1000
        
        print(f"✅ YOLO ONNX inference successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Inference time: {inference_time:.1f}ms")
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def test_pipeline_latency():
    """Test complete pipeline latency"""
    print("\n🚀 Testing complete pipeline latency...")
    
    try:
        # Load both models
        gridformer_session = ort.InferenceSession("models/gridformer_trained.onnx")
        yolo_session = ort.InferenceSession("models/yolov8s.onnx")
        
        # Get model info
        gf_input = gridformer_session.get_inputs()[0].name
        gf_output = gridformer_session.get_outputs()[0].name
        yolo_input = yolo_session.get_inputs()[0].name
        yolo_output = yolo_session.get_outputs()[0].name
        
        # Test multiple iterations
        total_times = []
        for i in range(10):
            test_image = np.random.rand(1, 3, 448, 448).astype(np.float32)
            
            start_time = time.time()
            
            # GridFormer inference
            restored = gridformer_session.run([gf_output], {gf_input: test_image})
            
            # YOLO inference  
            detections = yolo_session.run([yolo_output], {yolo_input: test_image})
            
            total_time = (time.time() - start_time) * 1000
            total_times.append(total_time)
        
        avg_latency = np.mean(total_times)
        print(f"✅ Pipeline test successful!")
        print(f"   Average E2E latency: {avg_latency:.1f}ms")
        print(f"   Target latency: <350ms")
        print(f"   Status: {'✅ PASS' if avg_latency < 350 else '⚠️ NEEDS OPTIMIZATION'}")
        
        return avg_latency < 350
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 GridFormer + YOLO Model Testing")
    print("=" * 50)
    
    # Test individual models
    gf_success = test_gridformer_onnx()
    yolo_success = test_yolo_onnx()
    
    # Test pipeline if both models work
    if gf_success and yolo_success:
        pipeline_success = test_pipeline_latency()
        
        print(f"\n📊 Final Results:")
        print(f"   GridFormer: {'✅' if gf_success else '❌'}")
        print(f"   YOLO: {'✅' if yolo_success else '❌'}")
        print(f"   Pipeline: {'✅' if pipeline_success else '⚠️'}")
        
        if gf_success and yolo_success and pipeline_success:
            print(f"\n🎉 All tests passed! Ready for ROS2 integration.")
        else:
            print(f"\n⚠️ Some optimizations needed before deployment.")
    else:
        print(f"\n❌ Model loading failed. Check ONNX exports.")

if __name__ == "__main__":
    main()