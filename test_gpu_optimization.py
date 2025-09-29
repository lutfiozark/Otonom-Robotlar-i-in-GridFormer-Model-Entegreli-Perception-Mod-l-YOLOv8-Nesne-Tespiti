#!/usr/bin/env python3
"""
GPU Optimization Test Script
Test GPU-accelerated GridFormer and YOLO models
"""

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time
from pathlib import Path

def test_gpu_providers():
    """Test available GPU providers"""
    print("🔍 Testing GPU Providers...")
    providers = ort.get_available_providers()
    print(f"Available: {providers}")
    
    gpu_available = 'CUDAExecutionProvider' in providers
    trt_available = 'TensorrtExecutionProvider' in providers
    
    print(f"✅ CUDA: {gpu_available}")
    print(f"✅ TensorRT: {trt_available}")
    return gpu_available, trt_available

def test_gridformer_gpu():
    """Test GridFormer with TensorRT/GPU optimization"""
    print("\n🔧 Testing GridFormer GPU Optimization...")
    
    model_path = "models/gridformer_optimized_384.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False, 0
    
    try:
        # Create session with TensorRT/GPU providers
        providers = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,  # 2GB
                'trt_fp16_enable': True,
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 1
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Log which provider is used
        used_providers = session.get_providers()
        print(f"🔥 GridFormer using: {used_providers[0]}")
        
        # Get input info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test input (384x384)
        test_input = np.random.rand(1, 3, 384, 384).astype(np.float32)
        
        # Warm up for TensorRT compilation
        if 'TensorrtExecutionProvider' in used_providers:
            print("🏗️  Compiling TensorRT engine (first run)...")
            _ = session.run([output_name], {input_name: test_input})
            print("✅ TensorRT engine compiled!")
        
        # Benchmark multiple runs
        times = []
        for i in range(10):
            start_time = time.time()
            outputs = session.run([output_name], {input_name: test_input})
            inference_time = (time.time() - start_time) * 1000
            times.append(inference_time)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        
        print(f"✅ GridFormer GPU test successful!")
        print(f"   Input: {test_input.shape}")
        print(f"   Output: {outputs[0].shape}")
        print(f"   Avg time: {avg_time:.1f}ms")
        print(f"   Min time: {min_time:.1f}ms")
        
        return True, avg_time
        
    except Exception as e:
        print(f"❌ GridFormer GPU test failed: {e}")
        return False, 0

def test_yolo_gpu():
    """Test YOLO with GPU optimization"""
    print("\n🔧 Testing YOLO GPU Optimization...")
    
    model_path = "models/yolov8s_optimized_416.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False, 0
    
    try:
        # Create session with GPU providers
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Log which provider is used
        used_providers = session.get_providers()
        print(f"🔥 YOLO using: {used_providers[0]}")
        
        # Get input info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test input (416x416)
        test_input = np.random.rand(1, 3, 416, 416).astype(np.float32)
        
        # Benchmark multiple runs
        times = []
        for i in range(10):
            start_time = time.time()
            outputs = session.run([output_name], {input_name: test_input})
            inference_time = (time.time() - start_time) * 1000
            times.append(inference_time)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        
        print(f"✅ YOLO GPU test successful!")
        print(f"   Input: {test_input.shape}")
        print(f"   Output: {outputs[0].shape}")
        print(f"   Avg time: {avg_time:.1f}ms")
        print(f"   Min time: {min_time:.1f}ms")
        
        return True, avg_time
        
    except Exception as e:
        print(f"❌ YOLO GPU test failed: {e}")
        return False, 0

def test_pipeline_gpu():
    """Test complete GPU-optimized pipeline"""
    print("\n🚀 Testing Complete GPU Pipeline...")
    
    try:
        # Load both models
        gf_path = "models/gridformer_optimized_384.onnx"
        yolo_path = "models/yolov8s_optimized_416.onnx"
        
        # GridFormer session
        gf_providers = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        gf_session = ort.InferenceSession(gf_path, providers=gf_providers)
        
        # YOLO session
        yolo_providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        yolo_session = ort.InferenceSession(yolo_path, providers=yolo_providers)
        
        # Get model info
        gf_input = gf_session.get_inputs()[0].name
        gf_output = gf_session.get_outputs()[0].name
        yolo_input = yolo_session.get_inputs()[0].name
        yolo_output = yolo_session.get_outputs()[0].name
        
        print(f"🔥 Pipeline providers:")
        print(f"   GridFormer: {gf_session.get_providers()[0]}")
        print(f"   YOLO: {yolo_session.get_providers()[0]}")
        
        # Test pipeline multiple times
        pipeline_times = []
        for i in range(10):
            # Simulate camera input (640x480 → resize to model inputs)
            camera_image = np.random.rand(480, 640, 3).astype(np.uint8)
            
            start_time = time.time()
            
            # GridFormer preprocessing and inference
            gf_resized = cv2.resize(camera_image, (384, 384))
            gf_normalized = gf_resized.astype(np.float32) / 255.0
            gf_tensor = np.transpose(gf_normalized, (2, 0, 1))
            gf_tensor = np.expand_dims(gf_tensor, axis=0)
            
            gf_output_tensor = gf_session.run([gf_output], {gf_input: gf_tensor})
            
            # YOLO preprocessing and inference
            yolo_resized = cv2.resize(camera_image, (416, 416))
            yolo_normalized = yolo_resized.astype(np.float32) / 255.0
            yolo_tensor = np.transpose(yolo_normalized, (2, 0, 1))
            yolo_tensor = np.expand_dims(yolo_tensor, axis=0)
            
            yolo_output_tensor = yolo_session.run([yolo_output], {yolo_input: yolo_tensor})
            
            total_time = (time.time() - start_time) * 1000
            pipeline_times.append(total_time)
        
        avg_latency = np.mean(pipeline_times)
        min_latency = np.min(pipeline_times)
        max_latency = np.max(pipeline_times)
        
        print(f"\n📊 GPU Pipeline Results:")
        print(f"   Average E2E: {avg_latency:.1f}ms")
        print(f"   Min E2E: {min_latency:.1f}ms")
        print(f"   Max E2E: {max_latency:.1f}ms")
        print(f"   Target: <350ms")
        
        success = avg_latency < 350
        print(f"   Status: {'✅ TARGET ACHIEVED!' if success else '⚠️ Still needs work'}")
        
        return success, avg_latency
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False, 0

def main():
    """Main test function"""
    print("🚀 GPU OPTIMIZATION VALIDATION")
    print("=" * 50)
    
    # Test providers
    gpu_available, trt_available = test_gpu_providers()
    
    if not gpu_available:
        print("❌ GPU not available - optimization tests skipped")
        return
    
    # Test individual models
    gf_success, gf_time = test_gridformer_gpu()
    yolo_success, yolo_time = test_yolo_gpu()
    
    # Test pipeline if both models work
    if gf_success and yolo_success:
        pipeline_success, pipeline_time = test_pipeline_gpu()
        
        print(f"\n🎯 FINAL GPU OPTIMIZATION RESULTS:")
        print(f"   GridFormer: {gf_time:.1f}ms {'✅' if gf_time < 150 else '⚠️'}")
        print(f"   YOLO: {yolo_time:.1f}ms {'✅' if yolo_time < 200 else '⚠️'}")
        print(f"   Pipeline: {pipeline_time:.1f}ms {'✅' if pipeline_success else '❌'}")
        
        # Compare with CPU baseline
        print(f"\n📈 Performance Comparison:")
        print(f"   CPU Baseline: ~1666ms")
        print(f"   GPU Optimized: {pipeline_time:.1f}ms")
        improvement = ((1666 - pipeline_time) / 1666) * 100
        print(f"   Improvement: {improvement:.1f}%")
        
        if pipeline_success:
            print(f"\n🎉 GPU OPTIMIZATION SUCCESSFUL!")
            print(f"💡 Ready for ROS2 Docker deployment!")
        else:
            print(f"\n⚠️  Still needs further optimization.")
            print(f"💡 Try: Smaller models, batch processing, or pure TensorRT.")
    else:
        print(f"\n❌ Model loading failed. Check ONNX files.")

if __name__ == "__main__":
    main()