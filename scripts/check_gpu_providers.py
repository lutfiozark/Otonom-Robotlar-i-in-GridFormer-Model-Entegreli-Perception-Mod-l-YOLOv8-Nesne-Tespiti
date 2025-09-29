#!/usr/bin/env python3
"""Quick GPU/TensorRT provider check for ONNX Runtime and models.

Usage:
  python scripts/check_gpu_providers.py \
    --gridformer models/gridformer_optimized_384.onnx \
    --yolo models/yolov8s_optimized_416.onnx

If model paths are missing, it will still print provider availability.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check ONNX Runtime providers and model sessions")
    parser.add_argument("--gridformer", default="models/gridformer_optimized_384.onnx")
    parser.add_argument("--yolo", default="models/yolov8s_optimized_416.onnx")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as e:
        print(f"❌ onnxruntime not available: {e}")
        return 1

    print("🧩 ONNX Runtime providers:")
    for p in providers:
        print(f"  - {p}")

    # Try GridFormer session
    gf_path = Path(args.gridformer)
    if gf_path.exists():
        prio = []
        if "TensorrtExecutionProvider" in providers:
            prio.append(("TensorrtExecutionProvider", {"trt_max_workspace_size": 2147483648, "trt_fp16_enable": True}))
        if "CUDAExecutionProvider" in providers:
            prio.append("CUDAExecutionProvider")
        prio.append("CPUExecutionProvider")
        try:
            sess = ort.InferenceSession(str(gf_path), providers=prio)
            used = sess.get_providers()[0]
            print(f"✅ GridFormer session created with: {used}")
        except Exception as e:
            print(f"❌ GridFormer session error: {e}")
    else:
        print(f"ℹ️ GridFormer ONNX not found: {gf_path}")

    # Try YOLO session
    yolo_path = Path(args.yolo)
    if yolo_path.exists():
        prio = []
        if "TensorrtExecutionProvider" in providers:
            prio.append(("TensorrtExecutionProvider", {"trt_max_workspace_size": 2147483648, "trt_fp16_enable": True}))
        if "CUDAExecutionProvider" in providers:
            prio.append("CUDAExecutionProvider")
        prio.append("CPUExecutionProvider")
        try:
            sess = ort.InferenceSession(str(yolo_path), providers=prio)
            used = sess.get_providers()[0]
            print(f"✅ YOLO session created with: {used}")
        except Exception as e:
            print(f"❌ YOLO session error: {e}")
    else:
        print(f"ℹ️ YOLO ONNX not found: {yolo_path}")

    print("\nIf TensorRT/CUDA providers are missing, install:")
    print("- NVIDIA GPU Driver + CUDA Toolkit (matching cuBLAS DLLs)")
    print("- TensorRT Runtime (adds onnxruntime_providers_tensorrt.dll deps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


