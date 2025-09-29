#!/usr/bin/env python3
"""Optimize trained models for deployment."""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def optimize_gridformer_model(model_path, output_dir, img_size=448):
    """Optimize GridFormer model."""
    print(f"🧠 Optimizing GridFormer model...")

    if not Path(model_path).exists():
        print(f"❌ GridFormer model not found: {model_path}")
        return False

    try:
        # Export to ONNX first
        onnx_path = Path(output_dir) / f"gridformer_{img_size}.onnx"
        export_cmd = [
            "python", "scripts/export_models.py",
            "--gridformer-model", model_path,
            "--output-dir", output_dir,
            "--img-size", str(img_size),
            "--skip-tensorrt"  # We'll do TRT separately
        ]

        print(f"   Exporting to ONNX: {onnx_path}")
        result = subprocess.run(export_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"   ✅ ONNX export successful")

            # Convert to TensorRT if available
            if onnx_path.exists():
                trt_path = convert_to_tensorrt(
                    str(onnx_path), output_dir, img_size)
                if trt_path:
                    print(f"   ✅ TensorRT conversion successful: {trt_path}")
                    return True

        else:
            print(f"   ❌ ONNX export failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"   ❌ GridFormer optimization failed: {e}")
        return False


def optimize_yolo_model(model_path, output_dir, img_size=448):
    """Optimize YOLO model."""
    print(f"🎯 Optimizing YOLO model...")

    if not Path(model_path).exists():
        print(f"❌ YOLO model not found: {model_path}")
        return False

    try:
        # Export YOLO to ONNX
        onnx_path = Path(output_dir) / f"yolo_{img_size}.onnx"
        export_cmd = [
            "yolo", "export",
            f"model={model_path}",
            "format=onnx",
            f"imgsz={img_size}",
            "device=cpu",  # Export on CPU for stability
            "half=False",  # FP32 for better compatibility
            "dynamic=False",
            "simplify=True",
            "opset=12"
        ]

        print(f"   Exporting to ONNX: {onnx_path}")
        result = subprocess.run(export_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Find exported ONNX file (YOLO creates it in model directory)
            model_dir = Path(model_path).parent
            exported_onnx = model_dir / f"{Path(model_path).stem}.onnx"

            if exported_onnx.exists():
                # Move to output directory
                final_onnx = Path(output_dir) / f"yolo_{img_size}.onnx"
                exported_onnx.rename(final_onnx)
                print(f"   ✅ ONNX export successful")

                # Convert to TensorRT
                trt_path = convert_to_tensorrt(
                    str(final_onnx), output_dir, img_size)
                if trt_path:
                    print(f"   ✅ TensorRT conversion successful: {trt_path}")
                    return True
            else:
                print(f"   ❌ ONNX file not found after export")
                return False
        else:
            print(f"   ❌ YOLO export failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"   ❌ YOLO optimization failed: {e}")
        return False


def convert_to_tensorrt(onnx_path, output_dir, img_size):
    """Convert ONNX to TensorRT."""
    print(f"🔧 Converting to TensorRT...")

    try:
        # Use shell script for TensorRT conversion
        script_path = "scripts/onnx_to_trt.sh"
        if not Path(script_path).exists():
            print(f"   ⚠️  TensorRT script not found: {script_path}")
            return None

        trt_cmd = [
            "bash", script_path, onnx_path,
            "--output-dir", output_dir,
            "--img-size", str(img_size),
            "--fp16"
        ]

        result = subprocess.run(trt_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Find generated TRT file
            basename = Path(onnx_path).stem
            trt_path = Path(output_dir) / f"{basename}_{img_size}_fp16.trt"

            if trt_path.exists():
                return str(trt_path)
            else:
                print(f"   ❌ TensorRT file not found after conversion")
                return None
        else:
            print(f"   ⚠️  TensorRT conversion failed (using ONNX only)")
            print(f"      Error: {result.stderr}")
            return None

    except Exception as e:
        print(f"   ⚠️  TensorRT conversion error: {e}")
        return None


def test_optimized_models(output_dir):
    """Test optimized models."""
    print(f"🧪 Testing optimized models...")

    test_script = "scripts/test_optimized_models.py"
    if Path(test_script).exists():
        test_cmd = ["python", test_script, "--model-dir", output_dir]
        result = subprocess.run(test_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"   ✅ Model tests passed")
            return True
        else:
            print(f"   ⚠️  Model tests failed: {result.stderr}")
            return False
    else:
        print(f"   ⚠️  Test script not found, skipping tests")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Optimize trained models for deployment')
    parser.add_argument('--gridformer-model', default='models/gridformer/best_model.pth',
                        help='GridFormer model path')
    parser.add_argument('--yolo-model', default='models/yolo/weather_detection/weights/best.pt',
                        help='YOLO model path')
    parser.add_argument(
        '--output-dir', default='models/optimized', help='Output directory')
    parser.add_argument('--img-size', type=int, default=448,
                        help='Image size for optimization')
    parser.add_argument('--skip-gridformer', action='store_true',
                        help='Skip GridFormer optimization')
    parser.add_argument('--skip-yolo', action='store_true',
                        help='Skip YOLO optimization')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip model tests')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Model Optimization Pipeline")
    print(f"   Output directory: {output_dir}")
    print(f"   Image size: {args.img_size}")
    print(f"   GTX 1650 optimized (FP16, <1.2GB)")
    print("=" * 50)

    success_count = 0
    total_count = 0

    # Optimize GridFormer
    if not args.skip_gridformer:
        total_count += 1
        if optimize_gridformer_model(args.gridformer_model, str(output_dir), args.img_size):
            success_count += 1

    # Optimize YOLO
    if not args.skip_yolo:
        total_count += 1
        if optimize_yolo_model(args.yolo_model, str(output_dir), args.img_size):
            success_count += 1

    # Test models
    if not args.skip_tests and success_count > 0:
        test_optimized_models(str(output_dir))

    # Summary
    print("=" * 50)
    print(f"📊 Optimization Summary:")
    print(f"   Successful: {success_count}/{total_count}")

    if success_count == total_count:
        print(f"✅ All models optimized successfully!")
        print(f"\n🎯 Next Steps:")
        print(f"1. Update ROS nodes to use optimized models")
        print(f"2. Test pipeline: python scripts/test_pipeline.py")
        print(f"3. Start demo: ros2 launch launch/warehouse_demo.launch.py")
    else:
        print(f"⚠️  Some optimizations failed. Check logs above.")

    print(f"\n📁 Optimized models location: {output_dir}")


if __name__ == "__main__":
    main()
