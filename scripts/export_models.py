#!/usr/bin/env python3
"""Export trained models to optimized formats (ONNX, TensorRT)."""

from gridformer import GridFormerModel
import argparse
import os
import sys
import subprocess
from pathlib import Path
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_gridformer_onnx(model_path: str, output_path: str, img_size: int = 448):
    """Export GridFormer model to ONNX format."""
    print(f"📦 Exporting GridFormer to ONNX...")

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GridFormerModel()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from: {model_path}")
    else:
        print(f"⚠️  No checkpoint found, using pretrained weights")

    model.eval()
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ GridFormer exported to: {output_path}")

    # Verify ONNX model
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verification passed")
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")

    return str(output_path)


def export_yolo_onnx(model_path: str, img_size: int = 448):
    """Export YOLO model to ONNX format."""
    print(f"📦 Exporting YOLO to ONNX...")

    if not Path(model_path).exists():
        print(f"❌ YOLO model not found: {model_path}")
        return None

    # Load YOLO model
    model = YOLO(model_path)

    # Export to ONNX
    export_path = model.export(
        format='onnx',
        imgsz=img_size,
        half=False,  # FP32 for better compatibility
        dynamic=False,
        simplify=True,
        opset=12
    )

    print(f"✅ YOLO exported to: {export_path}")
    return export_path


def onnx_to_tensorrt(onnx_path: str, output_path: str, fp16: bool = True):
    """Convert ONNX model to TensorRT format."""
    print(f"🔄 Converting ONNX to TensorRT...")

    if not Path(onnx_path).exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return None

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build TensorRT command
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}',
        '--verbose',
        '--buildOnly'
    ]

    if fp16:
        cmd.append('--fp16')
        print("   Using FP16 precision for GTX 1650 optimization")

    # Add workspace size limit for GTX 1650 (4GB VRAM)
    cmd.extend(['--workspace=1024'])  # 1GB workspace

    # Run TensorRT conversion
    try:
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)

        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ TensorRT engine created: {output_path}")
            print(f"   Size: {file_size_mb:.1f}MB")

            if file_size_mb > 1200:  # 1.2GB limit for GTX 1650
                print(
                    f"⚠️  Engine size ({file_size_mb:.1f}MB) may exceed GTX 1650 VRAM limits")
                print("   Consider using smaller input size or FP16 precision")

        return str(output_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ TensorRT conversion failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"❌ trtexec not found. Please install TensorRT.")
        print("   Alternative: Use ONNX Runtime with TensorRT provider")
        return None


def test_onnx_runtime(onnx_path: str, use_tensorrt: bool = False):
    """Test ONNX model with ONNX Runtime."""
    print(f"🧪 Testing ONNX model with ONNX Runtime...")

    # Setup providers
    providers = []
    if use_tensorrt and torch.cuda.is_available():
        providers.append('TensorrtExecutionProvider')
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    try:
        # Create inference session
        session = ort.InferenceSession(onnx_path, providers=providers)

        print(f"✅ ONNX Runtime session created")
        print(f"   Providers: {session.get_providers()}")

        # Get input shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Create test input
        if input_shape[0] == 'batch_size':
            input_shape[0] = 1

        test_input = torch.randn(*input_shape).numpy()

        # Run inference
        import time
        start_time = time.time()
        outputs = session.run(None, {input_name: test_input})
        inference_time = time.time() - start_time

        print(f"✅ Inference test passed")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Inference time: {inference_time*1000:.2f}ms")

        return True

    except Exception as e:
        print(f"❌ ONNX Runtime test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export models to optimized formats')
    parser.add_argument('--gridformer-model', default='models/gridformer/best_model.pth',
                        help='GridFormer model checkpoint')
    parser.add_argument('--yolo-model', default='models/yolo/weather_detection/weights/best.pt',
                        help='YOLO model path')
    parser.add_argument(
        '--output-dir', default='models/exported', help='Output directory')
    parser.add_argument('--img-size', type=int, default=448,
                        help='Image size for export')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--skip-tensorrt', action='store_true',
                        help='Skip TensorRT conversion')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test existing models')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Model Export Pipeline")
    print(f"   Output directory: {output_dir}")
    print(f"   Image size: {args.img_size}")
    print(f"   FP16: {args.fp16}")

    exported_models = {}

    if not args.test_only:
        # Export GridFormer
        print(f"\n" + "="*50)
        print(f"GRIDFORMER EXPORT")
        print(f"="*50)

        gridformer_onnx = output_dir / f"gridformer_{args.img_size}.onnx"
        exported_models['gridformer_onnx'] = export_gridformer_onnx(
            args.gridformer_model,
            str(gridformer_onnx),
            args.img_size
        )

        # Export YOLO
        print(f"\n" + "="*50)
        print(f"YOLO EXPORT")
        print(f"="*50)

        if Path(args.yolo_model).exists():
            yolo_onnx = export_yolo_onnx(args.yolo_model, args.img_size)
            if yolo_onnx:
                # Move to output directory
                yolo_onnx_dest = output_dir / f"yolo_{args.img_size}.onnx"
                Path(yolo_onnx).rename(yolo_onnx_dest)
                exported_models['yolo_onnx'] = str(yolo_onnx_dest)
        else:
            print(f"⚠️  YOLO model not found: {args.yolo_model}")

        # Convert to TensorRT
        if not args.skip_tensorrt:
            print(f"\n" + "="*50)
            print(f"TENSORRT CONVERSION")
            print(f"="*50)

            for model_name, onnx_path in exported_models.items():
                if onnx_path and Path(onnx_path).exists():
                    trt_path = output_dir / \
                        f"{model_name.replace('_onnx', '')}_{args.img_size}.trt"
                    trt_engine = onnx_to_tensorrt(
                        onnx_path, str(trt_path), args.fp16)
                    if trt_engine:
                        exported_models[f"{model_name.replace('_onnx', '_trt')}"] = trt_engine

    # Test exported models
    print(f"\n" + "="*50)
    print(f"MODEL TESTING")
    print(f"="*50)

    # Find existing models if test-only
    if args.test_only:
        for model_file in output_dir.glob("*.onnx"):
            model_name = model_file.stem
            exported_models[model_name] = str(model_file)

    for model_name, model_path in exported_models.items():
        if model_path and Path(model_path).exists() and model_path.endswith('.onnx'):
            print(f"\n🧪 Testing {model_name}...")
            test_onnx_runtime(model_path, use_tensorrt=False)

    # Summary
    print(f"\n" + "="*50)
    print(f"EXPORT SUMMARY")
    print(f"="*50)

    for model_name, model_path in exported_models.items():
        if model_path and Path(model_path).exists():
            file_size = Path(model_path).stat().st_size / (1024 * 1024)
            print(f"✅ {model_name}: {model_path} ({file_size:.1f}MB)")
        else:
            print(f"❌ {model_name}: Failed to export")

    print(f"\n🎯 GTX 1650 Optimization Tips:")
    print(f"   - Use fp16 precision: --fp16")
    print(f"   - Reduce image size if VRAM limited: --img-size 384")
    print(f"   - Monitor VRAM usage during inference")


if __name__ == "__main__":
    main()
