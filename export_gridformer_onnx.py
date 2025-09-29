#!/usr/bin/env python3
"""
GridFormer ONNX Export Script
Exports trained GridFormer model to ONNX format for TensorRT optimization
"""

import torch
import torch.onnx
import os
import argparse
from gridformer import GridFormer, create_gridformer_model


def export_gridformer_onnx(model_path: str = "models/gridformer.pth",
                           output_path: str = "models/gridformer.onnx",
                           input_size: tuple = (1, 3, 512, 512),
                           opset_version: int = 17,
                           dynamic_batch: bool = True):
    """
    Export GridFormer model to ONNX format

    Args:
        model_path: Path to trained PyTorch model (.pth)
        output_path: Output path for ONNX model
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_batch: Whether to use dynamic batch size
    """

    print("🚀 Starting GridFormer ONNX export...")
    print(f"   Model path: {model_path}")
    print(f"   Output path: {output_path}")
    print(f"   Input size: {input_size}")
    print(f"   Opset version: {opset_version}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create model instance
    model = create_gridformer_model()
    model.eval()

    # Load trained weights if available
    if os.path.exists(model_path):
        print(f"📥 Loading trained weights from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'params' in checkpoint:
                    print("⚠️  Checkpoint format not compatible with current model architecture")
                    print("   Using random initialization for ONNX export (for structure)")
                    print("   Please retrain or convert your model to the expected format")
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model weights loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading weights: {e}")
            print("   Using random initialization for ONNX export structure")
    else:
        print(f"⚠️  Model file not found at {model_path}")
        print("   Creating dummy model with random weights for demonstration")
        # Save a dummy model for the export process
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), f"{model_path.replace('.pth', '_structure.pth')}")
        print(f"💾 Saved model structure to {model_path.replace('.pth', '_structure.pth')}")

    # Create dummy input tensor
    dummy_input = torch.randn(input_size)
    print(f"🔧 Created dummy input tensor: {dummy_input.shape}")

    # Define dynamic axes for flexible input sizes
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}
        }

    # Export to ONNX
    print("🔄 Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,                          # PyTorch model
            dummy_input,                    # Input tensor
            output_path,                    # Output file path
            export_params=True,             # Store trained parameters
            opset_version=opset_version,    # ONNX opset version
            do_constant_folding=True,       # Optimize constants
            input_names=["input"],          # Input tensor name
            output_names=["output"],        # Output tensor name
            dynamic_axes=dynamic_axes       # Dynamic dimensions
        )

        print(f"✅ ONNX export successful!")
        print(f"   Output saved to: {output_path}")

        # Verify the exported model
        verify_onnx_model(output_path, dummy_input)

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise


def verify_onnx_model(onnx_path: str, test_input: torch.Tensor):
    """
    Verify the exported ONNX model

    Args:
        onnx_path: Path to ONNX model
        test_input: Test input tensor
    """
    try:
        import onnx
        import onnxruntime as ort

        print("🔍 Verifying ONNX model...")

        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)

        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        print(f"   Input name: {input_name}")
        print(f"   Output name: {output_name}")

        # Test inference
        ort_inputs = {input_name: test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        print(f"   ONNX output shape: {ort_outputs[0].shape}")
        print("✅ ONNX model verification successful!")

        # Get model size
        file_size = os.path.getsize(onnx_path)
        print(f"   Model size: {file_size / (1024*1024):.2f} MB")

    except ImportError:
        print("⚠️  ONNX/ONNXRuntime not available for verification")
        print("   Install with: pip install onnx onnxruntime")
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Export GridFormer model to ONNX")

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/gridformer.pth",
        help="Path to trained PyTorch model"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="models/gridformer.onnx",
        help="Output path for ONNX model"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        nargs=4,
        default=[1, 3, 512, 512],
        help="Input tensor size (batch, channels, height, width)"
    )

    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version"
    )

    parser.add_argument(
        "--no_dynamic_batch",
        action="store_true",
        help="Disable dynamic batch size"
    )

    args = parser.parse_args()

    # Export model
    export_gridformer_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=tuple(args.input_size),
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch
    )


if __name__ == "__main__":
    main()
