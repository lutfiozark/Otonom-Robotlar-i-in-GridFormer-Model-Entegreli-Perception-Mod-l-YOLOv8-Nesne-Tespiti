#!/usr/bin/env python3
"""
Load Trained GridFormer Model
Adapts the trained checkpoint to our GridFormer architecture
"""

import torch
import torch.nn as nn
import numpy as np
from gridformer import GridFormer, create_gridformer_model


def inspect_checkpoint(checkpoint_path: str):
    """Inspect the structure of the trained checkpoint"""
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")

        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"   {key}: {len(value)} parameters")
                # Show first few parameter names
                param_names = list(value.keys())[:5]
                print(f"      Sample params: {param_names}...")
            else:
                print(f"   {key}: {type(value)}")
    else:
        print(f"📋 Direct state_dict with {len(checkpoint)} parameters")

    return checkpoint


def adapt_weights_to_gridformer(checkpoint, target_model):
    """Adapt trained weights to our GridFormer architecture"""
    print("🔄 Adapting weights to GridFormer architecture...")

    # Get the parameter dict (try both 'params' and 'params_ema')
    if isinstance(checkpoint, dict):
        if 'params_ema' in checkpoint:
            source_state = checkpoint['params_ema']
            print("   Using EMA parameters")
        elif 'params' in checkpoint:
            source_state = checkpoint['params']
            print("   Using regular parameters")
        elif 'state_dict' in checkpoint:
            source_state = checkpoint['state_dict']
            print("   Using state_dict")
        else:
            source_state = checkpoint
            print("   Using direct checkpoint")
    else:
        source_state = checkpoint
        print("   Using direct state_dict")

    # Get target model state
    target_state = target_model.state_dict()

    print(f"   Source parameters: {len(source_state)}")
    print(f"   Target parameters: {len(target_state)}")

    # Try to match parameters by name or shape
    adapted_state = {}
    matched_count = 0

    for target_name, target_param in target_state.items():
        target_shape = target_param.shape

        # Try exact name match
        if target_name in source_state:
            source_param = source_state[target_name]
            if source_param.shape == target_shape:
                adapted_state[target_name] = source_param
                matched_count += 1
                continue

        # Try partial name matching (for adapted architectures)
        found_match = False
        for source_name, source_param in source_state.items():
            if source_param.shape == target_shape:
                # Simple heuristic: if shapes match and names are similar
                if any(part in source_name.lower() for part in target_name.lower().split('.')):
                    adapted_state[target_name] = source_param
                    matched_count += 1
                    found_match = True
                    print(f"   Matched: {target_name} ← {source_name}")
                    break

        if not found_match:
            # Keep original (random) weights for unmatched parameters
            adapted_state[target_name] = target_param
            print(f"   ⚠️  No match for: {target_name} {target_shape}")

    print(f"✅ Matched {matched_count}/{len(target_state)} parameters")

    return adapted_state


def create_adapted_model(checkpoint_path: str, output_path: str = "models/gridformer_adapted.pth"):
    """Create an adapted GridFormer model with trained weights"""

    # Inspect checkpoint
    checkpoint = inspect_checkpoint(checkpoint_path)

    # Create target GridFormer model
    print("\n🏗️  Creating GridFormer architecture...")
    target_model = create_gridformer_model()

    print(
        f"   GridFormer parameters: {sum(p.numel() for p in target_model.parameters()):,}")

    # Adapt weights
    adapted_state = adapt_weights_to_gridformer(checkpoint, target_model)

    # Load adapted weights
    target_model.load_state_dict(adapted_state, strict=False)

    # Save adapted model
    print(f"\n💾 Saving adapted model to: {output_path}")
    torch.save(target_model.state_dict(), output_path)

    # Test the adapted model
    print("\n🧪 Testing adapted model...")
    target_model.eval()
    test_input = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output = target_model(test_input)

    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

    return target_model


def main():
    """Main function"""
    print("🚀 GridFormer Model Adaptation")
    print("=" * 50)

    checkpoint_path = "models/gridformer_trained.pth"

    try:
        adapted_model = create_adapted_model(checkpoint_path)
        print("\n✅ Model adaptation completed successfully!")
        print("🎯 Ready to use with GridFormer node")

        # Export to ONNX with trained weights
        print("\n🔄 Exporting adapted model to ONNX...")
        from export_gridformer_onnx import export_gridformer_onnx

        export_gridformer_onnx(
            model_path="models/gridformer_adapted.pth",
            output_path="models/gridformer_adapted.onnx"
        )

    except Exception as e:
        print(f"❌ Error during adaptation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
