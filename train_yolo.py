#!/usr/bin/env python3
"""YOLO fine-tuning script for object detection in degraded weather conditions."""

import argparse
import os
import sys
import shutil
from pathlib import Path
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# MLOps imports
try:
    import mlflow
    from mlops.mlflow_utils import setup_mlflow, log_metrics
    # Disable MLflow on Windows due to path issues
    import os
    MLFLOW_AVAILABLE = False if os.name == 'nt' else True
    if os.name == 'nt':
        print("⚠️  MLflow disabled on Windows due to path compatibility issues")
except ImportError:
    MLFLOW_AVAILABLE = False


def prepare_yolo_dataset(data_dir: str, output_dir: str, use_gridformer_output: bool = False):
    """Prepare dataset in YOLO format from synthetic data."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Create YOLO directory structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Load synthetic data annotations
    annotations_file = data_path / 'annotations.yaml'
    if not annotations_file.exists():
        raise FileNotFoundError(
            f"Annotations file not found: {annotations_file}")

    with open(annotations_file, 'r') as f:
        annotations = yaml.load(f, Loader=yaml.FullLoader)

    # Generate bounding boxes for objects in warehouse scenes
    # For synthetic data, we'll create simple box annotations
    weather_conditions = annotations['weather_conditions']

    # Group by scene_id
    scene_groups = {}
    for item in weather_conditions:
        scene_id = item['scene_id']
        if scene_id not in scene_groups:
            scene_groups[scene_id] = []
        scene_groups[scene_id].append(item)

    # Process each scene group
    image_count = 0
    for scene_id, items in scene_groups.items():
        for item in items:
            # Skip clear images if we're using degraded conditions
            if not use_gridformer_output and item['weather'] == 'clear':
                continue

            # Determine split (80% train, 10% val, 10% test)
            if image_count % 10 < 8:
                split = 'train'
            elif image_count % 10 == 8:
                split = 'val'
            else:
                split = 'test'

            # Load image
            weather = item['weather']
            filename = item['filename']
            source_path = data_path / weather / filename

            if not source_path.exists():
                print(f"Warning: Image not found: {source_path}")
                continue

            # Copy image to YOLO structure
            image_name = f"scene_{scene_id:04d}_{weather}_{image_count:06d}.jpg"
            dest_image_path = output_path / split / 'images' / image_name
            shutil.copy2(source_path, dest_image_path)

            # Generate synthetic bounding box annotations
            # For warehouse scenes, create boxes for typical objects
            label_path = output_path / split / 'labels' / \
                f"{image_name.replace('.jpg', '.txt')}"

            # Load image to get dimensions
            img = cv2.imread(str(source_path))
            h, w = img.shape[:2]

            # Generate 2-4 random boxes per image (obstacles, boxes, etc.)
            num_boxes = np.random.randint(2, 5)

            # Generate warehouse-specific bounding boxes
            img = cv2.imread(str(source_path))
            h, w = img.shape[:2]
            boxes = generate_warehouse_boxes(h, w)

            with open(label_path, 'w') as f:
                for box in boxes:
                    # Write in YOLO format: class_id center_x center_y width height
                    f.write(f"{box['class_id']} {box['center_x']:.6f} {box['center_y']:.6f} "
                            f"{box['width']:.6f} {box['height']:.6f}\n")

            image_count += 1

    print(f"✅ Prepared {image_count} images for YOLO training")
    print(f"📁 Dataset saved to: {output_path}")

    # Create data.yaml for YOLO
    yolo_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,  # Number of classes
        # Class names
        'names': ['red_cube', 'blue_cube', 'green_cube', 'pallet']
    }

    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)

    return str(output_path / 'data.yaml')


def train_yolo_model(data_config: str, args):
    """Train YOLO model."""
    print(f"🧠 Loading YOLO model: {args.model}")

    # Load pre-trained model
    model = YOLO(args.model)

    # Configure training parameters for GTX 1650
    train_args = {
        'data': data_config,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'device': 'cpu' if args.cpu_only else 0,
        'workers': args.workers,
        'project': args.save_dir,
        'name': 'weather_detection',
        'save_period': 5,  # Save checkpoint every 5 epochs
        'val': True,
        'plots': True,
        'verbose': True,
        'exist_ok': True,

        # Optimizations for GTX 1650
        'half': not args.cpu_only,  # Use FP16 on GPU
        'cache': False,  # Don't cache to save RAM
        'rect': False,   # Disable rectangular training
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Close mosaic augmentation in last 10 epochs

        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }

    print(f"🚀 Starting YOLO training...")
    print(f"   Dataset: {data_config}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Batch size: {args.batch_size}")

    # Train the model
    results = model.train(**train_args)

    # Get best model path
    best_model_path = Path(args.save_dir) / \
        'weather_detection' / 'weights' / 'best.pt'

    print(f"✅ Training completed!")
    print(f"💾 Best model saved: {best_model_path}")

    return results, str(best_model_path)


def validate_model(model_path: str, data_config: str, args):
    """Validate trained model."""
    print(f"🔍 Validating model: {model_path}")

    # Load trained model
    model = YOLO(model_path)

    # Run validation
    val_results = model.val(
        data=data_config,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device='cpu' if args.cpu_only else 0,
        half=not args.cpu_only,
        plots=True,
        save_json=True
    )

    # Extract metrics
    metrics = {
        'mAP50': float(val_results.box.map50),
        'mAP50-95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
        'F1': float(val_results.box.f1)
    }

    print(f"📊 Validation Results:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    return metrics


def export_model(model_path: str, args):
    """Export model to ONNX format."""
    print(f"📦 Exporting model to ONNX...")

    model = YOLO(model_path)

    # Export to ONNX
    export_path = model.export(
        format='onnx',
        imgsz=args.imgsz,
        half=not args.cpu_only,
        dynamic=False,
        simplify=True,
        opset=12
    )

    print(f"✅ Model exported to: {export_path}")
    return export_path


def generate_warehouse_boxes(img_height, img_width):
    """Generate realistic warehouse object bounding boxes."""
    boxes = []

    # Generate 2-6 random objects per image
    num_objects = np.random.randint(2, 7)

    for _ in range(num_objects):
        # Object class: 0=red_cube, 1=blue_cube, 2=green_cube, 3=pallet
        class_id = np.random.randint(0, 4)

        # Object size based on type
        if class_id == 3:  # pallet (larger)
            box_w = np.random.uniform(0.08, 0.15)  # 8-15% of image width
            box_h = np.random.uniform(0.06, 0.12)  # 6-12% of image height
        else:  # cubes (smaller)
            box_w = np.random.uniform(0.04, 0.08)  # 4-8% of image width
            box_h = np.random.uniform(0.04, 0.08)  # 4-8% of image height

        # Position (avoid placing on walls/ceiling)
        center_x = np.random.uniform(0.25, 0.75)  # Middle area
        center_y = np.random.uniform(0.5, 0.9)    # Floor area

        # Ensure box is within image bounds
        center_x = np.clip(center_x, box_w/2, 1 - box_w/2)
        center_y = np.clip(center_y, box_h/2, 1 - box_h/2)

        boxes.append({
            'class_id': class_id,
            'center_x': center_x,
            'center_y': center_y,
            'width': box_w,
            'height': box_h
        })

    return boxes


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO for weather-degraded object detection')
    parser.add_argument('--data-dir', default='data/synthetic',
                        help='Synthetic dataset directory')
    parser.add_argument('--yolo-data-dir', default='data/yolo_dataset',
                        help='YOLO format dataset directory')
    parser.add_argument('--model', default='yolov8s.pt',
                        help='YOLO model to fine-tune')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (GTX 1650 optimized)')
    parser.add_argument('--imgsz', type=int, default=448,
                        help='Image size (GTX 1650 optimized)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--save-dir', default='models/yolo',
                        help='Model save directory')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--use-gridformer', action='store_true',
                        help='Use GridFormer processed images')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export existing model')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing model')

    args = parser.parse_args()

    # Setup device info
    if args.cpu_only:
        print("🖥️  Using CPU for training")
    else:
        if torch.cuda.is_available():
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            print(f"   VRAM: {memory_gb:.1f}GB")
        else:
            print("🖥️  CUDA not available, using CPU")
            args.cpu_only = True

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    if MLFLOW_AVAILABLE:
        setup_mlflow("yolo_training")
        mlflow.start_run()
        mlflow.log_params(vars(args))

    try:
        # Prepare YOLO dataset
        if not args.validate_only and not args.export_only:
            print(f"📁 Preparing YOLO dataset...")
            data_config = prepare_yolo_dataset(
                args.data_dir,
                args.yolo_data_dir,
                args.use_gridformer
            )
        else:
            data_config = str(Path(args.yolo_data_dir) / 'data.yaml')

        # Train model
        if not args.validate_only and not args.export_only:
            results, best_model_path = train_yolo_model(data_config, args)

            # Log training metrics to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    'final_mAP50': float(results.box.map50),
                    'final_mAP50-95': float(results.box.map),
                    'final_loss': float(results.box.loss)
                })
        else:
            # Use existing model
            best_model_path = str(
                save_dir / 'weather_detection' / 'weights' / 'best.pt')

        # Validate model
        if Path(best_model_path).exists():
            val_metrics = validate_model(best_model_path, data_config, args)

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics(val_metrics)

        # Export model
        if not args.validate_only:
            export_path = export_model(best_model_path, args)

            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(export_path)

        print(f"✅ YOLO training pipeline completed!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "failed")
        raise
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


if __name__ == "__main__":
    main()
