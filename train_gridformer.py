#!/usr/bin/env python3
"""GridFormer fine-tuning script for weather adaptation."""

import argparse
import os
import sys
import time
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    from mlops.mlflow_utils import setup_mlflow, log_metrics
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from gridformer import GridFormerModel


class WeatherDataset(Dataset):
    """Dataset for weather degraded and clean image pairs."""

    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load synthetic data annotations
        annotations_file = self.data_dir / 'annotations.yaml'
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                self.annotations = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_file}")

        # Filter data by weather conditions (exclude 'clear' for degraded images)
        self.image_pairs = []
        weather_conditions = self.annotations['weather_conditions']

        # Group by scene_id to create clean-degraded pairs
        scene_groups = {}
        for item in weather_conditions:
            scene_id = item['scene_id']
            if scene_id not in scene_groups:
                scene_groups[scene_id] = {}
            scene_groups[scene_id][item['weather']] = item

        # Create pairs: clean image as target, degraded as input
        for scene_id, weather_dict in scene_groups.items():
            if 'clear' in weather_dict:
                clean_item = weather_dict['clear']
                for weather, degraded_item in weather_dict.items():
                    if weather != 'clear':
                        self.image_pairs.append({
                            'input': degraded_item,  # Degraded image
                            'target': clean_item,    # Clean image
                            'weather': weather,
                            'intensity': degraded_item['intensity']
                        })

        print(f"Loaded {len(self.image_pairs)} image pairs for {split}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        # Load degraded (input) image
        input_filename = pair['input']['filename']
        weather_type = pair['weather']
        input_path = self.data_dir / weather_type / input_filename

        # Load clean (target) image
        target_filename = pair['target']['filename']
        target_path = self.data_dir / 'clear' / target_filename

        # Load images
        input_image = cv2.imread(str(input_path))
        target_image = cv2.imread(str(target_path))

        if input_image is None or target_image is None:
            raise FileNotFoundError(
                f"Could not load images: {input_path}, {target_path}")

        # Convert BGR to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return {
            'input': input_image,
            'target': target_image,
            'weather': weather_type,
            'intensity': pair['intensity']
        }


def get_transforms(image_size=448):
    """Get image transformations for training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class GridFormerTrainer:
    """GridFormer training class."""

    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Progress reporting
            if batch_idx % 10 == 0:
                progress = 100. * batch_idx / num_batches
                print(f'Epoch {epoch}: [{batch_idx}/{num_batches} ({progress:.1f}%)]'
                      f'\tLoss: {loss.item():.6f}')

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate PSNR
                psnr = calculate_psnr(outputs, targets)

                total_loss += loss.item()
                total_psnr += psnr.item() if psnr != float('inf') else 40.0

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        self.val_losses.append(avg_loss)
        self.val_psnrs.append(avg_psnr)

        print(f'Validation - Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f}dB')

        return avg_loss, avg_psnr

    def save_checkpoint(self, epoch, val_loss, save_path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_psnrs': self.val_psnrs,
            'args': self.args
        }
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Train GridFormer for weather adaptation')
    parser.add_argument(
        '--data-dir', default='data/synthetic', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (GTX 1650 optimized)')
    parser.add_argument('--learning-rate', type=float,
                        default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-5, help='Weight decay')
    parser.add_argument('--lr-step-size', type=int,
                        default=3, help='LR scheduler step size')
    parser.add_argument('--lr-gamma', type=float,
                        default=0.5, help='LR scheduler gamma')
    parser.add_argument('--imgsz', type=int, default=448,
                        help='Image size (GTX 1650 optimized)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument(
        '--save-dir', default='models/gridformer', help='Model save directory')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU training')

    args = parser.parse_args()

    # Setup device
    if args.cpu_only:
        device = torch.device('cpu')
        print("🖥️  Using CPU for training")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            print(f"   VRAM: {memory_gb:.1f}GB")
        else:
            print("🖥️  CUDA not available, using CPU")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    if MLFLOW_AVAILABLE:
        setup_mlflow("gridformer_training")
        mlflow.start_run()
        mlflow.log_params(vars(args))

    try:
        # Load dataset
        print(f"📁 Loading dataset from {args.data_dir}")
        transform = get_transforms(args.imgsz)

        # For now, use all data for training (in production, split train/val)
        dataset = WeatherDataset(args.data_dir, transform=transform)

        # Split dataset (80% train, 20% val)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == 'cuda'
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == 'cuda'
        )

        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")

        # Load model
        print(f"🧠 Loading GridFormer model...")
        model = GridFormerModel()
        model = model.to(device)

        # Initialize trainer
        trainer = GridFormerTrainer(model, device, args)

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume and Path(args.resume).exists():
            print(f"📥 Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(
                checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            trainer.train_losses = checkpoint.get('train_losses', [])
            trainer.val_losses = checkpoint.get('val_losses', [])
            trainer.val_psnrs = checkpoint.get('val_psnrs', [])

        # Training loop
        print(f"🚀 Starting training for {args.epochs} epochs...")
        best_val_loss = float('inf')

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_psnr = trainer.validate(val_loader, epoch)

            # Update learning rate
            trainer.scheduler.step()

            epoch_time = time.time() - epoch_start

            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f}dB, '
                  f'Time: {epoch_time:.1f}s')

            # Log to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_psnr': val_psnr,
                    'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                }, step=epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_dir / 'best_model.pth'
                trainer.save_checkpoint(epoch, val_loss, best_model_path)

                if MLFLOW_AVAILABLE:
                    mlflow.pytorch.log_model(model, "best_model")

            # Save latest checkpoint
            latest_checkpoint_path = save_dir / 'latest_checkpoint.pth'
            trainer.save_checkpoint(epoch, val_loss, latest_checkpoint_path)

        print(
            f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
        print(f"💾 Models saved in: {save_dir}")

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
