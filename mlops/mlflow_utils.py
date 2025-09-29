#!/usr/bin/env python3
"""
MLflow utilities for GridFormer Robot project
Experiment tracking, model logging, and performance monitoring
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import json


class GridFormerMLflowLogger:
    """MLflow logger for GridFormer experiments"""

    def __init__(self,
                 experiment_name: str = "gridformer-robot",
                 tracking_uri: str = "file:///workspace/mlruns"):
        """Initialize MLflow logger

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Start a new MLflow run

        Args:
            run_name: Optional name for the run
            tags: Optional tags dictionary
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.run

    def log_system_info(self):
        """Log system information"""
        import platform
        import psutil
        import torch

        system_info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            })

        mlflow.log_params(system_info)

    def log_model_config(self, config: Dict[str, Any]):
        """Log model configuration

        Args:
            config: Configuration dictionary
        """
        mlflow.log_params(config)

    def log_training_metrics(self,
                             epoch: int,
                             train_loss: float,
                             val_loss: Optional[float] = None,
                             psnr: Optional[float] = None,
                             ssim: Optional[float] = None):
        """Log training metrics

        Args:
            epoch: Training epoch
            train_loss: Training loss
            val_loss: Validation loss
            psnr: PSNR metric
            ssim: SSIM metric
        """
        metrics = {"train_loss": train_loss}

        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if psnr is not None:
            metrics["psnr"] = psnr
        if ssim is not None:
            metrics["ssim"] = ssim

        mlflow.log_metrics(metrics, step=epoch)

    def log_inference_metrics(self,
                              fps: float,
                              latency_ms: float,
                              memory_usage_mb: float,
                              batch_size: int = 1):
        """Log inference performance metrics

        Args:
            fps: Frames per second
            latency_ms: Average latency in milliseconds
            memory_usage_mb: Memory usage in MB
            batch_size: Batch size used
        """
        metrics = {
            "inference_fps": fps,
            "inference_latency_ms": latency_ms,
            "memory_usage_mb": memory_usage_mb,
            "batch_size": batch_size
        }

        mlflow.log_metrics(metrics)

    def log_detection_metrics(self,
                              map50: float,
                              map75: float,
                              precision: float,
                              recall: float,
                              f1_score: float):
        """Log object detection metrics

        Args:
            map50: mAP@0.5
            map75: mAP@0.75
            precision: Average precision
            recall: Average recall
            f1_score: F1 score
        """
        metrics = {
            "detection_map50": map50,
            "detection_map75": map75,
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": f1_score
        }

        mlflow.log_metrics(metrics)

    def log_navigation_metrics(self,
                               success_rate: float,
                               average_time: float,
                               path_length: float,
                               collision_count: int):
        """Log navigation performance metrics

        Args:
            success_rate: Navigation success rate (0-1)
            average_time: Average navigation time
            path_length: Average path length
            collision_count: Number of collisions
        """
        metrics = {
            "nav_success_rate": success_rate,
            "nav_average_time": average_time,
            "nav_path_length": path_length,
            "nav_collision_count": collision_count
        }

        mlflow.log_metrics(metrics)

    def log_image_comparison(self,
                             original: np.ndarray,
                             restored: np.ndarray,
                             filename: str = "comparison"):
        """Log image comparison (before/after restoration)

        Args:
            original: Original degraded image
            restored: Restored image
            filename: Base filename for saved images
        """
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original (Degraded)")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Restored")
        axes[1].axis('off')

        plt.tight_layout()

        # Save and log
        comparison_path = f"/tmp/{filename}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(comparison_path)
        plt.close()

        # Also log individual images
        original_path = f"/tmp/{filename}_original.png"
        restored_path = f"/tmp/{filename}_restored.png"

        cv2.imwrite(original_path, original)
        cv2.imwrite(restored_path, restored)

        mlflow.log_artifact(original_path)
        mlflow.log_artifact(restored_path)

        # Clean up temp files
        os.remove(comparison_path)
        os.remove(original_path)
        os.remove(restored_path)

    def log_model(self, model, model_name: str = "gridformer"):
        """Log PyTorch model

        Args:
            model: PyTorch model
            model_name: Name for the logged model
        """
        mlflow.pytorch.log_model(model, model_name)

    def log_tensorrt_engine(self, engine_path: str, model_name: str = "gridformer_trt"):
        """Log TensorRT engine as artifact

        Args:
            engine_path: Path to TensorRT engine file
            model_name: Name for the artifact
        """
        mlflow.log_artifact(engine_path, artifact_path=model_name)

    def log_config_file(self, config_path: str):
        """Log configuration file

        Args:
            config_path: Path to configuration file
        """
        mlflow.log_artifact(config_path)

    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()


class ExperimentComparison:
    """Compare multiple MLflow experiments"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)

    def get_best_run(self, metric: str, ascending: bool = False) -> mlflow.entities.Run:
        """Get the best run based on a metric

        Args:
            metric: Metric name to compare
            ascending: If True, lower values are better

        Returns:
            Best MLflow run
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )

        if len(runs) == 0:
            raise ValueError("No runs found in experiment")

        best_run_id = runs.iloc[0]['run_id']
        return mlflow.get_run(best_run_id)

    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> Dict:
        """Compare specific runs

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            comparison[run_id] = {
                "name": run.info.run_name,
                "metrics": {metric: run.data.metrics.get(metric) for metric in metrics},
                "start_time": run.info.start_time,
                "end_time": run.info.end_time
            }

        return comparison

    def generate_report(self, output_path: str = "/workspace/experiment_report.html"):
        """Generate HTML experiment report

        Args:
            output_path: Path to save the report
        """
        runs_df = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id])

        html_content = f"""
        <html>
        <head>
            <title>GridFormer Experiment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #007ACC; }}
            </style>
        </head>
        <body>
            <h1>GridFormer Robot Experiment Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total runs: {len(runs_df)}</p>
            
            <h2>Run Summary</h2>
            {runs_df.to_html(classes='table table-striped', escape=False)}
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Report saved to: {output_path}")


def setup_mlflow_experiment(experiment_name: str = "gridformer-robot") -> GridFormerMLflowLogger:
    """Convenience function to setup MLflow experiment

    Args:
        experiment_name: Name of the experiment

    Returns:
        Configured MLflow logger
    """
    logger = GridFormerMLflowLogger(experiment_name)
    return logger


if __name__ == "__main__":
    # Example usage
    logger = setup_mlflow_experiment()

    with logger.start_run("test_run"):
        logger.log_system_info()
        logger.log_training_metrics(epoch=1, train_loss=0.5, psnr=25.0)
        logger.log_inference_metrics(
            fps=30.0, latency_ms=33.3, memory_usage_mb=512.0)

    print("✅ MLflow test completed!")
