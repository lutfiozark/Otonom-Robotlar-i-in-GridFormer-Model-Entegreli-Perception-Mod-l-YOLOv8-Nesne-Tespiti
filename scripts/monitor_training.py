#!/usr/bin/env python3
"""Monitor training progress for GridFormer and YOLO."""

import time
import psutil
import GPUtil
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


class TrainingMonitor:
    """Monitor training progress and system resources."""

    def __init__(self, log_file="training_monitor.json"):
        self.log_file = Path(log_file)
        self.start_time = time.time()
        self.logs = []

    def get_system_info(self):
        """Get current system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # GPU usage (if available)
        gpu_info = None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_info = {
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                }
        except Exception:
            pass

        # Disk usage
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100

        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'gpu': gpu_info,
            'disk_percent': disk_percent
        }

    def check_training_processes(self):
        """Check if training processes are running."""
        python_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'train_gridformer.py' in cmdline or 'train_yolo.py' in cmdline:
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline,
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return python_processes

    def check_model_files(self):
        """Check for training output files."""
        files_status = {
            'gridformer_model': None,
            'yolo_model': None,
            'mlflow_runs': None
        }

        # Check GridFormer outputs
        gridformer_paths = [
            'models/gridformer/best_model.pth',
            'models/gridformer/latest_checkpoint.pth'
        ]

        for path in gridformer_paths:
            if Path(path).exists():
                stat = Path(path).stat()
                files_status['gridformer_model'] = {
                    'path': path,
                    'size_mb': stat.st_size / (1024*1024),
                    'modified': stat.st_mtime
                }
                break

        # Check YOLO outputs
        yolo_paths = [
            'models/yolo/weather_detection/weights/best.pt',
            'models/yolo/weather_detection/weights/last.pt'
        ]

        for path in yolo_paths:
            if Path(path).exists():
                stat = Path(path).stat()
                files_status['yolo_model'] = {
                    'path': path,
                    'size_mb': stat.st_size / (1024*1024),
                    'modified': stat.st_mtime
                }
                break

        # Check MLflow runs
        mlruns_path = Path('mlruns')
        if mlruns_path.exists():
            runs = list(mlruns_path.glob('*/*/'))
            files_status['mlflow_runs'] = len(runs)

        return files_status

    def log_status(self):
        """Log current status."""
        status = {
            'system': self.get_system_info(),
            'processes': self.check_training_processes(),
            'files': self.check_model_files(),
            'elapsed_time': time.time() - self.start_time
        }

        self.logs.append(status)

        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

        return status

    def print_status(self, status):
        """Print formatted status."""
        print(f"\n🕐 Elapsed: {status['elapsed_time']/60:.1f} minutes")
        print(f"💻 CPU: {status['system']['cpu_percent']:.1f}%")
        print(f"🧠 Memory: {status['system']['memory_percent']:.1f}% "
              f"({status['system']['memory_used_gb']:.1f}GB)")

        if status['system']['gpu']:
            gpu = status['system']['gpu']
            print(f"🎮 GPU: {gpu['utilization']:.1f}% | "
                  f"VRAM: {gpu['memory_percent']:.1f}% "
                  f"({gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB) | "
                  f"Temp: {gpu['temperature']:.0f}°C")

        # Training processes
        if status['processes']:
            print(f"\n🚀 Active Training:")
            for proc in status['processes']:
                script_name = 'GridFormer' if 'gridformer' in proc['cmdline'] else 'YOLO'
                print(f"   {script_name} (PID: {proc['pid']}) | "
                      f"CPU: {proc['cpu_percent']:.1f}% | "
                      f"MEM: {proc['memory_percent']:.1f}%")
        else:
            print(f"\n⏸️  No training processes detected")

        # Model files
        files = status['files']
        print(f"\n📁 Model Files:")

        if files['gridformer_model']:
            gf = files['gridformer_model']
            modified_mins = (time.time() - gf['modified']) / 60
            print(
                f"   GridFormer: {gf['size_mb']:.1f}MB (updated {modified_mins:.1f}m ago)")
        else:
            print(f"   GridFormer: Not found")

        if files['yolo_model']:
            yolo = files['yolo_model']
            modified_mins = (time.time() - yolo['modified']) / 60
            print(
                f"   YOLO: {yolo['size_mb']:.1f}MB (updated {modified_mins:.1f}m ago)")
        else:
            print(f"   YOLO: Not found")

        if files['mlflow_runs']:
            print(f"   MLflow Runs: {files['mlflow_runs']}")
        else:
            print(f"   MLflow: No runs")

    def run_continuous(self, interval=30):
        """Run continuous monitoring."""
        print(f"🔍 Starting training monitor (interval: {interval}s)")
        print(f"📊 Log file: {self.log_file}")
        print(f"Press Ctrl+C to stop")

        try:
            while True:
                status = self.log_status()
                self.print_status(status)
                print(f"{'='*60}")
                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped")
            print(f"📊 Logs saved to: {self.log_file}")

    def generate_report(self):
        """Generate training report with plots."""
        if not self.logs:
            print("❌ No logs available for report")
            return

        # Extract time series data
        timestamps = [log['elapsed_time'] /
                      60 for log in self.logs]  # Convert to minutes
        cpu_usage = [log['system']['cpu_percent'] for log in self.logs]
        memory_usage = [log['system']['memory_percent'] for log in self.logs]

        gpu_usage = []
        gpu_memory = []
        gpu_temp = []

        for log in self.logs:
            if log['system']['gpu']:
                gpu_usage.append(log['system']['gpu']['utilization'])
                gpu_memory.append(log['system']['gpu']['memory_percent'])
                gpu_temp.append(log['system']['gpu']['temperature'])
            else:
                gpu_usage.append(0)
                gpu_memory.append(0)
                gpu_temp.append(0)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Monitor Report', fontsize=16)

        # CPU Usage
        axes[0, 0].plot(timestamps, cpu_usage, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)

        # Memory Usage
        axes[0, 1].plot(timestamps, memory_usage, 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True, alpha=0.3)

        # GPU Usage
        if any(gpu_usage):
            axes[1, 0].plot(timestamps, gpu_usage, 'r-',
                            linewidth=2, label='Utilization')
            axes[1, 0].plot(timestamps, gpu_memory, 'orange',
                            linewidth=2, label='Memory')
            axes[1, 0].set_title('GPU Usage')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('GPU %')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No GPU Data', ha='center',
                            va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GPU Usage - No Data')

        # GPU Temperature
        if any(gpu_temp):
            axes[1, 1].plot(timestamps, gpu_temp, 'm-', linewidth=2)
            axes[1, 1].set_title('GPU Temperature')
            axes[1, 1].set_xlabel('Time (minutes)')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No GPU Data', ha='center',
                            va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('GPU Temperature - No Data')

        plt.tight_layout()

        # Save report
        report_path = 'training_monitor_report.png'
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        print(f"📊 Report saved: {report_path}")

        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--interval', type=int, default=30,
                        help='Monitoring interval (seconds)')
    parser.add_argument('--report', action='store_true',
                        help='Generate report from existing logs')
    parser.add_argument('--once', action='store_true',
                        help='Check status once and exit')

    args = parser.parse_args()

    monitor = TrainingMonitor()

    if args.report:
        monitor.generate_report()
    elif args.once:
        status = monitor.log_status()
        monitor.print_status(status)
    else:
        monitor.run_continuous(args.interval)


if __name__ == "__main__":
    main()
