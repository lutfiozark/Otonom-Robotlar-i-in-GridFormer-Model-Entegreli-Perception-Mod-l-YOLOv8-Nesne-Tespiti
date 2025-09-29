#!/usr/bin/env python3
"""Simple training monitor without GPUtil dependency."""

import time
import psutil
import os
from pathlib import Path


def check_training_status():
    """Check current training status."""
    print(f"🔍 Training Status Check - {time.strftime('%H:%M:%S')}")
    print("=" * 50)

    # Check Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python.exe', 'python']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train_gridformer.py' in cmdline or 'train_yolo.py' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    print(f"💻 System:")
    print(f"   CPU: {cpu_percent:.1f}%")
    print(
        f"   Memory: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

    # Training processes
    if python_processes:
        print(f"\n🚀 Active Training:")
        for proc in python_processes:
            script_name = 'GridFormer' if 'gridformer' in proc['cmdline'] else 'YOLO'
            print(f"   {script_name} (PID: {proc['pid']})")
    else:
        print(f"\n⏸️  No active training processes")

    # Check model files
    print(f"\n📁 Model Files:")

    # GridFormer
    gf_paths = ['models/gridformer/best_model.pth',
                'models/gridformer/latest_checkpoint.pth']
    gf_found = False
    for path in gf_paths:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024*1024)
            print(f"   GridFormer: {path} ({size_mb:.1f}MB)")
            gf_found = True
            break
    if not gf_found:
        print(f"   GridFormer: Not found")

    # YOLO
    yolo_paths = [
        'models/yolo/weather_detection/weights/best.pt', 'yolov8s.pt']
    yolo_found = False
    for path in yolo_paths:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024*1024)
            print(f"   YOLO: {path} ({size_mb:.1f}MB)")
            yolo_found = True
            break
    if not yolo_found:
        print(f"   YOLO: Not found")

    # MLflow runs
    mlruns_path = Path('mlruns')
    if mlruns_path.exists():
        runs = list(mlruns_path.glob('*/*/'))
        print(f"   MLflow: {len(runs)} runs")
    else:
        print(f"   MLflow: No runs")

    print("=" * 50)


if __name__ == "__main__":
    check_training_status()
