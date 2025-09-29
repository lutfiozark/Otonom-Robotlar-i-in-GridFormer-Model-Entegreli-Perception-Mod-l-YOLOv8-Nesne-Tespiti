#!/usr/bin/env python3
"""Validate project setup and dependencies."""

import sys
import subprocess
import importlib
import torch
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({memory_gb:.1f}GB)")

        if memory_gb >= 4.0:
            print("✅ Sufficient VRAM for GTX 1650+ requirements")
        else:
            print("⚠️  Limited VRAM - consider using reduced image sizes")
        return True
    else:
        print("❌ No CUDA GPU available")
        return False


def check_ros2():
    """Check ROS 2 installation."""
    try:
        result = subprocess.run(['ros2', '--version'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ ROS 2: {version}")
            return True
        else:
            print("❌ ROS 2 not properly installed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ROS 2 not found")
        return False


def check_project_structure():
    """Check if project has required structure."""
    required_files = [
        'requirements.txt',
        'gridformer.py',
        'env.py',
        'tests/__init__.py',
        'data/data.yaml',
        'LICENSE',
        'CONTRIBUTING.md'
    ]

    required_dirs = [
        'tests',
        'data',
        'perception',
        'navigation',
        'scripts'
    ]

    print("\n📁 Project Structure:")
    all_good = True

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_good = False

    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            all_good = False

    return all_good


def main():
    """Main validation function."""
    print("🔍 Weather-Adaptive Navigation System - Setup Validation")
    print("=" * 60)

    checks = []

    # Python version
    print("\n🐍 Python Environment:")
    checks.append(check_python_version())

    # Core packages
    print("\n📦 Core Dependencies:")
    core_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('onnx', 'onnx'),
        ('onnxruntime-gpu', 'onnxruntime'),
        ('ultralytics', 'ultralytics'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pyyaml', 'yaml'),
        ('pytest', 'pytest')
    ]

    for package, import_name in core_packages:
        checks.append(check_package(package, import_name))

    # ROS 2 packages
    print("\n🤖 ROS 2 Dependencies:")
    ros_packages = [
        ('rclpy', 'rclpy'),
        ('sensor-msgs', 'sensor_msgs'),
        ('geometry-msgs', 'geometry_msgs'),
        ('nav-msgs', 'nav_msgs')
    ]

    for package, import_name in ros_packages:
        checks.append(check_package(package, import_name))

    # GPU check
    print("\n🎮 GPU Support:")
    checks.append(check_gpu())

    # ROS 2 installation
    print("\n🤖 ROS 2 Installation:")
    checks.append(check_ros2())

    # Project structure
    checks.append(check_project_structure())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\n✨ Your environment is ready for development!")
        print("\nNext steps:")
        print("1. Run tests: python scripts/run_tests.py --type quick")
        print("2. Generate synthetic data: python data/generate_synthetic_data.py")
        print("3. Train models: python gridformer_yolo_pipeline.py")
        return 0
    else:
        print(f"⚠️  Some checks failed ({passed}/{total})")
        print(f"\n📋 To fix issues:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install ROS 2 Humble: https://docs.ros.org/en/humble/Installation.html")
        print("3. Install CUDA drivers for GPU support")
        return 1


if __name__ == "__main__":
    sys.exit(main())
