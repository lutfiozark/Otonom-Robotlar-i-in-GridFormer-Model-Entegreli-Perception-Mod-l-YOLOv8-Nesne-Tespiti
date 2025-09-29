#!/usr/bin/env python3
"""
GridFormer Robot Test Script
Quick test of the entire system
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🧪 {text}")
    print(f"{'='*60}")


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")

    required_packages = [
        'cv2', 'numpy', 'torch', 'ultralytics',
        'pybullet', 'mlflow', 'gymnasium'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is missing")

    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_info("Install with: pip install -r requirements.txt")
        return False

    print_success("All dependencies are installed!")
    return True


def test_pybullet_env():
    """Test PyBullet environment"""
    print_header("Testing PyBullet Environment")

    try:
        import env

        # Test environment creation
        warehouse_env = env.WarehouseEnv(render_mode="DIRECT")
        warehouse_env.connect()
        warehouse_env.setup_scene()

        print_success("Environment created successfully")

        # Test image capture
        image = warehouse_env.get_camera_image()
        print_success(f"Camera image captured: {image.shape}")

        # Test FPS benchmark
        print_info("Running 3-second FPS benchmark...")
        fps = warehouse_env.get_fps_benchmark(duration=3.0)
        print_success(f"Benchmark FPS: {fps:.1f}")

        warehouse_env.disconnect()
        print_success("PyBullet environment test completed!")
        return True

    except Exception as e:
        print_error(f"PyBullet environment test failed: {e}")
        return False


def test_model_loading():
    """Test model loading capabilities"""
    print_header("Testing Model Loading")

    try:
        # Test YOLOv8 loading
        from ultralytics import YOLO

        print_info("Testing YOLOv8 model loading...")
        model = YOLO('yolov8n.pt')  # Load nano model for testing
        print_success("YOLOv8 model loaded successfully")

        # Test dummy inference
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print_success("YOLOv8 inference test passed")

        return True

    except Exception as e:
        print_error(f"Model loading test failed: {e}")
        return False


def test_mlflow_setup():
    """Test MLflow setup"""
    print_header("Testing MLflow Setup")

    try:
        from mlops.mlflow_utils import setup_mlflow_experiment

        # Test MLflow logger creation
        logger = setup_mlflow_experiment("test-experiment")
        print_success("MLflow logger created successfully")

        # Test run creation
        with logger.start_run("test-run"):
            logger.log_system_info()
            logger.log_inference_metrics(
                fps=30.0, latency_ms=33.3, memory_usage_mb=512.0)
            print_success("MLflow logging test passed")

        return True

    except Exception as e:
        print_error(f"MLflow test failed: {e}")
        return False


def test_docker_build():
    """Test Docker build"""
    print_header("Testing Docker Build")

    try:
        print_info("Building Docker image (this may take a while)...")

        result = subprocess.run(
            ['docker', 'build', '-t', 'gridformer-robot-test', '.'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print_success("Docker image built successfully")

            # Test container creation
            print_info("Testing container creation...")
            result = subprocess.run(
                ['docker', 'run', '--rm', 'gridformer-robot-test',
                    'echo', 'Container test'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print_success("Docker container test passed")
                return True
            else:
                print_error(f"Container test failed: {result.stderr}")
                return False
        else:
            print_error(f"Docker build failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print_error("Docker build timed out")
        return False
    except Exception as e:
        print_error(f"Docker test failed: {e}")
        return False


def test_file_structure():
    """Test if all required files exist"""
    print_header("Testing File Structure")

    required_files = [
        'README.md',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'env.py',
        'perception/gridformer_node.py',
        'perception/yolov8_node.py',
        'navigation/rl_agent_node.py',
        'launch/warehouse_demo.launch.py',
        'scripts/onnx_to_trt.sh',
        'mlops/mlflow_utils.py'
    ]

    missing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"{file_path} exists")
        else:
            missing_files.append(file_path)
            print_error(f"{file_path} is missing")

    if missing_files:
        print_error(f"Missing files: {', '.join(missing_files)}")
        return False

    print_success("All required files exist!")
    return True


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='GridFormer Robot Test Suite')
    parser.add_argument('--skip-docker', action='store_true',
                        help='Skip Docker tests')
    parser.add_argument('--quick', action='store_true',
                        help='Run only quick tests')
    args = parser.parse_args()

    print_header("GridFormer Robot Test Suite")

    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", check_dependencies),
        ("PyBullet Environment", test_pybullet_env),
        ("Model Loading", test_model_loading),
        ("MLflow Setup", test_mlflow_setup),
    ]

    if not args.skip_docker and not args.quick:
        tests.append(("Docker Build", test_docker_build))

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")

    # Final report
    print_header("Test Results")
    print(f"Passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print_success("🎉 All tests passed! GridFormer Robot is ready!")
        return 0
    else:
        print_error(f"❌ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
