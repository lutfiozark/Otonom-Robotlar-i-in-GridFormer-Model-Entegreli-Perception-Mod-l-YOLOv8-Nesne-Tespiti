#!/usr/bin/env python3
"""Test runner script with different configurations for various environments."""

import argparse
import subprocess
import sys
import os
import torch
from pathlib import Path


def check_gpu_availability():
    """Check if CUDA GPU is available."""
    return torch.cuda.is_available()


def check_ros2_installation():
    """Check if ROS 2 is installed and sourced."""
    try:
        result = subprocess.run(['ros2', '--version'],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description or command}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description or command}")
        print(f"Error code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tests with different configurations")
    parser.add_argument('--type', choices=['unit', 'integration', 'all', 'quick'],
                        default='quick', help='Type of tests to run')
    parser.add_argument('--gpu', action='store_true', help='Run GPU tests')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only tests')
    parser.add_argument('--ros', action='store_true',
                        help='Include ROS 2 tests')
    parser.add_argument('--coverage', action='store_true',
                        help='Generate coverage report')
    parser.add_argument('--verbose', '-v',
                        action='store_true', help='Verbose output')
    parser.add_argument('--markers', help='Pytest markers to include/exclude')

    args = parser.parse_args()

    # Check system capabilities
    has_gpu = check_gpu_availability() and not args.cpu_only
    has_ros2 = check_ros2_installation()

    print(f"🖥️  System Check:")
    print(f"   GPU Available: {'✅' if has_gpu else '❌'}")
    print(f"   ROS 2 Available: {'✅' if has_ros2 else '❌'}")
    print(f"   Python: {sys.version.split()[0]}")

    # Base pytest command
    cmd_parts = ['python', '-m', 'pytest']

    if args.verbose:
        cmd_parts.append('-v')

    # Test selection based on type
    if args.type == 'unit':
        cmd_parts.extend(['-m', 'unit'])
    elif args.type == 'integration':
        cmd_parts.extend(['-m', 'integration'])
    elif args.type == 'quick':
        cmd_parts.extend(['-m', 'not slow'])
    elif args.type == 'all':
        pass  # Run all tests

    # GPU/CPU selection
    if not has_gpu or args.cpu_only:
        cmd_parts.extend(['-m', 'not gpu'])
        print("🚫 Skipping GPU tests (not available or CPU-only mode)")
    elif args.gpu:
        cmd_parts.extend(['-m', 'gpu'])
        print("🎮 Running GPU tests only")

    # ROS 2 selection
    if not has_ros2:
        if 'not gpu' in cmd_parts:
            # Find the index of 'not gpu' and replace with 'not gpu and not ros'
            idx = cmd_parts.index('not gpu')
            cmd_parts[idx] = 'not gpu and not ros'
        else:
            cmd_parts.extend(['-m', 'not ros'])
        print("🚫 Skipping ROS 2 tests (not available)")
    elif args.ros:
        cmd_parts.extend(['-m', 'ros'])
        print("🤖 Running ROS 2 tests only")

    # Custom markers
    if args.markers:
        cmd_parts.extend(['-m', args.markers])

    # Coverage
    if args.coverage:
        cmd_parts.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])

    # Add test directory
    cmd_parts.append('tests/')

    # Environment setup
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())

    if args.cpu_only:
        env['CUDA_VISIBLE_DEVICES'] = ''

    # Run linting first (quick checks)
    if args.type in ['all', 'quick']:
        print("\n🔍 Running code quality checks...")

        # Check if tools are available
        linting_commands = [
            ('python -m flake8 . --count --statistics', 'Flake8 linting'),
            ('python -m black --check .', 'Black formatting check'),
            ('python -m isort --check-only .', 'Import sorting check')
        ]

        for cmd, desc in linting_commands:
            try:
                subprocess.run(cmd, shell=True, check=True,
                               capture_output=True)
                print(f"✅ {desc} passed")
            except subprocess.CalledProcessError:
                print(
                    f"⚠️  {desc} failed (install with: pip install {desc.split()[0]})")

    # Run main tests
    print(f"\n🧪 Running tests: {' '.join(cmd_parts)}")

    try:
        result = subprocess.run(cmd_parts, env=env)

        if result.returncode == 0:
            print(f"\n🎉 All tests passed!")

            if args.coverage:
                print(f"📊 Coverage report generated in htmlcov/")

        else:
            print(f"\n💥 Some tests failed (exit code: {result.returncode})")

        return result.returncode

    except KeyboardInterrupt:
        print(f"\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
