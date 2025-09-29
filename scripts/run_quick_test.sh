#!/bin/bash
# Quick test script for GridFormer Robot
# Usage: bash scripts/run_quick_test.sh

set -e

echo "🚀 GridFormer Robot Quick Test"
echo "================================"

# Check if we're in the right directory
if [[ ! -f "env.py" ]]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Test 1: Environment variables
echo "📋 Test 1: Basic PyBullet test"
python3 -c "
import pybullet as p
print('✅ PyBullet import successful')
client = p.connect(p.DIRECT)
print('✅ PyBullet connection successful')
p.disconnect()
print('✅ PyBullet test completed')
"

# Test 2: Basic environment
echo "📋 Test 2: Environment creation test"
python3 -c "
import env
warehouse_env = env.WarehouseEnv(render_mode='DIRECT')
warehouse_env.connect()
warehouse_env.setup_scene()
print('✅ Environment setup successful')
image = warehouse_env.get_camera_image()
print(f'✅ Camera image captured: {image.shape}')
warehouse_env.disconnect()
print('✅ Environment test completed')
"

# Test 3: Check critical files
echo "📋 Test 3: File structure check"
files=(
    "requirements.txt"
    "Dockerfile" 
    "docker-compose.yml"
    "env.py"
    "perception/gridformer_node.py"
    "perception/yolov8_node.py"
    "launch/warehouse_demo.launch.py"
    "scripts/onnx_to_trt.sh"
)

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo ""
echo "🎉 Quick test completed successfully!"
echo "To run the full test suite: python3 run_test.py"
echo "To start the system: python3 env.py --render" 