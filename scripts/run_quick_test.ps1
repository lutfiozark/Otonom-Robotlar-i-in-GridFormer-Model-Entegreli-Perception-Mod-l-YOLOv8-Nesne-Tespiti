# GridFormer Robot Quick Test - PowerShell Version
# Usage: .\scripts\run_quick_test.ps1

Write-Host "GridFormer Robot Quick Test (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "env.py")) {
    Write-Host "Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Test 1: Basic PyBullet test
Write-Host "Test 1: Basic PyBullet test" -ForegroundColor Yellow
try {
    python -c "import pybullet as p; print('PyBullet import successful'); client = p.connect(p.DIRECT); print('PyBullet connection successful'); p.disconnect(); print('PyBullet test completed')"
} catch {
    Write-Host "PyBullet test failed: $_" -ForegroundColor Red
    exit 1
}

# Test 2: Basic environment
Write-Host "Test 2: Environment creation test" -ForegroundColor Yellow
try {
    python -c "import env; warehouse_env = env.WarehouseEnv(render_mode='DIRECT'); warehouse_env.connect(); warehouse_env.setup_scene(); print('Environment setup successful'); image = warehouse_env.get_camera_image(); print(f'Camera image captured: {image.shape}'); warehouse_env.disconnect(); print('Environment test completed')"
} catch {
    Write-Host "Environment test failed: $_" -ForegroundColor Red
    exit 1
}

# Test 3: File structure check
Write-Host "Test 3: File structure check" -ForegroundColor Yellow
$files = @(
    "requirements.txt",
    "Dockerfile", 
    "docker-compose.yml",
    "env.py",
    "perception\gridformer_node.py",
    "perception\yolov8_node.py",
    "launch\warehouse_demo.launch.py",
    "scripts\onnx_to_trt.sh"
)

$missing = @()
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "$file exists" -ForegroundColor Green
    } else {
        Write-Host "$file missing" -ForegroundColor Red
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing files found!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Quick test completed successfully!" -ForegroundColor Green
Write-Host "To run the full test suite: python run_test.py" -ForegroundColor Cyan
Write-Host "To start the system: python env.py --render" -ForegroundColor Cyan 
