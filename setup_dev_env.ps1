# GridFormer Robot Development Environment Setup - Windows
# Usage: .\setup_dev_env.ps1

Write-Host "🛠️  GridFormer Robot Development Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Check Python version
Write-Host "📋 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "📋 Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "📋 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "📋 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "📋 Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create necessary directories
Write-Host "📋 Creating necessary directories..." -ForegroundColor Yellow
$dirs = @("models", "logs", "data", "config")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    } else {
        Write-Host "✅ Directory exists: $dir" -ForegroundColor Green
    }
}

# Download YOLOv8 nano model for testing
Write-Host "Downloading YOLOv8 nano model for testing..." -ForegroundColor Yellow
try {
    python -c "from ultralytics import YOLO; import os; os.makedirs('models', exist_ok=True); model = YOLO('yolov8n.pt'); print('YOLOv8 nano model downloaded')"
} catch {
    Write-Host "YOLOv8 download failed, will download on first use" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Development environment setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run quick test: .\scripts\run_quick_test.ps1" -ForegroundColor White
Write-Host "2. Test environment: python env.py --render" -ForegroundColor White
Write-Host "3. Full test suite: python run_test.py --quick" -ForegroundColor White
Write-Host ""
Write-Host "Note: For ROS 2 functionality, use WSL2 or Docker" -ForegroundColor Yellow 