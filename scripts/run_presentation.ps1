# Presentation Demo Launcher (Windows)
# Usage: .\scripts\run_presentation.ps1

Write-Host "GridFormer Robot - Presentation Demo" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Ensure output dir exists
if (-not (Test-Path "docs/figures")) {
  New-Item -ItemType Directory -Path "docs/figures" | Out-Null
}

# Resolve Python executables
$venvPython = Join-Path ".venv\Scripts" "python.exe"
$useVenv = Test-Path $venvPython
if (-not $useVenv) {
  Write-Host "No venv found. Creating .venv and installing requirements..." -ForegroundColor Yellow
  py -3 -m venv .venv
  $useVenv = Test-Path $venvPython
}

if ($useVenv) {
  Write-Host "Using venv Python: $venvPython" -ForegroundColor Yellow

  # Ensure requirements are installed inside venv
  & $venvPython -c "import numpy" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing minimal demo dependencies into venv..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install numpy opencv-python onnxruntime
  }

  $pythonExe = $venvPython
} else {
  Write-Host "Falling back to system Python." -ForegroundColor Yellow
  $pythonExe = "python"
}

# Run demo generator (weather + restoration + detections) in 3-panel view
$argsList = @(
  "scripts/present_weather_demo.py",
  "--gridformer-model", "models/gridformer_optimized_384.onnx",
  "--yolo-onnx", "models/yolov8s_optimized_416.onnx",
  "--yolo-weights", "models/yolov8n.pt",
  "--width", "640",
  "--height", "480",
  "--segment-seconds", "5",
  "--output", "docs/figures/demo_nav.mp4"
)

Write-Host "Running demo (headless)..." -ForegroundColor Yellow
& $pythonExe @argsList

if ($LASTEXITCODE -ne 0) {
  Write-Host "Demo failed." -ForegroundColor Red
  exit 1
}

if (Test-Path "docs/figures/demo_nav.mp4") {
  Write-Host "✅ Demo video saved to docs/figures/demo_nav.mp4" -ForegroundColor Green
} else {
  Write-Host "⚠️ Demo did not produce a video file" -ForegroundColor Yellow
}

Write-Host "Done." -ForegroundColor Green


