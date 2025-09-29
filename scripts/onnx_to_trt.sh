#!/bin/bash
# ONNX to TensorRT conversion script for GTX 1650 optimization

set -e

# Default parameters
ONNX_FILE=""
OUTPUT_DIR="models/exported"
IMG_SIZE=448
FP16=true
WORKSPACE_SIZE=1024  # 1GB for GTX 1650
BATCH_SIZE=1

# Help function
show_help() {
    echo "Usage: $0 <onnx_file> [options]"
    echo ""
    echo "Convert ONNX model to TensorRT engine optimized for GTX 1650"
    echo ""
    echo "Arguments:"
    echo "  onnx_file              Path to ONNX model file"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR       Output directory (default: models/exported)"
    echo "  --img-size SIZE        Image size (default: 448)"
    echo "  --fp16                 Use FP16 precision (default: true)"
    echo "  --workspace SIZE       Workspace size in MB (default: 1024)"
    echo "  --batch-size SIZE      Batch size (default: 1)"
    echo "  --help                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 models/gridformer.onnx"
    echo "  $0 models/yolo.onnx --img-size 640 --fp16"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --img-size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --fp16)
            FP16=true
            shift
            ;;
        --no-fp16)
            FP16=false
            shift
            ;;
        --workspace)
            WORKSPACE_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$ONNX_FILE" ]; then
                ONNX_FILE="$1"
            else
                echo "Multiple input files not supported"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input file is provided
if [ -z "$ONNX_FILE" ]; then
    echo "Error: ONNX file path is required"
    show_help
    exit 1
fi

# Check if input file exists
if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: ONNX file not found: $ONNX_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename
BASENAME=$(basename "$ONNX_FILE" .onnx)
if [ "$FP16" = true ]; then
    TRT_FILE="${OUTPUT_DIR}/${BASENAME}_${IMG_SIZE}_fp16.trt"
else
    TRT_FILE="${OUTPUT_DIR}/${BASENAME}_${IMG_SIZE}_fp32.trt"
fi

echo "🔄 Converting ONNX to TensorRT..."
echo "   Input: $ONNX_FILE"
echo "   Output: $TRT_FILE"
echo "   Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "   Precision: $([ "$FP16" = true ] && echo "FP16" || echo "FP32")"
echo "   Workspace: ${WORKSPACE_SIZE}MB"
echo "   Batch size: $BATCH_SIZE"

# Build TensorRT command
TRTEXEC_CMD="trtexec \
    --onnx=\"$ONNX_FILE\" \
    --saveEngine=\"$TRT_FILE\" \
    --workspace=${WORKSPACE_SIZE} \
    --minShapes=input:${BATCH_SIZE}x3x${IMG_SIZE}x${IMG_SIZE} \
    --optShapes=input:${BATCH_SIZE}x3x${IMG_SIZE}x${IMG_SIZE} \
    --maxShapes=input:${BATCH_SIZE}x3x${IMG_SIZE}x${IMG_SIZE} \
    --verbose \
    --buildOnly"

# Add FP16 flag if enabled
if [ "$FP16" = true ]; then
    TRTEXEC_CMD="$TRTEXEC_CMD --fp16"
fi

echo ""
echo "🚀 Running TensorRT conversion..."
echo "Command: $TRTEXEC_CMD"
echo ""

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "❌ Error: trtexec not found"
    echo "   Please install TensorRT and ensure trtexec is in PATH"
    echo "   Alternative: Use ONNX Runtime with TensorRT provider"
    exit 1
fi

# Run conversion
if eval $TRTEXEC_CMD; then
    echo ""
    echo "✅ Conversion successful!"
    
    # Check output file
    if [ -f "$TRT_FILE" ]; then
        FILE_SIZE=$(ls -lh "$TRT_FILE" | awk '{print $5}')
        FILE_SIZE_MB=$(du -m "$TRT_FILE" | cut -f1)
        
        echo "📄 Output file: $TRT_FILE"
        echo "📊 File size: $FILE_SIZE (${FILE_SIZE_MB}MB)"
        
        # Check GTX 1650 compatibility
        if [ "$FILE_SIZE_MB" -gt 1200 ]; then
            echo "⚠️  Warning: File size (${FILE_SIZE_MB}MB) may exceed GTX 1650 VRAM limits"
            echo "   Consider using smaller image size or FP16 precision"
        else
            echo "✅ File size compatible with GTX 1650 (4GB VRAM)"
        fi
        
        # Test engine (optional)
        echo ""
        echo "🧪 Testing TensorRT engine..."
        TEST_CMD="trtexec --loadEngine=\"$TRT_FILE\" --warmUp=100 --duration=10 --iterations=100"
        
        if eval $TEST_CMD; then
            echo "✅ Engine test passed!"
        else
            echo "⚠️  Engine test failed (engine may still be usable)"
        fi
        
    else
        echo "❌ Error: Output file not created"
        exit 1
    fi
else
    echo ""
    echo "❌ Conversion failed!"
    echo "   Check TensorRT installation and ONNX model compatibility"
    exit 1
fi

echo ""
echo "🎯 GTX 1650 Optimization Notes:"
echo "   - Use FP16 precision for better performance"
echo "   - Keep engine size < 1.2GB"
echo "   - Monitor VRAM usage during inference"
echo "   - Consider batch size = 1 for memory efficiency" 