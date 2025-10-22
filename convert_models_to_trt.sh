#!/bin/bash
# TensorRT 模型转换脚本
# 将 ONNX 模型转换为 TensorRT 引擎文件

set -e  # 遇到错误立即退出

TRTEXEC=/usr/src/tensorrt/bin/trtexec
ASSETS_DIR="$(cd "$(dirname "$0")/assets" && pwd)"

echo "=========================================="
echo "TensorRT Model Conversion Script"
echo "=========================================="
echo "Assets directory: $ASSETS_DIR"
echo "TensorRT version: $($TRTEXEC --help | head -1)"
echo ""

# 检查 ONNX 文件是否存在
check_onnx() {
    if [ ! -f "$1" ]; then
        echo "⚠️  Warning: $1 not found, skipping..."
        return 1
    fi
    return 0
}

# 转换函数
convert_model() {
    local onnx_file=$1
    local precision=$2  # fp32, fp16, int8
    local engine_file="${onnx_file%.onnx}_${precision}.engine"
    
    if [ ! -f "$onnx_file" ]; then
        echo "⚠️  Skipping $onnx_file (not found)"
        return
    fi
    
    echo "🔧 Converting: $(basename $onnx_file) -> $(basename $engine_file)"
    echo "   Precision: $precision"
    
    local extra_args=""
    case $precision in
        fp16)
            extra_args="--fp16"
            ;;
        int8)
            extra_args="--int8 --best"
            echo "   ⚠️  INT8 requires calibration data for best results"
            ;;
        fp32)
            extra_args=""
            ;;
    esac
    
    $TRTEXEC \
        --onnx="$onnx_file" \
        --saveEngine="$engine_file" \
        $extra_args \
        --verbose 2>&1 | grep -E "(Input|Output|Layer|Performance summary|Throughput|GPU Compute Time|Serialized Engine|PASSED|FAILED)"
    
    if [ -f "$engine_file" ]; then
        local size=$(du -h "$engine_file" | cut -f1)
        echo "   ✅ Success! Engine size: $size"
        echo ""
    else
        echo "   ❌ Failed to create engine file"
        echo ""
    fi
}

cd "$ASSETS_DIR"

# 让用户选择精度
echo "请选择转换精度："
echo "  1) FP32 (最高精度，最慢，最大文件)"
echo "  2) FP16 (推荐 - 平衡精度和速度)"
echo "  3) INT8 (最快，最小文件，精度稍降)"
echo "  4) 全部转换 (FP16 + INT8)"
read -p "请输入选项 [1-4] (默认: 2): " choice
choice=${choice:-2}

case $choice in
    1) PRECISION="fp32" ;;
    2) PRECISION="fp16" ;;
    3) PRECISION="int8" ;;
    4) PRECISION="all" ;;
    *) echo "无效选项，使用 FP16"; PRECISION="fp16" ;;
esac

echo ""
echo "开始转换模型..."
echo "=========================================="

# 自瞄相关模型
if [ "$PRECISION" = "all" ]; then
    for prec in fp16 int8; do
        echo ">>> Converting with $prec precision"
        check_onnx "yolo11.onnx" && convert_model "$ASSETS_DIR/yolo11.onnx" $prec
        check_onnx "yolov8.onnx" && convert_model "$ASSETS_DIR/yolov8.onnx" $prec
        check_onnx "yolov5.onnx" && convert_model "$ASSETS_DIR/yolov5.onnx" $prec
        check_onnx "tiny_resnet.onnx" && convert_model "$ASSETS_DIR/tiny_resnet.onnx" $prec
    done
else
    check_onnx "yolo11.onnx" && convert_model "$ASSETS_DIR/yolo11.onnx" $PRECISION
    check_onnx "yolov8.onnx" && convert_model "$ASSETS_DIR/yolov8.onnx" $PRECISION
    check_onnx "yolov5.onnx" && convert_model "$ASSETS_DIR/yolov5.onnx" $PRECISION
    check_onnx "tiny_resnet.onnx" && convert_model "$ASSETS_DIR/tiny_resnet.onnx" $PRECISION
fi

# Buff 能量机关模型
if check_onnx "best2-sim.onnx"; then
    if [ "$PRECISION" = "all" ]; then
        for prec in fp16 int8; do
            convert_model "$ASSETS_DIR/best2-sim.onnx" $prec
        done
    else
        convert_model "$ASSETS_DIR/best2-sim.onnx" $PRECISION
    fi
fi

echo "=========================================="
echo "✅ 模型转换完成！"
echo ""
echo "生成的引擎文件："
ls -lh *.engine 2>/dev/null || echo "没有找到 .engine 文件"
echo ""
echo "下一步："
echo "  1. 更新配置文件 (configs/*.yaml) 中的模型路径"
echo "     例如: yolo11_model_path: ../assets/yolo11_fp16.engine"
echo "  2. 测试运行: ./build/standard"
echo "=========================================="
