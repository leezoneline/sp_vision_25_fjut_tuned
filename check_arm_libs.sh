#!/bin/bash
# 检查 ARM64 架构下的库文件配置

echo "=== 系统架构检查 ==="
echo "Architecture: $(uname -m)"
echo ""

echo "=== 大恒相机库文件检查 ==="
DAHENG_LIB_PATH="./io/daheng/lib"

if [ "$(uname -m)" = "aarch64" ]; then
    LIB_DIR="${DAHENG_LIB_PATH}/armv8"
elif [ "$(uname -m)" = "x86_64" ]; then
    LIB_DIR="${DAHENG_LIB_PATH}/x86_64"
else
    echo "不支持的架构: $(uname -m)"
    exit 1
fi

echo "库目录: ${LIB_DIR}"
echo ""

if [ -d "${LIB_DIR}" ]; then
    echo "✓ 库目录存在"
    echo ""
    echo "库文件列表:"
    ls -lh "${LIB_DIR}"/*.so 2>/dev/null || echo "✗ 未找到 .so 文件"
    echo ""
    
    # 检查关键库文件
    if [ -f "${LIB_DIR}/libgxiapi.so" ]; then
        echo "✓ libgxiapi.so 存在"
        file "${LIB_DIR}/libgxiapi.so"
    else
        echo "✗ libgxiapi.so 不存在"
    fi
    
    if [ -f "${LIB_DIR}/liblog4cplus_gx.so" ]; then
        echo "✓ liblog4cplus_gx.so 存在"
        file "${LIB_DIR}/liblog4cplus_gx.so"
    else
        echo "✗ liblog4cplus_gx.so 不存在"
    fi
else
    echo "✗ 库目录不存在: ${LIB_DIR}"
    exit 1
fi

echo ""
echo "=== 其他相机库检查 ==="

# 检查海康库
if [ "$(uname -m)" = "aarch64" ]; then
    HIKROBOT_DIR="./io/hikrobot/lib/arm64"
    MINDVISION_DIR="./io/mindvision/lib/arm64"
else
    HIKROBOT_DIR="./io/hikrobot/lib/amd64"
    MINDVISION_DIR="./io/mindvision/lib/amd64"
fi

echo "海康库目录: ${HIKROBOT_DIR}"
[ -d "${HIKROBOT_DIR}" ] && echo "✓ 存在" || echo "✗ 不存在"

echo "迈德威视库目录: ${MINDVISION_DIR}"
[ -d "${MINDVISION_DIR}" ] && echo "✓ 存在" || echo "✗ 不存在"

echo ""
echo "=== 编译建议 ==="
echo "1. 清理旧的编译文件: rm -rf build/"
echo "2. 重新配置: cmake -B build"
echo "3. 编译: make -C build/ -j\$(nproc)"
echo "4. 检查链接: ldd build/standard | grep -E 'gxiapi|log4cplus'"
