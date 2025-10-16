# ARM64 架构适配说明

## 修改概述

本次修改为项目添加了对 ARM64 架构（如 Jetson Orin NX）的完整支持，特别是大恒相机库的适配。

## 修改内容

### 1. io/CMakeLists.txt

#### 修改点 1: 架构检测和库路径设置
```cmake
# 原来
set(DAHENG_LIB_ARCH "arm64")  # 注意：Daheng可能没有arm64库

# 修改后
set(DAHENG_LIB_ARCH "armv8")  # ARM64架构使用armv8库
```

#### 修改点 2: 库目录链接
```cmake
# 原来
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC daheng/lib/${DAHENG_LIB_ARCH})
else()
  message(WARNING "Daheng SDK libs only provided for x86_64 in repo.")
endif()

# 修改后
target_link_directories(io PUBLIC daheng/lib/${DAHENG_LIB_ARCH})
```

#### 修改点 3: RPATH 设置
```cmake
# 原来
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CAMERA_SDK_RPATH "$ORIGIN/../io/hikrobot/lib/arm64:$ORIGIN/../io/mindvision/lib/arm64")

# 修改后
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CAMERA_SDK_RPATH "$ORIGIN/../io/hikrobot/lib/arm64:$ORIGIN/../io/mindvision/lib/arm64:$ORIGIN/../io/daheng/lib/armv8")
```

### 2. CMakeLists.txt (根目录)

#### 修改点: RPATH 设置
```cmake
# 原来
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_BUILD_RPATH "$ORIGIN/../io/hikrobot/lib/arm64:$ORIGIN/../io/mindvision/lib/arm64")
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../io/hikrobot/lib/arm64:$ORIGIN/../io/mindvision/lib/arm64")

# 修改后
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_BUILD_RPATH "$ORIGIN/../io/hikrobot/lib/arm64:$ORIGIN/../io/mindvision/lib/arm64:$ORIGIN/../io/daheng/lib/armv8")
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../io/hikrobot/lib/arm64:$ORIGIN/../io/mindvision/lib/arm64:$ORIGIN/../io/daheng/lib/armv8")
```

## 库文件结构

项目现在支持以下相机库的 ARM64 版本：

```
io/
├── hikrobot/
│   └── lib/
│       ├── amd64/      # x86_64 架构
│       └── arm64/      # ARM64 架构
├── mindvision/
│   └── lib/
│       ├── amd64/      # x86_64 架构
│       └── arm64/      # ARM64 架构
└── daheng/
    └── lib/
        ├── x86_64/     # x86_64 架构
        └── armv8/      # ARM64 架构 (新增)
            ├── libgxiapi.so
            ├── liblog4cplus_gx.so
            ├── GxGVTL.cti
            └── GxU3VTL.cti
```

## 编译说明

### 在 ARM64 设备上编译

1. 确保已安装所有依赖
2. 清理之前的编译文件（如果有）:
   ```bash
   rm -rf build/
   ```

3. 重新编译:
   ```bash
   cmake -B build
   make -C build/ -j$(nproc)
   ```

4. CMake 会自动检测架构并使用正确的库路径

### 验证架构检测

编译时会看到类似输出：
```
-- The C compiler identification is GNU ...
-- The CXX compiler identification is GNU ...
-- Detected system processor: aarch64
```

## 注意事项

1. **OpenVINO 推理引擎**: 在 Jetson 设备上，建议考虑使用 TensorRT 替代 OpenVINO 以获得更好的性能

2. **CUDA 支持**: 确保 OpenCV 编译时包含 CUDA 支持以充分利用 GPU 加速

3. **运行时库路径**: 使用 `$ORIGIN` 相对路径，确保程序可以在任何位置运行而无需修改 LD_LIBRARY_PATH

4. **相机权限**: 在 Linux 上可能需要配置 udev 规则以获得 USB 相机访问权限

## 测试清单

- [ ] 编译成功无错误
- [ ] 相机驱动正常初始化
- [ ] YOLO 模型推理正常
- [ ] 实时性能满足要求（目标 ≥60 FPS）
- [ ] 串口通信正常

## 性能优化建议

1. 使用 NVIDIA JetPack SDK
2. 启用 GPU 加速推理（TensorRT）
3. 优化相机参数（分辨率、帧率、曝光）
4. 考虑使用多线程并行处理

## 相关资源

- [大恒相机 Linux SDK 下载](https://gb.daheng-imaging.com/CN/Software/Cameras/Linux/)
- [OpenVINO ARM 支持文档](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- [NVIDIA TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
