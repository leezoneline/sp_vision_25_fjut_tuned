cmake_minimum_required(VERSION 3.16)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)

# Find TensorRT (system installation via deb packages)
set(TensorRT_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include directory")
set(TensorRT_LIBRARIES "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT library directory")

if(NOT EXISTS ${TensorRT_INCLUDE_DIRS}/NvInfer.h)
    message(FATAL_ERROR "TensorRT headers not found at ${TensorRT_INCLUDE_DIRS}. Please install libnvinfer-dev package.")
endif()

include_directories(${TensorRT_INCLUDE_DIRS})
link_directories(${TensorRT_LIBRARIES})

find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
    yolo11_buff.cpp
)

target_link_libraries(auto_buff auto_aim nvinfer nvonnxparser ${CUDA_LIBRARIES} cudart ${CERES_LIBRARIES})