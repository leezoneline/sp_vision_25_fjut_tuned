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

# Include fmt headers
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    trt_infer.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    # multithread/mt_detector.cpp  # TODO: Convert to TensorRT
    planner/planner.cpp
)

target_link_libraries(auto_aim io nvinfer nvonnxparser ${CUDA_LIBRARIES} cudart tinympcstatic fmt::fmt)