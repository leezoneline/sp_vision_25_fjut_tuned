cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(omniperception OBJECT 
    decider.cpp
    perceptron.cpp
)
target_link_libraries(omniperception openvino::runtime spdlog::spdlog fmt::fmt)
target_include_directories(omniperception BEFORE PRIVATE
    $<TARGET_PROPERTY:fmt::fmt,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)