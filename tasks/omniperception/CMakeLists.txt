cmake_minimum_required(VERSION 3.16)

add_library(omniperception OBJECT 
    decider.cpp
    perceptron.cpp
)
target_link_libraries(omniperception spdlog::spdlog fmt::fmt)
target_include_directories(omniperception BEFORE PRIVATE
    $<TARGET_PROPERTY:fmt::fmt,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)