cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

# 创建目标 io
add_library(io STATIC 
    hikrobot/hikrobot.cpp    
    mindvision/mindvision.cpp  
    usbcamera/usbcamera.cpp  
  daheng/daheng.cpp
    camera.cpp
    cboard.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
)

# 根据架构设置库路径后缀
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(HIKROBOT_LIB_ARCH "amd64")
  set(MINDVISION_LIB_ARCH "amd64")
  set(DAHENG_LIB_ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(HIKROBOT_LIB_ARCH "arm64")
  set(MINDVISION_LIB_ARCH "arm64")
  set(DAHENG_LIB_ARCH "arm64")  # 注意：Daheng可能没有arm64库
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}!")
endif()

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
target_link_directories(io PUBLIC hikrobot/lib/${HIKROBOT_LIB_ARCH})

# mindvision
target_include_directories(io PUBLIC mindvision/include)
target_link_directories(io PUBLIC mindvision/lib/${MINDVISION_LIB_ARCH})

# daheng
target_include_directories(io PUBLIC daheng/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC daheng/lib/${DAHENG_LIB_ARCH})
else()
  message(WARNING "Daheng SDK libs only provided for x86_64 in repo.")
endif()

target_link_libraries(io MvCameraControl MVSDK gxiapi log4cplus_gx usb-1.0 yaml-cpp serial spdlog::spdlog fmt::fmt)

# 设置 RPATH，使可执行文件能找到相对路径下的动态库
# BUILD_RPATH: 用于构建目录中的可执行文件
# INSTALL_RPATH: 用于安装后的可执行文件
set_target_properties(io PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/lib/${HIKROBOT_LIB_ARCH}:${CMAKE_CURRENT_SOURCE_DIR}/mindvision/lib/${MINDVISION_LIB_ARCH}:${CMAKE_CURRENT_SOURCE_DIR}/daheng/lib/${DAHENG_LIB_ARCH}"
)

# 为链接此库的可执行文件设置 RPATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

target_include_directories(io BEFORE PRIVATE
  $<TARGET_PROPERTY:fmt::fmt,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)

################## 检查 ROS 环境 ##################
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

# 如果找不到 ROS 环境，则跳过 ROS 相关的部分
if(NOT ament_cmake_FOUND OR NOT rclcpp_FOUND OR NOT std_msgs_FOUND OR NOT rosidl_typesupport_cpp_FOUND OR NOT sp_msgs_FOUND)
    message(WARNING "ROS2 not found, skipping ROS2 specific code.")
else()
    message(STATUS "ROS2 environment found, compiling ROS2-related code.")
    
    # 添加 ROS2 相关的文件
    target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)

    # 设置 ROS2 依赖
    include_directories(${std_msgs_INCLUDE_DIRS})
    ament_target_dependencies(io rclcpp std_msgs sp_msgs)
endif()
