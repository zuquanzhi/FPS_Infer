cmake_minimum_required(VERSION 3.5)
project(fps_infer_node)

# 默认使用C99标准
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# 默认使用C++17标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找 ROS 2 和其他依赖库
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVINO REQUIRED)

# 包含头文件目录
include_directories(
  include
)

# 添加可执行文件
add_executable(fps_infer_node
  src/fps_infer_node.cpp
)

# 添加目标依赖
target_link_libraries(fps_infer_node
  ${OpenCV_LIBS}               # OpenCV 库
  Eigen3::Eigen                # Eigen 库
  ${InferenceEngine_LIBRARIES} # OpenVINO 库
)

# 添加 include 路径
target_include_directories(fps_infer_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}
)

# 安装可执行文件
install(TARGETS fps_infer_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装配置文件（例如模型、配置和启动文件）
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# 注册节点
ament_package()
