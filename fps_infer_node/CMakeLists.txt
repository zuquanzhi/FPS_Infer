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

# 引入 ament_cmake
find_package(ament_cmake REQUIRED)

# 设置 OpenVINO 和 nGraph 的路径
set(InferenceEngine_DIR "/opt/intel/openvino_2023/runtime/cmake")
set(ngraph_DIR "/opt/intel/openvino_2023/runtime/cmake")

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(rclcpp REQUIRED)

# 包含头文件目录
include_directories(
  include
  ${CMAKE_SOURCE_DIR}/../config
)

# 添加可执行文件
add_executable(fps_infer_node
  src/fps_infer_node.cpp
)

# 添加目标依赖
ament_target_dependencies(
  fps_infer_node
  rclcpp          # 添加 rclcpp 作为依赖
  OpenCV
  Eigen3
)

target_link_libraries(fps_infer_node
  ${InferenceEngine_LIBRARIES}
  ${NGRAPH_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

# 安装可执行文件
install(TARGETS fps_infer_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装配置文件
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
