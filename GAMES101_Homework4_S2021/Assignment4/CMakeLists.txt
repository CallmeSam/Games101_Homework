cmake_minimum_required(VERSION 3.10)
project(BezierCurve)
set(OpenCV_DIR D:/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} D:/eigen-3.4.0)
link_directories(${OpenCV_LIBRARY_DIRS})
set(CMAKE_CXX_STANDARD 14)

add_executable(BezierCurve main.cpp)

target_link_libraries(BezierCurve ${OpenCV_LIBRARIES})
