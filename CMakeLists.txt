cmake_minimum_required(VERSION 3.5)
project(prune_tiny_yolo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(CUDA)



include_directories("./include" "./src")
file(GLOB darknet_src ./src/*.c ./src/*.cu)
add_library(darknet ${darknet_src} )
target_link_libraries(darknet pthread m)



file(GLOB example_src ./examples/*.c)


add_executable(object_detection ${example_src})
target_link_libraries(object_detection darknet)



