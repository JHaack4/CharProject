cmake_minimum_required(VERSION 3.10)
project (CharProject)

file(GLOB SOURCES "src/*.cpp")
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -flto -fopenmp -march=native -mtune=native")
add_executable(run_image_proc ${SOURCES} src/run_image_proc.cpp)
target_link_libraries(run_image_proc opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
