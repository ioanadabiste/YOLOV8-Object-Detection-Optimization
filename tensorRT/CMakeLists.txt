cmake_minimum_required(VERSION 3.10)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(CUDAToolkit REQUIRED)

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

add_executable(jetson_app main.cpp)

target_link_libraries(jetson_app 
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer       # Biblioteca principala TensorRT
    nvonnxparser  # Parserul ONNX (daca il vei folosi pe viitor)
)

message(STATUS "Build - Jetson Orin Nano.")
