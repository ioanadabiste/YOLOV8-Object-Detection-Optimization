cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(yolov8_libtorch_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# -------------- OpenCV --------------
set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- libtorch --------------
list(APPEND CMAKE_PREFIX_PATH "D:/libtorch")
set(Torch_DIR "D:/libtorch/share/cmake/Torch")

# !!! ADD THIS LINE HERE !!!
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
# -------------- Fix for CUDA 12+ on Windows --------------
find_package(CUDAToolkit REQUIRED)

# If CUDA::nvToolsExt is missing (removed in CUDA 12), define it manually
if(NOT TARGET CUDA::nvToolsExt)
    message(STATUS "CUDA 12 detected: Manually defining CUDA::nvToolsExt")
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    target_include_directories(CUDA::nvToolsExt INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
endif()
# ---------------------------------------------------------
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("${TORCH_LIBRARIES}")
message("${TORCH_INCLUDE_DIRS}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET yolov8_libtorch_example
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:yolov8_libtorch_example>)
# endif (MSVC)

include_directories(${TORCH_INCLUDE_DIRS})

add_executable(yolov8_libtorch_inference "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(yolov8_libtorch_inference ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET yolov8_libtorch_inference PROPERTY CXX_STANDARD 17)