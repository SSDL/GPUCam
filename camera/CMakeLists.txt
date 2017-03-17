cmake_minimum_required (VERSION 2.8)

project(cameraDemo)

# required packages and directories (OpenCV + XIMEA API)
find_package(OpenCV REQUIRED)
find_library(LIBM3API_LIB NAMES libm3api.so)

include_directories(${OpenCV_INCLUDE_DIRS})

# problem with opencv + cuda on tegra, this setting required
# must explicitly build with -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF the first time
message(STATUS "CUDA_USE_STATIC_CUDA_RUNTIME    " ${CUDA_USE_STATIC_CUDA_RUNTIME})

# executables
add_executable(singleCameraDemo
               singleCameraDemo.cpp
               xiApiPlusOcv.hpp
               xiApiPlusOcv.cpp)

# link libraries
target_link_libraries(singleCameraDemo ${OpenCV_LIBS} ${LIBM3API_LIB})
