cmake_minimum_required (VERSION 2.8)

project(satDetect)

# required package OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# problem with opencv + cuda on tegra, this setting required
# must explicitly build with -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF the first time
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# executables
add_executable(findCubesat findCubesat.cpp)
add_executable(distortionCorrection distortionCorrection.cpp)

# link libraries
target_link_libraries(findCubesat ${OpenCV_LIBS})
target_link_libraries(distortionCorrection ${OpenCV_LIBS})
