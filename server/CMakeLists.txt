cmake_minimum_required(VERSION 2.8)

project( server )

find_package( OpenCV REQUIRED )

find_library(LIBM3API_LIB NAMES libm3api.so)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

add_executable( server server.cpp xiApiPlusOcv.hpp xiApiPlusOcv.cpp)

target_link_libraries( server ${OpenCV_LIBS} ${LIBM3API_LIB})

set_property(TARGET server PROPERTY CXX_STANDARD 11)