cmake_minimum_required (VERSION 2.8)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set (CMAKE_C_FLAGS         "${CMAKE_C_FLAGS} -Wall -g -Ofast -march=native ")
set (CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -g -Ofast -march=native ")

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (/usr/local/cuda/include/)

add_subdirectory (src)
