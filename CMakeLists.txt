project(JEDI)

cmake_minimum_required (VERSION 3.2)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

find_package( OpenMP )
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_C_FLAGS       "${CMAKE_C_FLAGS} -DOPENMP")
endif()


set (CMAKE_C_FLAGS         "${CMAKE_C_FLAGS} -Wall -g -Ofast -march=native ")
set (CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -g -Ofast -march=native ")

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (/usr/local/cuda/include/ /usr/local/include/tkDNN/)

add_subdirectory (src)
