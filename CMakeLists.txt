cmake_minimum_required(VERSION 2.8.4)
project(Mapper)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -DDEBUG -g -msse2 -msse3 -msse4 -ffast-math -std=c++11")
add_definitions(-DCOMPILEDWITHC11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(LIBS
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
)

set(SOURCE_FILES
        src/inliers.cpp
        src/inliers.h
        )


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(Mapper
        ${SOURCE_FILES}
        )

target_link_libraries(Mapper
${LIBS}
)

