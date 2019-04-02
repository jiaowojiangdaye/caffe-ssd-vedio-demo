cmake_minimum_required(VERSION 3.13)
project(test)
set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV 3 REQUIRED )
#find_package( Boost 1.58 REQUIRED )
#if (NOT Boost_FOUND)
#        message(FATAL_ERROR "没找到boost")
#endif ()

set(Caffe_Root /home/aaron/mydisk/envs/caffe_lib/caffe)
set(Caffe_INCLUDE_DIRS
        ${Caffe_Root}/include
        ${Caffe_Root}/src
        /usr/local/cuda/include)

set(Caffe_LIBRARIES
        caffe
        glog
        protobuf)

set(TOOLS_LIBRARIES
        boost_system
        boost_filesystem
        gflags)

include_directories(
        ${Caffe_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        /usr/include
        /usr/local/include
#        ${Boost_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
        ${Caffe_Root}/build/lib
#        ${Boost_LIBRARY_DIRS}
        )

#add_library(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR})

AUX_SOURCE_DIRECTORY(. SRC_LIST)
ADD_EXECUTABLE(run main.cpp ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(run
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Caffe_LIBRARIES}
        ${TOOLS_LIBRARIES}
        )