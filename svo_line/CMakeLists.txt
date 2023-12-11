project(svo_direct)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(catkin_simple REQUIRED)

find_package(OpenCV 4.1 REQUIRED  
 core imgproc ximgproc line_descriptor)

catkin_simple(ALL_DEPS_REQUIRED)



include(SvoSetup)

set(HEADERS
  
)

set(SOURCES
  src/patch_warp.cpp
  
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE -g)



cs_add_executable(test_lsd_derection test/test_lsd_derection.cpp)
target_link_libraries(test_lsd_derection ${PROJECT_NAME})



################################################################################
cs_install()
cs_export()

