project(svo_line)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(catkin_simple REQUIRED)
find_package(yaml-cpp REQUIRED)

set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV 4.1 REQUIRED  
 core imgproc ximgproc line_descriptor)


# set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
# find_package(OpenCV REQUIRED core imgproc highgui )


catkin_simple(ALL_DEPS_REQUIRED)



include(SvoSetup)

set(HEADERS
include/svo/line/auxiliar.h
include/svo/line/bitarray_custom.hpp
include/svo/line/bitops_custom.hpp
include/svo/line/config.h
include/svo/line/descriptor_custom.hpp
include/svo/line/gridStructure.h
include/svo/line/line_descriptor_custom.hpp
include/svo/line/line_match.h
include/svo/line/matching.h
include/svo/line/precomp_custom.hpp
include/svo/line/stereoFeatures.h
include/svo/line/types_custom.hpp
)

set(SOURCES
src/binary_descriptor_custom.cpp
src/binary_descriptor_matcher.cpp
src/config.cpp
src/gridStructure.cpp
src/line_match.cpp
src/LSDDetector_custom.cpp
src/matching.cpp
src/stereoFeatures.cpp

  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} )
target_compile_options(${PROJECT_NAME} PRIVATE -g)



# cs_add_executable(test_lsd_derection test/test_lsd_derection.cpp)
# target_link_libraries(test_lsd_derection ${PROJECT_NAME})



################################################################################
cs_install()
cs_export()

