cmake_minimum_required(VERSION 3.0 )
project(ORB_Study)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)


find_package(OpenCV REQUIRED)
message("OpenCV VIRSION = ${OpenCV_VERSION}")
message("OpenCV INCLUDE = ${OpenCV_INCLUDE_DIRS}")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(fmt REQUIRED)



include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_executable(${PROJECT_NAME}
        src/System.cc
        src/ORBextractor.cc

        include/System.h
        include/ORBextractor.h
)

add_subdirectory(Thirdparty/g2o)

set(LIBRARIES

${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
boost_serialization
crypto
fmt::fmt
glog gflags
pthread
)

target_link_libraries(${PROJECT_NAME}
${LIBRARIES}

)

