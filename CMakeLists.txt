cmake_minimum_required(VERSION 3.10)

project(displacement_calc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native -fPIC -fopenmp")

if(NOT DEFINED PYBIND11_DIR)
    message(FATAL_ERROR "PYBIND11_DIR is not defined. Set the path to pybind11Config.cmake.")
endif()

# Find pybind11 using the PYBIND11_DIR path
find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR} NO_DEFAULT_PATH)

set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen)

# Add your source files
add_library(displacement_calc MODULE src/cpp/displacement_calc.cpp)

# Link pybind11 and Eigen
target_link_libraries(displacement_calc PRIVATE pybind11::module)
target_include_directories(displacement_calc PRIVATE ${EIGEN3_INCLUDE_DIR})

# Set the output directory for the shared library
set_target_properties(displacement_calc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/georeferencer
    PREFIX ""
)

install(TARGETS displacement_calc DESTINATION ${CMAKE_SOURCE_DIR}/src/python/georeferencer)
