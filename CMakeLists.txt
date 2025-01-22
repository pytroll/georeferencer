cmake_minimum_required(VERSION 3.10)
# Project name
project(displacement_calc)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Check if PYBIND11_DIR is set, and use it if found
if(NOT DEFINED PYBIND11_DIR)
    message(FATAL_ERROR "PYBIND11_DIR is not defined. Set the path to pybind11Config.cmake.")
endif()

# Find pybind11 using the PYBIND11_DIR path
find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR} NO_DEFAULT_PATH)

# Eigen (assuming you have the source already downloaded)
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen)

# Add your source files
add_library(displacement_calc MODULE src/cpp/displacement_calc.cpp)

# Link pybind11 and Eigen
target_link_libraries(displacement_calc PRIVATE pybind11::module)
target_include_directories(displacement_calc PRIVATE ${EIGEN3_INCLUDE_DIR})

# Install target
install(TARGETS displacement_calc DESTINATION ${CMAKE_SOURCE_DIR}/src/python/georeferencer)
