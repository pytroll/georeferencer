cmake_minimum_required(VERSION 3.15)
project(displacement_calc LANGUAGES CXX)

# Use pybind11 for building Python bindings
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Add the C++ source file
pybind11_add_module(displacement_calc displacement_calc.cpp)
install(TARGETS displacement_calc DESTINATION .)


cmake_minimum_required(VERSION 3.10)

# Project name
project(displacement_calc)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find pybind11 (you may need to update this if it's not installed system-wide)
find_package(pybind11 REQUIRED)

# Eigen (assuming you have the source already downloaded)
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen)

# Add your source files
add_library(displacement_calc MODULE src/cpp/displacement_calc.cpp)

# Link pybind11 and Eigen
target_link_libraries(displacement_calc PRIVATE pybind11::module)
target_include_directories(displacement_calc PRIVATE ${EIGEN3_INCLUDE_DIR})

# Install target
install(TARGETS displacement_calc DESTINATION ${CMAKE_SOURCE_DIR}/src/python/georeferencer)
