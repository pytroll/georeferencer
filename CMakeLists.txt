cmake_minimum_required(VERSION 3.10)

project(displacement_calc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native -flto -ftree-vectorize")
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-fPIC -fopenmp")

set(PYBIND11_FINDPYTHON ON)

if(NOT DEFINED PYBIND11_DIR)
    message(FATAL_ERROR "PYBIND11_DIR is not defined. Set the path to pybind11Config.cmake.")
endif()
find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR} NO_DEFAULT_PATH)

find_package(OpenMP REQUIRED)

set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen)

add_library(displacement_calc MODULE src/cpp/displacement_calc.cpp)

target_link_libraries(displacement_calc PRIVATE pybind11::module OpenMP::OpenMP_CXX)

target_include_directories(displacement_calc PRIVATE ${EIGEN3_INCLUDE_DIR})

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(OUTPUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/georeferencer")
else()
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/python/georeferencer")
endif()

set_target_properties(displacement_calc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    PREFIX ""
)

# Install step (important for packaging)
install(TARGETS displacement_calc LIBRARY DESTINATION georeferencer)

