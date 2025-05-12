cmake_minimum_required(VERSION 3.10)

project(displacement_calc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-O3 -Wall -shared -std=c++17 -fPIC -fopenmp -march=native -flto -ftree-vectorize")

if(NOT DEFINED PYBIND11_DIR)
    message(FATAL_ERROR "PYBIND11_DIR is not defined. Set the path to pybind11Config.cmake.")
endif()

find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR} NO_DEFAULT_PATH)

set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen)

add_library(displacement_calc MODULE src/cpp/displacement_calc.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(displacement_calc PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(displacement_calc PRIVATE pybind11::module)
target_include_directories(displacement_calc PRIVATE ${EIGEN3_INCLUDE_DIR})

set_target_properties(displacement_calc PROPERTIES PREFIX "")

add_custom_command(TARGET displacement_calc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:displacement_calc>
        ${CMAKE_SOURCE_DIR}/src/python/georeferencer/$<TARGET_FILE_NAME:displacement_calc>
    COMMENT "Copying displacement_calc.so to src/python/georeferencer/"
)

install(TARGETS displacement_calc
        LIBRARY DESTINATION georeferencer)
