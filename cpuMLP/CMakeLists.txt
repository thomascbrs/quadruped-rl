cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp-mlp CXX)

set(CMAKE_PREFIX_PATH libtorch ${CMAKE_PREFIX_PATH})

add_subdirectory(pybind11 pybind11)
include_directories("eigen3" ${TORCH_INCLUDE_DIRS})

pybind11_add_module(policy cpuMLP.cpp cpuMLP.hpp)
pybind11_add_module(stateEst cpuMLP.cpp cpuMLP.hpp)
pybind11_add_module(stateEst3 cpuMLP.cpp cpuMLP.hpp)

add_executable(cpp-mlp main.cpp)
set_property(TARGET cpp-mlp PROPERTY CXX_STANDARD 14)
set_property(TARGET policy PROPERTY CXX_STANDARD 14)
set_property(TARGET stateEst PROPERTY CXX_STANDARD 14)
set_property(TARGET stateEst3 PROPERTY CXX_STANDARD 14)
