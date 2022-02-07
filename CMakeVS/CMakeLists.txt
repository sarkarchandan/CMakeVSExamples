# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("CMakeVS" VERSION 1.0 DESCRIPTION "First VS CMake project" LANGUAGES CXX)

set(TARGET myexe)

# Include sub-projects.
add_subdirectory ("CMakeVS")
