# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8..3.22)

if (${CMAKE_VERSION} VERSION_LESS 3.22)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
	cmake_policy(VERSION 3.22)
endif()

project ("CMakeCUDA" VERSION 1.0 DESCRIPTION "First CMake CUDA Project" LANGUAGES CXX CUDA)

# Include sub-projects.
include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\include")
add_subdirectory ("CMakeCUDA")
