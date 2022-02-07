# CMakeList.txt : CMake project for CMakeCUDA, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8..3.22)

if (${CMAKE_VERSION} VERSION_LESS 3.22)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
	cmake_policy(VERSION 3.22)
endif()

if (NOT TARGET)
	set(TARGET vectoradd)
endif()

add_executable(${TARGET} src/vecadd.cu src/main.cpp)

if(NOT DEFINED CMAKE_CXX_STANDARD)
	set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

if (NOT DEFINED CMAKE_CUDA_STANDARD)
	set_target_properties(${TARGET} PROPERTIES CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()

# Enable separable compilation for CUDA
# More on separable compilation: https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/
set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_include_directories(${TARGET} PRIVATE include)
