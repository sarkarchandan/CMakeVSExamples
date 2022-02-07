# CMakeList.txt : CMake project for CMakeVS, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

if (NOT TARGET)
	set(TARGET myexe)
endif()

add_executable(${TARGET} src/main.cpp)

set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 11)

target_include_directories(${TARGET} PRIVATE include)


				
