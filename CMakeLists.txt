# CMake version
cmake_minimum_required(VERSION 3.0)

set(myproject "micrograd")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build")
# Project name
project(${myproject})

# Specify the required C++ standard
if(CMAKE_VERSION GREATER "3.1.2") 
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
else()
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    endif()    
endif()    

add_subdirectory(src)

add_executable(${PROJECT_NAME} main.cpp)      
target_link_libraries(${PROJECT_NAME} module_lib neuron_lib value_lib)