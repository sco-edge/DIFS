cmake_minimum_required(VERSION 3.7)

project(infaas-root)

# ============================================================
# Global build options
# ============================================================
option(ENABLE_TRTIS "Enable Triton/TRTIS support" OFF)
option(BUILD_ONLY_WORKER "Only build Worker" OFF)
option(BUILD_ONLY_MD "Only build Metadata Store" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Include paths (global)
# ============================================================
include_directories(BEFORE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    /usr/local/include
)

# Extra module search path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ============================================================
# Dependencies: Protobuf + gRPC
# ============================================================
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
if(NOT _GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "grpc_cpp_plugin not found.")
endif()

set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)

# ============================================================
# Disable TRTIS if OFF
# ============================================================
if(NOT ENABLE_TRTIS)
    add_definitions(-DDISABLE_TRTIS)
endif()

# ============================================================
# Add submodules
# ============================================================
if(LOCAL_MODE)
  message(STATUS "LOCAL_MODE: Skipping protos/ directory")
else()
  add_subdirectory(protos)
endif()
add_subdirectory(src)


if(NOT LOCAL_MODE)
  message(STATUS "Skipping metadata-store tests (LOCAL_MODE)")
  add_subdirectory(src/metadata-store)
endif()
