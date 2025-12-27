project(infaas-worker)

find_package(Protobuf REQUIRED)
# find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gRPC REQUIRED)

# AWS ALWAYS OFF in offline mode
add_compile_definitions(AWS_SDK_DISABLED)
option(ENABLE_AWS_AUTOSCALING "Enable AWS-based autoscaling" OFF)

if(ENABLE_AWS_AUTOSCALING)
    find_package(AWSSDK REQUIRED COMPONENTS s3)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # PNB: (2025.12.27)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(PROTOS_PATH "${CMAKE_CURRENT_BINARY_DIR}/../../protos")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# GPU detection
set(_has_nvidia OFF)
if (EXISTS "/proc/driver/nvidia/version")
    set(_has_nvidia ON)
    add_definitions(-DINFAAS_GPU_WORKER)
    find_package(CUDA REQUIRED)
endif()

# Inferentia detection
set(_has_inferentia OFF)
if (EXISTS "/run/infa/")
    set(_has_inferentia ON)
    add_definitions(-DINFAAS_NEURON_WORKER)
endif()


# link_directories(/usr/local/lib /usr/local/lib64)

# ------------------------------------------------------------
# Worker shared library
# ------------------------------------------------------------
add_library(inf-worker SHARED query_client.cc)

target_link_libraries(inf-worker
    redis-md
    infaas-protos-internal
    grpc++_unsecure
    grpc
    gpr
    protobuf::libprotobuf # ${PROTOBUF_LIBRARY} # PNB: (2025.12.27)
)


# ------------------------------------------------------------
# TRTIS (disabled by default)
# ------------------------------------------------------------
option(ENABLE_TRTIS "Enable Triton/TRTIS support" OFF)

if(ENABLE_TRTIS)
    add_library(trtis-request SHARED trtis_request.cc trtis_model_config.cc)
    # target_link_libraries(trtis-request nv-trtis-protos ${PROTOBUF_LIBRARY} curl grpc++_unsecure grpc gpr) # ${PROTOBUF_LIBRARY} # PNB: (2025.12.27)
    target_link_libraries(trtis-request nv-trtis-protos  protobuf::libprotobuf  curl grpc++_unsecure grpc gpr)
else()
    add_library(trtis-request INTERFACE)
endif()

# ------------------------------------------------------------
# Worker internal library (THIS FIXES YOUR LINK ERRORS)
# ------------------------------------------------------------
set(worker-util_SOURCES
    common_model_util.cc
    autoscaler.cc
    ${CMAKE_SOURCE_DIR}/utils/filesystem_utils.cpp   # PNB:
)

add_library(worker-util STATIC ${worker-util_SOURCES})

# include_directories("${CMAKE_SOURCE_DIR}/src") # PNB: (2025.12.27)
target_include_directories(inf-worker
  PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

## include_directories("${CMAKE_SOURCE_DIR}/src") # PNB: (2025.12.27)
#target_include_directories(worker-util PUBLIC
  #${CMAKE_SOURCE_DIR}   # PNB: (2025.12.27)
  #${CMAKE_SOURCE_DIR}/src
  #${PROJECT_BINARY_DIR}/protos/internal
#)

# PNB: (2025.12.28)
target_include_directories(worker-util PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/utils
  ${PROJECT_BINARY_DIR}/protos/internal
)

target_link_libraries(worker-util
    trtis-request
    redis-md
    infaas-protos-internal
    grpc++_unsecure
    grpc
    gpr
    protobuf::libprotobuf # ${PROTOBUF_LIBRARY} # PNB: (2025.12.27)
    #    $<$<BOOL:${ENABLE_AWS_AUTOSCALING}>:${AWSSDK_LINK_LIBRARIES}>
)

# ------------------------------------------------------------
# Worker executables
# ------------------------------------------------------------
add_executable(query_executor query_executor.cc)
add_executable(query_heartbeat query_heartbeat.cc)

target_link_libraries(query_executor
	worker-util
    absl::synchronization
    absl::strings
    absl::base
    absl::time
    absl::status
    ${OpenCV_LIBS}
    $<$<BOOL:${_has_nvidia}>:cuda>)
target_link_libraries(query_heartbeat
    inf-worker
)

# ------------------------------------------------------------
# GPU daemon
# ------------------------------------------------------------
# if(_has_nvidia)
#     cuda_add_executable(gpu_daemon gpu_daemon.cc OPTIONS -std=c++11)
#     target_link_libraries(gpu_daemon redis-md trtis-request)
# endif()

# PNB Place here in favor of the original commented block just above(2025.12.26)
if(ENABLE_TRTIS)
if(_has_nvidia)
    cuda_add_executable(gpu_daemon gpu_daemon.cc OPTIONS -std=c++11)
    target_link_libraries(gpu_daemon redis-md trtis-request)
endif()
endif()

# ------------------------------------------------------------
# Inferentia test
# ------------------------------------------------------------
if(_has_inferentia)
    add_executable(inferentia_query_test inferentia_query_test.cc)
    target_link_libraries(inferentia_query_test inf-worker ${OpenCV_LIBS})
endif()

