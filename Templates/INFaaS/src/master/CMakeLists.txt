project(infaas-master)

find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

option(ENABLE_AWS_AUTOSCALING "Enable AWS autoscaling" OFF)

if(ENABLE_AWS_AUTOSCALING)
    find_package(AWSSDK REQUIRED COMPONENTS ec2 s3)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories(/usr/local/include)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../../protos")

link_directories(/usr/local/lib /usr/local/lib64)

# ------------------------------------------------------------
# Master shared library
# ------------------------------------------------------------
add_library(inf-master SHARED
    queryfe_client.cc
    modelreg_client.cc
)

target_link_libraries(inf-master
    redis-md
    infaas-protos
    ${PROTOBUF_LIBRARY}
)

link_libraries(grpc++_unsecure grpc gpr)

# ------------------------------------------------------------
# Master executables
# ------------------------------------------------------------
add_executable(modelreg_server modelreg_server.cc)
add_executable(modelreg_heartbeat modelreg_heartbeat.cc)

add_executable(queryfe_server queryfe_server.cc)
add_executable(queryfe_heartbeat queryfe_heartbeat.cc)

# ------------------------------------------------------------
# AWS daemon only if enabled
# ------------------------------------------------------------
if(ENABLE_AWS_AUTOSCALING)
    add_executable(master_vm_daemon master_vm_daemon.cc)
    target_link_libraries(master_vm_daemon
        redis-md
        inf-worker
        ${AWSSDK_LINK_LIBRARIES}
    )
endif()

# ------------------------------------------------------------
# Link targets (worker-util required!)
# ------------------------------------------------------------
target_link_libraries(modelreg_server
    inf-master
    worker-util
)

target_link_libraries(modelreg_heartbeat
    inf-master
    worker-util
)

target_link_libraries(queryfe_server
    inf-master
    inf-worker
    worker-util
)

target_link_libraries(queryfe_heartbeat
    inf-master
)

# ------------------------------------------------------------
# Output dirs
# ------------------------------------------------------------
set_target_properties(modelreg_server modelreg_heartbeat
    queryfe_server queryfe_heartbeat
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(ENABLE_AWS_AUTOSCALING)
    set_target_properties(master_vm_daemon PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
