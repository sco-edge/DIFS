cmake_minimum_required(VERSION 3.10)

find_package(Protobuf REQUIRED)

find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
if(NOT GRPC_CPP_PLUGIN)
  message(FATAL_ERROR "grpc_cpp_plugin not found")
endif()

set(PROTO_FILES
  infaas_request_status.proto
  diffusion_service.proto
  query.proto
  model.proto
)

# ---- Generate protobuf + gRPC sources ----
set(GENERATED_PROTO_SRCS "")
set(GENERATED_PROTO_HDRS "")
set(GENERATED_GRPC_SRCS "")
set(GENERATED_GRPC_HDRS "")

foreach(proto ${PROTO_FILES})
  get_filename_component(proto_name ${proto} NAME_WE)

  list(APPEND GENERATED_PROTO_SRCS
    ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.cc)
  list(APPEND GENERATED_PROTO_HDRS
    ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.h)

  list(APPEND GENERATED_GRPC_SRCS
    ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.cc)
  list(APPEND GENERATED_GRPC_HDRS
    ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.h)
endforeach()

add_custom_command(
  OUTPUT
    ${GENERATED_PROTO_SRCS}
    ${GENERATED_PROTO_HDRS}
    ${GENERATED_GRPC_SRCS}
    ${GENERATED_GRPC_HDRS}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
  ARGS
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTO_FILES}
  DEPENDS ${PROTO_FILES}
)

# ---- Single library definition (ONLY ONCE) ----
add_library(infaas-protos-internal STATIC
  ${PROTO_FILES}
  ${GENERATED_PROTO_SRCS}
  ${GENERATED_GRPC_SRCS}
)

target_include_directories(infaas-protos-internal PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(infaas-protos-internal
  protobuf::libprotobuf
  grpc++
  grpc
)

