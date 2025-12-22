INCLUDE(GRPCGenerateCPP)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
  request_reply.proto
  modelreg.proto
  queryfe.proto
  diffusion_service.proto # PNB: Diffusion model implementation related (2025.12.19)
)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS
  modelreg.proto
  queryfe.proto
  diffusion_service.proto # PNB: Diffusion model implementation related (2025.12.19)
)

add_library(infaas-protos
  ${PROTO_SRCS}
  ${PROTO_GRPC_SRCS}
)

target_include_directories(infaas-protos PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_subdirectory(internal)
add_subdirectory(trtis)
