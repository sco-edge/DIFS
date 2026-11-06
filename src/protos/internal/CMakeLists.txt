add_libray(infaas-protos-diffusion
   diffusion_service.pb.cc
   diffusion_service.grpc.pb.cc
)


target_link_libraries(worker-util
   infaas-protos-diffusion
)

target_link_libraries(infaas-protos-diffusion
    protobuf::libprotobuf
    gRPC::grpc++
)

target_include_directories(infaas-protos-diffusion PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
