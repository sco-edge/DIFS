/*
  Author: KB
  Purpose: for Worker RPC Implementation (Diffusion)
  Date: 2026.01.15

 */

#include "diffusion_service_impl.h"
#include "common_model_util.h"

namespace infaas {
namespace internal {

grpc::Status DiffusionServiceImpl::Generate(
    grpc::ServerContext*,
    const DiffusionRequest* request,
    DiffusionReply* response) {

  if (request->prompt().empty()) {
    response->set_status("ERROR");
    response->set_error_msg("Prompt is empty");
    return grpc::Status::OK;
  }

  try {
    std::string output_path;
    std::string png_bytes;

    int ret = DiffusionModelManager::Generate(
        request->model(),
        request->prompt(),
        request->steps(),
        request->width(),
        request->height(),
        png_bytes,
        output_path);

    if (ret != 0) {
      response->set_status("ERROR");
      response->set_error_msg("Diffusion generation failed");
      return grpc::Status::OK;
    }

    response->set_image_png(png_bytes);
    response->set_image_path(output_path);
    response->set_status("OK");
    return grpc::Status::OK;

  } catch (const std::exception& e) {
    response->set_status("ERROR");
    response->set_error_msg(e.what());
    return grpc::Status::OK;
  }
}

}  // namespace internal
}  // namespace infaas
