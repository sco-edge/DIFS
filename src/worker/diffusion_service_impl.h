/*
  Author: KB
  Purpose: for Worker RPC Implementation (Diffusion)
  Date: 2026.01.15

 */

#pragma once

#include "diffusion_service.grpc.pb.h"

namespace infaas {
namespace internal {

class DiffusionServiceImpl final : public DiffusionService::Service {
 public:
  grpc::Status Generate(
      grpc::ServerContext* context,
      const DiffusionRequest* request,
      DiffusionReply* response) override;
};

}  // namespace internal
}  // namespace infaas
