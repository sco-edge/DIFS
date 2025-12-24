#ifdef ENABLE_TRTIS

#include "common_model_util.h"
#include "trtis_request.h"

#include <iostream>
#include <thread>

namespace trtis = nvidia::inferenceserver;
namespace trtisc = nvidia::inferenceserver::client;

namespace infaas {
namespace internal {

static const std::string trtis_grpc_url = "localhost:8001";

/*
 * ===========================
 * TRTIS MODEL STATE HELPERS
 * ===========================
 */

int GpuModelState(const std::string& model_name) {
  trtis::ModelReadyState state = trtis::MODEL_UNAVAILABLE;

  std::unique_ptr<trtisc::ServerStatusContext> ctx;
  trtisc::Error err =
      trtisc::ServerStatusGrpcContext::Create(&ctx, trtis_grpc_url, false);

  if (!err.IsOk()) {
    std::cerr << "[TRTIS] Failed to create status context: "
              << err.Message() << std::endl;
    return state;
  }

  trtis::ServerStatus server_status;
  err = ctx->GetServerStatus(&server_status);
  if (!err.IsOk()) {
    std::cerr << "[TRTIS] GetServerStatus failed: "
              << err.Message() << std::endl;
    return state;
  }

  const auto& model_status_map = server_status.model_status();
  auto it = model_status_map.find(model_name);
  if (it == model_status_map.end()) {
    return trtis::MODEL_UNAVAILABLE;
  }

  const auto& version_map = it->second.version_status();
  if (version_map.empty()) {
    return trtis::MODEL_UNAVAILABLE;
  }

  return version_map.begin()->second.ready_state();
}

int8_t WaitGpuModelState(const std::string& model_name,
                         trtis::ModelReadyState desired_state,
                         unsigned interval,
                         unsigned max_retries) {
  for (unsigned i = 0; i < max_retries; ++i) {
    auto state = static_cast<trtis::ModelReadyState>(
        GpuModelState(model_name));

    if (state == desired_state) {
      return 0;
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(interval));
  }

  std::cerr << "[TRTIS] WaitGpuModelState timed out for model: "
            << model_name << std::endl;
  return -1;
}

/*
 * ===========================
 * TRTIS MODEL MANAGER
 * ===========================
 */

int8_t GpuModelManager::LoadModelTRTIS(const std::string& model_name) {
  std::cerr << "[TRTIS] Loading model: " << model_name << std::endl;
  return WaitGpuModelState(model_name, trtis::MODEL_READY, 1000, 10);
}

int8_t GpuModelManager::UnloadModelTRTIS(const std::string& model_name) {
  std::cerr << "[TRTIS] Unloading model: " << model_name << std::endl;
  return WaitGpuModelState(model_name, trtis::MODEL_UNAVAILABLE, 500, 10);
}

}  // namespace internal
}  // namespace infaas

#endif  // ENABLE_TRTIS
