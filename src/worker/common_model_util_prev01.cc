#include "common_model_util.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "redis_md.h"

namespace infaas {
namespace internal {

/*
 * ===========================
 * TRTIS-INDEPENDENT UTILITIES
 * ===========================
 *
 * This file MUST compile without Triton/TRTIS headers.
 * All TRTIS logic is delegated to common_model_util_trtis.cc
 */

/* ----------------------------
 * Safe default GPU model state
 * ----------------------------
 */
#ifndef ENABLE_TRTIS

// Dummy enum replacement to avoid Triton headers
enum DummyModelState {
  MODEL_UNAVAILABLE = 0,
  MODEL_READY = 1
};

// Stub: TRTIS not available
int GpuModelState(const std::string& model_name) {
  std::cerr << "[GPU Worker] TRTIS disabled, model unavailable: "
            << model_name << std::endl;
  return MODEL_UNAVAILABLE;
}

// Stub: always fail safely
int8_t WaitGpuModelState(const std::string& model_name,
                         int /*desired_state*/,
                         unsigned /*interval*/,
                         unsigned /*max_retries*/) {
  std::cerr << "[GPU Worker] TRTIS disabled, cannot wait for model state: "
            << model_name << std::endl;
  return -1;
}

#endif  // ENABLE_TRTIS

/*
 * ===========================
 * NON-TRTIS MODEL MANAGEMENT
 * ===========================
 *
 * These functions are allowed to exist regardless of TRTIS
 */

GpuModelManager::GpuModelManager() = default;
GpuModelManager::~GpuModelManager() = default;

int8_t GpuModelManager::LoadModel(
    std::string model_name,
    std::string /*model_path*/,
    std::unique_ptr<RedisMetadata>& /*redis*/,
    std::unique_ptr<Aws::S3::S3Client>& /*s3*/) {

#ifndef ENABLE_TRTIS
  std::cerr << "[GPU Worker] LoadModel skipped (TRTIS disabled): "
            << model_name << std::endl;
  return -1;
#else
  return LoadModelTRTIS(model_name);
#endif
}

int8_t GpuModelManager::UnloadModel(
    const std::string& model_name,
    std::unique_ptr<RedisMetadata>& /*redis*/) {

#ifndef ENABLE_TRTIS
  std::cerr << "[GPU Worker] UnloadModel skipped (TRTIS disabled): "
            << model_name << std::endl;
  return -1;
#else
  return UnloadModelTRTIS(model_name);
#endif
}

}  // namespace internal
}  // namespace infaas
