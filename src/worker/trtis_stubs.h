#pragma once
#include <string>
#include <memory>
#include <map>
#include <vector>

// GLOBAL TRTIS STUBS - Used by ALL files
namespace nvidia {
namespace inferenceserver {
enum ModelReadyState {
  MODEL_UNAVAILABLE = 0,
  MODEL_LOADING = 1,
  MODEL_READY = 2
};

enum ServerReadyState {
  SERVER_UNAVAILABLE = 0,
  SERVER_STARTING = 1,
  SERVER_READY = 2,
  SERVER_FAILED = 3
};

struct ServerStatus {};
}  // inferenceserver
}  // nvidia

namespace nvidia {
namespace inferenceserver {
namespace grpc {
class Error {
 public:
  bool IsOk() const { return true; }
  std::string Message() const { return ""; }
};

class ServerStatusContext {
 public:
  static nvidia::inferenceserver::grpc::Error Create(
      std::unique_ptr<nvidia::inferenceserver::grpc::ServerStatusContext>* ctx,
      const std::string& url, bool verbose = false);
  nvidia::inferenceserver::grpc::Error GetServerStatus(
      nvidia::inferenceserver::ServerStatus* server_status);
};

class ServerStatusGrpcContext {};

class InferContext {
 public:
  class Options {
   public:
    static nvidia::inferenceserver::grpc::Error Create(
        std::unique_ptr<nvidia::inferenceserver::grpc::InferContext::Options>* options);
    void SetBatchSize(int32_t batch_size) {}
    void AddRawResult(const std::string& output_name) {}
  };
  
  std::vector<std::string> Inputs() const { return {}; }
  std::vector<std::string> Outputs() const { return {}; }
  nvidia::inferenceserver::grpc::Error SetRunOptions(const Options& options) { return Error(); }
  nvidia::inferenceserver::grpc::Error Run(
      std::map<std::string, std::unique_ptr<nvidia::inferenceserver::grpc::InferContext::Result>>* results) { return Error(); }
  
  struct Result {};
};

class InferGrpcContext {};
}  // grpc
}  // inferenceserver
}  // nvidia

// TYPE ALIASES
namespace trtis = nvidia::inferenceserver;
namespace trtisc = nvidia::inferenceserver::grpc;

// FUNCTION STUBS
trtis::ModelReadyState GpuModelState(const std::string& model_name);
int8_t WaitGpuModelState(const std::string& model_name, trtis::ModelReadyState model_state, unsigned interval, int max_tries = 10);
