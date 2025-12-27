#pragma once

#include <cstdint>
#include <string>

#ifdef AWS_SDK_DISABLED

namespace localfs {

struct ClientConfiguration {
  uint32_t connectTimeoutMs = 0;
  uint32_t requestTimeoutMs = 0;
  std::string root_dir = "";
  // Optional: keep region / endpoint semantics
  std::string endpoint;
};

class S3Client {
public:
    explicit S3Client(const ClientConfiguration& cfg) : cfg_(cfg) {}
private:
    ClientConfiguration cfg_;
};


// Drop-in stand-in for Aws::Client::ClientConfiguration
struct LocalClientConfig {
  uint32_t connectTimeoutMs = 0;
  uint32_t requestTimeoutMs = 0;
  std::string root_dir = "";
  // Optional: keep region / endpoint semantics
  std::string endpoint;
};

}// namespace localfs

// namespace Aws {
// namespace Client {

// // Alias so existing code still sees Aws::Client::ClientConfiguration
// using ClientConfiguration = localfs::ClientConfiguration;

// }  // namespace Client
// }  // namespace Aws

#else
#error "AWS SDK should not be enabled in offline/local mode"
#endif
