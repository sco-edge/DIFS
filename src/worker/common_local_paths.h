#ifndef COMMON_LOCAL_PATHS_H
#define COMMON_LOCAL_PATHS_H

#include <string>

namespace infaas {
namespace internal {

// Local-only fallback paths (used when S3 / remote paths are disabled)
static const std::string kLocalModelBaseDir = "/tmp/infaas/models";
static const std::string kLocalInputBaseDir = "/tmp/infaas/input";
static const std::string kLocalOutputBaseDir = "/tmp/infaas/output";

}  // namespace internal
}  // namespace infaas

#endif
