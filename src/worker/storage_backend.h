/*
  Author: KB
  Purpose: Created because "local_storage_backend.h" needs it to properly function and compile
  Date: Sun 28 Dec 01:15:23 KST 2025
 */

#pragma once

#include <string>
#include <vector>

namespace infaas {
namespace internal {

/**
 * Abstract storage interface.
 * Replaces AWS S3 without changing higher-level logic.
 */
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    // Check if object exists
    virtual bool exists(const std::string& path) = 0;

    // Read binary object
    virtual bool read(
        const std::string& path,
        std::vector<uint8_t>& out) = 0;

    // Write binary object
    virtual bool write(
        const std::string& path,
        const std::vector<uint8_t>& data) = 0;
};

} // namespace internal
} // namespace infaas
