/*
Author: KB
Purpose: To replace AWS storage functionality in local version
Date: Sat 27 Dec 23:50:23 KST 2025
 */

#pragma once
#include "storage_backend.h"   // THIS WAS MISSING
#include <string>
#include <filesystem>
#include <fstream>
/* struct LocalStorageBackend : public StorageBackend { */
/*     std::string root; */

/*     explicit LocalStorageBackend(std::string r) : root(std::move(r)) {} */

/*     bool exists(const std::string& path) override { */
/*         return std::filesystem::exists(root + "/" + path); */
/*     } */
/* }; */


namespace infaas {
namespace internal {

struct LocalStorageBackend : public StorageBackend {

    bool exists(const std::string& path) override {
        return std::filesystem::exists(path);
    }

    bool read(const std::string& path,
              std::vector<uint8_t>& out) override {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        out.assign(std::istreambuf_iterator<char>(f), {});
        return true;
    }

    bool write(const std::string& path,
               const std::vector<uint8_t>& data) override {
        std::filesystem::create_directories(
            std::filesystem::path(path).parent_path());
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
        return true;
    }
};

} // namespace internal
} // namespace infaas
