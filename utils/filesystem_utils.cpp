#include "filesystem_utils.h"
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace filesystem_utils {

// ============================================================================
//  FileCopyOutcome methods must work even with move semantics
// ============================================================================

FileCopyOutcome::FileCopyOutcome(bool success, const std::string& error_message)
    : success(success), error_message(error_message) {}

// ============================================================================
// get_file_data – local equivalent of S3 GetObject
// ============================================================================
FileCopyOutcome get_file_data(const std::string& src) {
    try {
        if (!fs::exists(src)) {
            return FileCopyOutcome(false, "Source file does not exist: " + src);
        }

        std::ifstream file(src, std::ios::binary);
        if (!file.is_open()) {
            return FileCopyOutcome(false, "Failed to open file: " + src);
        }

        FileCopyOutcome outcome(true);
        outcome.file_stream = std::move(file);   // FIXED: move stream into struct
        return outcome;

    } catch (const std::ifstream::failure& e) {
        return FileCopyOutcome(false, "Error reading file: " + std::string(e.what()));
    }
}

// ============================================================================
//  ListObjectsOutcome constructors — unchanged but validated
// ============================================================================

ListObjectsOutcome::ListObjectsOutcome(const std::vector<std::string>& files)
    : success_(true), files_(files) {}

ListObjectsOutcome::ListObjectsOutcome(const std::string& error_message)
    : success_(false), error_message_(error_message) {}

// ============================================================================
//  ListLocalFiles (returning outcome for better AWS compatibility)
// ============================================================================
ListObjectsOutcome ListLocalFiles(const std::string& directory_path) {
    std::vector<std::string> file_list;

    try {
        if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
            return ListObjectsOutcome("Directory does not exist or is not a directory: " +
                                      directory_path);
        }

        for (const auto& entry : fs::directory_iterator(directory_path)) {
            if (fs::is_regular_file(entry)) {
                file_list.push_back(entry.path().string());
            }
        }

        return ListObjectsOutcome(file_list);

    } catch (const fs::filesystem_error& e) {
        return ListObjectsOutcome(e.what());
    }
}

// ============================================================================
//  list_local_path — INFaaS still uses this legacy format
// ============================================================================
void list_local_path(const std::string& directory, std::vector<std::string>& file_list) {
    try {
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cerr << "Error: Directory does not exist or is not a directory: "
                      << directory << std::endl;
            return;
        }

        for (const auto& entry : fs::directory_iterator(directory)) {
            if (fs::is_regular_file(entry)) {
                file_list.push_back(entry.path().string());
            }
        }

    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// ============================================================================
// copy_local_file — local equivalent of S3 CopyObject
// ============================================================================
FileCopyOutcome copy_local_file(const std::string& src, const std::string& dst) {
    try {
        if (!fs::exists(src)) {
            return FileCopyOutcome(false, "Source file does not exist: " + src);
        }

        fs::create_directories(fs::path(dst).parent_path());

        fs::copy(src, dst, fs::copy_options::overwrite_existing);

        return FileCopyOutcome(true);

    } catch (const fs::filesystem_error& e) {
        return FileCopyOutcome(false, "Error copying file: " + std::string(e.what()));
    }
}

} // namespace filesystem_utils
