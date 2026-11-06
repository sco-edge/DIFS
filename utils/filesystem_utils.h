#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

#include <string>
#include <vector>
#include <fstream>

namespace filesystem_utils {

// ----------------------------------------------------------------------------
// Legacy interface used by INFaaS
// ----------------------------------------------------------------------------
void list_local_path(const std::string& directory, std::vector<std::string>& file_list);

// ----------------------------------------------------------------------------
// FileCopyOutcome structure (local S3-style outcome)
// ----------------------------------------------------------------------------
struct FileCopyOutcome {
    bool success;
    std::string error_message;
    std::ifstream file_stream;

    FileCopyOutcome(bool success, const std::string& error_message = "");

    bool IsSuccess() const { return success; }
    std::ifstream& GetResult() { return file_stream; }
    const std::string& GetErrorMessage() const { return error_message; }
    const std::string& GetError() const { return error_message; }
};

// ----------------------------------------------------------------------------
// ListObjectsOutcome — AWS ListObjects-style wrapper for local filesystem
// ----------------------------------------------------------------------------
class ListObjectsOutcome {
public:
    ListObjectsOutcome(const std::vector<std::string>& files);
    ListObjectsOutcome(const std::string& error_message);

    bool IsSuccess() const { return success_; }
    const std::vector<std::string>& GetFiles() const { return files_; }
    const std::string& GetErrorMessage() const { return error_message_; }
    const std::string& GetError() const { return error_message_; }

private:
    bool success_;
    std::vector<std::string> files_;
    std::string error_message_;
};

// ----------------------------------------------------------------------------
// API functions
// ----------------------------------------------------------------------------
ListObjectsOutcome ListLocalFiles(const std::string& directory_path);
FileCopyOutcome get_file_data(const std::string& src);
FileCopyOutcome copy_local_file(const std::string& src, const std::string& dst);

} // namespace filesystem_utils

#endif // FILESYSTEM_UTILS_H
