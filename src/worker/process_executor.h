#pragma once

#ifndef INFAAS_PROCESS_EXECUTOR_H_
#define INFAAS_PROCESS_EXECUTOR_H_

#include <string>
#include <vector>

namespace infaas {
namespace internal {

/**
 * Forks a child process and executes a command.
 *
 * @param argv        Command and arguments (argv[0] = executable)
 * @param stdout_out  Captured stdout (optional, may be nullptr)
 * @param stderr_out  Captured stderr (optional, may be nullptr)
 *
 * @return Exit code of the child process, or -1 on failure
 */
int ForkAndExec(const std::vector<std::string>& argv,
                std::string* stdout_out,
                std::string* stderr_out);

}  // namespace internal
}  // namespace infaas

#endif  // INFAAS_PROCESS_EXECUTOR_H_

struct ModelSpec {
  std::string model_name;
  std::string framework;
  std::string task;
  std::string exec_path;
  std::string entry_point;
  std::string env_path;
};
