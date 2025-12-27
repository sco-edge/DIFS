/*
 * Copyright 2018-2021 Board of Trustees of Stanford University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <curl/curl.h>
#include <dirent.h>
#include <grpcpp/grpcpp.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <random>
#include <cstdlib>
#include <sstream>

// PNB 2025.11.28 - Local paths
#include "common_local_paths.h"

#ifndef LOCAL_MODE
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/DeleteObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/PutObjectRequest.h>
#endif

#include "common_model_util.h"
#include "include/base64.h"
//#include "include/constants.h"
#include "constants.h"  // PNB: (2025.11.28)
#include "include/json.hpp"
#include "query.grpc.pb.h"
#include "query_client.h"
#include "autoscaler.h" // PNB: (2025.11.28)
#include "query.grpc.pb.h" // PNB: (2025.12.27)
#include "query.pb.h" // PNB: (2025.12.27)

#ifdef INFAAS_NEURON_WORKER
#include "trtis_request.h"
#endif

using Aws::S3::S3Client;
using grpc::ClientContext;
using grpc::Status;
using json = nlohmann::json;

//PNB: To have access to the symbols within "infaas::internal:[object]" namespace (2025.12.28)
using infaas::internal::Query;
using infaas::internal::QueryOnlineRequest;
using infaas::internal::QueryOnlineResponse;
using infaas::internal::QueryOfflineRequest;
using infaas::internal::QueryOfflineResponse;
using infaas::internal::InfaasRequestStatusEnum;
using infaas::internal::Autoscaler;

// ================================================
// CONSTANTS AND GLOBAL VARIABLES
// ================================================
static const std::string trtis_grpc_url = "localhost:8001";
static const std::string local_model_dir = "tmp/models";
static const std::string local_trt_model_dir = "tmp/trt_models";
static const std::string local_input_dir = "tmp/infaas_input";
static const std::string local_output_dir = "tmp/infaas_output";
static const std::string local_input_leaf_dir = "infer";
static const std::string trt_pbtxt_file = "config.pbtxt";
static const std::string batching_parameters_file = "infaas_path/src/worker/batching_parameters.txt";
static const std::string batching_parameters_file_tmp = "tmp/batching_parameters.txt";  // PNB 2025.11.28
static const std::string GNMT_NVPY = "gnmt-nvpy";
static const int first_cpu_port = 9001;  // The first port used by CPU container
static const int first_infa_port = 4001;  // The first port used by Inferentia container. 5000 apart from CPU ones.
static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
static const int offline_batch = 4;  // This might change.
static const int offline_cpu_thresh = 40;  // Offline can run iff CPU util < 40%, might change.
static const int sleep_interval = 1000;  // Offline poll per 1 sec.
static const int DOCKER_STOP_TIME = 1;  // Wait 1 sec before killing a container
static const bool CPU_ADAPTIVE_BATCHING = false;
static const bool OFFLINE_CONTROL = true;  // if true, run to avoid interference
static const int WARMUP_QUERIES = 10;

#ifdef INFAAS_NEURON_WORKER
static const int WORKER_NEURON_CORES = 4;
#endif

// RUNTIME GLOBALS - FIXED FOR LOCAL_MODE
std::string bucketprefix = "s3://";
std::string workername = "local_worker";

uint64_t time2 = 0;
// ================================================
// CRITICAL UTILITY FUNCTIONS - FULL IMPLEMENTATIONS
// ================================================
inline uint64_t get_curr_timestamp() { 
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

inline double get_duration_ms(uint64_t a, uint64_t b) { 
    return (b - a) / 1000.0; 
}

inline bool file_exist(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}

inline std::string exec_cmd(const std::string& cmd) {
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buffer[128];
    while (fgets(buffer, 128, pipe) != NULL) result += buffer;
    pclose(pipe);
    return result;
}

inline int8_t createdir(const std::string& path) {
    return mkdir(path.c_str(), 0755);
}

inline int8_t rmdir(const std::string& path) {
    return rmdir(path.c_str());
}

// S3 UTILITY STUBS FOR LOCAL_MODE
inline int8_t parse_s3_url(const std::string& url, std::string& bucket, std::string& objname) {
    size_t pos = url.find("://");
    if (pos == std::string::npos) return -1;
    std::string path = url.substr(pos + 3);
    size_t slash = path.find('/');
    if (slash == std::string::npos) return -1;
    bucket = path.substr(0, slash);
    objname = path.substr(slash + 1);
    return 0;
}

inline int8_t list_s3_path(const std::string& bucket, const std::string& objname, 
                          std::unique_ptr<S3Client>& s3c, std::vector<std::string>& keynames) {
    // LOCAL_MODE: Skip S3, return empty
    keynames.clear();
    return 0;
}

inline int8_t download_s3_local(const std::string& srcbucket, const std::string& objname,
                               std::vector<std::string>::iterator start, 
                               std::vector<std::string>::iterator end,
                               const std::string& dsturl, std::unique_ptr<S3Client>& s3c) {
    // LOCAL_MODE: Skip download
    return 0;
}

inline int8_t upload_local_s3(const std::string& localbasedir, const std::vector<std::string>& fnames,
                             const std::string& dstbucket, const std::string& dstobjname,
                             std::unique_ptr<S3Client>& s3c) {
    // LOCAL_MODE: Skip upload
    return 0;
}

inline int8_t list_local_path(const std::string& localbasedir, std::vector<std::string>& fnames) {
    DIR* dir = opendir(localbasedir.c_str());
    if (!dir) return -1;
    fnames.clear();
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) fnames.push_back(entry->d_name);
    }
    closedir(dir);
    return 0;
}

// DOCKER STUBS - Always succeed for LOCAL_MODE
inline bool WaitDockerInstanceStop(const std::string& name) { return true; }
inline bool WaitDockerModelReady(const std::string& modelname, const std::string& framework, int portnum, int interval) { 
    return true; 
}

// TRTIS STUBS FOR LOCAL_MODE + Diffusion
namespace nvidia {
namespace inferenceserver {
enum ModelReadyState { MODEL_UNAVAILABLE = 0, MODEL_LOADING = 1, MODEL_READY = 2 };
enum ServerReadyState { SERVER_UNAVAILABLE = 0, SERVER_READY = 2 };
struct ServerStatus {};

namespace grpc {
class Error {
public:
    bool IsOk() const { return true; }
    std::string Message() const { return ""; }
};
struct Result {};
class InferContext {
public:
    class Options {};
    static Error Create(std::unique_ptr<Options>& o) { return Error(); }
    void SetBatchSize(int32_t batchsize) {}
    void AddRawResult(const std::string& outputname, std::vector<std::string>& inputs) {}
    std::vector<std::string> Inputs() const { return {}; }
    std::vector<std::string> Outputs() const { return {}; }
    Error SetRunOptions(const Options& options) { return Error(); }
    Error Run(std::map<std::string, std::unique_ptr<Result>>& results) { return Error(); }
};
}  // grpc
}  // inferenceserver
}  // nvidia

namespace trtis = nvidia::inferenceserver;

inline trtis::ModelReadyState GpuModelState(const std::string& model_name) {
    return trtis::MODEL_READY;
}

inline int8_t WaitGpuModelState(const std::string& model_name, 
                               trtis::ModelReadyState model_state, 
                               unsigned interval, int max_tries = 10) {
    return 0;  // Always success
}

// Diffusion model runner for LOCAL_MODE
bool RunDiffusionModel(const std::string& modelname, const std::string& inputpath, 
                      const std::string& outputpath, std::string& errormsg) {
    std::stringstream cmd;
    cmd << "python3 ./scripts/run_diffusion.py --model " << modelname 
        << " --input " << inputpath << " --output " << outputpath;
    
    int ret = system(cmd.str().c_str());
    if (ret != 0) {
        errormsg = "Diffusion execution failed with code " + std::to_string(ret);
        return false;
    }
    return true;
}

// ================================================
// CPU MODEL MANAGER - FULL IMPLEMENTATION
// ================================================
namespace CpuModelManager {
    std::map<std::string, std::deque<std::string>> modeltonamesonline;
    std::map<std::string, std::deque<std::string>> modeltonamesoffline;
    std::map<std::string, int> nametoport;
    std::set<int, std::greater<int>> usedports;
    std::mutex updatemutex;
    
    size_t numReplicas(const std::string& modelname) {
        return modeltonamesonline[modelname].size();
    }
    
    int8_t LoadModel(const std::string& srcurl, const std::string& modelname,
                    std::unique_ptr<RedisMetadata>& rmd, std::unique_ptr<S3Client>& s3c,
                    const std::string& containername, bool foronline) {
        uint64_t time1 = get_curr_timestamp(), time2;
        
        std::map<std::string, std::deque<std::string>>* modeltonames = 
            foronline ? &modeltonamesonline : &modeltonamesoffline;
            
        if (srcurl.empty()) {
            std::cerr << "CPU Manager: Empty source URL!" << std::endl;
            return -1;
        }
        if (modelname.empty()) {
            std::cerr << "CPU Manager: Empty model name!" << std::endl;
            return -1;
        }
        
        std::string instancename = containername.empty() ? modelname : containername;
        std::cout << "Loading model " << modelname << " on CPU, instance/container name " 
                  << instancename << std::endl;
        
        int portnum;
        {
            std::lock_guard<std::mutex> lock(updatemutex);
            if (nametoport.find(instancename) != nametoport.end()) {
                std::cout << "Instance " << instancename << " is being loaded." << std::endl;
                return 1;
            }
            
            if (usedports.size() == 0) {
                portnum = first_cpu_port;
            } else {
                portnum = *usedports.begin() - 1;  // last port + 1
            }
            usedports.insert(portnum);
            nametoport[instancename] = portnum;
        }
        
        // Download files if doesn't exist
        std::string dsturl = local_model_dir + "/" + modelname;
        if (dsturl.back() != '/') dsturl += '/';
        
        if (!file_exist(dsturl)) {
            uint64_t time3 = get_curr_timestamp(), time4;
            
            // Parse source bucket name and object name
            size_t preind = srcurl.find(bucketprefix);
            if (preind == std::string::npos || preind != 0) {
                std::cerr << "CPU Manager: Not a valid bucket source " << srcurl << std::endl;
                return -1;
            }
            
            std::string srcbucket, objname;
            parse_s3_url(srcurl, srcbucket, objname);
            std::cout << "CPU Load model src bucket name " << srcbucket 
                      << " object " << objname << std::endl;
            
            // List all names and download. We assume that a model is within a directory.
            if (objname.back() != '/') objname += '/';
            std::vector<std::string> keynames;
            if (list_s3_path(srcbucket, objname, s3c, keynames) != 0) {
                std::cerr << "CPU Manager: failed to list the files" << std::endl;
                return -1;
            }
            if (keynames.size() == 0) {
                std::cerr << "CPU Manager: keynames size 0 " << objname << std::endl;
                return -1;
            }
            
            if (download_s3_local(srcbucket, objname, keynames.begin(), keynames.end(), 
                                dsturl, s3c) != 0) {
                std::cerr << "CPU Manager: Failed to load model " << modelname 
                          << " to worker " << workername << std::endl;
                return -1;
            }
            
            time4 = get_curr_timestamp();
            printf("[common_model_util.cc] CPU LoadModel - copy files %.4lf ms.\n",
                   get_duration_ms(time3, time4));
        }
        
        // Start the container
        auto framework = rmd->get_model_info(modelname, "framework");
        int inputdim = std::stoi(rmd->get_model_info(modelname, "imgdim"));
        int ready = -1;
        char docker_cmd[512];
        
        // Limit the cpu usage to cpupercontainer
        int cpupercontainer = 1;
        // Get the number of CPUs requirement from the suffix of the name
        size_t rpos = modelname.rfind('_');
        if (rpos != std::string::npos) {
            std::string numstr = modelname.substr(rpos + 1);
            size_t endpos = numstr.find_first_not_of("0123456789");
            if (endpos != std::string::npos) numstr = numstr.substr(0, endpos);
            if (!numstr.empty()) cpupercontainer = std::stoi(numstr);
        }
        std::cout << "Num of cpu for container " << cpupercontainer << std::endl;
        
        // Make sure the instance name is available
        if (!WaitDockerInstanceStop(instancename)) {
            std::cerr << "Instance " << instancename << " is not stopped!" << std::endl;
            {
                std::lock_guard<std::mutex> lock(updatemutex);
                auto it = nametoport.find(instancename);
                if (it != nametoport.end()) {
                    portnum = it->second;
                    nametoport.erase(it);
                    usedports.erase(portnum);
                }
            }
            return -1;
        }
        
        if (framework == "pytorch") {
            std::string offlinenice = OFFLINE_CONTROL ? "ON" : "OFF";
            sprintf(docker_cmd, 
                   "docker run --rm -it -d -p%d:%d --cpus=%.1f --name=%s "
                   "--ipc=host --cap-add=sys_nice -e OFFLINENICE=%s "
                   "-v%s:/tmp/model -v%s:/tmp/infaas_input -v%s:/tmp/infaas_output "
                   "qianl15/infaas-pytorch:latest workspace/containerstart.sh "
                   "pytorchcontainer.py %d %s %d",
                   portnum, portnum, (float)cpupercontainer, instancename.c_str(), 
                   offlinenice.c_str(), local_model_dir.c_str(), local_input_dir.c_str(), 
                   local_output_dir.c_str(), inputdim, modelname.c_str(), portnum);
            ready = system(docker_cmd);
        } else if (framework == "tensorflow-cpu") {
            if (CPU_ADAPTIVE_BATCHING) {
                sprintf(docker_cmd, 
                       "docker run --rm -it -d -p%d:8501 --cpus=%.1f --name=%s "
                       "--ipc=host --cap-add=sys_nice -v%s:/models/%s "
                       "-e MODEL_NAME=%s -v%s:/models/%s/batching_parameters.txt "
                       "-t tensorflow/serving --enable-batching=true "
                       "--batching_parameters_file=models/%s/batching_parameters.txt",
                       portnum, (float)cpupercontainer, instancename.c_str(), 
                       local_model_dir.c_str(), modelname.c_str(), modelname.c_str(), 
                       batching_parameters_file.c_str(), modelname.c_str(), modelname.c_str());
            } else {
                sprintf(docker_cmd, 
                       "docker run --rm -it -d -p%d:8501 --cpus=%.1f --name=%s "
                       "--ipc=host --cap-add=sys_nice -v%s:/models -e MODEL_NAME=%s "
                       "tensorflow/serving",
                       portnum, (float)cpupercontainer, instancename.c_str(), 
                       local_model_dir.c_str(), modelname.c_str());
            }
            ready = system(docker_cmd);
        } else if (framework.find(GNMT_NVPY) != std::string::npos) {
            std::string dockerver = "docker";
            int beamsize = 1;
            std::string modelmath = "fp32";
            std::string usecuda = "--no-cuda";
            
            // Decide CPU or GPU
            size_t found = modelname.find("gpu");
            if (found != std::string::npos) {
                dockerver = "nvidia-docker";
                usecuda = "--cuda";
            }
            
            // Decide beam size
            found = modelname.rfind('_');
            if (found != std::string::npos) {
                std::string numstr = modelname.substr(found + 1);
                size_t endpos = numstr.find_first_not_of("0123456789");
                if (endpos != std::string::npos) numstr = numstr.substr(0, endpos);
                if (!numstr.empty()) beamsize = std::stoi(numstr);
            }
            
            // Decide precision
            found = modelname.find("fp16");
            if (found != std::string::npos) modelmath = "fp16";
            
            sprintf(docker_cmd, "%s run --rm -it -d -p%d:%d --cpus=4 --name=%s "
                   "--ipc=host -v%s:/tmp/model qianl15/gnmt-infaas:latest "
                   "./gnmtcontainer.py --model %s --port %d %s --math %s --beam-size %d",
                   dockerver.c_str(), portnum, portnum, instancename.c_str(), 
                   local_model_dir.c_str(), modelname.c_str(), portnum, usecuda.c_str(), 
                   modelmath.c_str(), beamsize);
            ready = system(docker_cmd);
        } else {
            std::cerr << "Unsupport framework " << framework << std::endl;
            ready = -1;
        }
        
        if (ready == -1) {
            std::cerr << "CPU Manager: Docker cmd failed or unsupported" << std::endl;
            {
                std::lock_guard<std::mutex> lock(updatemutex);
                auto it = nametoport.find(instancename);
                if (it != nametoport.end()) {
                    portnum = it->second;
                    nametoport.erase(it);
                    usedports.erase(portnum);
                }
            }
            return -1;
        }
        
        // Check the container is running
        sprintf(docker_cmd, "docker ps -aqf name=%s", instancename.c_str());
        std::string retstr = exec_cmd(docker_cmd);  // Should return the docker ID
        
        if (retstr.empty()) {
            std::cerr << "CPU Manager: Failed to start model instance " << instancename 
                      << " to worker " << workername << std::endl;
            {
                std::lock_guard<std::mutex> lock(updatemutex);
                auto it = nametoport.find(instancename);
                if (it != nametoport.end()) {
                    portnum = it->second;
                    nametoport.erase(it);
                    usedports.erase(portnum);
                }
            }
            return -1;
        }
        
        // Wait for model to be ready
        if (!WaitDockerModelReady(modelname, framework, portnum, 100)) {
            std::cerr << "CPU Manager: Heartbeat failed! Retry..." << std::endl;
        }
        if (!WaitDockerModelReady(modelname, framework, portnum, 1000)) {
            std::lock_guard<std::mutex> lock(updatemutex);
            auto it = nametoport.find(instancename);
            if (it != nametoport.end()) {
                portnum = it->second;
                nametoport.erase(it);
                usedports.erase(portnum);
            }
            return -1;
        }
        
        time2 = get_curr_timestamp();
        printf("[common_model_util.cc] CPU LoadModel - total time %.4lf ms.\n",
               get_duration_ms(time1, time2));
        
        // Update instance name to the map. Push to the queue only if everything is fine.
        {
            std::lock_guard<std::mutex> lock(updatemutex);
            // If it is the first instance of that model variant and it is for online query,
            // then update to metadata store.
            if (modeltonames->find(modelname) == modeltonames->end() || 
                modeltonames->at(modelname).empty()) {
                if (foronline) {
                    rmd->add_running_model(workername, modelname);
                    std::string parentmod = rmd->get_parent_model(modelname);
                    rmd->unset_parent_scaledown(workername, parentmod);
                }
            }
            (*modeltonames)[modelname].push_back(instancename);
        }
        std::cout << "Model is ready to serve with docker " << retstr << std::endl;
        return 0;
    }
    
    int8_t UnloadModel(const std::string& modelname, std::unique_ptr<RedisMetadata>& rmd,
                      const std::string& containername, bool foronline) {
        if (modelname.empty()) {
            std::cerr << "CPU Manager: Empty model name!" << std::endl;
            return -1;
        }
        std::cout << "Unload model instance of " << modelname << " from CPU." << std::endl;
        
        std::map<std::string, std::deque<std::string>>* modeltonames = 
            foronline ? &modeltonamesonline : &modeltonamesoffline;
        std::string instancename;
        
        {
            std::lock_guard<std::mutex> lock(updatemutex);
            // We assume the model is not available to serve queries once we started unloading it.
            auto it = modeltonames->find(modelname);
            if (it == modeltonames->end() || it->second.empty()) {
                std::cout << "Model " << modelname << " has already unloaded." << std::endl;
                return 0;
            }
            
            std::deque<std::string>& namesqueue = it->second;
            // Remove the instance from back if unspecified name.
            instancename = containername.empty() ? namesqueue.back() : containername;
            std::cout << "Removing instance " << instancename << std::endl;
            
            auto deqit = std::find(namesqueue.begin(), namesqueue.end(), instancename);
            auto ntpit = nametoport.find(instancename);
            if (ntpit == nametoport.end()) {
                if (deqit == namesqueue.end()) {
                    std::cout << "Instance " << instancename << " is unloaded (not found)" << std::endl;
                    return 0;
                } else {
                    std::cerr << "Inconsistency found! instance name is not in nametoport "
                              << "but appears in modeltonames." << std::endl;
                    return -1;
                }
            }
            
            namesqueue.erase(deqit);
            int portnum = ntpit->second;
            nametoport.erase(ntpit);
            usedports.erase(portnum);
        }
        
        int8_t isrunning = -1;
        if (foronline) {
            isrunning = rmd->is_model_running(modelname, workername);
            if (!isrunning) {
                std::cerr << "Model is not running! inconsistency found." << std::endl;
                return -1;
            }
        }
        
        if (foronline && modeltonames->find(modelname) != modeltonames->end() && 
            modeltonames->at(modelname).empty()) {
            // Set model load/unload flag.
            int8_t res = rmd->set_model_load_unload(modelname);
            if (res != 0) {
                std::cerr << "Failed to set modelloadunload flag for " << modelname << std::endl;
            }
            res = rmd->remove_running_model(workername, modelname);
            if (res != 0) {
                std::cerr << "Failed to remove running model" << std::endl;
                return -1;
            }
        }
        
        // Clean up docker container
        char docker_cmd[200];
        sprintf(docker_cmd, "docker stop -t %d %s", DOCKER_STOP_TIME, instancename.c_str());
        std::string retstr = exec_cmd(docker_cmd);
        
        if (retstr.find(instancename) == std::string::npos) {
            std::cerr << "CPU Manager: Docker cmd failed " << retstr 
                      << " expected " << instancename << std::endl;
            std::cerr << "Find at pos " << retstr.find(instancename) << std::endl;
            return -1;
        }
        
        if (foronline) {
            int8_t res = rmd->unset_model_load_unload(modelname);
            if (res != 0) {
                std::cerr << "Failed to unset modelloadunload flag for " << modelname << std::endl;
            }
        }
        
        Autoscaler::setAvgBatch(modelname, 0);
        return 0;
    }
    
    int8_t QueryModelOnline(const std::string& modelname, const QueryOnlineRequest& request,
                           QueryOnlineResponse& reply, std::unique_ptr<RedisMetadata>& rmd,
                           std::unique_ptr<S3Client>& s3c) {
        if (modelname.empty()) {
            std::cerr << "CPU Manager: model name is empty!" << std::endl;
            return -1;
        }
        
        // Need to wait until model is loaded.
        int numtry = 0, maxtry = 60;
        int sleepinterval = 500;  // Sleep 500 ms.
        while (numtry < maxtry) {
            if (modeltonamesonline.find(modelname) != modeltonamesonline.end() &&
                modeltonamesonline[modelname].size() > 0) {
                break;
            }
            
            // Load the model
            std::string srcurl = bucketprefix + infaas_buckets_dir + modelname;
            auto res = LoadModel(srcurl, modelname, rmd, s3c, modelname, true);
            if (res == 0) {
                break;  // Loaded by this thread
            } else if (res == 1) {
                // Model is being loaded by others, still need to wait.
                std::cout << "Model " << modelname << " is being loaded by others but not available" << std::endl;
            } else {
                std::cerr << "CPU Manager QueryModelOnline - Failed to load model " << modelname << std::endl;
                return res;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepinterval));
            numtry++;
            std::cout << "Try " << numtry << " failed, try again." << std::endl;
        }
        if (numtry >= maxtry) {
            std::cerr << "CPU Manager: Timed out waiting for model loading..." << std::endl;
            return -1;
        }
        
        // Schedule in a round robin way - get one from the front and push to the back.
        std::string instancename;
        {
            std::lock_guard<std::mutex> lock(updatemutex);
            instancename = modeltonamesonline[modelname].front();
            modeltonamesonline[modelname].pop_front();
            modeltonamesonline[modelname].push_back(instancename);
        }
        
        int portnum = nametoport[instancename];
        std::cout << "Serve with instance " << instancename << " port number " 
                  << std::to_string(portnum) << std::endl;
        
        auto framework = rmd->get_model_info(modelname, "framework");
	Address destaddr;
	destaddr.ip = "localhost";
	destaddr.port = portnum;
	
        if (framework == "pytorch" || framework.find(GNMT_NVPY) != std::string::npos) {
        
            grpc::ChannelArguments arguments;
            arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
            arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);

            std::string addr = destaddr.ip + ":" + destaddr.port;
            std::unique_ptr<Query::Stub> stub = Query::NewStub(
                grpc::CreateCustomChannel(addr, grpc::InsecureChannelCredentials(), arguments));
            
            ClientContext context;
            uint64_t time1 = get_curr_timestamp(), time2;
            Status status = stub->QueryOnline(&context, request, &reply);
            time2 = get_curr_timestamp();
            printf("[common_model_util.cc] Pytorch inference time %.4lf ms.\n",
                   get_duration_ms(time1, time2));
        } else if (framework == "tensorflow-cpu") {
            const google::protobuf::RepeatedPtrField<std::string>& rawinput = request.raw_input();
            const auto& raw_input = request.raw_input(); // PNB: (2025.12.27)
	    size_t batchsize = raw_input.size();
            
            CURL* curl = curl_easy_init();
            if (!curl) {
                std::cerr << "failed to post request to model " << modelname << std::endl;
                return -1;
            }
            
            uint64_t time1 = get_curr_timestamp();
            std::string curlreqs = "{\"instances\":[";
            
            // Encode each input as base64
            for (size_t idx = 0; idx < batchsize; idx++) {

	      // PNB: (2025.12.27)
		const std::string& s = raw_input.Get(idx);
                const char* rawptr = s.data();
                unsigned int rawsize = s.size();
		
		
                unsigned char const* bytestoencode = 
                reinterpret_cast<unsigned char const*>(rawptr);
                std::string base64str = base64_encode(bytestoencode, rawsize);
                std::replace(base64str.begin(), base64str.end(), '+', '-');
                std::replace(base64str.begin(), base64str.end(), '/', '_');
                
                if (idx > 0) curlreqs += ",";
                curlreqs += "\"b64:" + base64str + "\"";
            }
            curlreqs += "]}";
            
            std::cout << "curlreqs string size " << curlreqs.size() << std::endl;
            uint64_t timeb64encode = get_curr_timestamp();
            printf("[common_model_util.cc] TF-CPU base64 encode time %.4lf ms.\n",
                   get_duration_ms(time1, timeb64encode));
            
            std::string readbuff;
            std::string tfurl = "http://localhost:" + std::to_string(portnum) + 
                               "/v1/models/" + modelname + ":predict";
            
            // Now post the request
            struct curl_slist* curllist = NULL;
            curllist = curl_slist_append(curllist, "Content-Type: application/json");
            curllist = curl_slist_append(curllist, "charset: utf-8");
            
            curl_easy_setopt(curl, CURLOPT_URL, tfurl.c_str());
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curllist);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, curlreqs.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, curlreqs.length());
            curl_easy_setopt(curl, CURLOPT_POST, 1);
            
            // Callback function to write response
            auto CurlWriteCallBack = [](void* contents, size_t size, size_t nmemb, std::string* userp) {
                userp->append((char*)contents, size * nmemb);
                return size * nmemb;
            };
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallBack);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readbuff);
            curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
            
            CURLcode curlres = curl_easy_perform(curl);
            
            // Process output
            if (readbuff.size() > 1000) std::cout << "Readbuff " << readbuff << std::endl;
            json bodyjs = json::parse(readbuff);
            auto predjs = bodyjs["predictions"];
            
            // One batch at a time.
            float* predv = nullptr;
            for (json::iterator it = predjs.begin(); it != predjs.end(); ++it) {
                std::cout << "Size " << it->size() << std::endl;
                if (predv) delete[] predv;
                predv = new float[it->size()];
                
                // Turn the string of floating points into raw bytes.
                int cntit = 0;
                for (auto& i : *it) {
                    predv[cntit++] = std::stof(i.dump());
                }
                reply.add_raw_output(reinterpret_cast<const char*>(predv), 
                                  it->size() * sizeof(float));
            }
            if (predv) delete[] predv;
            
            curl_easy_cleanup(curl);
            curl_slist_free_all(curllist);
            
            time2 = get_curr_timestamp();
            printf("[common_model_util.cc] TF-CPU inference time %.4lf ms.\n",
                   get_duration_ms(time1, time2));
        } else {
            std::cerr << "Don't support framework " << framework << std::endl;
            return -1;
        }
        
        std::cout << "rawoutput batch size " << reply.raw_output_size() 
                  << " each dimension " << reply.raw_output(0).size() << std::endl;
        return 0;
    }
    
    int8_t QueryModelOffline(const std::string& modelname, const QueryOfflineRequest& request,
                            std::unique_ptr<RedisMetadata>& rmd, std::unique_ptr<S3Client>& s3c) {
        std::string inputurl = bucketprefix + request.input_url();
        std::string outputurl = bucketprefix + request.output_url();
        std::string submitter = request.submitter();
        
        // NOTE: one level deeper for the folder.
        std::string localinstancename = modelname + "_" + submitter;
        std::string localinput = local_input_dir + "/" + localinstancename + "/" + local_input_leaf_dir;
        std::string localoutput = local_output_dir + "/" + localinstancename;
        
        // Create input and output directory.
        auto res = createdir(localoutput);
        if (res != 0) {
            std::cerr << "Failed to create local output directory " << localoutput 
                      << " errno " << errno << std::endl;
            return -1;
        }
        res = createdir(localinput);
        if (res != 0) {
            std::cerr << "Failed to create local input directory " << localinput << std::endl;
            return -1;
        }
        
        // Fix/pad the folder names.
        if (inputurl.back() != '/') inputurl += '/';
        if (outputurl.back() != '/') outputurl += '/';
        
        printf("Process offline request... input %s, output %s, model %s, submitter %s.\n",
               inputurl.c_str(), outputurl.c_str(), modelname.c_str(), submitter.c_str());
        printf("Local input dir %s local output dir %s\n", localinput.c_str(), localoutput.c_str());
        fflush(stdout);
        
        // 0. Load model if not loaded. Need to wait until model is loaded.
        int numtry = 0, maxtry = 60;
        int sleepinterval = 500;  // Sleep 500 ms.
        while (numtry < maxtry) {
            if (modeltonamesoffline.find(modelname) != modeltonamesoffline.end() &&
                modeltonamesoffline[modelname].size() > 0) {
                break;
            }
            
            std::string srcurl = bucketprefix + infaas_buckets_dir + modelname;
            auto res = LoadModel(srcurl, modelname, rmd, s3c, modelname, false);
            if (res == 0) {
                break;
            } else if (res == 1) {
                std::cout << "Model " << modelname << " is being loaded by others but not available" << std::endl;
            } else {
                std::cerr << "CPU Manager QueryModelOffline - Failed to load model " << modelname << std::endl;
                return -1;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepinterval));
            numtry++;
        }
        
        // Schedule in a round robin way
        std::string instancename;
        {
            std::lock_guard<std::mutex> lock(updatemutex);
            instancename = modeltonamesoffline[modelname].front();
            modeltonamesoffline[modelname].pop_front();
            modeltonamesoffline[modelname].push_back(instancename);
        }
        
        int portnum = nametoport[instancename];
        std::cout << "Offline serve with instance " << instancename << " port number " 
                  << std::to_string(portnum) << std::endl;
        
        
        grpc::ChannelArguments arguments;
        arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
        arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);

	// PNB: (2025.12.27)
	Address destaddr;
	destaddr.ip = "localhost";
	destaddr.port = std::to_string(portnum);
	
	std::string addr = destaddr.ip + ":" + destaddr.port;
        std::unique_ptr<Query::Stub> stub = Query::NewStub(
            grpc::CreateCustomChannel(addr, grpc::InsecureChannelCredentials(), arguments));
        
        QueryOfflineRequest containerrequest;
        QueryOfflineResponse reply;
        containerrequest.set_input_url(localinstancename);
        containerrequest.add_model(modelname);
        containerrequest.set_output_url(localinstancename);
        containerrequest.mutable_slo()->CopyFrom(request.slo());
        containerrequest.set_submitter(submitter);
        
        // 1. Get the names of all input files (just file names, not full path).
        std::string objname, srcbucket;
        std::vector<std::string> inputnames;
        parse_s3_url(inputurl, srcbucket, objname);
        std::cout << "Offline srcbucket " << srcbucket << ", objname " << objname << std::endl;
        
        if (list_s3_path(srcbucket, objname, s3c, inputnames) != 0) {
            std::cerr << "Offline: Failed to list input bucket." << std::endl;
            return -1;
        }
        
        // 2. Process one batch at a time.
        int totalnum = inputnames.size();
        for (int iter = 0; iter < totalnum; iter += offline_batch) {
            int batch = std::min(iter + offline_batch, totalnum) - iter;
            
            // Download batch files from the bucket.
            if (download_s3_local(srcbucket, objname, inputnames.begin() + iter,
                                inputnames.begin() + iter + batch, localinput, s3c) != 0) {
                std::cerr << "Failed to download batch " << iter << std::endl;
                return -1;
            }
            
            // Execute: Wait until the CPU util is below the threshold
            if (OFFLINE_CONTROL) {
                double cpuutil = rmd->get_cpu_util(workername);
                std::cout << "[common_model_util.cc] cpu util from offline " << cpuutil << std::endl;
                bool hasblacklisted = false;  // CommonModelUtilHasBlacklisted();
                
                while (cpuutil > offline_cpu_thresh || hasblacklisted) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
                    cpuutil = rmd->get_cpu_util(workername);
                    hasblacklisted = false;
                    std::cout << "[common_model_util.cc] try again, cpu util from offline " 
                              << cpuutil << " blacklisted " << hasblacklisted << std::endl;
                }
            } else {
                std::cout << "No control over offline, run anyway." << std::endl;
            }
            
            ClientContext context;  // context should not be reused!
            Status status = stub->QueryOffline(&context, containerrequest, &reply);
            if (!status.ok() || !reply.status().status() == InfaasRequestStatusEnum::SUCCESS) {
                std::cerr << "Offline Execution failed RPC " << status.error_message() 
                          << " INFaaS " << reply.status().msg() << std::endl;
            }
        }
        
        // Upload output directory
        std::vector<std::string> outputnames;
        std::string command, retstr;
        std::string outobjname, outsrcbucket;
        parse_s3_url(outputurl, outsrcbucket, outobjname);
        std::cout << "Offline output bucket " << outsrcbucket << ", outobjname " << outobjname << std::endl;
        
        if (list_local_path(localoutput, outputnames) != 0) {
            std::cerr << "Failed to list local output directory " << localoutput << std::endl;
        } else {
            // Upload to S3.
            if (upload_local_s3(localoutput, outputnames, outsrcbucket, outobjname, s3c) != 0) {
                std::cerr << "Failed to upload to " << outputurl << std::endl;
            }
        }
        
        command = "rm -r " + localoutput;
        retstr = exec_cmd(command.c_str());
        if (!retstr.empty()) {
            std::cerr << "Command " << command << " returned error " << retstr << std::endl;
        }
        
        // Cleanup input directory for the next batch.
        command = "rm -r " + localinput;
        retstr = exec_cmd(command.c_str());
        if (!retstr.empty()) {
            std::cerr << "Command " << command << " returned error " << retstr << std::endl;
        }
        
        // Remove input/output directories
        if (rmdir(localoutput.c_str()) != 0) {
            std::cerr << "Failed to remove localoutput" << std::endl;
            return -1;
        }
        std::string localinputdir = local_input_dir + "/" + localinstancename;
        if (rmdir(localinputdir.c_str()) != 0) {
            std::cerr << "Failed to remove " << localinput << std::endl;
            return -1;
        }
        std::cout << "Removed input and output directories" << std::endl;
        
        // Unload the offline model.
        res = UnloadModel(modelname, rmd, modelname, false);
        if (res != 0) {
            std::cerr << "Failed to unload offline model " << modelname << std::endl;
            return -1;
        }
        return 0;
    }
}

// ================================================
// INFA MODEL MANAGER (Inferentia) - SIMPLIFIED
// ================================================
#ifdef INFAAS_NEURON_WORKER
namespace InfaModelManager {
    std::map<std::string, std::deque<std::string>> modeltonames;
    std::map<std::string, int> nametoport;
    std::set<int, std::greater<int>> usedports;
    std::mutex updatemutex;
    int usedcores = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> unirand(0, 1000);
    
    size_t numReplicas(const std::string& modelname) {
        return modeltonames[modelname].size();
    }
    
    int numUsedCores() {
        return usedcores;
    }
    
    int numRequiredCores(const std::string& modelname) {
        int neuronspercontainer = 1;
        size_t rpos = modelname.rfind('_');
        if (rpos != std::string::npos) {
            std::string numstr = modelname.substr(rpos + 1);
            size_t endpos = numstr.find_first_not_of("0123456789");
            if (endpos != std::string::npos) numstr = numstr.substr(0, endpos);
            if (!numstr.empty()) neuronspercontainer = std::stoi(numstr);
        }
        return neuronspercontainer;
    }
    
    int cleanLoadFailure(const std::string& modelname, const std::string& instancename) {
        int neuronspercontainer = numRequiredCores(modelname);
        {
            std::lock_guard<std::mutex> lock(updatemutex);
            auto it = nametoport.find(instancename);
            if (it == nametoport.end()) {
                std::cout << "Model " << instancename << " has already unloaded." << std::endl;
                return -1;
            }
            int portnum = it->second;
            nametoport.erase(it);
            usedports.erase(portnum);
            usedcores -= neuronspercontainer;
        }
        return -1;
    }
    
    // Add other InfaModelManager functions here following similar pattern...
}
#endif

// ================================================
// GPU MODEL MANAGER STUB (for completeness)
// ================================================
#ifdef INFAAS_GPU_WORKER
int8_t QueryGeneralModel(const std::string& modelname, const QueryOnlineRequest& request,
                        QueryOnlineResponse& reply) {
    // GPU TRTIS implementation would go here
    // For now, stub returns success
    return 0;
}
#endif

// ================================================
// BLACKLIST CHECKER STUB
// ================================================
bool CommonModelUtilHasBlacklisted() {
    return false;
}
