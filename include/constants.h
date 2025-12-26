//~ // PNB: Create by me mainly to enable more intelligent managment of the LOCAL_MODE flag with queryfe_server.cc
#include <string>
#ifndef INFAAS_CONSTANTS_H
#define INFAAS_CONSTANTS_H

// Run INFaaS in single-node mode without AWS
//static const bool LOCAL_MODE = true;
#define LOCAL_MODE 1  // Ensure this is defined somewhere before the relevant code

// No AWS S3 bucket prefix in LOCAL mode
static const std::string bucket_prefix = "";

// No S3 bucket for storing models/configs
inline const std::string infaas_buckets_dir = "/tmp/infaas_bucket";

// No S3 bucket for storing models/configs
// static const char* model_location ="/local/models/"
static const char* infaas_models_dir = "/tmp/models"; // (2025.12.01)

// Disable framework auto-selection via AWS metadata
static const char* offline_framework = "LOCAL";

// Local log storage directory used by worker / queryfe / autoscaler
static const std::string infaas_log_dir = "/tmp/infaas_logs";

// Use batch size 1 in LOCAL mode unless manually changed
static const int MAX_ONLINE_BATCH = 1;

// Disable autoscaling (master_vm_daemon removed)
static const bool ENABLE_AUTOSCALING = false;

// For codepaths that expect AWS region/zone
static const char* region = "local";
static const char* zone   = "local-zone-1a";


// Master/controller addresses
static const char* MASTER_IP    = "127.0.0.1";
static const int   MODELREG_PORT = 50051;
static const int   QUERYFE_PORT  = 50052;

// Redis metadata store
static const char* REDIS_HOST = "127.0.0.1";
static const int   REDIS_PORT = 6379;

// Local model/config DB names (if your code uses them)
static const char* MODELDB  = "local_modeldb";
static const char* CONFIGDB = "local_configdb";

#endif

