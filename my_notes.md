# Author: KB
# Purpose: To track activities and codes run and why. 

* I renamed my repository from INFaaS to diffusion model inference serving (DIFS)(TIME: Wed  3 Dec 23:07:35 KST 2025)


* _INFaaS Codebase RE(RE) and code analysis_
| Activity                              | Reason                                          | Related Code(s) |
|---------------------------------------+-------------------------------------------------+-----------------|
| I tried to use the "Sourcetrail" to   | I eventually stopped                            |                 |
| analyse the codebase;                 | that line of action because I wasn't conversant |                 |
|                                       | with the tool. Later after I got the intial     |                 |
|                                       | code to work on a local machine. I revisted a   |                 |
|                                       | tutorial to get use to the tool.                |                 |
|                                       |                                                 |                 |
|                                       |                                                 |                 |
| Created "download_supported_model.py" | INFaaS didn't come with any of their models     |                 |
|                                       | so I downloaded similar models using the python |                 |
|                                       | script to dry run the INFaaS github code        |                 |


* _Versions and Why they were saved_
| File                                             | Date    |  Time | Reason                                                         |
|--------------------------------------------------+---------+-------+----------------------------------------------------------------|
| ../INFaaS_DIFS_BkUps/INFaaS_BkUps/INFaaS_BkUp_v1 | Nov  21 | 13:30 | I was going to start editting the working distributed version  |
|                                                  |         |       | of INFaaS into a local copy so I needed a backup               |
| ../INFaaS_DIFS_BkUps/INFaaS_BkUps/INFaaS_BkUp_v2 | Nov  29 | 00:51 |                                                                |
|--------------------------------------------------+---------+-------+----------------------------------------------------------------|
| ../INFaaS_DIFS_BkUps/INFaaS_BkUps/INFaaS_BkUp_v3 | Dec   1 | 17:11 | This was to backup the finally working local version of INFaaS |
|--------------------------------------------------+---------+-------+----------------------------------------------------------------|
| ../INFaaS_DIFS_BkUps/DIFS_BkUps/DIFS_BkUp_v2/    | Dec 18  | 15:53 | Backed up because I intend to integrate the Stable Diffusion   |
|                                                  |         |       | Model into DIFS                                                |


* AWS CHEATSHEET
| Use                                                       | Command                                                                                                                                                                                                                           |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To find my AWS account ID (which identifies my AWS        | aws sts get-caller-identity                                                                                                                                                                                                       |
| account globally)                                         |                                                                                                                                                                                                                                   |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To find my region                                         | aws configure get region                                                                                                                                                                                                          |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To change my region                                       | aws configure set region us-west-2                                                                                                                                                                                                |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To copy AMI to another region                             | aws ec2 copy-image (e.g. aws ec2 copy-image   --source-region us-east-1  --source-image-id ami-0123456789abcdef0   --region us-west-2   --name "INFaaS-Worker-Copy"   --description "Copy of INFaaS worker image from us-east-1") |
|                                                           |                                                                                                                                                                                                                                   |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To find my IAM username                                   | aws iam get-user                                                                                                                                                                                                                  |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                           |                                                                                                                                                                                                                                   |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To SSH into the AMI machine that I launch on AWS from my  | ssh -i "my_security/infaas_instance1_key1.pem" ubuntu@3.106.137.23                                                                                                                                                                |
| local machine                                             |                                                                                                                                                                                                                                   |
|                                                           | _NB: details after creating and lauching AMI_                                                                                                                                                                                     |
|                                                           | type of base image used in launch: ubuntu                                                                                                                                                                                         |
|                                                           | public key: infaas_intance1_key1.pem  (I change the permissions i.e 'chmod 400 infaas_instance1_key1.pem' in order to use it for ssh)                                                                                             |
|                                                           | public IP: 3.106.137.23                                                                                                                                                                                                           |
|-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To copy the 'start_infaas.sh' file from the local machine | scp -i "my_security/infaas_instance1_key1.pem" \                                                                                                                                                                                  |
| to the AMI machine that I have lauched on AWS via SSH     | start_infaas.sh ubuntu@3.106.137.23:/home/ubuntu/WORK/PROGRAMMING_WORLD/RESEARCH_PROJECTS/Templates/INFaaS                                                                                                                        |
|                                                           |                                                                                                                                                                                                                                   |
|                                                           |                                                                                                                                                                                                                                   |
|                                                           |                                                                                                                                                                                                                                   |




* _CHANGES TO MAKE IT WORK ON LOCAL MACHINE_
** To get rid of AWS support and S3 dependent code
*** CODE REPLACEMENT/ADDITION
#+begin_src c++
    // void parse_s3_url(const std::string& src_url, std::string* src_bucket,
  //                   std::string* obj_name) {
  //   // *obj_name = src_url.substr(bucket_prefix.size());
  //   *obj_name = src_url;
  //   auto pre_ind = obj_name->find("/");  // exclude bucket name
  //   *src_bucket = obj_name->substr(0, pre_ind);
  //   *obj_name = obj_name->substr(pre_ind + 1);
  // }

  # PNB: Replace the immediate above commented section with this in order to remove AWS SDK & S3 support(2025.11.21)
  void parse_s3_url(const std::string& src_url, std::string* src_bucket,
                    std::string* obj_name) {
    // LOCAL MODE: Treat entire URL as local file path
    // Supports: file:///path/to/file.jpg  OR  /path/to/file.jpg
    std::string path = src_url;

    // Handle file:// prefix (optional)
    if (path.find("file://") == 0) {
      path = path.substr(7);  // Remove "file://"
    }

    ,*src_bucket = "";           // Unused in local mode
    ,*obj_name = path;           // Full local path
    std::cout << "[LOCAL MODE] Reading input from: " << path << std::endl;
  }
#+end_src


**** Added the following line to include/constants.h.templ file
#+begin_src c++
  #define master_ip "127.0.0.1"
  #define modelreg_port "50051"
  #define queryfe_port "50052"
  #define redis_host "127.0.0.1"
  #define redis_port 6379
#+end_src

**** Although '#include "include/constants.h"' is in the queryfe_server.cc file, no folder and it's file existed like that. I created both the folder and the file 
**** to enable me set the local mode flag and to make parsing smarter to queryfe_server.cc
**** Later changed the commenting out of the original "parse_s3_url(...)" function and its redefinition by using an if statement to
**** check for the setting of the "LOCAL_MODE" flag and relying on that to activate either the function in "local mode" or "AWS mode (i.e., distributed mode)"

*** DELETED/COMMENTED LINES
# file: src/master/queryfe_server.cc
#+begin_src C++
  #include <aws/core/Aws.h>
  #include <aws/s3/S3Client.h>
  #include <aws/s3/model/ListObjectsV2Request.h>
#+end_src

**** Deleted all occurrences of 'AWS::' in all files (e.g. files within the master and worker folders which matched that description)
**** The files that received this particular edit were: master_vm_daemon.cc, queryfe_server.cc, modelreg_server.cc, query_executor.cc, and query_test.cc
**** put a global if..fi block arount the script in 'scripts/install_aws_cpp_sdk' to ensure that it does not ever run as we will be in Local mode

*** INSTALLED
**** sudo apt install protobuf-compiler libprotobuf-dev 
**** sudo apt install libhiredis-dev
**** sudo apt install libev-dev
**** sudo apt install libcurl4-openssl-dev
**** INSTALLING OPENCV 2.4
git clone https://github.com/opencv/opencv.git --branch 2.4 --single-branch opencv-2.4
cd opencv-2.4
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. (or "cmake -DWITH_CUDA=OFF -DWITH_CUBLAS=OFF -DWITH_CUFFT=OFF .." if CUDA is causing issues)
make -j$(nproc)
sudo make install
**** INSTALLING AWS SDK for C++
- SDK is often not packaged in older versions so the best approach is to build from source:
  sudo apt install cmake libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev git build-essential
- Clone AWS SDK:
  cd ~
  git clone https://github.com/aws/aws-sdk-cpp.git
  cd aws-sdk-cpp
  git submodule update --init --recursive

* _DEBUGGING ACTIONS_
** MAIN ERROR SOURCES
- Libraries: Protobuf, gRPC
** ACTIONS
- I have crated a series of files bearing the name formats 'rebuild_all_v<version number>.sh' to handle the
  series of corrections that I have attempted
*** _Diffusion Model Integration (2025.12.18)_
- PROTOBUF RELATED:
  - Created new file "protos/infaas_query.proto" to handle implement the query interface
-
  
** POSSIBLE QUESTIONS TO ATTEMPT TO RESOLVE THE PROBLEMS
- Is it possible for INFaaS to work if one of the problematic libraries is eliminated?
- How can we get compactible versions online that don't keep making the problem recur?
- Is there are issue in our altered code that is causing the problem to occur?

* _KEY ABBREVIATIONS_
EC2: Elastic Computing Cloud
S3: Simple Storage Service
