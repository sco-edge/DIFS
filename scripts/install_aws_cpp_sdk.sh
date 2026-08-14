#!/bin/bash

#PNB: This outermost if..fi block was added to ensure that this code here doesn't run as we are not in local mode (Fri 21 Nov 21:13:09 KST 2025); It only runs when we reactivate all AWS dependent lines which have been commented in the code

if [[ "$LOCAL_MODE" != "1" ]]; then

set -x
set -e

cmake_inst_path="/opt/INFaaS/thirdparty/aws-sdk-cpp"
if [ $# -ne 1 ]; then
  echo "Using "${cmake_inst_path}" for cmake"
else
  cmake_inst_path=$1
fi

# Need to install higher version cmake
install_path="${HOME}/aws-sdk-cpp"
mkdir -p ${install_path}
cd ${install_path}

cmake -DCMAKE_BUILD_TYPE=Release  ${cmake_inst_path}
make -j8
sudo make install

fi
