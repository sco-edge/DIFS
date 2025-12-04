#!/bin/bash
set -e
set -o pipefail

# -----------------------------
# CONFIGURE PATHS
# -----------------------------
PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/INFaaS
INSTALL_PREFIX=/usr/local
BUILD_DIR=build

# -----------------------------
# STEP 1: Remove old installations
# -----------------------------
echo "==> Removing old Protobuf, gRPC, and Abseil installations..."
sudo rm -rf $INSTALL_PREFIX/include/google \
            $INSTALL_PREFIX/include/grpcpp \
            $INSTALL_PREFIX/include/absl
sudo rm -rf $INSTALL_PREFIX/lib/libprotobuf* \
            $INSTALL_PREFIX/lib/libgrpc* \
            $INSTALL_PREFIX/lib/libabsl* \
            $INSTALL_PREFIX/lib/cmake/protobuf \
            $INSTALL_PREFIX/lib/cmake/grpc \
            $INSTALL_PREFIX/lib/cmake/absl
sudo ldconfig

# Clean INFaaS build
echo "==> Cleaning INFaaS build directory..."
cd $INFAAS_SRC
rm -rf $BUILD_DIR
mkdir $BUILD_DIR && cd $BUILD_DIR

# -----------------------------
# STEP 2: Build and install Protobuf
# -----------------------------
echo "==> Building Protobuf..."
cd $PROTOBUF_SRC
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dprotobuf_BUILD_TESTS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

make -j$(nproc)
sudo make install
sudo ldconfig

# -----------------------------
# STEP 3: Build and install gRPC
# -----------------------------
echo "==> Building gRPC..."
cd $GRPC_SRC
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=ON \
  -DProtobuf_DIR=$INSTALL_PREFIX/lib/cmake/protobuf \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

make -j$(nproc)
sudo make install
sudo ldconfig

# -----------------------------
# STEP 4: Build INFaaS
# -----------------------------
echo "==> Building INFaaS..."
cd $INFAAS_SRC/$BUILD_DIR

cmake .. \
  -DProtobuf_DIR=$INSTALL_PREFIX/lib/cmake/protobuf \
  -DgRPC_DIR=$INSTALL_PREFIX/lib/cmake/grpc \
  -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# -----------------------------
# STEP 5: Verify installations
# -----------------------------
echo "==> Verification..."
echo "Protobuf version:"
protoc --version

echo "INFaaS build directory contents:"
ls -1 $INFAAS_SRC/$BUILD_DIR

echo "gRPC include files:"
ls -1 $INSTALL_PREFIX/include/grpcpp

echo "Abseil include files:"
ls -1 $INSTALL_PREFIX/include/absl

echo "==> Build complete! INFaaS should now compile successfully."
