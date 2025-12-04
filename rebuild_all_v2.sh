#!/bin/bash
set -e
set -o pipefail

# -------------------------------------------------
# CONFIGURE VERSIONS
# -------------------------------------------------

# PROTOBUF_VERSION="v5.26.1"
# PROTOBUF_VERSION="v5.29.0"
PROTOBUF_VERSION="v3.12.4"   # Best compatible for INFaaS

# GRPC_VERSION="v1.59.0"
# GRPC_VERSION="v1.65.0"
GRPC_VERSION="v1.32.0"       # Matches protobuf 3.12

INSTALL_PREFIX="/usr/local"

# -------------------------------------------------
# PATHS
# -------------------------------------------------
PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/INFaaS
BUILD_DIR=build

export PATH="$INSTALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# -------------------------------------------------
# CLEANUP OLD BUILDS & INSTALLS
# -------------------------------------------------
echo "==> Cleaning old Protobuf/gRPC/INFaaS builds..."

rm -rf $PROTOBUF_SRC $GRPC_SRC $INFAAS_SRC/$BUILD_DIR

echo "==> Removing previous installs..."
sudo rm -rf \
  $INSTALL_PREFIX/include/google \
  $INSTALL_PREFIX/include/grpc* \
  $INSTALL_PREFIX/include/absl \
  $INSTALL_PREFIX/lib/libprotobuf* \
  $INSTALL_PREFIX/lib/libgrpc* \
  $INSTALL_PREFIX/lib/libabsl* \
  $INSTALL_PREFIX/lib/cmake/protobuf \
  $INSTALL_PREFIX/lib/cmake/grpc \
  $INSTALL_PREFIX/lib/cmake/absl

sudo ldconfig

# -------------------------------------------------
# STEP 1: BUILD PROTOBUF
# -------------------------------------------------
echo "==> Cloning and building Protobuf $PROTOBUF_VERSION"
cd ~
git clone https://github.com/protocolbuffers/protobuf.git $PROTOBUF_SRC
cd $PROTOBUF_SRC
git checkout $PROTOBUF_VERSION
git submodule update --init --recursive

mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)
sudo make install
sudo ldconfig

# -------------------------------------------------
# STEP 2: BUILD gRPC
# -------------------------------------------------
echo "==> Cloning and building gRPC $GRPC_VERSION"
cd ~
git clone --recurse-submodules https://github.com/grpc/grpc.git $GRPC_SRC

cd $GRPC_SRC
git checkout $GRPC_VERSION
git submodule update --init --recursive

mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_INSTALL=ON \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)
sudo make install
sudo ldconfig

# -------------------------------------------------
# STEP 3: BUILD INFaaS
# -------------------------------------------------
echo "==> Building INFaaS..."
cd $INFAAS_SRC
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR && cd $BUILD_DIR

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DgRPC_DIR="$INSTALL_PREFIX/lib/cmake/grpc" \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)

echo "----------------------------------------------------"
echo "   ✅ ALL COMPONENTS BUILT SUCCESSFULLY"
echo "----------------------------------------------------"
