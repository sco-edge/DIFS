#!/bin/bash
set -e
set -o pipefail

# -------------------------------------------------
# CONFIG (kept identical to your original script)
# -------------------------------------------------
#PROTOBUF_VERSION="v3.12.4"
PROTOBUF_VERSION="v3.21.12"

GRPC_VERSION="v1.48.0"
#GRPC_VERSION="v1.60.0"
INSTALL_PREFIX="/usr/local"

PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/RESEARCH_PROJECTS/Templates/INFaaS
BUILD_DIR=build

export PATH="$INSTALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# Colors (kept same)
GREEN="\033[1;32m"
RED="\033[1;31m"
BLUE="\033[1;34m"
NC="\033[0m"

trap 'echo -e "${RED}❌ ERROR occurred on line $LINENO${NC}"' ERR

# -------------------------------------------------
echo -e "${BLUE}==> Installing required build dependencies...${NC}"
sudo apt-get update -y
sudo apt-get install -y \
    build-essential cmake git autoconf libtool pkg-config \
    zlib1g-dev libssl-dev libcurl4-openssl-dev libc-ares-dev python3

# -------------------------------------------------
echo -e "${BLUE}==> Cleaning old local build directories...${NC}"
rm -rf "$PROTOBUF_SRC" "$GRPC_SRC" "$INFAAS_SRC/$BUILD_DIR"

# -------------------------------------------------
echo -e "${RED}==> Removing ALL old Protobuf/gRPC installations (DESTRUCTIVE CLEAN)...${NC}"
sudo rm -rf \
    $INSTALL_PREFIX/include/google \
    $INSTALL_PREFIX/include/grpc* \
    $INSTALL_PREFIX/include/absl \
    $INSTALL_PREFIX/lib/libprotobuf* \
    $INSTALL_PREFIX/lib/libproto* \
    $INSTALL_PREFIX/lib/libgrpc* \
    $INSTALL_PREFIX/lib/libgpr* \
    $INSTALL_PREFIX/lib/cmake/protobuf \
    $INSTALL_PREFIX/lib/cmake/grpc \
    $INSTALL_PREFIX/lib/cmake/absl \
    $INSTALL_PREFIX/bin/protoc \
    $INSTALL_PREFIX/bin/grpc_cpp_plugin

sudo ldconfig

# -------------------------------------------------
# BUILD PROTOBUF
# -------------------------------------------------
echo -e "${GREEN}=== BUILDING PROTOBUF $PROTOBUF_VERSION ===${NC}"
git config --global advice.detachedHead false

rm -rf "$PROTOBUF_SRC"
git clone https://github.com/protocolbuffers/protobuf.git "$PROTOBUF_SRC"
cd "$PROTOBUF_SRC"

git fetch --all --tags
git checkout "$PROTOBUF_VERSION"
git submodule update --init --recursive

cd cmake
mkdir -p build && cd build

echo -e "${BLUE}==> Configuring Protobuf...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo -e "${BLUE}==> Compiling Protobuf...${NC}"
make -j$(nproc)
sudo make install
sudo ldconfig
echo -e "${GREEN}✔ Protobuf installed${NC}"
protoc --version

# -------------------------------------------------
# BUILD gRPC
# -------------------------------------------------
echo -e "${GREEN}=== BUILDING gRPC $GRPC_VERSION ===${NC}"

rm -rf "$GRPC_SRC"
git clone --recurse-submodules https://github.com/grpc/grpc "$GRPC_SRC"
cd "$GRPC_SRC"

git fetch --all --tags
git checkout "$GRPC_VERSION"
git submodule update --init --recursive

mkdir -p build && cd build

echo -e "${BLUE}==> Configuring gRPC...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DgRPC_ABSL_PROVIDER=module \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo -e "${BLUE}==> Compiling gRPC...${NC}"
make -j$(nproc)
sudo make install
sudo ldconfig

echo -e "${GREEN}✔ gRPC installed${NC}"

# -------------------------------------------------
# BUILD INFaaS
# -------------------------------------------------
echo -e "${BLUE}==> Cleaning previous INFaaS build artifacts...${NC}"
rm -rf "$INFAAS_SRC/$BUILD_DIR"
find "$INFAAS_SRC" -name "*.pb.cc" -o -name "*.pb.h" | xargs -r rm -f

echo -e "${GREEN}=== BUILDING INFaaS ===${NC}"

cd "$INFAAS_SRC"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DgRPC_DIR="$INSTALL_PREFIX/lib/cmake/grpc" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=11 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCURL_INCLUDE_DIR=/usr/include \
    -DCURL_LIBRARY=/usr/lib/x86_64-linux-gnu/libcurl.so

echo -e "${BLUE}==> Compiling INFaaS...${NC}"
make -j$(nproc)

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}✔ BUILD COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}=================================================${NC}"
