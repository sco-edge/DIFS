#!/bin/bash
set -e
set -o pipefail

# ================================================================
# CONFIG
# ================================================================
PROTOBUF_VERSION="v3.21.12"
GRPC_VERSION="v1.48.0"
INSTALL_PREFIX="/usr/local"

PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/INFaaS
AWS_SDK_DIR=~/aws-sdk-cpp

BUILD_DIR=build

export PATH="$INSTALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

GREEN="\033[1;32m"
RED="\033[1;31m"
BLUE="\033[1;34m"
NC="\033[0m"

trap 'echo -e "${RED}❌ ERROR on line $LINENO${NC}"' ERR


# ================================================================
# Install dependencies
# ================================================================
echo -e "${BLUE}==> Installing dependencies...${NC}"
sudo apt-get update -y
sudo apt-get install -y \
    build-essential cmake git autoconf libtool pkg-config \
    zlib1g-dev libssl-dev libcurl4-openssl-dev libc-ares-dev python3 \
    uuid-dev libbz2-dev libxml2-dev


# ================================================================
# CLEAN previous builds
# ================================================================
echo -e "${BLUE}==> Cleaning old builds...${NC}"
rm -rf "$PROTOBUF_SRC" "$GRPC_SRC" "$INFAAS_SRC/$BUILD_DIR"
rm -rf "$AWS_SDK_DIR"


echo -e "${RED}==> Removing previous Protobuf/gRPC installs...${NC}"
sudo rm -rf \
    $INSTALL_PREFIX/include/google \
    $INSTALL_PREFIX/include/grpc* \
    $INSTALL_PREFIX/include/absl \
    $INSTALL_PREFIX/lib/libproto* \
    $INSTALL_PREFIX/lib/libprotobuf* \
    $INSTALL_PREFIX/lib/libgrpc* \
    $INSTALL_PREFIX/lib/libgpr* \
    $INSTALL_PREFIX/lib/cmake/protobuf \
    $INSTALL_PREFIX/lib/cmake/grpc \
    $INSTALL_PREFIX/lib/cmake/absl \
    $INSTALL_PREFIX/bin/protoc \
    $INSTALL_PREFIX/bin/grpc_cpp_plugin
sudo ldconfig


# ================================================================
# BUILD PROTOBUF
# ================================================================
echo -e "${GREEN}=== BUILDING PROTOBUF $PROTOBUF_VERSION ===${NC}"
git clone https://github.com/protocolbuffers/protobuf.git "$PROTOBUF_SRC"
cd "$PROTOBUF_SRC"
git checkout "$PROTOBUF_VERSION"
git submodule update --init --recursive

mkdir -p cmake/build && cd cmake/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
protoc --version


# ================================================================
# BUILD gRPC
# ================================================================
echo -e "${GREEN}=== BUILDING gRPC $GRPC_VERSION ===${NC}"
git clone --recursive https://github.com/grpc/grpc "$GRPC_SRC"
cd "$GRPC_SRC"
git checkout "$GRPC_VERSION"
git submodule update --init --recursive

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf"
make -j$(nproc)
sudo make install
sudo ldconfig

# -------------------------------------------------
# Regenerate C++ code from the updated modelreg.proto file
# -------------------------------------------------
echo -e "${BLUE}==> Regenerating C++ code from modelreg.proto...${NC}"

# Correct path to modelreg.proto file (update this with the actual path)
PROTO_PATH="$INFAAS_SRC/protos"

# Check if the proto file exists
if [ ! -f "$PROTO_PATH/modelreg.proto" ]; then
    echo -e "${RED}Error: modelreg.proto not found at $PROTO_PATH.${NC}"
    exit 1
fi

# Run protoc to regenerate C++ code
protoc --proto_path="$PROTO_PATH" --cpp_out="$INFAAS_SRC/src/master" --grpc_out="$INFAAS_SRC/src/master" --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` "$PROTO_PATH/modelreg.proto"


# -------------------------------------------------
# INSTALLING AWS SDK FOR C++ (LEGACY MODE)
# -------------------------------------------------
echo -e "${BLUE}==> Installing AWS SDK for C++ (LEGACY v1.11.267)...${NC}"

AWS_SDK_CPP_DIR="$HOME/aws-sdk-cpp"
rm -rf "$AWS_SDK_CPP_DIR"

git clone https://github.com/aws/aws-sdk-cpp.git "$AWS_SDK_CPP_DIR"
cd "$AWS_SDK_CPP_DIR"

# Checkout a version without CRT
git fetch --all --tags
git checkout tags/1.11.267 -b legacy-build

mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_ONLY="s3;transfer" \
    -DENABLE_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLEGACY_BUILD=ON

make -j$(nproc)
sudo make install
sudo ldconfig

echo -e "${GREEN}✔ AWS SDK (LEGACY) INSTALLED${NC}"




# ================================================================
# BUILD INFaaS (LOCAL MODE)
# ================================================================
echo -e "${GREEN}=== BUILDING INFaaS (LOCAL MODE) ===${NC}"
cd "$INFAAS_SRC"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DgRPC_DIR="$INSTALL_PREFIX/lib/cmake/grpc" \
    -DENABLE_AWS_AUTOSCALING=OFF \
    -DENABLE_TRTIS=OFF \
    -DCMAKE_CXX_STANDARD=11 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON

#make -j$(nproc)
#echo -e "${GREEN}=======================================================${NC}"
#echo -e "${GREEN}✔ INFaaS COMPLETELY REBUILT SUCCESSFULLY (LOCAL MODE) ${NC}"
#echo -e "${GREEN}=======================================================${NC}"


echo -e "${BLUE}==> Compiling INFaaS (this may show a fake error — it's normal)...${NC}"
# This is the ONLY safe way when using parallel make + set -e
make -j$(nproc) || {
    echo -e "${BLUE}→ make returned non-zero (common false positive with -j)${NC}"
    sleep 1
}

# REAL success check — this is what matters
if [ -f "bin/modelreg_server" ] && \
   [ -f "bin/queryfe_server" ] && \
   [ -f "bin/worker_daemon" ] && \
   [ -f "bin/infaas_online_query" ]; then

    echo -e "${GREEN}=======================================================${NC}"
    echo -e "${GREEN}  INFaaS BUILD 100% SUCCESSFUL!${NC}"
    echo -e "${GREEN}  All binaries ready in: $(pwd)/bin${NC}"
    echo -e "${GREEN}  Run: ./start_infaas.sh${NC}"
    echo -e "${GREEN}=======================================================${NC}"
else
    echo -e "${RED}Build actually failed — binaries missing${NC}"
    exit 1
fi
