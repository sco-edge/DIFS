#!/bin/bash
set -e
set -o pipefail

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PROTOBUF_VERSION="v3.12.4"
GRPC_VERSION="v1.48.0"
INSTALL_PREFIX="/usr/local"

PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/INFaaS
BUILD_DIR=build

export PATH="$INSTALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# Colors
GREEN="\033[1;32m"
RED="\033[1;31m"
BLUE="\033[1;34m"
NC="\033[0m"

trap 'echo -e "${RED}❌ ERROR occurred on line $LINENO${NC}"' ERR

# -------------------------------------------------
echo -e "${BLUE}==> Cleaning old local build directories...${NC}"
rm -rf "$PROTOBUF_SRC" "$GRPC_SRC" "$INFAAS_SRC/$BUILD_DIR"

# -------------------------------------------------
echo -e "${BLUE}==> Removing old installations from /usr/local...${NC}"
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

# IMPORTANT — Protobuf 3.12.4 uses CMakeLists.txt inside cmake/
cd cmake
mkdir -p build && cd build

echo -e "${BLUE}==> Configuring Protobuf...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo -e "${BLUE}==> Compiling Protobuf...${NC}"
make -j$(nproc)
sudo make install
sudo ldconfig

echo -e "${GREEN}✔ Protobuf installed${NC}"

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

# Force gRPC to use its own Abseil copy (to avoid version mismatches)
echo -e "${BLUE}==> Configuring gRPC...${NC}"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_BUILD_CSHARP_EXT=OFF \
    \
    -DgRPC_ABSL_PROVIDER=module \
    -DgRPC_CARES_PROVIDER=package \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DgRPC_RE2_PROVIDER=package \
    -DgRPC_SSL_PROVIDER=package \
    -DgRPC_ZLIB_PROVIDER=package \
    \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DOPENSSL_ROOT_DIR=/usr \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    \
    -DCMAKE_C_FLAGS="-Wno-stringop-overflow -Wno-stringop-truncation" \
    -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow -Wno-stringop-truncation"


echo -e "${BLUE}==> Compiling gRPC...${NC}"
make -j$(nproc)
sudo make install
sudo ldconfig

echo -e "${GREEN}✔ gRPC installed${NC}"

# -------------------------------------------------
# BUILD INFaaS
# -------------------------------------------------
echo -e "${GREEN}=== BUILDING INFaaS ===${NC}"

cd "$INFAAS_SRC"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DgRPC_DIR="$INSTALL_PREFIX/lib/cmake/grpc" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}✔ BUILD COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}=================================================${NC}"
