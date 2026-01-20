#!/bin/bash

# Author: KB
# Purpose: For downloading all dependencies and building the entire project

set -e
set -o pipefail

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PROTOBUF_VERSION="v3.21.12"
GRPC_VERSION="v1.48.0"
INSTALL_PREFIX="/usr/local"
PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS
BUILD_DIR=build

export PATH="$INSTALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# Colors
GREEN="\033[1;32m"
RED="\033[1;31m"
BLUE="\033[1;34m"
NC="\033[0m"

trap 'echo -e "${RED}ERROR occurred on line $LINENO${NC}"' ERR

# -------------------------------------------------
echo -e "${BLUE}==> Installing required build dependencies...${NC}"
sudo apt-get update -y
sudo apt-get install -y \
    build-essential cmake git autoconf libtool pkg-config \
    zlib1g-dev libssl-dev libcurl4-openssl-dev libc-ares-dev python3

# -------------------------------------------------
echo -e "${BLUE}==> HARD CLEAN: removing build + generated protobuf artifacts${NC}"

rm -rf "$PROTOBUF_SRC" "$GRPC_SRC" "$INFAAS_SRC/$BUILD_DIR"

# 🔥 Critical: remove any stale generated diffusion proto files
find "$INFAAS_SRC" -name "*diffusion_service*.pb.*" -delete || true

# -------------------------------------------------
echo -e "${RED}==> Removing ALL old Protobuf/gRPC installations${NC}"
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
git clone https://github.com/protocolbuffers/protobuf.git "$PROTOBUF_SRC"
cd "$PROTOBUF_SRC"
git checkout "$PROTOBUF_VERSION"
git submodule update --init --recursive

cd cmake
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)
sudo make install
sudo ldconfig
echo -e "${GREEN}✔ Protobuf installed${NC}"
protoc --version

# -------------------------------------------------
# BUILD gRPC
# -------------------------------------------------
echo -e "${GREEN}=== BUILDING gRPC $GRPC_VERSION ===${NC}"
git clone --recurse-submodules https://github.com/grpc/grpc "$GRPC_SRC"
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
    -DgRPC_ABSL_PROVIDER=module \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)
sudo make install
sudo ldconfig
echo -e "${GREEN}✔ gRPC installed${NC}"

# --------------------------------------------------------------------
# REGENERATE C++ protos; modelreg.proto ONLY (no diffusion, no TRTIS)
# --------------------------------------------------------------------
echo -e "${BLUE}==> Regenerating modelreg.proto only${NC}"

cd "$INFAAS_SRC"

# PROTO_PATH="$INFAAS_SRC/protos"
PROTO_PATH="protos"

if [ ! -f "$PROTO_PATH/modelreg.proto" ]; then
    echo -e "${RED}Error: modelreg.proto not found at $PROTO_PATH${NC}"
    exit 1
fi

# PNB: for cpp base protoc compilation  (2026.01.06)
protoc \
  -I protos \
  --cpp_out="$INFAAS_SRC/src/master" \
  --grpc_out="$INFAAS_SRC/src/master" \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  "$PROTO_PATH/modelreg.proto"



# --------------------------------------------------------------------
# REGENERATE C++ protos; now I add in diffusion (2026.01.10)
# --------------------------------------------------------------------

# PNB: for cpp base protoc compilation  (2026.01.10)
protoc \
  -I protos \
  --cpp_out="src/protos/internal" \
  --grpc_out="src/protos/internal" \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  "protos/internal/diffusion_service.proto"




# -------------------------------------------------
# REGENERATE Python protos (query + status)
# -------------------------------------------------

# PNB: for python base protoc compilation  (2026.01.06)

# python -m grpc_tools.protoc  -I=protos   \
#        --python_out="$PROTO_PATH/python_protos" \
#        --grpc_python_out="$PROTO_PATH/python_protos" \
#        "$PROTO_PATH/query.proto" "$PROTO_PATH/infaas_request_status.proto"

echo -e "${BLUE}==> Regenerating Python protobufs${NC}"

cd "$INFAAS_SRC"

PY_PROTO_OUT="protos/python_protos"
mkdir -p "$PY_PROTO_OUT"

python3 -m grpc_tools.protoc \
  -I protos \
  --python_out="$PY_PROTO_OUT" \
  --grpc_python_out="$PY_PROTO_OUT" \
  protos/query.proto \
  protos/infaas_request_status.proto

# -------------------------------------------------
# BUILD DIFS (LOCAL MODE, NO TRTIS, NO DIFFUSION)
# -------------------------------------------------
echo -e "${GREEN}=== BUILDING DIFS (LOCAL MODE) ===${NC}"
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
    -DENABLE_AWS_AUTOSCALING=OFF \
    -DENABLE_TRTIS=OFF \
    -DBUILD_TRTIS_PROTO=OFF \
    -DBUILD_TRITON_PROTO=OFF \
    -DENABLE_DIFFUSION=ON # PNB Turn on code to build difussion (2026.01.10)

# First pass (may show harmless parallel error)
make -j$(nproc) || true

# 🔁 Mandatory reconfigure to flush stale proto deps
cmake .

# Final build
make -j$(nproc) || true

# -------------------------------------------------
# VERIFY REQUIRED BINARIES
# -------------------------------------------------
BIN_DIR="$(pwd)/bin"
echo -e "${BLUE}==> Verifying binaries in $BIN_DIR${NC}"

if [ -f "$BIN_DIR/modelreg_server" ] && \
   [ -f "$BIN_DIR/queryfe_server" ]; then

    echo -e "${GREEN}=============================================================${NC}"
    echo -e "${GREEN}✔ DIFS LOCAL BUILD SUCCESSFUL${NC}"
    echo -e "${GREEN}✔ Diffusion + TRTIS fully disabled${NC}"
    echo -e "${GREEN}✔ Core services ready${NC}"
    echo -e "${GREEN}=============================================================${NC}"
else
    echo -e "${RED}Build failed — missing core binaries${NC}"
    ls -la "$BIN_DIR"
    exit 1
fi
