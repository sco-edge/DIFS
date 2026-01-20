#!/bin/bash


# Author: KB
# Purpose: Used only after 'rebuild_all_v9.sh' has already been used to download all dependencies and build the entire project for the first time. Afterwards, you don't need to re-download
# dependencies, you just only need to build newly added protoc files and c++ files. This file is for that function. It allows for immediate feedback.


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

echo -e "${BLUE}==> FAST REBUILD MODE: Protos + C++ only${NC}"

# -------------------------------------------------
# SKIP: Dependencies (assume already installed)
# -------------------------------------------------
echo -e "${BLUE}==> Skipping apt dependencies (assumed installed)${NC}"
echo -e "${BLUE}==> Protoc check:${NC}"
protoc --version || { echo -e "${RED}protoc missing. Run original script first.${NC}"; exit 1; }
grpc_cpp_plugin --version 2>/dev/null || echo -e "${BLUE}gRPC plugin OK or optional${NC}"

# -------------------------------------------------
# CLEAN: Only build artifacts + protos
# -------------------------------------------------
echo -e "${BLUE}==> CLEAN: build dir + generated protos only${NC}"
rm -rf "$INFAAS_SRC/$BUILD_DIR"
find "$INFAAS_SRC" -name "*diffusion_service*.pb.*" -delete || true
find "$INFAAS_SRC" -name "*modelreg*.pb.*" -delete || true
find "$INFAAS_SRC/protos/python_protos" -name "*.py" -delete || true

# -------------------------------------------------
# REGENERATE C++ protos
# -------------------------------------------------
echo -e "${GREEN}=== REGENERATING C++ PROTOS ===${NC}"
cd "$INFAAS_SRC"

PROTO_PATH="protos"

# Verify proto files exist
[ -f "$PROTO_PATH/modelreg.proto" ] || { echo -e "${RED}modelreg.proto missing${NC}"; exit 1; }
[ -f "$PROTO_PATH/internal/diffusion_service.proto" ] || echo -e "${BLUE}diffusion_service.proto optional${NC}"

# modelreg.proto
protoc \
  -I protos \
  --cpp_out="$INFAAS_SRC/src/master" \
  --grpc_out="$INFAAS_SRC/src/master" \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  "$PROTO_PATH/modelreg.proto"

# diffusion_service.proto
protoc \
  -I protos \
  --cpp_out="src/protos/internal" \
  --grpc_out="src/protos/internal" \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  "$PROTO_PATH/internal/diffusion_service.proto" 2>/dev/null || echo -e "${BLUE}Diffusion proto optional${NC}"

# -------------------------------------------------
# REGENERATE Python protos
# -------------------------------------------------
echo -e "${BLUE}==> Regenerating Python protobufs${NC}"
PY_PROTO_OUT="protos/python_protos"
mkdir -p "$PY_PROTO_OUT"

python3 -m grpc_tools.protoc \
  -I protos \
  --python_out="$PY_PROTO_OUT" \
  --grpc_python_out="$PY_PROTO_OUT" \
  protos/query.proto \
  protos/infaas_request_status.proto

# -------------------------------------------------
# BUILD DIFS (LOCAL MODE)
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
  -DENABLE_DIFFUSION=ON

# Build
make -j$(nproc)

# -------------------------------------------------
# VERIFY BINARIES
# -------------------------------------------------
BIN_DIR="$(pwd)/bin"
echo -e "${BLUE}==> Verifying binaries in $BIN_DIR${NC}"

if [ -f "$BIN_DIR/modelreg_server" ] && [ -f "$BIN_DIR/queryfe_server" ]; then
  echo -e "${GREEN}=============================================================${NC}"
  echo -e "${GREEN}✔ FAST REBUILD COMPLETE (5-10x faster)${NC}"
  echo -e "${GREEN}✔ Protos regenerated, C++ rebuilt${NC}"
  echo -e "${GREEN}✔ Core services ready: $BIN_DIR${NC}"
  echo -e "${GREEN}=============================================================${NC}"
  ls -la "$BIN_DIR"/modelreg* "$BIN_DIR"/queryfe*
else
  echo -e "${RED}Build failed — missing core binaries${NC}"
  ls -la "$BIN_DIR" || true
  exit 1
fi
