#!/bin/bash
set -e
set -o pipefail

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

PROTOBUF_VERSION="v3.12.4"

#GRPC_VERSION="v1.32.0"
GRPC_VERSION="v1.28.3"

INSTALL_PREFIX="/usr/local"

PROTOBUF_SRC=~/protobuf
GRPC_SRC=~/grpc
INFAAS_SRC=~/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/INFaaS
BUILD_DIR=build

export PATH="$INSTALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# -------------------------------------------------
# STEP 0a: CLEAN OLD BUILD DIRECTORIES
# -------------------------------------------------
echo "==> Cleaning old build directories..."
rm -rf ~/protobuf ~/grpc "$INFAAS_SRC/$BUILD_DIR"

# -------------------------------------------------
# STEP 0b: REMOVE OLD INSTALLATIONS
# -------------------------------------------------
echo "==> Removing old Protobuf, gRPC, and Abseil from /usr/local..."

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
# STEP 1: BUILD PROTOBUF v3.12.4 (AUTOTOOLS BUILD)
# -------------------------------------------------
echo ""
echo "======================================="
echo " BUILDING PROTOBUF $PROTOBUF_VERSION"
echo "======================================="

cd ~
git clone https://github.com/protocolbuffers/protobuf.git "$PROTOBUF_SRC"
cd "$PROTOBUF_SRC"

git fetch --all --tags
git checkout $PROTOBUF_VERSION
git submodule update --init --recursive

echo "==> Running autotools build (correct for 3.12.x)..."
./autogen.sh
./configure --prefix="$INSTALL_PREFIX" CXXFLAGS="-fPIC"

make -j$(nproc)
sudo make install
sudo ldconfig

# -------------------------------------------------
# STEP 2: BUILD gRPC v1.32.0 (CMAKE BUILD)
# -------------------------------------------------
echo ""
echo "======================================="
echo " BUILDING gRPC $GRPC_VERSION"
echo "======================================="

cd ~
git clone --recurse-submodules https://github.com/grpc/grpc.git grpc
cd grpc
git checkout "$GRPC_VERSION"
git submodule update --init --recursive

mkdir -p build && cd build

echo "==> Removing -Werror flags..."
sed -i 's/-Werror//g' third_party/boringssl-with-bazel/CMakeLists.txt || true
sed -i 's/-Werror//g' CMakeLists.txt || true
sed -i 's/-Werror//g' cmake/*.cmake || true


cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_ZLIB_PROVIDER=package \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)
sudo make install
sudo ldconfig

# -------------------------------------------------
# STEP 3: BUILD INFaaS
# -------------------------------------------------
echo ""
echo "======================================="
echo " BUILDING INFaaS"
echo "======================================="

cd "$INFAAS_SRC"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR && cd $BUILD_DIR

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    -DProtobuf_DIR="$INSTALL_PREFIX/lib/cmake/protobuf" \
    -DgRPC_DIR="$INSTALL_PREFIX/lib/cmake/grpc" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)

echo ""
echo "======================================="
echo " ✅ BUILD COMPLETED SUCCESSFULLY"
echo "======================================="
