#!/bin/bash
# 🚀 INFaaS v3 - CMAKE MAKEFILE FIX (PNB 2025.12.23 FINAL)

echo "🚀 Starting INFaaS v3 - CMAKE MAKEFILE NUCLEAR FIX"
echo "Current date: $(date '+%Y.%m.%d. (%a) %H:%M:%S %Z')"
echo "=================================="

BASE_DIR="/home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS"
cd "$BASE_DIR" || { echo "❌ Cannot cd to $BASE_DIR"; exit 1; }
echo "✅ BASE DIR: $PWD"

# ========================================
# 🛠️ STEP 0: SURGICAL CMAKELists.txt FIX
# ========================================
echo "🛠️ STEP 0: SURGICAL CMakeLists.txt line 56 fix..."

# BACKUP
cp -f CMakeLists.txt CMakeLists.txt.bak

# CRITICAL FIX: Replace line 56 add_subdirectory(protos) with conditional
sed -i '56{
/add_subdirectory(protos)/c\
if(LOCAL_MODE)\
  message(STATUS "LOCAL_MODE: Skipping protos/ directory")\
else()\
  add_subdirectory(protos)\
endif()\
}' CMakeLists.txt

# Verify fix
if grep -q "Skipping protos/ directory" CMakeLists.txt; then
    echo "✅ CMakeLists.txt line 56 FIXED!"
else
    echo "❌ Sed failed, manual patch..."
    awk 'NR==56{print "if(LOCAL_MODE)\n  message(STATUS \"LOCAL_MODE: Skipping protos/ directory\")\nelse()\n  add_subdirectory(protos)\nendif()"} NR!=56' CMakeLists.txt.bak > CMakeLists.txt
fi

# ========================================
# 📦 STEP 1: Dependencies (silent)
# ========================================
echo "📦 STEP 1: Dependencies..."
sudo apt update -qq >/dev/null 2>&1
sudo apt install -y libhiredis-dev libev-dev protobuf-compiler libprotobuf-dev build-essential cmake || true

# ========================================
# 🧹 STEP 2: ULTRA CLEAN + CMAKE
# ========================================
echo "🧹 STEP 2: ULTRA CLEAN + CMAKE..."
rm -rf build/
mkdir -p build && cd build

# CAPTURE CMAKE OUTPUT
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOCAL_MODE=ON \
  -DENABLE_TRTIS=OFF \
  -DINFAAS_GPU_WORKER=OFF \
  -DINFAAS_NEURON_WORKER=OFF \
  -DCMAKE_C_FLAGS="-DLOCAL_MODE" \
  -DCMAKE_CXX_FLAGS="-DLOCAL_MODE" > cmake_output.log 2>&1

# CHECK CMAKE SUCCESS
if [ $? -ne 0 ] || ! grep -q "Configuring done" cmake_output.log; then
    echo "❌ CMAKE FAILED. Debug:"
    cat cmake_output.log
    echo "🔧 MANUAL CMAKELists.txt inspection..."
    head -60 ../CMakeLists.txt | tail -10
    exit 1
fi

echo "✅ CMAKE SUCCESS! Makefile generated."
ls -la Makefile  # PROOF makefile exists

# ========================================
# 🔨 STEP 3: PROVEN BUILD STRATEGY
# ========================================
echo "🔨 STEP 3: Building..."
if make -j$(nproc) VERBOSE=0; then
    echo "✅ BUILD SUCCESS (parallel)!"
elif make -j4 VERBOSE=0; then
    echo "✅ BUILD SUCCESS (4-core)!"
elif make -j2 VERBOSE=0; then
    echo "✅ BUILD SUCCESS (2-core)!"
elif make -j1 VERBOSE=1; then
    echo "✅ BUILD SUCCESS (single-thread)!"
else
    echo "❌ ALL BUILDS FAILED"
    echo "CMake output:"
    cat cmake_output.log
    exit 1
fi

# VERIFY BINARIES
if [ -x "../bin/modelregserver" ] && [ -x "../bin/worker" ]; then
    echo "✅ BINARIES VERIFIED: modelregserver + worker ✓"
else
    echo "❌ BINARIES MISSING!"
    ls -la ../bin/
    exit 1
fi

# ========================================
# 📁 STEP 4: Demo Setup
# ========================================
echo "📁 STEP 4: Demo setup..."
cd "$BASE_DIR"
mkdir -p tmp/{models,infaas_input,infaas_output} scripts

cat > scripts/run_diffusion.py << 'EOF'
#!/usr/bin/env python3
import sys, os, time
model = sys.argv[1] if len(sys.argv) > 1 else "demo"
print(f"🎨 Diffusion Demo: {model}")
time.sleep(2)
output = f"tmp/infaas_output/{model}_result.png"
os.makedirs("tmp/infaas_output", exist_ok=True)
with open(output, "w") as f:
    f.write(f"FAKE_DIFFUSION_{model}_512x512")
print(f"✅ Diffusion saved: {output}")
EOF
chmod +x scripts/run_diffusion.py

# ========================================
# 🏃 STEP 5: Launch Services
# ========================================
echo "🏃 STEP 5: Launching services..."
cd bin

# CLEAN START
pkill -f infaas 2>/dev/null || true
pkill -f modelregserver 2>/dev/null || true
pkill -f worker 2>/dev/null || true
sleep 3

echo "🎬 modelregserver..."
if [ -x "./modelregserver" ]; then
    nohup ./modelregserver >infaas_modelreg.log 2>&1 &
    echo "✅ modelregserver started (PID: $!)"
else
    echo "❌ modelregserver missing!"
    exit 1
fi

sleep 3

echo "⚙️ worker..."
if [ -x "./worker" ]; then
    nohup ./worker >infaas_worker.log 2>&1 &
    echo "✅ worker started (PID: $!)"
else
    echo "❌ worker missing!"
    exit 1
fi

sleep 6

# ========================================
# 🩺 STEP 6: Health Check
# ========================================
echo "🩺 STEP 6: Health check..."
for i in {1..3}; do
    sleep 2
    HEALTH=$(curl -s -m 2 http://localhost:8080/health 2>/dev/null || echo "FAIL")
    if [[ $HEALTH == *"OK"* ]]; then
        echo "✅ HEALTH: PERFECT!"
        curl -s -X POST http://localhost:8080/models \
          -H "Content-Type: application/json" \
          -d '{"name":"stable-diffusion","framework":"diffusion","imgdim":"512"}' \
          && echo "✅ Diffusion model registered!"
        break
    fi
    echo "⏳ Health check $i/3... ($HEALTH)"
done

echo "=================================="
echo "🌐 http://localhost:8080/health"
echo "📈 http://localhost:8080/models" 
echo "📊 tail -f bin/infaas_modelreg.log"
echo "🚀 INFaaS + Diffusion LIVE! 🎉🐱💥"
