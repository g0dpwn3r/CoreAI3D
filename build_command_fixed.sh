#!/bin/bash
set -e

# Navigate to build directory
mkdir -p CoreAI3D/build
cd CoreAI3D/build

# Configure CMake with vcpkg
cmake .. \
    -G "Ninja" \
    -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_OVERLAY_PORTS=/app/CoreAI3D/custom-ports \
    -DVCPKG_TARGET_TRIPLET=x64-linux-release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH=/app/vcpkg/installed/x64-linux-release

echo "CMake configuration completed successfully"
