# Use Ubuntu 22.04 LTS for stability with vcpkg/ICU builds
# ubuntu:latest includes GCC 15 which has compatibility issues with ICU's configure script
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Fix for ICU build failures (Locale issues)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install build dependencies
# Note: 'cmake' is explicitly included here
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    tar \
    curl \
    zip \
    unzip \
    pkg-config \
    autoconf \
    automake \
    nasm    \
    libtool \
    bison \
    flex \
    gawk \
    perl \
    python3 \
    python-is-python3 \
    git \
    git-lfs \
    ninja-build \
    ca-certificates \
    linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Initialize git-lfs
RUN git lfs install

# Clone repositories
RUN git clone https://github.com/g0dpwn3r/vcpkg.git
RUN git clone -b v0.8 https://github.com/g0dpwn3r/CoreAI3D.git

# Update LFS
RUN cd CoreAI3D && git lfs pull

# Bootstrap Vcpkg
RUN cd vcpkg && ./bootstrap-vcpkg.sh && ./vcpkg integrate install

# ---------------------------------------------------------------------
# FIX: Create the Release-Only Triplet in the MAIN triplets folder
# ---------------------------------------------------------------------
RUN echo "set(VCPKG_TARGET_ARCHITECTURE x64)" > /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_CRT_LINKAGE dynamic)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_LIBRARY_LINKAGE static)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_CMAKE_SYSTEM_NAME Linux)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_BUILD_TYPE release)" >> /app/vcpkg/triplets/x64-linux-release.cmake

# Configure CMake
# We explicitly point to our new custom triplet: -DVCPKG_TARGET_TRIPLET=x64-linux-release
RUN mkdir -p CoreAI3D/build && cd CoreAI3D/build && \
    cmake .. \
    -G "Ninja" \
    -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_OVERLAY_PORTS=/app/CoreAI3D/custom-ports \
    -DVCPKG_TARGET_TRIPLET=x64-linux-release


# Build
RUN cd CoreAI3D/build && ninja

# Default command
CMD ["/app/CoreAI3D/build/CoreAI3D"]