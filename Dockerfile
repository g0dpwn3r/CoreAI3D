FROM ubuntu:latest

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Restrict vcpkg concurrency to prevent Out-Of-Memory (OOM) kills on heavy compiles
ENV VCPKG_MAX_CONCURRENCY=2

# -----------------------------------------------------------------------------
# MASSIVE DEPENDENCY INSTALLATION
# We group these by category so you can see what is covered.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# MASSIVE DEPENDENCY INSTALLATION (Updated)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    # --- Core Build Tools & Post-Processing ---
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    curl \
    zip \
    unzip \
    tar \
    pkg-config \
    autoconf \
    automake \
    libtool \
    autoconf-archive \
    bison \
    flex \
    gawk \
    perl \
    nasm \
    yasm \
    gperf \
    m4 \
    gettext \
    patchelf \
    # --- Python & System Utilities ---
    python3 \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-jinja2 \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    # --- MySQL & Database Dependencies ---
    libaio-dev \
    libncurses-dev \
    libtinfo-dev \
    libtirpc-dev \
    # --- X11 / GUI / Windowing ---
    libx11-dev \
    libxft-dev \
    libxext-dev \
    libxi-dev \
    libxtst-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxdamage-dev \
    libxcomposite-dev \
    libxfixes-dev \
    libxcb1-dev \
    libxrender-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    # --- Audio & Media ---
    libasound2-dev \
    libpulse-dev \
    libudev-dev \
    # --- SystemD & DBus ---
    libdbus-1-dev \
    libcap-dev \
    libcap2-bin \
    libmount-dev \
    libfixez    \C 8u 7ytrfdsz  b
    libblkid-dev \
    libglib2.0-dev \
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
# FORCE RELEASE-ONLY BUILD
# ---------------------------------------------------------------------
RUN echo "set(VCPKG_TARGET_ARCHITECTURE x64)" > /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_CRT_LINKAGE dynamic)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_LIBRARY_LINKAGE static)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_CMAKE_SYSTEM_NAME Linux)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_BUILD_TYPE release)" >> /app/vcpkg/triplets/x64-linux-release.cmake

# Configure CMake
RUN mkdir -p CoreAI3D/build && cd CoreAI3D/build && \
    cmake .. \
    -G "Ninja" \
    -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_OVERLAY_PORTS=/app/CoreAI3D/custom-ports \
    -DVCPKG_TARGET_TRIPLET=x64-linux-release

# Build
RUN cd CoreAI3D/build && ninja

# Default command
CMD ["/app/CoreAI3D/build/CoreAI3D"]