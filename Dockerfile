FROM ubuntu:latest

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CXXFLAGS="-Wno-error=override-init"
ENV CFLAGS="-Wno-error=override-init"

# Restrict vcpkg concurrency to prevent Out-Of-Memory (OOM) kills on heavy compiles

ENV VCPKG_MAX_CONCURRENCY=1
# Force vcpkg to retry downloads more aggressively and wait longer for GitHub
ENV VCPKG_KEEP_GOING=1 \
    VCPKG_MAX_CONCURRENCY=1 \
    CURL_MAX_RETRIES=10 \
    CURL_RETRY_DELAY=5
# -----------------------------------------------------------------------------
# MASSIVE DEPENDENCY INSTALLATION
# We group these by category so you can see what is covered.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# MASSIVE DEPENDENCY INSTALLATION (Updated)
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

# Install system-level build dependencies required by libsystemd in vcpkg
RUN apt-get update && apt-get install -y \
    pkg-config \
    gperf \
    python3-jinja2 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------------------
# FORCE RELEASE-ONLY BUILD
# ---------------------------------------------------------------------
RUN echo "set(VCPKG_TARGET_ARCHITECTURE x64)" > /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_CRT_LINKAGE dynamic)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_LIBRARY_LINKAGE static)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_CMAKE_SYSTEM_NAME Linux)" >> /app/vcpkg/triplets/x64-linux-release.cmake && \
    echo "set(VCPKG_BUILD_TYPE release)" >> /app/vcpkg/triplets/x64-linux-release.cmake

# Configure CMake
# Configure CMake and force log dump on vcpkg # Configure CMake and isolate the exact error on failure
# Configure CMake and output the main manifest log on failure
# Configure CMake and output the exact utfcpp logs on failure
# Configure CMake and output the main manifest log on failure
# Configure CMake and output the exact libsystemd error
RUN mkdir -p CoreAI3D/build && cd CoreAI3D/build && \
    (cmake .. \
    -G "Ninja" \
    -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_OVERLAY_PORTS=/app/CoreAI3D/custom-ports \
    -DVCPKG_TARGET_TRIPLET=x64-linux-release || \
    (echo -e "\n\n=== LIBSYSTEMD RAW ERROR LOG ===" && \
    tail -n 200 /app/vcpkg/buildtrees/libsystemd/package-x64-linux-release-rel-out.log && \
    exit 1))
# Build
RUN cd CoreAI3D/build && ninja

# Default command
CMD ["/app/CoreAI3D/build/CoreAI3D"]