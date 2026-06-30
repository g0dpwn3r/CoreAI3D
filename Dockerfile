# syntax=docker/dockerfile:1

# ==============================================================================
# Stage 1: System dependencies
# Only rebuilds when apt package list changes.
# ==============================================================================
FROM ubuntu:24.04 AS system-deps

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
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
    python3 \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-venv \
    python3-jinja2 \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libaio-dev \
    libncurses-dev \
    libtinfo-dev \
    libtirpc-dev \
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
    libasound2-dev \
    libpulse-dev \
    libudev-dev \
    libdbus-1-dev \
    libcap-dev \
    libcap2-bin \
    libmount-dev \
    libblkid-dev \
    libglib2.0-dev \
    libstdc++6 \
    libltdl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# ==============================================================================
# Stage 2: vcpkg bootstrap
# Only rebuilds when vcpkg fork/branch changes.
# ==============================================================================
FROM system-deps AS vcpkg-bootstrap

# Clone vcpkg from your fork (full history for compatibility)
RUN git clone https://github.com/g0dpwn3r/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

RUN printf '%s\n' \
    'set(VCPKG_TARGET_ARCHITECTURE x64)' \
    'set(VCPKG_CRT_LINKAGE dynamic)' \
    'set(VCPKG_LIBRARY_LINKAGE static)' \
    'set(VCPKG_CMAKE_SYSTEM_NAME Linux)' \
    'set(VCPKG_BUILD_TYPE release)' \
    'set(VCPKG_C_FLAGS "-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types")' \
    'set(VCPKG_CXX_FLAGS "-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types")' \
    > /opt/vcpkg/triplets/x64-linux-release.cmake

RUN printf '%s\n' \
    'set(VCPKG_TARGET_ARCHITECTURE x64)' \
    'set(VCPKG_CRT_LINKAGE dynamic)' \
    'set(VCPKG_LIBRARY_LINKAGE static)' \
    'set(VCPKG_CMAKE_SYSTEM_NAME Linux)' \
    'set(VCPKG_BUILD_TYPE release)' \
    'set(VCPKG_C_FLAGS "-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types")' \
    'set(VCPKG_CXX_FLAGS "-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types")' \
    > /opt/vcpkg/triplets/community/x64-linux-release.cmake

# ==============================================================================
# Stage 3: vcpkg dependency installation
# Rebuilds ONLY when vcpkg.json or custom-ports change.
# Cache mounts persist downloads/buildtrees across builds so even when
# vcpkg.json changes, only the delta recompiles.
# ==============================================================================
FROM vcpkg-bootstrap AS vcpkg-install

WORKDIR /app

COPY vcpkg.json ./
COPY custom-ports/ ./custom-ports/

ENV VCPKG_MAX_CONCURRENCY=1 \
    VCPKG_KEEP_GOING=1 \
    CURL_MAX_RETRIES=10 \
    CURL_RETRY_DELAY=5 \
    CXXFLAGS="-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types" \
    CFLAGS="-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types"

RUN --mount=type=cache,target=/root/.cache/vcpkg \
    --mount=type=cache,target=/opt/vcpkg/downloads \
    --mount=type=cache,target=/opt/vcpkg/buildtrees \
    --mount=type=cache,target=/opt/vcpkg/packages \
    { /opt/vcpkg/vcpkg install \
        --x-manifest-root=/app \
        --overlay-ports=/app/custom-ports \
        --triplet=x64-linux-release \
        --vcpkg-root=/opt/vcpkg \
        2>&1 ; echo $? > /tmp/vcpkg_rc ; } | tee /tmp/vcpkg-build.log ; \
      VCPKG_RC=$(cat /tmp/vcpkg_rc 2>/dev/null || echo 1) ; \
      if [ "$VCPKG_RC" -ne 0 ]; then \
        echo "" ; \
        echo "========================================" ; \
        echo " VCPKG BUILD FAILED (exit code $VCPKG_RC)" ; \
        echo "========================================" ; \
        echo "" ; \
        echo ">>> LAST 200 LINES OF BUILD OUTPUT <<<" ; \
        tail -200 /tmp/vcpkg-build.log ; \
        exit 1 ; \
      fi

# ==============================================================================
# Stage 4: Application build
# This is the ONLY stage that rebuilds when you edit source code.
# All vcpkg dependencies are copied from the previous stage as pre-built
# static libraries, so no recompilation of dependencies ever occurs.
# ==============================================================================
FROM system-deps AS builder

COPY --from=vcpkg-install /opt/vcpkg /opt/vcpkg

WORKDIR /app/CoreAI3D

COPY CMakeLists.txt ./
COPY CoreAI3D/ ./CoreAI3D/
COPY custom-ports/ ./custom-ports/
COPY config.json* ./
COPY test/ ./test/

ENV CXXFLAGS="-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types" \
    CFLAGS="-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types"

RUN mkdir -p build && cd build && \
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DVCPKG_OVERLAY_PORTS=/app/CoreAI3D/custom-ports \
        -DVCPKG_TARGET_TRIPLET=x64-linux-release \
        -DSKIP_POST_BUILD=ON \
    && ninja \
    || { \
        echo ""; \
        echo "========================================"; \
        echo " APPLICATION BUILD FAILURE"; \
        echo "========================================"; \
        if [ -f CMakeFiles/CMakeError.log ]; then \
            echo "--- CMakeError.log ---"; \
            tail -80 CMakeFiles/CMakeError.log; \
        fi; \
        exit 1; \
    }

# ==============================================================================
# Stage 5: Minimal runtime image (~200MB vs ~15GB full build image)
# ==============================================================================
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgcc-s1 \
    ca-certificates \
    libglib2.0-0t64 \
    libgl1 \
    libglu1-mesa \
    libasound2t64 \
    libpulse0 \
    libx11-6 \
    libxext6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxdamage1 \
    libxcomposite1 \
    libxfixes3 \
    libxrender1 \
    libxi6 \
    libxtst6 \
    libdbus-1-3 \
    libudev1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/CoreAI3D/build/CoreAI3D/CoreAI3D /app/CoreAI3D
COPY --from=builder /app/CoreAI3D/build/CoreAI3D/config.json /app/config.json

WORKDIR /app

EXPOSE 8080 8081

CMD ["./CoreAI3D"]
