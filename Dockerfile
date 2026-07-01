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
     'set(VCPKG_CXX_FLAGS "-Wno-error=override-init")' \
     > /opt/vcpkg/triplets/x64-linux.cmake

RUN printf '%s\n' \
     'set(VCPKG_TARGET_ARCHITECTURE x64)' \
     'set(VCPKG_CRT_LINKAGE dynamic)' \
     'set(VCPKG_LIBRARY_LINKAGE static)' \
     'set(VCPKG_CMAKE_SYSTEM_NAME Linux)' \
     'set(VCPKG_BUILD_TYPE release)' \
     'set(VCPKG_C_FLAGS "-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types")' \
     'set(VCPKG_CXX_FLAGS "-Wno-error=override-init")' \
     > /opt/vcpkg/triplets/community/x64-linux.cmake

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

# Validate host architecture — the x64 triplet requires an x86_64 CPU
RUN case "$(uname -m)" in x86_64|amd64) echo "OK: building on x86_64 ($(uname -m))" ;; \
    *) echo "ERROR: this Dockerfile requires an x86_64 host; detected $(uname -m)" ; exit 1 ;; esac

ENV VCPKG_MAX_CONCURRENCY=4 \
    VCPKG_KEEP_GOING=1 \
    CURL_MAX_RETRIES=10 \
    CURL_RETRY_DELAY=5

RUN --mount=type=cache,target=/root/.cache/vcpkg \
    --mount=type=cache,target=/opt/vcpkg/downloads \
    --mount=type=cache,target=/opt/vcpkg/buildtrees \
    --mount=type=cache,target=/opt/vcpkg/packages \
    ARCH_FILE=/opt/vcpkg/buildtrees/.last_build_arch; \
    if [ -f "$ARCH_FILE" ] && [ "$(cat "$ARCH_FILE")" != "$(uname -m)" ]; then \
        echo "Host architecture changed from $(cat "$ARCH_FILE") to $(uname -m); clearing stale vcpkg buildtrees"; \
        rm -rf /opt/vcpkg/buildtrees/*; \
    fi; \
    mkdir -p /opt/vcpkg/buildtrees; \
    uname -m > "$ARCH_FILE"; \
    { /opt/vcpkg/vcpkg install \
        --x-manifest-root=/app \
        --overlay-ports=/app/custom-ports \
        --triplet=x64-linux \
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
        echo "" ; \
        echo ">>> MOST RECENT CMAKE CONFIG LOGS <<<" ; \
        find /opt/vcpkg/buildtrees -type f \( \
                -name "config-*-out.log" -o \
                -name "config-*-err.log" -o \
                -name "config-*-CMakeConfigureLog.yaml.log" \
            \) -printf "%T@ %p\n" 2>/dev/null | \
            sort -rn | head -5 | cut -d' ' -f2- | \
            while IFS= read -r logfile; do \
                sz=$(wc -c < "$logfile" 2>/dev/null || echo 0) ; \
                echo "" ; \
                echo "=== $logfile ($sz bytes) ===" ; \
                tail -150 "$logfile" ; \
            done ; \
        exit 1 ; \
      fi

# ==============================================================================
# Stage 4: Application build
# This is the ONLY stage that rebuilds when you edit source code.
# All vcpkg dependencies are copied from the previous stage as pre-built
# static libraries, so no recompilation of dependencies ever occurs.
# ==============================================================================
FROM system-deps AS builder
ARG PROJECT_NAME=CoreAI3D
ARG EXECUTABLE_NAME=${PROJECT_NAME}

COPY --from=vcpkg-install /opt/vcpkg /opt/vcpkg
COPY --from=vcpkg-install /app/vcpkg_installed /app/${PROJECT_NAME}/vcpkg_installed

WORKDIR /app/${PROJECT_NAME}

SHELL ["/bin/bash", "-c"]

COPY vcpkg.json ./
COPY CMakeLists.txt ./
COPY ${PROJECT_NAME}/ ./${PROJECT_NAME}/
COPY custom-ports/ ./custom-ports/
COPY config.json* ./
COPY test/ ./test/

ENV CXXFLAGS="-Wno-error=override-init" \
    CFLAGS="-Wno-error=override-init -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types"

RUN --mount=type=cache,target=/root/.cache/vcpkg \
    mkdir -p build && cd build && \
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DVCPKG_OVERLAY_PORTS=/app/${PROJECT_NAME}/custom-ports \
        -DVCPKG_TARGET_TRIPLET=x64-linux \
        -DVCPKG_INSTALLED_DIR=/app/${PROJECT_NAME}/vcpkg_installed 2>&1 | tee /tmp/cmake.log; \
    CMAKE_RC=${PIPESTATUS[0]}; \
    if [ "$CMAKE_RC" -ne 0 ]; then \
        echo ""; \
        echo "========================================"; \
        echo " CMAKE CONFIGURATION FAILURE (exit $CMAKE_RC)"; \
        echo "========================================"; \
        echo ""; \
        echo ">>> Installed packages check <<<"; \
        ls -la /app/${PROJECT_NAME}/vcpkg_installed/x64-linux/share/ 2>/dev/null | head -30 || echo "No share dir"; \
        echo ""; \
        echo ">>> Last 100 lines of cmake output <<<"; \
        tail -100 /tmp/cmake.log; \
        echo ""; \
        echo ">>> CMakeError.log <<<"; \
        cat CMakeFiles/CMakeError.log 2>/dev/null || echo "No CMakeError.log"; \
        exit 1; \
    fi

RUN cd build && ninja -j$(nproc) 2>&1 | tee /tmp/ninja.log; \
    NINJA_RC=${PIPESTATUS[0]}; \
    if [ "$NINJA_RC" -ne 0 ]; then \
        echo ""; \
        echo "========================================"; \
        echo " NINJA BUILD FAILURE (exit $NINJA_RC)"; \
        echo "========================================"; \
        echo ""; \
        echo ">>> Last 150 lines of ninja output <<<"; \
        tail -150 /tmp/ninja.log; \
        exit 1; \
    fi

# ==============================================================================
# Stage 5: Minimal runtime image (~200MB vs ~15GB full build image)
# ==============================================================================
FROM ubuntu:24.04 AS runtime
ARG PROJECT_NAME=CoreAUDIO
ARG EXECUTABLE_NAME=${PROJECT_NAME}

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

COPY --from=builder /app/${PROJECT_NAME}/build/${PROJECT_NAME}/${EXECUTABLE_NAME} /app/${PROJECT_NAME}
COPY --from=builder /app/${PROJECT_NAME}/build/${PROJECT_NAME}/config.json /app/config.json

WORKDIR /app

EXPOSE 8080 8081

CMD ["/bin/sh", "-c", "./${EXECUTABLE_NAME}"]
