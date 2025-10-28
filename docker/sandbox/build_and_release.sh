#!/bin/bash

# CoreAI3D Docker Sandbox Build and Release Script
# Builds CoreAI3D release and creates Docker images for all supported platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker/sandbox"
BUILD_DIR="${PROJECT_ROOT}/out/build"
RELEASE_DIR="${PROJECT_ROOT}/release"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-coreai3d}"

# Supported platforms
PLATFORMS=(
    "ubuntu:ubuntu"
    "debian:debian"
    "centos:centos"
    "fedora:fedora"
    "alpine:alpine"
    "arch:arch"
    "opensuse:opensuse"
    "kali:kali"
)

# Windows platforms (separate handling)
WINDOWS_PLATFORMS=(
    "windows:windows"
)

echo -e "${BLUE}CoreAI3D Docker Sandbox Builder${NC}"
echo "=================================="

# Function to check if release exists
check_release_exists() {
    local version="$1"
    if [ -d "${RELEASE_DIR}/CoreAI3D-${version}" ]; then
        echo -e "${GREEN}✓ Release ${version} found${NC}"
        return 0
    else
        echo -e "${RED}✗ Release ${version} not found${NC}"
        return 1
    fi
}

# Function to build CoreAI3D release
build_release() {
    local version="$1"
    echo -e "${YELLOW}Building CoreAI3D ${version} release...${NC}"

    cd "${PROJECT_ROOT}"

    # Create build directory
    mkdir -p "${BUILD_DIR}"

    # Configure and build
    cd "${BUILD_DIR}"
    cmake "${PROJECT_ROOT}" -DCMAKE_BUILD_TYPE=Release -DUSE_MYSQL=OFF
    cmake --build . --config Release --parallel $(nproc)

    # Create release directory
    mkdir -p "${RELEASE_DIR}"
    local release_path="${RELEASE_DIR}/CoreAI3D-${version}"

    if [ -d "${release_path}" ]; then
        rm -rf "${release_path}"
    fi

    mkdir -p "${release_path}"

    # Copy built executable and necessary files
    cp "bin/CoreAI3D" "${release_path}/"
    cp "${PROJECT_ROOT}/README.md" "${release_path}/"
    cp "${PROJECT_ROOT}/LICENSE" "${release_path}/"

    # Create basic config
    cat > "${release_path}/config.json" << EOF
{
    "version": "${version}",
    "build_type": "release",
    "features": {
        "mysql": false,
        "api_server": true,
        "training": true,
        "chat": true
    }
}
EOF

    echo -e "${GREEN}✓ Release ${version} built successfully${NC}"
}

# Function to build Docker image
build_docker_image() {
    local platform="$1"
    local dockerfile="$2"
    local version="$3"

    echo -e "${YELLOW}Building Docker image for ${platform}...${NC}"

    cd "${DOCKER_DIR}"

    # Copy release files to docker context
    local docker_context="${DOCKER_DIR}/context_${platform}"
    mkdir -p "${docker_context}"

    cp "${RELEASE_DIR}/CoreAI3D-${version}/CoreAI3D" "${docker_context}/"
    cp "${RELEASE_DIR}/CoreAI3D-${version}/config.json" "${docker_context}/"
    cp "${RELEASE_DIR}/CoreAI3D-${version}/README.md" "${docker_context}/"

    # Build Docker image
    docker build \
        --file "${dockerfile}" \
        --tag "${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:${version}" \
        --tag "${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:latest" \
        --build-arg COREAI3D_VERSION="${version}" \
        "${DOCKER_DIR}"

    # Clean up
    rm -rf "${docker_context}"

    echo -e "${GREEN}✓ Docker image for ${platform} built successfully${NC}"
}

# Function to build Windows Docker image
build_windows_docker_image() {
    local platform="$1"
    local dockerfile="$2"
    local version="$3"

    echo -e "${YELLOW}Building Windows Docker image for ${platform}...${NC}"

    # Note: Windows containers require Windows host
    # This is a placeholder for Windows container build
    echo -e "${BLUE}Note: Windows containers require Windows host system${NC}"
    echo -e "${YELLOW}Skipping Windows container build${NC}"
}

# Function to push Docker images
push_docker_images() {
    local version="$1"

    echo -e "${YELLOW}Pushing Docker images to registry...${NC}"

    for platform_info in "${PLATFORMS[@]}"; do
        IFS=':' read -r platform dockerfile <<< "${platform_info}"
        echo -e "${BLUE}Pushing ${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:${version}${NC}"
        docker push "${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:${version}"
        docker push "${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:latest"
    done

    echo -e "${GREEN}✓ All Docker images pushed successfully${NC}"
}

# Function to create docker-compose file
create_docker_compose() {
    local version="$1"

    echo -e "${YELLOW}Creating docker-compose.yml...${NC}"

    cat > "${PROJECT_ROOT}/docker-compose.sandbox.yml" << EOF
version: '3.8'

services:
EOF

    for platform_info in "${PLATFORMS[@]}"; do
        IFS=':' read -r platform dockerfile <<< "${platform_info}"
        cat >> "${PROJECT_ROOT}/docker-compose.sandbox.yml" << EOF
  coreai3d-${platform}:
    image: ${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:${version}
    container_name: coreai3d-sandbox-${platform}
    volumes:
      - ./training_data:/workspace/training_data
      - ./models:/workspace/models
    environment:
      - SANDBOX_TYPE=${platform}
      - COREAI3D_VERSION=${version}
    ports:
      - "808${platform//[^0-9]/}:8080"
    restart: unless-stopped
    command: ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

EOF
    done

    echo -e "${GREEN}✓ docker-compose.sandbox.yml created${NC}"
}

# Main build process
main() {
    local version="${1:-v1.0}"

    echo "Building CoreAI3D Docker Sandboxes - Version ${version}"
    echo "======================================================="

    # Check if release exists, build if not
    if ! check_release_exists "${version}"; then
        build_release "${version}"
    fi

    # Build Docker images for each platform
    for platform_info in "${PLATFORMS[@]}"; do
        IFS=':' read -r platform dockerfile <<< "${platform_info}"
        build_docker_image "${platform}" "Dockerfile.${dockerfile}" "${version}"
    done

    # Handle Windows platforms (if on Windows host)
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        for platform_info in "${WINDOWS_PLATFORMS[@]}"; do
            IFS=':' read -r platform dockerfile <<< "${platform_info}"
            build_windows_docker_image "${platform}" "Dockerfile.${dockerfile}" "${version}"
        done
    fi

    # Create docker-compose file
    create_docker_compose "${version}"

    # Ask to push images
    read -p "Push Docker images to registry? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        push_docker_images "${version}"
    fi

    echo -e "${GREEN}✓ All CoreAI3D Docker sandboxes built successfully!${NC}"
    echo ""
    echo "Available images:"
    for platform_info in "${PLATFORMS[@]}"; do
        IFS=':' read -r platform dockerfile <<< "${platform_info}"
        echo "  - ${DOCKER_REGISTRY}/coreai3d-sandbox-${platform}:${version}"
    done
    echo ""
    echo "To start all sandboxes: docker-compose -f docker-compose.sandbox.yml up -d"
}

# Run main function with provided version or default
main "$@"