#!/bin/bash

# CoreAI3D Docker Build Script
# Utilizes BuildKit for aggressive caching and multi-stage builds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or you don't have permission"
        exit 1
    fi
    
    # Check for BuildKit support
    if ! docker buildx version &> /dev/null; then
        log_warn "Docker Buildx not available, falling back to legacy builder"
        export DOCKER_BUILDKIT=0
    else
        export DOCKER_BUILDKIT=1
        log_success "BuildKit enabled"
    fi
    
    # Check for required files
    local required_files=("Dockerfile" "docker-compose.yml" "vcpkg.json" "CMakeLists.txt" "config.json")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done
    
    if [ ! -d "CoreAI3D" ]; then
        log_error "CoreAI3D source directory missing"
        exit 1
    fi
    
    if [ ! -d "custom-ports" ]; then
        log_error "custom-ports directory missing"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Build the coreai3d service
build_coreai3d() {
    log_info "Building coreai3d service..."
    
    local build_args=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cache)
                build_args="$build_args --no-cache"
                log_warn "Building without cache (this will take 1-2 hours)"
                shift
                ;;
            --progress=*)
                build_args="$build_args --progress=${1#*=}"
                shift
                ;;
            --target=*)
                build_args="$build_args --target=${1#*=}"
                log_info "Building only target stage: ${1#*=}"
                shift
                ;;
            *)
                log_warn "Unknown argument: $1"
                shift
                ;;
        esac
    done
    
    # Default progress output
    if [[ ! "$build_args" =~ "--progress=" ]]; then
        build_args="$build_args --progress=plain"
    fi
    
    log_info "Starting build with args: $build_args"
    log_info "First build will take 1-2 hours (vcpkg dependencies)"
    log_info "Subsequent builds will take 30-60 seconds (source code only)"
    
    # Build using docker-compose
    docker-compose build $build_args coreai3d
    
    log_success "Build completed successfully"
}

# Build individual stages for testing
build_stage() {
    local stage=$1
    
    if [ -z "$stage" ]; then
        log_error "Usage: $0 stage <stage-name>"
        log_info "Available stages: system-deps, vcpkg-bootstrap, vcpkg-install, builder, runtime"
        exit 1
    fi
    
    log_info "Building stage: $stage"
    
    DOCKER_BUILDKIT=1 docker build \
        --progress=plain \
        --target "$stage" \
        -t "coreai3d-test:$stage" \
        .
    
    log_success "Stage '$stage' built successfully"
}

# Run the application
run_app() {
    log_info "Starting CoreAI3D application..."
    
    docker-compose up -d mysql
    log_info "Waiting for MySQL to be ready..."
    sleep 10
    
    docker-compose up -d coreai3d
    
    log_success "Application started"
    log_info "API Server: http://localhost:1906"
    log_info "WebSocket Server: ws://localhost:1907"
    log_info "View logs: docker-compose logs -f coreai3d"
}

# Stop the application
stop_app() {
    log_info "Stopping CoreAI3D application..."
    docker-compose down
    log_success "Application stopped"
}

# Clean build artifacts
clean() {
    log_warn "This will remove all Docker images and build cache for CoreAI3D"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning Docker images..."
        docker-compose down --rmi all --volumes --remove-orphans
        
        log_info "Pruning build cache..."
        docker builder prune -f
        
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show build cache statistics
cache_stats() {
    log_info "Docker build cache statistics:"
    echo ""
    docker system df -v | grep -E "Build Cache|coreai3d" || true
    echo ""
    log_info "To clear cache: $0 clean"
}

# Main command router
case "${1:-build}" in
    build)
        shift || true
        check_prerequisites
        build_coreai3d "$@"
        ;;
    stage)
        shift
        check_prerequisites
        build_stage "$@"
        ;;
    run)
        check_prerequisites
        run_app
        ;;
    stop)
        stop_app
        ;;
    clean)
        clean
        ;;
    cache)
        cache_stats
        ;;
    help|--help|-h)
        echo "CoreAI3D Docker Build Script"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  build [options]     Build the coreai3d service (default)"
        echo "    --no-cache        Build without using cache (1-2 hours)"
        echo "    --progress=plain  Show detailed build output"
        echo "    --target=STAGE    Build only up to specific stage"
        echo ""
        echo "  stage <name>        Build individual stage for testing"
        echo "                      Stages: system-deps, vcpkg-bootstrap, vcpkg-install, builder, runtime"
        echo ""
        echo "  run                 Start the application"
        echo "  stop                Stop the application"
        echo "  clean               Remove all images and cache"
        echo "  cache               Show cache statistics"
        echo "  help                Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 build                    # Build with cache (fast after first build)"
        echo "  $0 build --no-cache         # Force full rebuild"
        echo "  $0 stage vcpkg-install      # Test vcpkg dependency installation"
        echo "  $0 run                      # Start the application"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
