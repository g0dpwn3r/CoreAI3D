# CoreAI3D Docker Sandboxes

This directory contains Docker configurations for running CoreAI3D in isolated sandbox environments across multiple Linux distributions and Windows.

## Supported Platforms

### Linux Distributions
- **Ubuntu** (22.04 LTS) - `Dockerfile.ubuntu`
- **Debian** (12) - `Dockerfile.debian`
- **CentOS** (8) - `Dockerfile.centos`
- **Fedora** (40) - `Dockerfile.fedora`
- **Alpine Linux** (3.20) - `Dockerfile.alpine`
- **Arch Linux** (latest) - `Dockerfile.arch`
- **openSUSE Leap** (15.5) - `Dockerfile.opensuse`
- **Kali Linux** (rolling) - `Dockerfile.kali`

### Windows
- **Windows 11** (Windows Server Core 2022) - `Dockerfile.windows`

## Features

Each sandbox includes:
- **CoreAI3D Release Build**: Pre-compiled executable optimized for the platform
- **Python Environment**: pyenv with Python 3.10 and AI/ML packages
- **Node.js Environment**: Latest LTS with web automation tools
- **Supervisor**: Process management for long-running services
- **Training Scripts**: Pre-installed training automation scripts
- **Isolated Workspace**: Separate user environment for security

## Quick Start

### Build All Sandboxes
```bash
# Build CoreAI3D release and all Docker images
./docker/sandbox/build_and_release.sh v1.0

# Or specify a custom version
./docker/sandbox/build_and_release.sh v2.1
```

### Run Individual Sandbox
```bash
# Ubuntu sandbox
docker run -d --name coreai3d-ubuntu \
  -v $(pwd)/training_data:/workspace/training_data \
  coreai3d/coreai3d-sandbox-ubuntu:latest

# Kali Linux sandbox
docker run -d --name coreai3d-kali \
  -v $(pwd)/training_data:/workspace/training_data \
  coreai3d/coreai3d-sandbox-kali:latest
```

### Run All Sandboxes with Docker Compose
```bash
# Start all sandboxes
docker-compose -f docker-compose.sandbox.yml up -d

# View running containers
docker ps

# Access a specific sandbox
docker exec -it coreai3d-ubuntu /bin/bash
```

## Configuration

### Environment Variables
- `SANDBOX_TYPE`: Platform identifier (ubuntu, kali, fedora, etc.)
- `COREAI3D_VERSION`: Version of CoreAI3D installed
- `HOME`: Sandbox user home directory

### Volumes
- `/workspace/training_data`: Mount point for training data
- `/workspace/models`: Mount point for trained models

### Ports
Each sandbox exposes port 8080 for the API server, mapped to unique host ports:
- Ubuntu: 808ubuntu
- Debian: 808debian
- CentOS: 808centos
- etc.

## Development

### Adding a New Platform
1. Create `Dockerfile.{platform}` based on existing templates
2. Update `build_and_release.sh` PLATFORMS array
3. Test the build: `./build_and_release.sh v1.0-test`

### Customization
Each Dockerfile can be customized for specific platform requirements:
- Package managers (apt, yum, pacman, etc.)
- System dependencies
- Python/Node.js versions
- Security hardening

## Security Considerations

- **Isolated User**: Each sandbox runs as a non-root user
- **Minimal Packages**: Only essential packages installed
- **No Privileged Access**: Containers run without privileged mode
- **Read-Only Base**: Core system files are immutable

## Troubleshooting

### Build Issues
```bash
# Check build logs
docker build --progress=plain -f Dockerfile.ubuntu .

# Clean up failed builds
docker system prune -f
```

### Runtime Issues
```bash
# Check container logs
docker logs coreai3d-ubuntu

# Access container shell
docker exec -it coreai3d-ubuntu /bin/bash

# Restart supervisor
docker exec coreai3d-ubuntu supervisorctl restart all
```

### Common Problems
- **MySQL Connection**: Ensure database connectivity for full features
- **GPU Access**: Add `--gpus all` for GPU-enabled containers
- **Memory Limits**: Increase Docker memory limits for large models
- **Port Conflicts**: Modify port mappings in docker-compose.yml

## Contributing

1. Test your changes on multiple platforms
2. Update documentation
3. Submit pull request with platform-specific testing

## License

Same as CoreAI3D project - see main LICENSE file.