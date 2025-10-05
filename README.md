# CoreAI3D

CoreAI3D is a comprehensive AI system featuring neural network training, prediction, real-time chat, and multi-modal processing capabilities. It provides a modular architecture with C++ core, Python client library, web interface, and Docker deployment options.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Docker Deployment](#docker-deployment)
  - [Native C++ Application](#native-c-application)
  - [Python Client](#python-client)
  - [Web Client](#web-client)
  - [Windows GUI](#windows-gui)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Neural Network Training**: Custom neural network implementation with configurable layers and neurons
- **Multi-modal Processing**: Support for vision, audio, text, and system data processing
- **Real-time Communication**: WebSocket-based real-time data streaming
- **REST API**: Comprehensive HTTP API for all AI modules
- **Database Integration**: MySQL support for data persistence and model storage
- **Docker Support**: Containerized deployment with sandbox environments
- **Cross-platform**: Windows and Linux support
- **Extensible Architecture**: Modular design for easy extension

## Prerequisites

### System Requirements
- **Operating System**: Windows 10+ or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for installation and data

### For Native C++ Build
- **CMake**: Version 3.20 or higher
- **C++ Compiler**: GCC 9+ (Linux) or MSVC 2019+ (Windows)
- **vcpkg**: Package manager for C++ dependencies
- **MySQL Server**: Version 8.0+ (optional, for database features)

### For Docker Deployment
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher

### For Python Client
- **Python**: Version 3.8 or higher
- **pip**: Python package installer

### For Web Client
- **Modern Web Browser**: Chrome 90+, Firefox 88+, or Safari 14+
- **WebSocket Support**: Required for real-time features

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-org/CoreAI3D.git
cd CoreAI3D
```

### Docker Setup (Recommended)

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   - API Server: http://localhost:8080
   - WebSocket: ws://localhost:8081
   - MySQL: localhost:3306 (X Protocol: 33060)

### Native C++ Build

1. **Install vcpkg** (if not already installed):
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh
   ./vcpkg integrate install
   ```

2. **Install dependencies**:
   ```bash
   vcpkg install --triplet x64-linux
   ```

3. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ..
   cmake --build . --config Release
   ```

### Python Client Setup

1. **Install dependencies**:
   ```bash
   pip install aiohttp websockets requests
   ```

2. **Optional: Install from requirements.txt** (if available):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Docker Deployment

The easiest way to run CoreAI3D is using Docker Compose, which sets up all components automatically.

#### Quick Start with Docker

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f coreai3d

# Stop services
docker-compose down
```

#### Docker Environment Variables

The Docker setup includes:
- **MySQL Database**: Pre-configured with test data
- **CoreAI3D API Server**: Runs on port 8080
- **WebSocket Server**: Runs on port 8081
- **Sandbox Environments**: Ubuntu, CentOS, Alpine for training

#### Training with Docker Sandboxes

```bash
# Access Ubuntu sandbox
docker-compose exec ubuntu-sandbox bash

# Run training script
python3 /app/docker/training/train_linux_operations.py
```

### Native C++ Application

The native application supports multiple modes: prediction, chat, and API server.

#### Command Line Options

```
CoreAI3D Options:
  -h [ --help ]                    produce help message
  -i [ --input-file ] arg          Input filename (CSV for prediction mode)
  -t [ --target-file ] arg         Optional target values file
  -d [ --delimiter ] arg (=,)      CSV delimiter (default: ,)
  -s [ --samples ] arg (=0)        Number of samples to process
  --language arg (=en)             Language code (en, nl, ru)
  --embedding-file arg (=embedding.txt)  Path to embedding file
  -e [ --epochs ] arg (=10)        Number of training epochs
  --learning-rate arg (=0.01)      Learning rate
  -l [ --layers ] arg (=3)         Number of hidden layers
  -n [ --neurons ] arg (=10)       Neurons per layer
  --min arg (=0.0)                 Min normalization value
  --max arg (=1.0)                 Max normalization value
  --input-size arg (=1)            Number of input features
  --output-size arg (=1)           Number of output features
  --db-host arg (=localhost)       Database host
  --db-port arg (=33060)           Database port
  --db-user arg (=user)            Database username
  --db-password arg (=password)    Database password
  --db-schema arg (=coreai_db)     Database schema
  --ssl-mode arg (=DISABLED)       SSL mode
  --dataset-name arg (=online-1a)  Dataset name
  --create-tables                  Create database tables
  --offline                        Run in offline mode
  --dataset-id arg (=-1)           Dataset ID for operations
  -o [ --output-csv ] arg          Output CSV file for results
  --contains-header                CSV has header row
  --contains-text                  CSV contains text data
  --start-chat                     Start chat mode
  --start-predict                  Start prediction mode
  --api-port arg (=8080)           API server port
```

#### Prediction Mode Example

```bash
# Train and predict with CSV data
./CoreAI3D --start-predict \
  --input-file data/gold.csv \
  --output-csv results/predictions.csv \
  --delimiter "," \
  --samples 1000 \
  --epochs 50 \
  --learning-rate 0.001 \
  --layers 5 \
  --neurons 25 \
  --input-size 2 \
  --output-size 1 \
  --contains-header
```

#### Chat Mode Example

```bash
# Start interactive chat
./CoreAI3D --start-chat \
  --language en \
  --embedding-file embeddings/en_embeddings.csv \
  --db-host localhost \
  --db-port 33060 \
  --db-user root \
  --db-password password
```

#### API Server Mode

```bash
# Start HTTP API server
./CoreAI3D --api-port 8080
```

### Python Client

The Python client provides async API access with automation helpers.

#### Basic Usage

```python
import asyncio
from coreai3d_client import CoreAI3DClient

async def main():
    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key'
    }) as client:
        # Health check
        health = await client.health_check()
        print(f"System healthy: {health}")

        # Get system metrics
        metrics = await client.get_system_metrics()
        print(f"CPU Usage: {metrics.data.get('cpuUsage', 0)}%")

        # Send chat message
        response = await client.send_chat_message("Hello AI!")
        print(f"Response: {response.data.get('content', '')}")

asyncio.run(main())
```

#### Automation Example

```python
from automation_helper import AutomationHelper

async def automate_tasks():
    async with CoreAI3DClient({...}) as client:
        async with AutomationHelper(client) as helper:
            # Batch image analysis
            results = await helper.analyze_image_batch([
                'image1.jpg', 'image2.png'
            ], ['classification', 'objects'])

            for result in results:
                print(f"✓ {result.message}")

asyncio.run(automate_tasks())
```

### Web Client

The web client provides browser-based access to CoreAI3D features.

#### Setup

1. **Serve static files** (using any web server):
   ```bash
   cd web
   python3 -m http.server 3000
   ```

2. **Open in browser**: http://localhost:3000

#### Usage Example

```javascript
const client = new CoreAI3DClient({
    baseURL: 'http://localhost:8080/api/v1',
    wsURL: 'ws://localhost:8081/ws',
    apiKey: 'your-api-key'
});

// Connect WebSocket
await client.connectWebSocket();

// Send chat message
const response = await client.sendChatMessage("Analyze this image", {
    attachments: ['image.jpg']
});

console.log(response);
```

### Windows GUI

The Windows GUI provides a user-friendly interface for CoreAI3D operations.

**Note**: GUI implementation is currently in development. Check the `windows_gui/` directory for updates.

## Configuration

### Environment Variables

- `COREAI3D_API_KEY`: API authentication key
- `COREAI3D_BASE_URL`: Base URL for API calls
- `COREAI3D_WS_URL`: WebSocket URL
- `COREAI3D_DB_HOST`: Database host
- `COREAI3D_DB_PORT`: Database port
- `COREAI3D_DB_USER`: Database username
- `COREAI3D_DB_PASSWORD`: Database password

### Docker Configuration

Edit `docker-compose.yml` to customize:
- Port mappings
- Environment variables
- Volume mounts
- Service dependencies

### API Configuration

The API server supports configuration via:
- Command line arguments
- Configuration files (planned)
- Environment variables

## API Reference

### REST Endpoints

#### Vision API
- `POST /api/v1/vision/analyze` - Analyze image
- `POST /api/v1/vision/detect` - Object detection
- `POST /api/v1/vision/ocr` - Optical character recognition
- `POST /api/v1/vision/faces` - Face analysis

#### Audio API
- `POST /api/v1/audio/speech-to-text` - Speech recognition
- `POST /api/v1/audio/text-to-speech` - Text synthesis
- `POST /api/v1/audio/analyze` - Audio analysis

#### System API
- `GET /api/v1/system/metrics` - System metrics
- `GET /api/v1/system/processes` - Running processes
- `POST /api/v1/system/capture` - Screen capture

#### Web API
- `POST /api/v1/web/search` - Web search
- `POST /api/v1/web/extract` - Content extraction

#### Math API
- `POST /api/v1/math/calculate` - Expression evaluation
- `POST /api/v1/math/optimize` - Function optimization

### WebSocket Events

- `chat_response` - Chat message responses
- `stream_data` - Real-time data streams
- `system_status` - System status updates
- `error` - Error notifications

## Troubleshooting

### Common Issues

#### Docker Issues
- **Port conflicts**: Change ports in `docker-compose.yml`
- **Memory issues**: Increase Docker memory limit
- **Build failures**: Clear Docker cache with `docker system prune`

#### C++ Build Issues
- **vcpkg not found**: Ensure vcpkg is in PATH
- **Missing dependencies**: Run `vcpkg install` for your platform
- **CMake errors**: Delete build directory and reconfigure

#### Python Client Issues
- **Import errors**: Install missing packages with pip
- **Connection refused**: Ensure API server is running
- **Timeout errors**: Increase timeout in client config

#### Web Client Issues
- **CORS errors**: Configure API server CORS settings
- **WebSocket connection failed**: Check firewall settings
- **Browser compatibility**: Use modern browser with WebSocket support

### Debug Mode

Enable debug logging:

```bash
# C++ application
./CoreAI3D --debug

# Python client
client = CoreAI3DClient({'debug': True, ...})
```

### Logs

- **API Server logs**: Check console output or Docker logs
- **Database logs**: Access MySQL container logs
- **Client logs**: Enable debug mode in client configurations

### Performance Tuning

- **Increase threads**: Use `--num-threads` option
- **Adjust batch sizes**: Configure batch processing parameters
- **Database optimization**: Tune MySQL configuration
- **Memory limits**: Adjust Docker memory settings

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `docker-compose -f docker-compose.test.yml up`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Build documentation
mkdocs build
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more information, visit the [documentation](docs/) or join our [community forum](https://community.coreai3d.org).