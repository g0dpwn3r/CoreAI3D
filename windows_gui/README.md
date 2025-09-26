# CoreAI3D Windows GUI Dashboard

A comprehensive Windows desktop application for managing AI training data, testing, debugging, and safely testing OS control operations using Sandboxie.

## Overview

The CoreAI3D Windows GUI Dashboard provides a complete admin interface for:

- **AI Training Data Management** - Organize, manage, and analyze training datasets
- **Code Testing & Debugging** - Test modules, API connectivity, and system automation
- **Sandboxie Integration** - Safely test OS control and automation in isolated environment
- **Real-time Monitoring** - Monitor system status, performance, and AI module health
- **Multi-modal AI Control** - Control vision, audio, system, web, and math modules
- **Comprehensive Logging** - Track all operations and debug issues

## Features

### 🗂️ Training Data Management
- Create and manage AI training datasets
- Support for multiple data types (images, audio, video, text, mixed)
- File organization and metadata management
- Dataset statistics and analysis
- Bulk file operations

### 🧪 Testing & Debugging
- API connectivity testing
- Module functionality testing
- System automation testing
- Performance benchmarking
- Load testing capabilities
- Comprehensive test reporting

### 🏠 Sandboxie Integration
- Safe testing environment for OS control
- Isolated automation testing
- File system operation testing
- Process management testing
- Network operation testing
- Security testing in sandbox

### 📊 System Monitoring
- Real-time system metrics
- Process monitoring
- Resource usage tracking
- Performance analysis
- Health status monitoring

### 🔧 Advanced Features
- Multi-threaded operations
- Async/await support for responsive UI
- Comprehensive error handling
- Settings management
- Logging and audit trails
- Plugin architecture ready

## Installation

### Prerequisites

1. **Windows 10/11** with Sandboxie installed
2. **Python 3.8+** installed
3. **CoreAI3D C++ Server** running
4. **Sandboxie** installed and configured

### Install Sandboxie

1. Download Sandboxie Plus from the official website
2. Run the installer and follow the setup wizard
3. Restart your computer if required
4. Configure Sandboxie settings as needed

### Install Dependencies

```bash
# Navigate to windows_gui directory
cd windows_gui

# Install Python dependencies
pip install -r requirements.txt

# For Sandboxie support, ensure:
# - Windows 10/11 with Sandboxie installed
# - Virtualization enabled in BIOS
# - At least 4GB RAM (8GB recommended)
```

### Run the Dashboard

```bash
# Basic startup
python dashboard.py

# Or run as module
python -m windows_gui.dashboard
```

## Configuration

### Initial Setup

1. **First Run**: The dashboard will prompt for initial configuration
2. **API Settings**: Configure connection to CoreAI3D server
3. **Sandboxie Settings**: Configure Sandboxie parameters
4. **Data Directory**: Set default location for training data

### Configuration File

The dashboard saves settings to:
```
%APPDATA%\CoreAI3D\Dashboard\settings.ini
```

Manual configuration:
```ini
[API]
api_url=http://localhost:8080/api/v1
ws_url=ws://localhost:8081/ws
api_key=your-api-key-here

[Training]
data_dir=training_data

[Sandboxie]
timeout=300
sandbox_name=CoreAI3D_Sandbox
host_folder=C:\

[Debug]
log_level=INFO
enable_diagnostics=true
```

## Usage Guide

### Dashboard Overview

The main interface consists of 6 tabs:

1. **Dashboard** - System status and quick actions
2. **Training Data** - Manage AI training datasets
3. **Testing** - Run tests and debug code
4. **Sandboxie Testing** - Sandboxie operations
5. **Debugging** - System information and debugging tools
6. **Settings** - Configuration management

### Training Data Management

#### Creating a Dataset

1. Go to **Training Data** tab
2. Click **New Dataset**
3. Enter dataset name and description
4. Select data type (images, audio, video, text, mixed)
5. Click **OK**

#### Adding Files to Dataset

1. Select dataset from the list
2. Click **Add File**
3. Browse and select files
4. Files are automatically organized by type

#### Managing Datasets

- **View Details**: Click on dataset to see information
- **Delete Dataset**: Select dataset and click **Delete Dataset**
- **Remove Files**: Select file in dataset and click **Remove File**

### Testing & Debugging

#### Running Tests

1. Go to **Testing** tab
2. Select test type from dropdown
3. Configure test parameters
4. Click **Run Test**
5. View results in the table below

#### Test Types Available

- **API Connectivity** - Test connection to CoreAI3D server
- **Module Functionality** - Test individual AI modules
- **System Automation** - Test OS control capabilities
- **Performance** - Benchmark system performance
- **Load Testing** - Test under load conditions

#### Viewing Test Results

- Results appear in the table immediately
- Click on result row for detailed information
- Export results for analysis
- Track test history

### Sandboxie Testing

#### Starting Sandboxie

1. Go to **Sandboxie Testing** tab
2. Configure sandbox settings:
    - **Sandbox Name**: Name for the sandbox
    - **Startup Command**: Commands to run on startup
    - **Networking**: Enable or disable network access
    - **Drop Rights**: Drop administrative privileges
3. Click **Start Sandboxie**
4. Wait for sandbox to initialize

#### Running Tests in Sandboxie

1. Select test type from the list
2. Click **Run Selected Test**
3. Monitor progress in the logs area
4. Results are logged automatically

#### Sandbox Test Types

- **OS Control Tests** - Test OS automation capabilities
- **File System Operations** - Test file management
- **Process Management** - Test process control
- **Network Operations** - Test network functionality
- **Registry Operations** - Test registry access
- **Security Tests** - Test security features

#### Stopping Sandboxie

1. Click **Stop Sandboxie** to terminate
2. Or close the sandboxed application windows
3. Review logs for test results

### System Monitoring

#### Real-time Monitoring

- **System Status**: CPU, memory, disk usage
- **Process Monitor**: Running processes and resources
- **Network Monitor**: Network activity and connections
- **Performance Metrics**: Detailed performance data

#### Health Dashboard

- **Overall Health Score**: 0-100 system health rating
- **Component Status**: Status of all system components
- **Alerts**: Warnings for issues detected
- **Recommendations**: Suggestions for optimization

### Debugging Tools

#### System Information

- **Hardware Details**: CPU, RAM, storage information
- **Software Environment**: OS, Python, dependencies
- **Network Configuration**: IP, ports, connections
- **Process List**: All running processes

#### Debug Operations

- **API Debug**: Test API connectivity and responses
- **Module Debug**: Test individual AI module functionality
- **System Debug**: Comprehensive system diagnostics
- **Log Analysis**: Review application logs

## Sandboxie Integration

### Safety Features

- **Isolated Environment**: Complete isolation from host system with file/registry virtualization
- **No Persistent Changes**: All changes lost when sandbox closes
- **Resource Limits**: Configurable CPU, memory, disk limits
- **Network Control**: Optional network access control
- **Privilege Reduction**: Drop administrative rights for security
- **Network Isolation**: Optional network access control

### Testing Capabilities

#### OS Control Testing
```python
# Safe OS control testing in sandbox
async def test_os_control():
    # Test file operations
    await client.create_file("test.txt")
    await client.write_file("test.txt", "Hello from sandbox!")

    # Test process management
    await client.start_process("notepad.exe")
    await client.list_processes()

    # Test registry operations
    await client.read_registry("HKEY_CURRENT_USER\\Software\\Test")
```

#### Automation Testing
```python
# Safe automation testing
async def test_automation():
    # Test UI automation
    await client.click_at(100, 100)
    await client.type_text("Automated test")

    # Test file automation
    await client.copy_file("source.txt", "dest.txt")
    await client.move_file("old_path", "new_path")

    # Test system automation
    await client.shutdown_system(delay=10)
```

### Configuration Examples

#### Basic Sandbox Config
```python
config = {
    'host_folder': 'C:\\Projects\\CoreAI3D',
    'startup_command': 'python test_script.py',
    'networking': 'enable',
    'vgpu': 'enable'
}
```

#### Advanced Sandbox Config
```python
config = {
    'host_folder': 'C:\\Development',
    'sandbox_folder': 'C:\\Work',
    'readonly': False,
    'startup_command': '''
        cd C:\\Work
        python -m pytest tests/ -v
        python automation_script.py
    ''',
    'logon_script': 'setup_environment.cmd',
    'audio_input': 'enable',
    'video_input': 'enable',
    'clipboard_redirection': 'enable'
}
```

## API Integration

### Connecting to CoreAI3D Server

1. **Configure API Settings**:
   - API URL: `http://localhost:8080/api/v1`
   - WebSocket URL: `ws://localhost:8081/ws`
   - API Key: Your authentication key

2. **Test Connection**:
   - Go to Testing tab
   - Run "API Connectivity" test
   - Check system status in Dashboard

3. **Monitor Connection**:
   - Real-time status in Dashboard tab
   - Automatic reconnection on failure
   - Health checks every 5 seconds

### Using Python Client

```python
from coreai3d_client import CoreAI3DClient
from automation_helper import AutomationHelper

async def main():
    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key'
    }) as client:

        # Test connection
        is_healthy = await client.health_check()
        print(f"Server healthy: {is_healthy}")

        # Get system metrics
        metrics = await client.get_system_metrics()
        print(f"CPU Usage: {metrics.data.get('cpuUsage', 0)}%")

        # Use automation helper
        async with AutomationHelper(client) as helper:
            result = await helper.monitor_system(duration=30)
            print(f"Monitoring result: {result.success}")

asyncio.run(main())
```

## Security Considerations

### Sandbox Security

- **Isolation**: Complete isolation from host system
- **No Data Persistence**: All data lost when sandbox closes
- **Limited Resources**: Configurable resource limits
- **Network Control**: Optional network access
- **Audit Logging**: All operations logged

### API Security

- **API Key Authentication**: Required for all requests
- **Session Management**: Secure session handling
- **Request Validation**: Input validation and sanitization
- **Rate Limiting**: Built-in rate limiting
- **Error Handling**: No sensitive data in error messages

### Data Security

- **Encrypted Storage**: Settings encrypted on disk
- **Secure File Handling**: Safe file operations
- **Access Control**: User permission checks
- **Audit Trails**: Complete operation logging

## Troubleshooting

### Common Issues

#### Sandboxie Not Starting
1. **Check Requirements**:
   - Windows 10/11 Pro, Enterprise, or Education
   - Virtualization enabled in BIOS
   - At least 4GB RAM available

2. **Enable Features**:
    - Sandboxie installed and running
    - Virtualization in BIOS/UEFI (if using hardware acceleration)

3. **Resource Check**:
   - Ensure sufficient disk space
   - Check RAM availability
   - Verify administrator privileges

#### API Connection Issues
1. **Server Status**:
   - Verify CoreAI3D server is running
   - Check API server logs
   - Test with curl or Postman

2. **Network Configuration**:
   - Check firewall settings
   - Verify port availability
   - Test network connectivity

3. **Authentication**:
   - Verify API key is correct
   - Check session management
   - Review authentication logs

#### Performance Issues
1. **System Resources**:
   - Monitor CPU and memory usage
   - Check disk I/O performance
   - Review network latency

2. **Application Settings**:
   - Adjust timeout settings
   - Configure connection pooling
   - Optimize thread settings

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run dashboard with debug output
python -c "
import sys
sys.path.append('windows_gui')
from dashboard import main
main()
"
```

### Log Files

- **Application Logs**: `%APPDATA%\CoreAI3D\Dashboard\logs\`
- **Sandbox Logs**: `%TEMP%\sandbox_logs\`
- **System Logs**: Windows Event Viewer
- **API Logs**: CoreAI3D server logs

## Performance Optimization

### System Requirements

- **Minimum**:
  - Windows 10/11 (64-bit)
  - 4GB RAM
  - 2GB free disk space
  - Intel/AMD CPU with virtualization support

- **Recommended**:
  - Windows 11 Pro
  - 16GB RAM
  - SSD storage
  - Multi-core CPU
  - Dedicated GPU (for AI tasks)

### Optimization Tips

1. **Memory Management**:
   - Close unused applications
   - Monitor memory usage in Task Manager
   - Restart dashboard periodically

2. **Disk Performance**:
   - Use SSD for data directories
   - Regular disk cleanup
   - Optimize file organization

3. **Network Optimization**:
   - Use local API server when possible
   - Minimize network requests
   - Enable connection pooling

## Advanced Usage

### Custom Test Scripts

Create custom test scripts in the `scripts/` directory:

```python
# custom_test.py
import asyncio
from coreai3d_client import CoreAI3DClient

async def custom_test():
    async with CoreAI3DClient({...}) as client:
        # Your custom test logic here
        response = await client.analyze_image('test_image.jpg')
        print(f"Custom test result: {response.success}")

        return response.data

if __name__ == "__main__":
    asyncio.run(custom_test())
```

### Plugin Development

Extend functionality with plugins:

```python
# plugins/my_plugin.py
class MyPlugin:
    def __init__(self, dashboard):
        self.dashboard = dashboard

    def activate(self):
        # Plugin activation code
        pass

    def deactivate(self):
        # Plugin deactivation code
        pass
```

### Automation Scripts

Create automation workflows:

```python
# automation/daily_health_check.py
import asyncio
from coreai3d_client import CoreAI3DClient
from automation_helper import AutomationHelper

async def daily_health_check():
    async with CoreAI3DClient({...}) as client:
        async with AutomationHelper(client) as helper:

            # System health check
            health = await helper.system_health_check()
            print(f"Health Score: {health.data['health_score']}")

            # Automated backup
            backup = await helper.automated_backup(
                ['documents', 'images'],
                'backups/daily'
            )
            print(f"Backup: {'Success' if backup.success else 'Failed'}")

            # Generate report
            report = {
                'timestamp': time.time(),
                'health_score': health.data['health_score'],
                'backup_status': backup.success
            }

            return report

if __name__ == "__main__":
    asyncio.run(daily_health_check())
```

## Support & Documentation

### Getting Help

1. **Built-in Help**:
   - Click Help → Documentation
   - Check tooltips and status messages
   - Review log files

2. **Troubleshooting Guide**:
   - See Troubleshooting section above
   - Check Windows Event Viewer
   - Review application logs

3. **Community Support**:
   - GitHub Issues (for bugs)
   - Documentation (for features)
   - Forums (for discussions)

### API Documentation

- **Interactive API Docs**: `http://localhost:8080/api/docs`
- **Python Client Docs**: See `python/README.md`
- **Web Interface Docs**: See `web/README.md`

### Contributing

1. **Fork the repository**
2. **Create feature branch**
3. **Make changes with tests**
4. **Submit pull request**

## License

This software is licensed under the MIT License. See LICENSE file for details.

## Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added Sandboxie integration
- **v1.2.0**: Enhanced testing and debugging features
- **v1.3.0**: Performance optimizations and bug fixes

---

**CoreAI3D Windows GUI Dashboard** - Complete admin interface for AI training data management, testing, debugging, and safe OS control testing with Sandboxie integration.