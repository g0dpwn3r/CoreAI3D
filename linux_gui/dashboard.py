#!/usr/bin/env python3
"""
CoreAI3D Linux GUI Dashboard
A comprehensive desktop application for managing AI training data, testing, debugging,
and safely testing OS control operations using Linux containers.

Author: CoreAI3D Development Team
Version: 1.3.0
License: MIT
"""

import sys
import os
import asyncio
import json
import logging
import threading
import time
import psutil
import subprocess
import shutil
import urllib.request
import urllib.parse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# WebSocket client import
import websocket

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QProgressBar, QComboBox, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QInputDialog, QMenuBar, QMenu, QStatusBar,
    QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem, QPlainTextEdit,
    QToolButton
)

# qasync import for unified event loop
import qasync

# Ensure qasync is properly configured
# Note: High DPI scaling attributes are not available in current qasync/PyQt6 version

# Matplotlib imports for neural network visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QObject, QSize, QRect,
    QSettings, QStandardPaths, QDir, QFileSystemWatcher, QUrl
)
from PyQt6.QtGui import (
    QFont, QIcon, QPixmap, QPalette, QColor, QLinearGradient,
    QPainter, QBrush, QAction, QDesktopServices
)

# CoreAI3D imports
from coreai3d_client import CoreAI3DClient, APIResponse
from automation_helper import AutomationHelper

# Create logs directory before configuring logging
os.makedirs('logs', exist_ok=True)

# Configure logging with reduced verbosity
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# The dashboard opens as a normal, resizable window (never full screen).
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# Resend REST endpoint used to deliver beta keys by email.
RESEND_API_URL = "https://api.resend.com/emails"

# Worker threads that could not be stopped in time are parked here so their
# QThread object is never destroyed while it is still running (which aborts
# the process).
_LINGERING_THREADS: List[Any] = []

# Custom Qt log handler for debug tab
class QtLogHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
    def emit(self, record):
        msg = self.format(record)
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda m=msg: self.text_edit.append(m))
        except Exception:
            pass

# Reduce verbosity of third-party libraries
logging.getLogger('websockets.client').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

def default_api_url() -> str:
    """CoreAI3D backend API base URL.

    Resolved from COREAI3D_API_URL when set; otherwise the bare-metal host
    127.0.0.1:1906 (the API container's published port on the HOST). The GUI
    runs on bare metal, so it must NOT talk to the container's internal 8080 or
    to the Docker service name "coreai3d".
    """
    return os.environ.get('COREAI3D_API_URL', 'http://127.0.0.1:1906/api/v1')


def default_ws_url() -> str:
    """CoreAI3D backend WebSocket URL.

    Honours COREAI3D_WS_URL when set; otherwise DERIVED from the API base URL
    (http -> ws, port 1906 -> 1907) so the two always move together. This is a
    fallback -- set COREAI3D_WS_URL explicitly if the mapping ever diverges.
    """
    explicit = os.environ.get('COREAI3D_WS_URL')
    if explicit:
        return explicit
    api = default_api_url()
    try:
        parsed = urllib.parse.urlparse(api)
        scheme = 'ws' if parsed.scheme == 'http' else 'wss'
        # The API container publishes 1906 -> 8080 and 1907 -> 8081, so the
        # WebSocket port is the API port + 1.
        port = parsed.port + 1 if parsed.port else 1907
        netloc = f'{parsed.hostname or "127.0.0.1"}:{port}'
        return urllib.parse.urlunparse((scheme, netloc, '', '', '', ''))
    except Exception:
        return 'ws://127.0.0.1:1907'


@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    api_url: str = field(default_factory=default_api_url)
    ws_url: str = field(default_factory=default_ws_url)
    api_key: str = ""
    resend_api_key: str = ""
    resend_from_email: str = ""
    beta_key: str = ""
    data_dir: str = "training_data"
    sandbox_timeout: int = 300
    log_level: str = "INFO"
    enable_diagnostics: bool = True
    max_connections: int = 10
    retry_attempts: int = 3
    connection_timeout: float = 90.0
    chat_timeout: int = 30
    admin_email: str = ""
    admin_password: str = ""

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_rx: int = 0
    network_tx: int = 0
    process_count: int = 0
    thread_count: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class TestResult:
    """Test execution result"""
    test_id: str
    test_type: str
    success: bool
    duration: float
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelInfo:
    """Model information from external platforms"""
    name: str
    type: str
    size: str
    version: str
    status: str
    platform: str
    download_url: str = ""
    description: str = ""
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_updated: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class ModelDiscoveryResult:
    """Result of model discovery operation"""
    models: List[ModelInfo]
    total_count: int
    current_page: int
    total_pages: int
    has_next: bool
    has_previous: bool
    platform: str
    timestamp: float = field(default_factory=time.time)

class SandboxManager(QObject):
    """Docker container management"""

    sandbox_started = pyqtSignal(str)
    sandbox_stopped = pyqtSignal(str)
    sandbox_error = pyqtSignal(str, str)

    def __init__(self, config: DashboardConfig):
        super().__init__()
        self.config = config
        self.sandbox_process: Optional[subprocess.Popen] = None
        self.sandboxed_processes: List[subprocess.Popen] = []
        self.is_running = False
        self.sandbox_name = "coreai3d_sandbox"
        self.sandbox_config: Optional[Dict[str, Any]] = None
        self.monitor_thread: Optional[threading.Thread] = None

    def create_sandbox_config(self, config: Dict[str, Any]) -> bool:
        """Create Docker container configuration"""
        try:
            # Docker uses command-line options and container configurations
            self.sandbox_config = {
                'container_name': config.get('sandbox_name', self.sandbox_name),
                'image': config.get('image', 'kali:latest'),
                'host_folder': config.get('host_folder', '/tmp'),
                'startup_command': config.get('startup_command', 'tail -f /dev/null'),
                'networking': config.get('networking', 'enable'),
                'privileged': config.get('privileged', False),
                'detach': True
            }
            return True

        except Exception as e:
            logger.error(f"Error creating container config: {str(e)}")
            return False

    def start_sandbox(self, config: Dict[str, Any]) -> bool:
        """Start Docker container with configuration"""
        try:
            if self.is_running:
                logger.warning("Container already running")
                return False

            if not self.create_sandbox_config(config):
                return False

            # Check if Docker is installed
            if not self._is_docker_installed():
                error_msg = "Docker is not installed or not found in PATH"
                logger.error(error_msg)
                self.sandbox_error.emit("Installation Error", error_msg)
                return False

            # Start container using Docker
            success = self._start_docker_container()
            if not success:
                return False

            self.is_running = True
            self.sandbox_started.emit("Docker container started successfully")
            logger.info("Docker container started")

            # Monitor container
            self.monitor_thread = threading.Thread(target=self._monitor_container, daemon=True)
            self.monitor_thread.start()

            return True

        except Exception as e:
            error_msg = f"Failed to start container: {str(e)}"
            logger.error(error_msg)
            self.sandbox_error.emit("Start Error", error_msg)
            return False

    def stop_sandbox(self) -> bool:
        """Stop Docker container"""
        try:
            if not self.is_running:
                return True

            # Stop the container
            success = self._terminate_docker_container()

            self.is_running = False
            if success:
                self.sandbox_stopped.emit("Docker container stopped successfully")
                logger.info("Docker container stopped")
            else:
                self.sandbox_stopped.emit("Docker container stopped with warnings")
                logger.warning("Docker container stopped with some issues")

            # Clean up monitor thread
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)

            return success

        except Exception as e:
            logger.error(f"Error stopping container: {str(e)}")
            return False

    def _is_docker_installed(self) -> bool:
        """Check if Docker is installed"""
        try:
            # Check for Docker installation
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error checking Docker installation: {str(e)}")
            return False

    def _start_docker_container(self) -> bool:
        """Start Docker container"""
        try:
            if not self.sandbox_config:
                logger.error("No sandbox config available")
                return False

            container_name = self.sandbox_config['container_name']

            # Check if a container with the same name already exists
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True
            )

            if container_name in result.stdout:
                logger.info(f"Container {container_name} already exists, attempting to start it")
                start_result = subprocess.run(
                    ['docker', 'start', container_name],
                    capture_output=True,
                    text=True
                )
                if start_result.returncode == 0:
                    time.sleep(2)
                    return True
                else:
                    logger.warning(f"Failed to start existing container, removing it: {start_result.stderr}")
                    subprocess.run(
                        ['docker', 'rm', '-f', container_name],
                        capture_output=True,
                        text=True
                    )

            # Pull the image if needed
            self._pull_image()

            # Start the container
            cmd = [
                'docker', 'run',
                '--name', container_name,
                '--rm'  # Remove container when stopped
            ]

            # Add networking option
            if self.sandbox_config.get('networking') == 'disable':
                cmd.append('--network')
                cmd.append('none')

            # Add privileged/user option: privileged maps to root, non-privileged to 1000:1000
            if self.sandbox_config.get('privileged', False):
                cmd.append('--privileged')
                cmd.append('--user')
                cmd.append('0:0')  # root
            else:
                cmd.append('--user')
                cmd.append('1000:1000')  # Non-root user

            # Add host folder mount
            host_folder = self.sandbox_config.get('host_folder')
            if host_folder and os.path.exists(host_folder):
                cmd.append('-v')
                cmd.append(f'{host_folder}:{host_folder}')

            # Add detach
            if self.sandbox_config.get('detach', True):
                cmd.append('-d')

            # Add image and command
            cmd.append(self.sandbox_config['image'])
            cmd.extend(['/bin/bash', '-c', self.sandbox_config['startup_command']])

            self.sandbox_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait a bit for container to start
            time.sleep(2)

            # Check if container is running
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True
            )

            return container_name in result.stdout

        except Exception as e:
            logger.error(f"Error starting Docker container: {str(e)}")
            return False

    def _pull_image(self):
        """Pull Docker image if needed"""
        try:
            if not self.sandbox_config:
                logger.error("No sandbox config available for image pull")
                return

            image = self.sandbox_config['image']
            # Check if image exists locally
            result = subprocess.run(
                ['docker', 'images', '-q', image],
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                # Pull the image
                logger.info(f"Pulling Docker image: {image}")
                subprocess.run(
                    ['docker', 'pull', image],
                    capture_output=True,
                    text=True
                )

        except Exception as e:
            logger.error(f"Error pulling Docker image: {str(e)}")

    def _terminate_docker_container(self) -> bool:
        """Terminate Docker container"""
        try:
            success = True

            if not self.sandbox_config:
                logger.warning("No sandbox config available for termination")
                return False

            container_name = self.sandbox_config['container_name']

            # Stop the container
            try:
                subprocess.run(
                    ['docker', 'stop', container_name],
                    capture_output=True,
                    timeout=30,
                    text=True
                )
            except subprocess.TimeoutExpired:
                logger.warning("Docker container stop timed out")
                # Force kill if needed
                try:
                    subprocess.run(
                        ['docker', 'kill', container_name],
                        capture_output=True,
                        timeout=10,
                        text=True
                    )
                except subprocess.TimeoutExpired:
                    success = False

            # Remove container explicitly in case --rm did not auto-remove
            try:
                subprocess.run(
                    ['docker', 'rm', '-f', container_name],
                    capture_output=True,
                    timeout=10,
                    text=True
                )
            except subprocess.TimeoutExpired:
                logger.warning("Docker container remove timed out")
                success = False

            # Clean up the subprocess if still running
            if self.sandbox_process and self.sandbox_process.poll() is None:
                self.sandbox_process.terminate()
                try:
                    self.sandbox_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.sandbox_process.kill()

            return success

        except Exception as e:
            logger.error(f"Error terminating container: {str(e)}")
            return False

    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Execute command in the running Docker container"""
        try:
            if not self.is_running or not self.sandbox_config:
                return False, "Container is not running"

            # Execute command in container using docker exec
            result = subprocess.run(
                ['docker', 'exec', self.sandbox_config['container_name'], '/bin/bash', '-c', command],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Return success status and combined output
            success = result.returncode == 0
            output = result.stdout
            if result.stderr:
                if output:
                    output += "\n" + result.stderr
                else:
                    output = result.stderr

            return success, output.strip() if output else ""

        except subprocess.TimeoutExpired:
            return False, "Command execution timed out"
        except Exception as e:
            logger.error(f"Error executing command in container: {str(e)}")
            return False, f"Error: {str(e)}"

    def _monitor_container(self):
        """Monitor container in background"""
        try:
            while self.is_running and self.sandbox_config:
                # Check if container is still running
                result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={self.sandbox_config["container_name"]}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True
                )

                if self.sandbox_config['container_name'] not in result.stdout:
                    # Container has stopped
                    self.is_running = False
                    self.sandbox_stopped.emit("Docker container terminated")
                    break

                time.sleep(2)
        except Exception as e:
            logger.error(f"Error monitoring container: {str(e)}")
            self.is_running = False
            self.sandbox_error.emit("Monitor Error", f"Container monitoring failed: {str(e)}")

    def _update_container_processes(self):
        """Update list of container processes"""
        try:
            if not self.is_running or not self.sandbox_config:
                return

            # Query Docker for running processes in our container
            result = subprocess.run(
                ['docker', 'exec', self.sandbox_config['container_name'], 'ps', 'aux'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Parse output to get process list
                # This is a simplified version
                self.sandboxed_processes = result.stdout.split('\n')[1:]  # Skip header

        except Exception as e:
            logger.error(f"Error updating container processes: {str(e)}")

class TrainingDataManager(QObject):
    """Training data management"""

    dataset_created = pyqtSignal(str, str)
    dataset_updated = pyqtSignal(str)
    file_added = pyqtSignal(str, str)

    def __init__(self, config: DashboardConfig):
        super().__init__()
        self.config = config
        self.datasets: Dict[str, Dict[str, Any]] = {}

    def create_dataset(self, name: str, description: str, data_type: str) -> bool:
        """Create new training dataset"""
        try:
            dataset_path = os.path.join(self.config.data_dir, name)
            os.makedirs(dataset_path, exist_ok=True)

            dataset_info = {
                'name': name,
                'description': description,
                'data_type': data_type,
                'created_at': datetime.now().isoformat(),
                'files': [],
                'metadata': {
                    'inputsize': 0,
                    'outputsize': 0,
                    'samplesize': 0
                }
            }

            # Save dataset metadata
            metadata_path = os.path.join(dataset_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)

            self.datasets[name] = dataset_info
            self.dataset_created.emit(name, f"Dataset '{name}' created successfully")
            logger.info(f"Created dataset: {name}")

            return True

        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return False

    def add_file_to_dataset(self, dataset_name: str, file_path: str, file_sizes: Dict[str, Any] = None) -> bool:
        """Add file to dataset"""
        try:
            if dataset_name not in self.datasets:
                logger.error(f"Dataset not found: {dataset_name}")
                return False

            dataset_path = os.path.join(self.config.data_dir, dataset_name)
            file_name = os.path.basename(file_path)

            # Copy file to dataset
            dest_path = os.path.join(dataset_path, file_name)
            shutil.copy2(file_path, dest_path)

            # Calculate dataset sizes for this file if not provided
            if file_sizes is None:
                file_sizes = {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}

            # Update dataset metadata
            file_info = {
                'name': file_name,
                'path': dest_path,
                'added_at': datetime.now().isoformat(),
                'size': os.path.getsize(dest_path),
                'inputsize': file_sizes['inputsize'],
                'outputsize': file_sizes['outputsize'],
                'samplesize': file_sizes['samplesize']
            }
            self.datasets[dataset_name]['files'].append(file_info)

            # Update overall dataset metadata with cumulative sizes
            self._update_dataset_sizes(dataset_name)

            self._save_dataset_metadata(dataset_name)
            self.file_added.emit(dataset_name, file_name)

            logger.info(f"Added file {file_name} to dataset {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding file to dataset: {str(e)}")
            return False

    def _update_dataset_sizes(self, dataset_name: str):
        """Update overall dataset sizes based on all files"""
        try:
            dataset = self.datasets[dataset_name]
            files = dataset['files']

            if not files:
                # No files, reset sizes
                dataset['metadata']['inputsize'] = 0
                dataset['metadata']['outputsize'] = 0
                dataset['metadata']['samplesize'] = 0
                return

            # For multiple files, aggregate sizes
            # This is a simplified approach - in practice, you might want more sophisticated aggregation
            total_inputsize = 0
            total_outputsize = 0
            total_samplesize = 0
            valid_files = 0

            for file_info in files:
                inputsize = file_info.get('inputsize', 'N/A')
                outputsize = file_info.get('outputsize', 'N/A')
                samplesize = file_info.get('samplesize', 'N/A')

                if inputsize != 'N/A' and outputsize != 'N/A' and samplesize != 'N/A':
                    total_inputsize = max(total_inputsize, inputsize)  # Use max for inputsize
                    total_outputsize = max(total_outputsize, outputsize)  # Use max for outputsize
                    total_samplesize += samplesize  # Sum samplesize
                    valid_files += 1

            if valid_files > 0:
                dataset['metadata']['inputsize'] = total_inputsize
                dataset['metadata']['outputsize'] = total_outputsize
                dataset['metadata']['samplesize'] = total_samplesize
            else:
                dataset['metadata']['inputsize'] = 'N/A'
                dataset['metadata']['outputsize'] = 'N/A'
                dataset['metadata']['samplesize'] = 'N/A'

        except Exception as e:
            logger.error(f"Error updating dataset sizes: {str(e)}")

    def _save_dataset_metadata(self, dataset_name: str):
        """Save dataset metadata to file"""
        try:
            dataset_path = os.path.join(self.config.data_dir, dataset_name)
            metadata_path = os.path.join(dataset_path, 'metadata.json')

            with open(metadata_path, 'w') as f:
                json.dump(self.datasets[dataset_name], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dataset metadata: {str(e)}")

class ModelDiscoveryService(QObject):
    """Service for discovering models from external platforms"""

    models_discovered = pyqtSignal(ModelDiscoveryResult)
    discovery_error = pyqtSignal(str, str)
    discovery_progress = pyqtSignal(str, int, int)  # platform, current_page, total_pages

    def __init__(self, config: DashboardConfig):
        super().__init__()
        self.config = config
        self.cache: Dict[str, ModelDiscoveryResult] = {}
        self.cache_timeout = 300  # 5 minutes
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.request_timeout = 30  # seconds

    async def discover_models(self, platform: str = "all", query: str = "",
                              model_type: str = "all", page: int = 1,
                              page_size: int = 25) -> ModelDiscoveryResult:
        """Discover models from specified platform(s) with comprehensive pagination"""
        try:
            cache_key = f"{platform}_{query}_{model_type}_{page}_{page_size}"

            # Check cache first
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result.timestamp < self.cache_timeout:
                    return cached_result

            all_models = []

            # Discover models from each platform with comprehensive fetching
            if platform in ["all", "kaggle"]:
                try:
                    kaggle_models = await self._discover_kaggle_models_comprehensive(query, model_type, page, page_size)
                    all_models.extend(kaggle_models)
                    logger.info(f"Discovered {len(kaggle_models)} models from Kaggle")
                except Exception as e:
                    logger.warning(f"Failed to discover Kaggle models: {str(e)}")
                    # Continue with other platforms

            if platform in ["all", "huggingface"]:
                try:
                    hf_models = await self._discover_huggingface_models_comprehensive(query, model_type, page, page_size)
                    all_models.extend(hf_models)
                    logger.info(f"Discovered {len(hf_models)} models from HuggingFace")
                except Exception as e:
                    logger.warning(f"Failed to discover HuggingFace models: {str(e)}")
                    # Continue with other platforms

            if platform in ["all", "local"]:
                local_models = self._get_local_models(query, model_type)
                all_models.extend(local_models)

            # Apply filtering
            filtered_models = self._filter_models(all_models, query, model_type)

            # For comprehensive discovery, we want all models, not just one page
            # But we still need to paginate for UI display
            total_count = len(filtered_models)

            # If this is page 1 and we have comprehensive results, store them
            if page == 1 and len(filtered_models) > page_size:
                # Cache the full result set for pagination
                full_cache_key = f"{platform}_{query}_{model_type}_full"
                full_result = ModelDiscoveryResult(
                    models=filtered_models,
                    total_count=total_count,
                    current_page=1,
                    total_pages=(total_count + page_size - 1) // page_size,
                    has_next=total_count > page_size,
                    has_previous=False,
                    platform=platform
                )
                full_result.timestamp = time.time()
                self.cache[full_cache_key] = full_result

            # Paginate results for current page
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_models = filtered_models[start_idx:end_idx]

            total_pages = (total_count + page_size - 1) // page_size

            result = ModelDiscoveryResult(
                models=page_models,
                total_count=total_count,
                current_page=page,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_previous=page > 1,
                platform=platform
            )

            # Cache result
            result.timestamp = time.time()
            self.cache[cache_key] = result

            logger.info(f"Model discovery completed: {len(page_models)} models on page {page}/{total_pages} (total: {total_count})")
            return result

        except Exception as e:
            logger.error(f"Error discovering models: {str(e)}")
            # Return empty result instead of raising exception
            return ModelDiscoveryResult(
                models=[],
                total_count=0,
                current_page=page,
                total_pages=0,
                has_next=False,
                has_previous=False,
                platform=platform
            )

    async def _discover_kaggle_models_comprehensive(self, query: str, model_type: str, page: int, page_size: int) -> List[ModelInfo]:
        """Discover models from Kaggle with comprehensive pagination across multiple pages"""
        try:
            models = []
            total_models_needed = page_size * 100  # Get 100x the requested page size for comprehensive discovery
            max_pages = 500  # Allow up to 500 pages to get comprehensive results

            # Try Kaggle API v1 models endpoint first
            search_url = "https://www.kaggle.com/api/v1/models/list"

            # Start from page 1 and go up to max_pages or until we have enough models
            for current_page in range(1, max_pages + 1):
                # Emit progress signal
                self.discovery_progress.emit("Kaggle", current_page, min(max_pages, 100))  # Cap progress display at 100 for UI

                params = {
                    'search': query,
                    'page': current_page,
                    'pageSize': 50,  # Increased page size for more efficient fetching
                    'sortBy': 'hottest'  # Kaggle API v1 uses 'hottest'
                }

                if model_type != "all":
                    # Map model types to Kaggle tags - expanded mapping for models
                    type_mapping = {
                        "vision": ["computer-vision", "image-classification", "object-detection", "image-processing", "image-segmentation", "pose-estimation"],
                        "audio": ["audio-data", "speech-recognition", "speech-processing", "music", "sound"],
                        "text": ["nlp", "text-data", "sentiment-analysis", "text-mining", "language-model", "translation"],
                        "neural": ["deep-learning", "neural-networks", "machine-learning", "tensorflow", "pytorch", "keras"],
                        "multimodal": ["multimodal", "vision-language", "audio-visual"],
                        "math": ["mathematics", "statistics", "numerical-analysis"],
                        "web": ["web-scraping", "api", "internet"]
                    }
                    if model_type in type_mapping:
                        params['tags'] = ','.join(type_mapping[model_type])

                # Make request with retry logic - using GET for Kaggle API v1
                url = f"{search_url}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'CoreAI3D-Dashboard/1.0',
                    'Accept': 'application/json'
                })

                success = False
                for attempt in range(self.max_retries):
                    try:
                        with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                            data = json.loads(response.read().decode('utf-8'))
                            # Kaggle API v1 returns a list directly, not an object with 'modelList'
                            if isinstance(data, list):
                                page_models = data
                            else:
                                page_models = data.get('modelList', [])
                            success = True
                            break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:  # Rate limited
                            if attempt < self.max_retries - 1:
                                logger.warning(f"Kaggle API rate limited, retrying in {self.retry_delay}s (attempt {attempt + 1}/{self.max_retries})")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            else:
                                logger.warning(f"Kaggle API rate limited after {current_page} pages, collected {len(models)} models")
                                break  # Break to try fallback
                        elif e.code == 404:
                            logger.warning(f"Kaggle API models endpoint not found, trying datasets endpoint")
                            # Try datasets endpoint as fallback
                            return await self._discover_kaggle_models_fallback(query, model_type, page, page_size)
                        elif e.code == 400:
                            logger.warning(f"Kaggle API bad request (400), trying fallback method")
                            # Try fallback method
                            return await self._discover_kaggle_models_fallback(query, model_type, page, page_size)
                        else:
                            logger.warning(f"Kaggle API error on page {current_page}: {e.code}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            break
                    except Exception as e:
                        logger.warning(f"Error fetching Kaggle page {current_page} (attempt {attempt + 1}): {str(e)}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        break

                if not success:
                    logger.warning(f"Failed to fetch Kaggle page {current_page} after {self.max_retries} attempts")
                    # Try fallback method
                    return await self._discover_kaggle_models_fallback(query, model_type, page, page_size)

                # Process the data
                if not page_models:
                    # No more results, stop pagination
                    break

                for item in page_models:
                    model_info = ModelInfo(
                        name=item.get('name', item.get('title', 'Unknown')),
                        type=self._classify_kaggle_model_type(item),
                        size="Unknown",  # Models API may not provide size
                        version=item.get('version', '1.0'),
                        status="Available",
                        platform="Kaggle",
                        download_url=f"https://www.kaggle.com/{item.get('owner', '')}/{item.get('name', '')}",
                        description=item.get('description', ''),
                        author=item.get('owner', ''),
                        downloads=item.get('downloads', 0),
                        likes=item.get('likes', 0),
                        last_updated=item.get('updated', ''),
                        tags=item.get('tags', [])
                    )
                    models.append(model_info)

                # Continue until we have enough models or hit page limit
                if len(models) >= total_models_needed:
                    logger.info(f"Kaggle discovery: Collected {len(models)} models across {current_page} pages")
                    break

            logger.info(f"Kaggle model discovery completed: {len(models)} models found")
            return models

        except Exception as e:
            logger.warning(f"Failed to discover Kaggle models: {str(e)}")
            # Try fallback method
            return await self._discover_kaggle_models_fallback(query, model_type, page, page_size)

    async def _discover_kaggle_models_fallback(self, query: str, model_type: str, page: int, page_size: int) -> List[ModelInfo]:
        """Fallback method for Kaggle model discovery using alternative approaches"""
        try:
            logger.info("Using fallback method for Kaggle model discovery")

            # Method 1: Try datasets endpoint with model-related search terms
            try:
                models = await self._discover_kaggle_datasets_as_models(query, model_type, page, page_size)
                if models:
                    logger.info(f"Fallback method 1 successful: found {len(models)} models")
                    return models
            except Exception as e:
                logger.warning(f"Fallback method 1 failed: {str(e)}")

            # Method 2: Return cached results if available
            if hasattr(self, 'cache') and 'kaggle_fallback' in self.cache:
                logger.info("Using cached Kaggle models as fallback")
                cached_result = self.cache['kaggle_fallback']
                if isinstance(cached_result, ModelDiscoveryResult):
                    return cached_result.models
                return cached_result

            # Method 3: Return sample models for demonstration
            logger.info("Using sample models as final fallback")
            return self._get_sample_kaggle_models(query, model_type, page_size)

        except Exception as e:
            logger.error(f"Fallback method failed: {str(e)}")
            return []

    async def _discover_kaggle_datasets_as_models(self, query: str, model_type: str, page: int, page_size: int) -> List[ModelInfo]:
        """Discover models by searching Kaggle datasets with model-related keywords"""
        try:
            models = []
            max_pages = 50  # Limit pages for fallback method

            # Use datasets endpoint but filter for model-related content
            search_url = "https://www.kaggle.com/api/v1/datasets/list"

            # Enhance search query with model-related terms
            model_search_terms = ["model", "neural", "network", "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "onnx", "h5", "pb", "tflite", "pt", "pth"]
            enhanced_query = f"{query} {' '.join(model_search_terms)}" if query else ' '.join(model_search_terms)

            for current_page in range(1, max_pages + 1):
                params = {
                    'search': enhanced_query,
                    'page': current_page,
                    'pageSize': 50,  # Increased page size for more efficient fetching
                    'sortBy': 'hottest'
                }

                if model_type != "all":
                    type_mapping = {
                        "vision": ["computer-vision", "image-classification", "object-detection"],
                        "audio": ["speech-recognition", "audio-data"],
                        "text": ["nlp", "text-data", "language-model"],
                        "neural": ["deep-learning", "neural-networks", "tensorflow", "pytorch"],
                        "multimodal": ["multimodal", "vision-language"],
                        "math": ["mathematics", "statistics"],
                        "web": ["web-scraping", "api"]
                    }
                    if model_type in type_mapping:
                        params['tags'] = ','.join(type_mapping[model_type])

                url = f"{search_url}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'CoreAI3D-Dashboard/1.0',
                    'Accept': 'application/json'
                })

                success = False
                for attempt in range(self.max_retries):
                    try:
                        with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                            data = json.loads(response.read().decode('utf-8'))
                            if isinstance(data, list):
                                page_models = data
                            else:
                                page_models = data.get('datasetList', [])
                            success = True
                            break
                    except urllib.error.HTTPError as e:
                        if e.code == 400:
                            logger.warning(f"Datasets fallback also got 400 error, parameters may be invalid")
                            return []
                        elif e.code == 429:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                        else:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        break

                if not success:
                    break

                if not page_models:
                    break

                for item in page_models:
                    # Check if this dataset contains model files
                    if self._is_model_dataset(item):
                        model_info = ModelInfo(
                            name=item.get('title', item.get('ref', 'Unknown')),
                            type=self._classify_kaggle_model_type(item),
                            size=f"{item.get('size', 0) / (1024*1024):.1f} MB",
                            version=item.get('versionNumber', '1.0'),
                            status="Available",
                            platform="Kaggle",
                            download_url=f"https://www.kaggle.com/{item.get('ref', '')}",
                            description=item.get('description', ''),
                            author=item.get('ownerUser', ''),
                            downloads=item.get('downloadCount', 0),
                            likes=item.get('voteCount', 0),
                            last_updated=item.get('lastUpdated', ''),
                            tags=item.get('tags', [])
                        )
                        models.append(model_info)

                if len(models) >= page_size * 50:  # Get 50x the requested size for comprehensive discovery
                    break

            logger.info(f"Kaggle datasets fallback found {len(models)} potential models")
            return models

        except Exception as e:
            logger.error(f"Error in datasets fallback: {str(e)}")
            return []

    def _is_model_dataset(self, item: Dict[str, Any]) -> bool:
        """Check if a Kaggle dataset contains model files"""
        try:
            # Check file types
            file_types = item.get('fileTypes', [])
            model_extensions = ['model', 'onnx', 'pb', 'tflite', 'pt', 'pth', 'h5', 'pickle', 'pkl']

            if any(ext in file_types for ext in model_extensions):
                return True

            # Check tags for model-related keywords
            tags = [tag.lower() for tag in item.get('tags', [])]
            model_keywords = ['model', 'neural', 'network', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras']

            if any(keyword in tag for keyword in model_keywords for tag in tags):
                return True

            # Check title and description
            title = item.get('title', '').lower()
            description = item.get('description', '').lower()

            if any(keyword in title or keyword in description for keyword in model_keywords):
                return True

            return False

        except Exception:
            return False

    def _get_sample_kaggle_models(self, query: str, model_type: str, page_size: int) -> List[ModelInfo]:
        """Return sample Kaggle models for demonstration when API is unavailable"""
        try:
            sample_models = [
                ModelInfo(
                    name="ResNet-50 Image Classification",
                    type="Vision",
                    size="98 MB",
                    version="1.0.0",
                    status="Available",
                    platform="Kaggle",
                    download_url="https://www.kaggle.com/pytorch/resnet50",
                    description="ResNet-50 model for image classification tasks",
                    author="PyTorch",
                    downloads=15000,
                    likes=450,
                    last_updated="2024-01-15",
                    tags=["computer-vision", "image-classification", "resnet", "pytorch"]
                ),
                ModelInfo(
                    name="BERT Base Uncased",
                    type="Text",
                    size="420 MB",
                    version="2.1.0",
                    status="Available",
                    platform="Kaggle",
                    download_url="https://www.kaggle.com/huggingface/bert-base-uncased",
                    description="BERT base model for natural language processing",
                    author="HuggingFace",
                    downloads=25000,
                    likes=1200,
                    last_updated="2024-01-20",
                    tags=["nlp", "bert", "transformers", "text"]
                ),
                ModelInfo(
                    name="YOLOv5 Small Object Detection",
                    type="Vision",
                    size="14 MB",
                    version="6.1.0",
                    status="Available",
                    platform="Kaggle",
                    download_url="https://www.kaggle.com/ultralytics/yolov5",
                    description="YOLOv5 small model for real-time object detection",
                    author="Ultralytics",
                    downloads=35000,
                    likes=2100,
                    last_updated="2024-01-10",
                    tags=["object-detection", "yolo", "computer-vision", "pytorch"]
                ),
                ModelInfo(
                    name="Whisper Tiny Speech Recognition",
                    type="Audio",
                    size="39 MB",
                    version="1.2.0",
                    status="Available",
                    platform="Kaggle",
                    download_url="https://www.kaggle.com/openai/whisper-tiny",
                    description="OpenAI Whisper tiny model for speech recognition",
                    author="OpenAI",
                    downloads=18000,
                    likes=890,
                    last_updated="2024-01-25",
                    tags=["speech-recognition", "audio", "whisper", "transformers"]
                ),
                ModelInfo(
                    name="DistilBERT Text Classification",
                    type="Text",
                    size="66 MB",
                    version="1.0.0",
                    status="Available",
                    platform="Kaggle",
                    download_url="https://www.kaggle.com/huggingface/distilbert-base-uncased",
                    description="DistilBERT model for efficient text classification",
                    author="HuggingFace",
                    downloads=12000,
                    likes=650,
                    last_updated="2024-01-18",
                    tags=["nlp", "distilbert", "text-classification", "efficient"]
                )
            ]

            # Filter by query and model type
            filtered_models = []
            for model in sample_models:
                # Text search
                if query:
                    search_text = f"{model.name} {model.description} {model.author}".lower()
                    if query.lower() not in search_text:
                        continue

                # Type filter
                if model_type != "all" and model.type.lower() != model_type.lower():
                    continue

                filtered_models.append(model)

            # Limit to requested page size
            return filtered_models[:page_size]

        except Exception as e:
            logger.error(f"Error creating sample models: {str(e)}")
            return []

    async def _discover_huggingface_models_comprehensive(self, query: str, model_type: str, page: int, page_size: int) -> List[ModelInfo]:
        """Discover models from HuggingFace with comprehensive pagination across multiple pages"""
        try:
            models = []
            total_models_needed = page_size * 200  # Get 200x the requested page size for comprehensive discovery
            max_pages = 1000  # Allow up to 1000 pages to get comprehensive results

            # HuggingFace API search - comprehensive discovery across multiple pages
            search_url = "https://huggingface.co/api/models"

            # Start from page 1 and go up to max_pages or until we have enough models
            for current_page in range(1, max_pages + 1):
                # Emit progress signal
                self.discovery_progress.emit("HuggingFace", current_page, min(max_pages, 100))  # Cap progress display at 100 for UI

                params = {
                    'search': query,
                    'limit': 100,  # HF API maximum limit per request
                    'sort': 'downloads',
                    'direction': '-1'
                }

                if model_type != "all":
                    # Map model types to HF pipeline tags - comprehensive mapping
                    pipeline_mapping = {
                        "vision": ["image-classification", "object-detection", "image-segmentation", "depth-estimation", "image-to-image", "image-to-text"],
                        "audio": ["audio-classification", "automatic-speech-recognition", "text-to-speech", "audio-to-audio", "speech-emotion-recognition"],
                        "text": ["text-classification", "text-generation", "question-answering", "translation", "summarization", "fill-mask", "token-classification", "text2text-generation", "conversational"],
                        "multimodal": ["visual-question-answering", "image-to-text", "text-to-image", "document-question-answering", "video-classification"],
                        "neural": ["feature-extraction", "sentence-similarity", "table-question-answering"],
                        "math": ["text-generation", "text2text-generation"],  # Math models often use text generation
                        "web": ["text-generation", "question-answering", "text2text-generation"]  # Web-related models
                    }
                    if model_type in pipeline_mapping:
                        # Use first pipeline tag for this request
                        params['pipeline_tag'] = pipeline_mapping[model_type][0]

                # Make request with retry logic
                url = f"{search_url}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'CoreAI3D-Dashboard/1.0'})

                success = False
                for attempt in range(self.max_retries):
                    try:
                        with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                            data = json.loads(response.read().decode('utf-8'))
                            success = True
                            break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:  # Rate limited
                            if attempt < self.max_retries - 1:
                                logger.warning(f"HuggingFace API rate limited, retrying in {self.retry_delay}s (attempt {attempt + 1}/{self.max_retries})")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            else:
                                logger.warning(f"HuggingFace API rate limited after {current_page} pages, collected {len(models)} models")
                                return models
                        elif e.code == 404:
                            logger.warning(f"HuggingFace API endpoint not found")
                            return models
                        else:
                            logger.warning(f"HuggingFace API error on page {current_page}: {e.code}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            break
                    except Exception as e:
                        logger.warning(f"Error fetching HuggingFace page {current_page} (attempt {attempt + 1}): {str(e)}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        break

                if not success:
                    logger.warning(f"Failed to fetch HuggingFace page {current_page} after {self.max_retries} attempts")
                    continue

                # Process the data
                if not data:
                    # No more results, stop pagination
                    break

                for item in data:
                    model_info = ModelInfo(
                        name=item.get('id', 'Unknown').split('/')[-1],
                        type=self._classify_hf_model_type(item),
                        size="Unknown",  # HF doesn't provide size in search
                        version=item.get('sha', 'main')[:8],
                        status="Available",
                        platform="HuggingFace",
                        download_url=f"https://huggingface.co/{item.get('id', '')}",
                        description=item.get('description', ''),
                        author=item.get('id', '').split('/')[0],
                        downloads=item.get('downloads', 0),
                        likes=item.get('likes', 0),
                        last_updated=item.get('lastModified', ''),
                        tags=item.get('tags', [])
                    )
                    models.append(model_info)

                # Continue until we have enough models or hit page limit
                if len(models) >= total_models_needed:
                    logger.info(f"HuggingFace discovery: Collected {len(models)} models across {current_page} pages")
                    break

            logger.info(f"HuggingFace model discovery completed: {len(models)} models found")
            return models

        except Exception as e:
            logger.warning(f"Failed to discover HuggingFace models: {str(e)}")
            return []

    def _get_local_models(self, query: str, model_type: str) -> List[ModelInfo]:
        """Get locally available models"""
        try:
            models_dir = os.path.join(self.config.data_dir, "models")
            if not os.path.exists(models_dir):
                return []

            models = []
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if it matches search criteria
                    if query.lower() in item.lower():
                        model_info = ModelInfo(
                            name=item,
                            type="Local",
                            size=self._get_dir_size(item_path),
                            version="1.0",
                            status="Downloaded",
                            platform="Local",
                            download_url=item_path,
                            description=f"Local model: {item}",
                            author="Local",
                            downloads=0,
                            likes=0,
                            last_updated=datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat(),
                            tags=[]
                        )
                        models.append(model_info)

            return models

        except Exception as e:
            logger.warning(f"Failed to get local models: {str(e)}")
            return []

    def _filter_models(self, models: List[ModelInfo], query: str, model_type: str) -> List[ModelInfo]:
        """Filter models based on search criteria"""
        filtered = []

        for model in models:
            # Text search
            if query:
                search_text = f"{model.name} {model.description} {model.author}".lower()
                if query.lower() not in search_text:
                    continue

            # Type filter
            if model_type != "all" and model.type.lower() != model_type.lower():
                continue

            filtered.append(model)

        return filtered

    def _classify_kaggle_model_type(self, item: Dict[str, Any]) -> str:
        """Classify Kaggle dataset type"""
        tags = item.get('tags', [])
        title = item.get('title', '').lower()

        if any(tag in ['computer-vision', 'image-classification'] for tag in tags) or 'vision' in title:
            return "Vision"
        elif any(tag in ['audio-data', 'speech'] for tag in tags) or 'audio' in title:
            return "Audio"
        elif any(tag in ['nlp', 'text'] for tag in tags) or 'text' in title:
            return "Text"
        elif any(tag in ['deep-learning', 'neural'] for tag in tags) or 'neural' in title:
            return "Neural"
        else:
            return "Dataset"

    def _classify_hf_model_type(self, item: Dict[str, Any]) -> str:
        """Classify HuggingFace model type"""
        pipeline = item.get('pipeline_tag', '')
        tags = item.get('tags', [])

        if pipeline in ['image-classification', 'object-detection', 'image-segmentation']:
            return "Vision"
        elif pipeline in ['audio-classification', 'automatic-speech-recognition']:
            return "Audio"
        elif pipeline in ['text-classification', 'text-generation', 'question-answering']:
            return "Text"
        elif pipeline in ['visual-question-answering', 'image-to-text']:
            return "Multimodal"
        elif 'bert' in item.get('id', '').lower() or 'gpt' in item.get('id', '').lower():
            return "Text"
        else:
            return "Other"

    def _get_dir_size(self, path: str) -> str:
        """Get directory size in human readable format"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)

            # Convert to human readable
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"

        except Exception:
            return "Unknown"

class CodeDebugger(QObject):
    """Code debugging and testing utilities"""

    test_completed = pyqtSignal(TestResult)
    debug_message = pyqtSignal(str, str)

    def __init__(self, client: CoreAI3DClient):
        super().__init__()
        self.client = client
        self.test_history: List[TestResult] = []

    async def run_api_test(self, test_type: str) -> TestResult:
        """Run API connectivity test"""
        start_time = time.time()

        try:
            if test_type == "health_check":
                result = await self.client.health_check()
                success = result.success
                message = "Health check passed" if success else "Health check failed"

            elif test_type == "system_metrics":
                result = await self.client.get_system_metrics()
                success = result.success
                message = f"Retrieved {len(result.data)} metrics"

            elif test_type == "module_status":
                result = await self.client.get_module_status()
                success = result.success
                message = f"Module status retrieved for {len(result.data)} modules"

            else:
                success = False
                message = f"Unknown test type: {test_type}"

            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"api_{test_type}_{int(time.time())}",
                test_type=test_type,
                success=success,
                duration=duration,
                message=message,
                data=result.data if success else {}
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"api_{test_type}_{int(time.time())}",
                test_type=test_type,
                success=False,
                duration=duration,
                message=str(e)
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

    async def run_module_test(self, module_name: str, test_data: Dict[str, Any]) -> TestResult:
        """Run module functionality test"""
        start_time = time.time()

        try:
            if module_name == "vision":
                result = await self.client.analyze_image(test_data.get('image_path', ''))
                success = result.success
                message = "Vision analysis completed" if success else "Vision analysis failed"

            elif module_name == "audio":
                result = await self.client.analyze_audio(test_data.get('audio_path', ''))
                success = result.success
                message = "Audio analysis completed" if success else "Audio analysis failed"

            elif module_name == "system":
                result = await self.client.get_system_metrics()
                success = result.success
                message = "System metrics retrieved" if success else "System metrics failed"

            elif module_name == "web":
                result = await self.client.search_web(test_data.get('query', ''))
                success = result.success
                message = "Web search completed" if success else "Web search failed"

            elif module_name == "math":
                result = await self.client.calculate(test_data.get('expression', ''))
                success = result.success
                message = "Math calculation completed" if success else "Math calculation failed"

            else:
                success = False
                message = f"Unknown module: {module_name}"

            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"module_{module_name}_{int(time.time())}",
                test_type=f"module_{module_name}",
                success=success,
                duration=duration,
                message=message,
                data=result.data if success else {}
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"module_{module_name}_{int(time.time())}",
                test_type=f"module_{module_name}",
                success=False,
                duration=duration,
                message=str(e)
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

    async def run_system_automation_test(self, task_type: str, parameters: Dict[str, Any]) -> TestResult:
        """Run system automation test"""
        start_time = time.time()

        try:
            result = await self.client.automate_task(task_type, parameters)
            success = result.success

            if success:
                message = f"System automation task '{task_type}' completed successfully"
            else:
                message = f"System automation task '{task_type}' failed"

            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"automation_{task_type}_{int(time.time())}",
                test_type="system_automation",
                success=success,
                duration=duration,
                message=message,
                data=result.data if success else {}
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"automation_{task_type}_{int(time.time())}",
                test_type="system_automation",
                success=False,
                duration=duration,
                message=str(e)
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

    async def run_performance_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Run performance benchmarking test"""
        start_time = time.time()

        try:
            # Extract test configuration
            test_type = test_config.get('test_type', 'cpu')
            duration = test_config.get('duration', 10)
            iterations = test_config.get('iterations', 1000)
            concurrency = test_config.get('concurrency', 1)

            # Call performance test API
            result = await self.client.run_performance_test({
                'testType': test_type,
                'duration': duration,
                'iterations': iterations,
                'concurrency': concurrency
            })

            success = result.success

            if success:
                # Calculate performance metrics
                metrics = result.data.get('metrics', {})
                message = f"Performance test completed: {test_type.upper()} - {metrics.get('operations_per_second', 0):.2f} ops/sec"
            else:
                message = f"Performance test failed: {test_type}"

            total_duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"performance_{test_type}_{int(time.time())}",
                test_type="performance",
                success=success,
                duration=total_duration,
                message=message,
                data=result.data if success else {}
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"performance_{test_config.get('test_type', 'unknown')}_{int(time.time())}",
                test_type="performance",
                success=False,
                duration=duration,
                message=str(e)
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

    async def run_load_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Run load testing with concurrent requests"""
        start_time = time.time()

        try:
            # Extract test configuration
            test_type = test_config.get('test_type', 'api')
            duration = test_config.get('duration', 30)
            concurrent_users = test_config.get('concurrent_users', 10)
            requests_per_user = test_config.get('requests_per_user', 100)
            endpoint = test_config.get('endpoint', '/health')

            # Simulate load by making concurrent requests
            tasks = []
            for user_id in range(concurrent_users):
                task = asyncio.create_task(self._simulate_user_load(
                    user_id, endpoint, requests_per_user, duration
                ))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate load test metrics
            successful_requests = sum(1 for r in results if not isinstance(r, Exception) and r)
            failed_requests = len(results) - successful_requests
            total_requests = concurrent_users * requests_per_user
            actual_duration = time.time() - start_time

            success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
            requests_per_second = successful_requests / actual_duration if actual_duration > 0 else 0

            success = success_rate >= 95  # Consider test successful if 95%+ requests succeed

            message = f"Load test completed: {successful_requests}/{total_requests} successful ({success_rate:.1f}%), {requests_per_second:.2f} req/sec"

            test_result = TestResult(
                test_id=f"load_{test_type}_{int(time.time())}",
                test_type="load",
                success=success,
                duration=actual_duration,
                message=message,
                data={
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'total_requests': total_requests,
                    'success_rate': success_rate,
                    'requests_per_second': requests_per_second,
                    'concurrent_users': concurrent_users,
                    'duration': actual_duration
                }
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_id=f"load_{test_config.get('test_type', 'unknown')}_{int(time.time())}",
                test_type="load",
                success=False,
                duration=duration,
                message=str(e)
            )

            self.test_history.append(test_result)
            self.test_completed.emit(test_result)

            return test_result

    async def _simulate_user_load(self, user_id: int, endpoint: str, requests_per_user: int, duration: float) -> bool:
        """Simulate load for a single user"""
        try:
            start_time = time.time()
            successful_requests = 0

            # Make requests for the specified duration or number of requests
            for i in range(requests_per_user):
                if time.time() - start_time > duration:
                    break

                try:
                    # Make API request
                    if endpoint == '/health':
                        result = await self.client.health_check()
                    elif endpoint == '/system/metrics':
                        result = await self.client.get_system_metrics()
                    else:
                        result = await self.client.get(endpoint)

                    if result.success:
                        successful_requests += 1

                    # Small delay between requests
                    await asyncio.sleep(0.1)

                except Exception:
                    # Count failed requests
                    pass

            return successful_requests > 0

        except Exception:
            return False




# ===========================================================================
# Robust API key response parser
# ===========================================================================
class ApiKeyParser:
    """
    Extract an API key from arbitrary backend response data.

    The backend may return the key under many different shapes depending on
    version, endpoint, or middleware wrapping.  This parser checks every
    plausible location so the "key was generated but not found" error never
    happens for a valid response.

    Search order:
      1. Top-level: api_key, apiKey, apikey, key, secret, token, access_token
      2. Nested under: data, result, payload, response, keys
      3. Lists (recursive)
      4. Last resort: first string value longer than 8 chars
    """

    TOP_LEVEL_KEYS = (
        "api_key", "apiKey", "apikey",
        "key", "secret", "token",
        "access_token", "accessToken",
    )

    NESTED_KEYS = (
        "data", "result", "payload", "response",
        "keys", "key", "apiKey", "api_key",
    )

    @classmethod
    def extract(cls, response_data: Any) -> Optional[str]:
        """Extract a single API key string from arbitrary response data."""
        if response_data is None:
            return None

        if isinstance(response_data, dict):
            # 1. Top-level key names
            for k in cls.TOP_LEVEL_KEYS:
                v = response_data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

            # 2. Nested containers
            for nk in cls.NESTED_KEYS:
                v = response_data.get(nk)
                found = cls._extract_from_container(v)
                if found:
                    return found

            # 3. Last resort: first non-empty string longer than 8 chars
            for v in response_data.values():
                if isinstance(v, str) and v.strip() and len(v) > 8:
                    return v.strip()

        if isinstance(response_data, list):
            for item in response_data:
                found = cls.extract(item)
                if found:
                    return found

        return None

    @classmethod
    def _extract_from_container(cls, container: Any) -> Optional[str]:
        """Recursively extract a key from a nested dict/list."""
        if container is None:
            return None

        if isinstance(container, dict):
            for k in cls.TOP_LEVEL_KEYS:
                v = container.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in container.values():
                found = cls._extract_from_container(v)
                if found:
                    return found

        elif isinstance(container, list):
            for item in container:
                found = cls._extract_from_container(item)
                if found:
                    return found

        return None

    @classmethod
    def parse_response(cls, response: Any) -> Optional[str]:
        """Parse an APIResponse object (or dict) and return the API key."""
        if response is None:
            return None
        data = getattr(response, 'data', None)
        if data is None:
            data = response if isinstance(response, dict) else None
        return cls.extract(data)


# ===========================================================================
# Show-once API key dialog
# ===========================================================================
class ApiKeyDialog(QDialog):
    """
    Modal dialog that displays a newly generated API key exactly once.

    The key is shown in full with a one-click Copy button and a strong
    warning that it will never be displayed again.  Once the dialog is
    dismissed, the key is not retained by this dialog.
    """

    def __init__(self, api_key: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Generated")
        self.setModal(True)
        self.setFixedSize(560, 400)
        self._api_key = api_key
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QLabel("🔑 New API Key Generated")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(header)

        # Warning
        warning = QLabel(
            "⚠️  This key will <b>never be shown again</b>. "
            "Copy it now and store it securely."
        )
        warning.setWordWrap(True)
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warning.setStyleSheet("color: #b45309; font-size: 13px;")
        layout.addWidget(warning)

        # Key display
        key_frame = QFrame()
        key_frame.setStyleSheet(
            "QFrame { background: #f8fafc; border: 1px solid #cbd5e1; "
            "border-radius: 8px; }"
        )
        key_layout = QVBoxLayout(key_frame)
        key_layout.setContentsMargins(16, 16, 16, 16)

        key_label = QLabel("API Key:")
        key_label.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 600;")
        key_layout.addWidget(key_label)

        self.key_edit = QLineEdit(self._api_key)
        self.key_edit.setReadOnly(True)
        self.key_edit.setStyleSheet(
            "QLineEdit { font-family: 'Courier New', monospace; "
            "font-size: 13px; padding: 8px; }"
        )
        key_layout.addWidget(self.key_edit)

        # Copy button
        self.copy_btn = QPushButton("📋 Copy to Clipboard")
        self.copy_btn.clicked.connect(self._copy_key)
        key_layout.addWidget(self.copy_btn)

        self.copy_status = QLabel("")
        self.copy_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.copy_status.setStyleSheet("color: #059669; font-size: 12px; font-weight: 600;")
        key_layout.addWidget(self.copy_status)

        layout.addWidget(key_frame)

        # Spacer
        layout.addStretch()

        # Acknowledge button
        self.ack_btn = QPushButton("I Have Saved the Key")
        self.ack_btn.setDefault(True)
        self.ack_btn.setStyleSheet(
            "QPushButton { background: #3b82f6; color: white; "
            "font-weight: 600; padding: 8px 16px; border-radius: 6px; }"
            "QPushButton:hover { background: #2563eb; }"
        )
        self.ack_btn.clicked.connect(self.accept)
        layout.addWidget(self.ack_btn)

    def _copy_key(self) -> None:
        cb = QApplication.clipboard()
        cb.setText(self._api_key)
        self.copy_btn.setText("✅ Copied!")
        self.copy_btn.setEnabled(False)
        self.copy_status.setText("Key copied to clipboard")

    def key(self) -> str:
        return self._api_key


# ===========================================================================
# QThread worker for async API calls (API key generation, etc.)
# ===========================================================================
class AsyncWorker(QObject):
    """
    Worker that runs an async client method in a dedicated QThread with its
    own asyncio event loop, completely independent of qasync.

    This avoids the "Event loop is closed" error that occurs when qasync's
    event loop is stopped while pending asyncio tasks still reference it.

    The coroutine factory is invoked *inside* the worker thread, so any
    asyncio object it creates is bound to the worker's own event loop.
    """

    finished = pyqtSignal(object)  # result of the coroutine
    error = pyqtSignal(str)

    def __init__(self, coro_func, *args, **kwargs):
        super().__init__()
        self.coro_func = coro_func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        """Run the coroutine in a new event loop on this worker thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.coro_func(*self.args, **self.kwargs)
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("AsyncWorker error")
            self.error.emit(str(e))
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)


class ApiKeyWorker(AsyncWorker):
    """Worker used for API key generation (thin alias of :class:`AsyncWorker`)."""


async def _call_blocking(func, *args, **kwargs):
    """Run a blocking callable inside a worker's own event loop."""
    return func(*args, **kwargs)


class CoreAI3DDashboard(QMainWindow):
    """Main dashboard window"""

    # Chat signals for thread-safe UI updates
    chat_response_received = pyqtSignal(str)

    # Generic marshalling signal: worker threads use it to schedule work that
    # has to run on the GUI thread (Qt forbids touching widgets off-thread).
    gui_task = pyqtSignal(object)

    # Resend REST endpoint used to deliver beta keys by email.
    RESEND_API_URL = RESEND_API_URL

    def __init__(self):
        super().__init__()

        # Load the local .env (if any) so that backend endpoints and the
        # mail provider settings are available no matter how the dashboard
        # was launched.
        self._load_env_file()

        # Initialize configuration
        self.config = self._load_config()

        # Initialize qasync event loop manager
        self.async_manager = None

        # Initialize components
        self.client: Optional[CoreAI3DClient] = None
        self.sandbox_manager = SandboxManager(self.config)
        self.training_manager = TrainingDataManager(self.config)
        self.debugger: Optional[CodeDebugger] = None

        # A current asyncio event loop must exist in the main thread:
        # qasync's asyncSlot wrapper calls asyncio.ensure_future(), which
        # raises RuntimeError -- and aborts the whole process through PyQt --
        # when no loop is installed.
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # Initialize WebSocket client
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self._ws_shutdown = False

        # Chat response handler
        self.chat_response_handler = None

        # In-flight HTTP chat request (cancellable from the Chat tab)
        self._chat_task: Optional[asyncio.Task] = None

        # Beta keys worker thread (network calls run off the GUI thread)
        self._beta_worker_thread: Optional[QThread] = None
        self._beta_worker: Optional[AsyncWorker] = None
        self._email_worker_thread: Optional[QThread] = None

        # Connect chat signal to handler
        self.chat_response_received.connect(self._on_chat_response_received)

        # Connect the generic GUI marshalling signal
        self.gui_task.connect(self._execute_gui_task)

        # Deduplication state for chat responses (HTTP + WS may both deliver)
        self._chat_response_lock = threading.Lock()
        self._shown_chat_response_keys = set()

        # Initialize training argument variables - consolidated learning settings
        self.norm_min = -1.0
        self.norm_max = 1.0
        self.neurons = 10
        self.samples = 0
        self.epochs = 10
        self.layers = 3
        self.learning_rate = 0.01
        self.current_dataset_loaded = False

        # Initialize training button to prevent AttributeError
        self.training_button = None

        # Initialize learning config with default values
        self.learning_config = {
            'neurons_per_layer': self.neurons,
            'training_samples': self.samples,
            'epochs': self.epochs,
            'hidden_layers': self.layers,
            'learning_rate': self.learning_rate,
            'normalization_min': self.norm_min,
            'normalization_max': self.norm_max,
            'output_size': 1,
            'contains_text': False,
            'has_header': True,
            'dataset_name': 'default_dataset'
        }

        # Global async task serialization - single lock for all async operations
        self.global_async_lock = asyncio.Lock()
        self.current_async_task: Optional[asyncio.Task] = None
        self.async_task_queue: List[asyncio.Task] = []
        self.task_serialization_lock = threading.Lock()
        self.training_lock = asyncio.Lock()

        # Connection test synchronization
        self.connection_test_running = False

        # Task management system - enhanced for proper async handling
        self.active_tasks: Set[str] = set()
        self.task_lock = threading.Lock()
        self.training_in_progress = False

        # Async task tracking for proper cleanup
        self.pending_tasks: Set[asyncio.Task] = set()
        self.task_cleanup_lock = asyncio.Lock()

        # Learning configuration settings
        self.learning_config = {
            'normalization_min': self.norm_min,
            'normalization_max': self.norm_max,
            'neurons_per_layer': self.neurons,
            'training_samples': self.samples,
            'epochs': self.epochs,
            'hidden_layers': self.layers,
            'learning_rate': self.learning_rate,
            'data_source': 'file',  # file, database, web, etc.
            'model_type': 'neural_network',
            'enable_early_stopping': True,
            'validation_split': 0.2,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss_function': 'mse',
            'input_size': 1,
            'output_size': 1,
            'activation_function': 'relu',
            'dropout_rate': 0.2
        }

        # Initialize UI
        self.setup_ui()
        self.setup_connections()
        self.setup_timers()

        # Initialize WebSocket connection
        self._setup_websocket_connection()

        # Load initial data
        self.load_datasets()

        # Prime the CPU counter so the first metrics refresh reports a real
        # value.  The refresh itself is driven by the metrics timer, which
        # keeps the start-up path free of blocking work.
        try:
            psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.debug(f"Could not prime CPU metrics: {e}")

        # Defer client initialization until event loop is running
        self.client_initialized = False
        self._client_initializing = False
        # Load API key from environment variable or leave empty for user to configure
        self.config.api_key = os.environ.get('COREAI3D_API_KEY', '')
        if not self.config.api_key:
            logger.debug("No API key configured - set COREAI3D_API_KEY environment variable or configure in Settings tab")

        logger.info("CoreAI3D Dashboard initialized")

    def _load_env_file(self) -> None:
        """Load ``KEY=VALUE`` pairs from a local .env file into os.environ.

        Variables that are already set in the environment always win, so an
        explicit export overrides the file.
        """
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if not os.path.isfile(env_path):
            return
        try:
            with open(env_path, 'r', encoding='utf-8') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            logger.debug(f"Could not read .env file: {e}")

    def _load_config(self) -> DashboardConfig:
        """Load configuration from settings"""
        settings = QSettings("CoreAI3D", "Dashboard")

        config = DashboardConfig()
        # Env var wins; then the user's last-saved Settings value; then the
        # 127.0.0.1 default. We no longer clobber a saved host on every load.
        if env_api := os.environ.get('COREAI3D_API_URL'):
            config.api_url = env_api
        else:
            config.api_url = settings.value("api_url", default_api_url())
        if env_ws := os.environ.get('COREAI3D_WS_URL'):
            config.ws_url = env_ws
        else:
            config.ws_url = settings.value("ws_url", default_ws_url())
        config.api_key = settings.value("api_key", config.api_key)
        config.data_dir = settings.value("data_dir", config.data_dir)
        config.sandbox_timeout = settings.value("sandbox_timeout", config.sandbox_timeout, type=int)
        config.log_level = settings.value("log_level", config.log_level)
        config.enable_diagnostics = settings.value("enable_diagnostics", config.enable_diagnostics, type=bool)
        config.connection_timeout = settings.value(
            "connection_timeout", config.connection_timeout, type=float
        )
        config.chat_timeout = settings.value("chat_timeout", config.chat_timeout, type=int)
        config.resend_api_key = settings.value("resend_api_key", config.resend_api_key)
        config.resend_from_email = settings.value("resend_from_email", config.resend_from_email)
        config.beta_key = settings.value("beta_key", config.beta_key)

        # Environment variables win over the persisted values so a deployment
        # can override the GUI configuration without touching QSettings.
        env_resend_key = os.environ.get('RESEND_API_KEY', '')
        if env_resend_key:
            config.resend_api_key = env_resend_key
        env_resend_from = os.environ.get('RESEND_FROM_EMAIL', '')
        if env_resend_from:
            config.resend_from_email = env_resend_from
        env_admin_email = os.environ.get('COREAI3D_ADMIN_EMAIL', '')
        if env_admin_email:
            config.admin_email = env_admin_email
        env_admin_password = os.environ.get('COREAI3D_ADMIN_PASSWORD', '')
        if env_admin_password:
            config.admin_password = env_admin_password

        # Validate and sanitize API key
        if config.api_key:
            config.api_key = self._validate_api_key(config.api_key)

        return config

    def _validate_api_key(self, api_key: str) -> str:
        """Validate and sanitize API key"""
        if not api_key or not isinstance(api_key, str):
            logger.warning("API key is empty or not a string")
            return ""

        # Remove any whitespace
        api_key = api_key.strip()

        # Check for invalid characters that could cause HTTP header issues
        invalid_chars = ['\n', '\r', '\t', '\0']
        for char in invalid_chars:
            if char in api_key:
                logger.warning(f"API key contains invalid character '{repr(char)}', removing it")
                api_key = api_key.replace(char, '')

        # Check for control characters
        if any(ord(c) < 32 or ord(c) > 126 for c in api_key):
            logger.warning("API key contains non-printable characters, this may cause issues")
            # Remove non-printable characters
            api_key = ''.join(c for c in api_key if ord(c) >= 32 and ord(c) <= 126)

        # Check length (reasonable bounds)
        if len(api_key) > 1000:
            logger.warning(f"API key is unusually long ({len(api_key)} characters), truncating")
            api_key = api_key[:1000]

        # Allow shorter API keys for testing/development
        if len(api_key) < 5 and api_key:
            logger.warning(f"API key is very short ({len(api_key)} characters), this may indicate an issue")

        logger.info(f"API key validation complete: length={len(api_key)}, configured={bool(api_key)}")
        return api_key

    def _save_config(self):
        """Save configuration to settings"""
        settings = QSettings("CoreAI3D", "Dashboard")

        settings.setValue("api_url", self.config.api_url)
        settings.setValue("ws_url", self.config.ws_url)
        settings.setValue("api_key", self.config.api_key)
        settings.setValue("data_dir", self.config.data_dir)
        settings.setValue("sandbox_timeout", self.config.sandbox_timeout)
        settings.setValue("log_level", self.config.log_level)
        settings.setValue("enable_diagnostics", self.config.enable_diagnostics)
        settings.setValue("connection_timeout", self.config.connection_timeout)
        settings.setValue("chat_timeout", self.config.chat_timeout)
        settings.setValue("resend_api_key", self.config.resend_api_key)
        settings.setValue("resend_from_email", self.config.resend_from_email)
        settings.setValue("beta_key", self.config.beta_key)

        # Persist the window geometry so the dashboard reopens at the size and
        # position the user last used.
        try:
            settings.setValue("window_geometry", self.saveGeometry())
            settings.setValue("window_state", self.saveState())
        except Exception as e:
            logger.debug(f"Could not save window geometry: {e}")

    def _restore_window_geometry(self):
        """Restore the saved window geometry, or centre a default-sized window.

        The window is never opened full screen: it is a normal, resizable
        1280x800 window centred on the available screen area.
        """
        settings = QSettings("CoreAI3D", "Dashboard")
        geometry = settings.value("window_geometry")
        state = settings.value("window_state")

        restored = False
        if geometry is not None:
            try:
                restored = bool(self.restoreGeometry(geometry))
            except Exception as e:
                logger.debug(f"Could not restore window geometry: {e}")
                restored = False

        if state is not None:
            try:
                self.restoreState(state)
            except Exception as e:
                logger.debug(f"Could not restore window state: {e}")

        if not restored:
            self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
            self._center_window()

    def _center_window(self):
        """Centre the window on the available area of the primary screen."""
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                return
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())
        except Exception as e:
            logger.debug(f"Could not centre window: {e}")

    def _is_main_thread(self) -> bool:
        """True when the caller runs on the GUI (main) thread.

        Qt objects such as QTimer must be created and started on the thread
        that owns them; creating one from a worker thread is undefined
        behaviour and produces the "Timers cannot be started from another
        thread" warnings.
        """
        try:
            app = QApplication.instance()
            if app is None:
                return True
            return QThread.currentThread() == app.thread()
        except Exception:
            return True

    def _create_gui_timer(self, parent=None) -> QTimer:
        """Create a QTimer that belongs to the GUI thread.

        Every timer in the dashboard is owned by the GUI thread.  Creating one
        from a worker thread is undefined behaviour, so the request is logged
        and the timer is created empty -- it must then be started from the GUI
        thread.
        """
        if not self._is_main_thread():
            logger.warning(
                "QTimer created off the GUI thread - it must be started from "
                "the GUI thread to avoid cross-thread timer warnings."
            )
        return QTimer(parent if parent is not None else self)

    def _run_on_gui_thread(self, func, *args, **kwargs) -> None:
        """Schedule `func` to run on the GUI thread.

        Worker threads must never touch widgets directly; this marshals the
        call onto the GUI thread through a queued signal.
        """
        try:
            self.gui_task.emit(lambda: func(*args, **kwargs))
        except Exception as e:
            logger.error(f"Could not schedule GUI task: {e}")

    @pyqtSlot(object)
    def _execute_gui_task(self, func) -> None:
        """Execute a task that was scheduled from a worker thread."""
        try:
            func()
        except Exception as e:
            logger.error(f"GUI task failed: {e}", exc_info=True)

    def _select_file(self, caption: str, on_selected, file_filter: str = "",
                     on_cancelled=None) -> None:
        """Open a file picker without blocking the GUI thread.

        `on_selected` is called with the chosen path (empty when the user
        cancels).
        """
        dialog = QFileDialog(self, caption)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_filter:
            dialog.setNameFilter(file_filter)
        dialog.fileSelected.connect(on_selected)
        if on_cancelled is not None:
            dialog.rejected.connect(on_cancelled)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _select_files(self, caption: str, on_selected, file_filter: str = "") -> None:
        """Open a multi-file picker without blocking the GUI thread."""
        dialog = QFileDialog(self, caption)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if file_filter:
            dialog.setNameFilter(file_filter)
        dialog.filesSelected.connect(on_selected)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _select_directory(self, caption: str, on_selected) -> None:
        """Open a directory picker without blocking the GUI thread."""
        dialog = QFileDialog(self, caption)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        handled = {'done': False}

        def handle(path):
            if handled['done']:
                return
            handled['done'] = True
            on_selected(path)

        # PyQt6 exposes fileSelected/urlSelected (there is no
        # directorySelected), so both are wired up and the first one wins.
        dialog.fileSelected.connect(handle)
        dialog.urlSelected.connect(
            lambda url: handle(url.toLocalFile() if url.isLocalFile() else url.toString())
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _select_save_file(self, caption: str, on_selected, default_name: str = "",
                          file_filter: str = "") -> None:
        """Open a save-file picker without blocking the GUI thread."""
        dialog = QFileDialog(self, caption)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if default_name:
            dialog.selectFile(default_name)
        if file_filter:
            dialog.setNameFilter(file_filter)
        dialog.fileSelected.connect(on_selected)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _prompt_text(self, title: str, label: str, on_accepted, default_text: str = "",
                     on_cancelled=None) -> None:
        """Ask the user for a text value without blocking the GUI thread.

        `on_accepted` receives the entered text.
        """
        dialog = QInputDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText(label)
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setTextValue(default_text)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.textValueSelected.connect(on_accepted)
        if on_cancelled is not None:
            dialog.rejected.connect(on_cancelled)
        dialog.open()

    def _prompt_item(self, title: str, label: str, items, on_accepted) -> None:
        """Ask the user to pick one of `items` without blocking the GUI thread."""
        dialog = QInputDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText(label)
        dialog.setComboBoxItems([str(item) for item in items])
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.textValueSelected.connect(on_accepted)
        dialog.open()

    def _confirm(self, title: str, message: str, on_confirmed, on_cancelled=None) -> None:
        """Ask the user to confirm an action without blocking the GUI thread.

        A modal question box freezes the dashboard until it is dismissed, so
        the confirmation is shown modeless and the action runs from the
        button handler.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(message)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def on_clicked(button):
            try:
                if box.button(button) == QMessageBox.StandardButton.Yes:
                    on_confirmed()
                elif on_cancelled is not None:
                    on_cancelled()
            except Exception as e:
                logger.error(f"Confirmation handler failed: {e}", exc_info=True)
                self._notify("Error", str(e), level="critical")
            box.deleteLater()

        box.buttonClicked.connect(on_clicked)
        box.show()

    def _notify(self, title: str, message: str, level: str = "info") -> None:
        """Show a non-blocking notification to the user.

        Modal message boxes freeze the GUI thread (and hang automated tests),
        so notifications are displayed asynchronously and mirrored to the
        status bar.
        """
        icons = {
            "info": QMessageBox.Icon.Information,
            "warning": QMessageBox.Icon.Warning,
            "critical": QMessageBox.Icon.Critical,
        }
        try:
            box = QMessageBox(self)
            box.setIcon(icons.get(level, QMessageBox.Icon.Information))
            box.setWindowTitle(title)
            box.setText(message)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            box.show()
        except Exception as e:
            logger.debug(f"Could not display notification: {e}")

        if hasattr(self, "status_bar") and self.status_bar is not None:
            summary = message.splitlines()[0] if message else title
            self.status_bar.showMessage(f"{title}: {summary}"[:200], 5000)

    def setup_ui(self):
        """Setup main user interface"""
        self.setWindowTitle("CoreAI3D Linux Dashboard")

        # The dashboard is a normal resizable window: restore the geometry the
        # user last used, or fall back to a centred 1280x800 window.
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_dashboard_tab()
        self.create_training_tab()
        self.create_training_configuration_tab()
        self.create_testing_tab()
        self.create_predict_tab()
        self.create_load_model_tab()
        self.create_sandbox_tab()
        self.create_debugging_tab()
        self.create_linux_sandbox_tab()
        self.create_model_downloads_tab()
        self.create_code_generation_tab()
        self.create_chat_tab()
        self.create_neural_tab()
        self.create_settings_tab()
        self.create_beta_keys_tab()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create menu bar
        self.create_menu_bar()

        # Apply the window geometry last: setting the central widget and the
        # tab contents resizes the window to the layout's size hint, so the
        # requested size has to be applied once the UI is complete.
        self._restore_window_geometry()

        # Restore a previously persisted beta key and re-check its activation
        # status against the backend on launch.
        self._restore_beta_key_state()

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        refresh_action = QAction('Refresh All', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.refresh_all)
        tools_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        docs_action = QAction('Documentation', self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_dashboard_tab(self):
        """Create dashboard tab with system overview"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Dashboard")

        layout = QVBoxLayout(tab)

        # System status group
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout(status_group)

        # Status indicators
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.memory_label = QLabel("Memory Usage: 0%")
        self.disk_label = QLabel("Disk Usage: 0%")
        self.network_label = QLabel("Network: 0 KB/s")

        status_layout.addWidget(self.cpu_label, 0, 0)
        status_layout.addWidget(self.memory_label, 0, 1)
        status_layout.addWidget(self.disk_label, 1, 0)
        status_layout.addWidget(self.network_label, 1, 1)

        # Progress bars
        self.cpu_bar = QProgressBar()
        self.memory_bar = QProgressBar()
        self.disk_bar = QProgressBar()

        status_layout.addWidget(self.cpu_bar, 2, 0)
        status_layout.addWidget(self.memory_bar, 2, 1)
        status_layout.addWidget(self.disk_bar, 3, 0)

        layout.addWidget(status_group)

        # Quick actions group
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)

        test_connection_btn = QPushButton("Test Connection")
        test_connection_btn.clicked.connect(self.test_connection)
        actions_layout.addWidget(test_connection_btn)

        refresh_metrics_btn = QPushButton("Refresh Metrics")
        refresh_metrics_btn.clicked.connect(self.update_system_metrics)
        actions_layout.addWidget(refresh_metrics_btn)

        layout.addWidget(actions_group)

        # Health status
        health_group = QGroupBox("System Health")
        health_layout = QVBoxLayout(health_group)

        self.health_score_label = QLabel("Overall Health Score: 100%")
        health_layout.addWidget(self.health_score_label)

        self.health_bar = QProgressBar()
        self.health_bar.setRange(0, 100)
        self.health_bar.setValue(100)
        health_layout.addWidget(self.health_bar)

        layout.addWidget(health_group)

    def create_training_tab(self):
        """Create training data management tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Training Data")

        layout = QVBoxLayout(tab)

        # Dataset list
        list_group = QGroupBox("Datasets")
        list_layout = QVBoxLayout(list_group)

        self.dataset_list = QListWidget()
        self.dataset_list.itemClicked.connect(self.on_dataset_selected)
        list_layout.addWidget(self.dataset_list)

        # Dataset buttons
        dataset_buttons = QHBoxLayout()

        new_dataset_btn = QPushButton("New Dataset")
        new_dataset_btn.clicked.connect(self.create_dataset)
        dataset_buttons.addWidget(new_dataset_btn)

        self.load_dataset_button = QPushButton("Load Dataset")
        self.load_dataset_button.clicked.connect(lambda: asyncio.ensure_future(self.load_dataset()))
        dataset_buttons.addWidget(self.load_dataset_button)

        # File type selection for upload
        self.upload_file_type_combo = QComboBox()
        self.upload_file_type_combo.addItems([
            "CSV", "JSON", "XML", "PDF", "DOCX", "DOC", "TXT",
            "MP4", "AVI", "MOV", "MKV", "MP3", "WAV", "FLAC", "AAC",
            "JPG", "JPEG", "PNG", "BMP", "GIF"
        ])
        dataset_buttons.addWidget(QLabel("Upload:"))
        dataset_buttons.addWidget(self.upload_file_type_combo)

        upload_file_btn = QPushButton("Upload File")
        upload_file_btn.clicked.connect(lambda: asyncio.ensure_future(self.upload_file()))
        dataset_buttons.addWidget(upload_file_btn)

        delete_dataset_btn = QPushButton("Delete Dataset")
        delete_dataset_btn.clicked.connect(self.delete_dataset)
        dataset_buttons.addWidget(delete_dataset_btn)

        list_layout.addLayout(dataset_buttons)
        layout.addWidget(list_group)

        # File Learning Controls
        file_learning_group = QGroupBox("Learn from Files")
        file_learning_layout = QVBoxLayout(file_learning_group)

        self.file_dataset_name = QLineEdit("file_learned_dataset")
        file_learning_form = QFormLayout()
        file_learning_form.addRow("Dataset Name:", self.file_dataset_name)

        self.file_learning_mode_combo = QComboBox()
        self.file_learning_mode_combo.addItems(["Single Directory", "Multiple Files"])
        self.file_learning_mode_combo.currentIndexChanged.connect(self._on_file_learning_mode_changed)
        file_learning_form.addRow("Learning Mode:", self.file_learning_mode_combo)

        self.file_output_size = QSpinBox()
        self.file_output_size.setRange(1, 1000)
        self.file_output_size.setValue(1)
        file_learning_form.addRow("Output Size:", self.file_output_size)

        file_learning_layout.addLayout(file_learning_form)

        directory_layout = QHBoxLayout()
        self.file_directory_edit = QLineEdit()
        self.file_directory_edit.setReadOnly(True)
        browse_directory_btn = QPushButton("Browse Directory...")
        browse_directory_btn.clicked.connect(self._browse_learning_directory)
        directory_layout.addWidget(self.file_directory_edit)
        directory_layout.addWidget(browse_directory_btn)
        self.file_directory_widget = QWidget()
        self.file_directory_widget.setLayout(directory_layout)
        file_learning_layout.addWidget(self.file_directory_widget)

        self.file_learning_list = QListWidget()
        self.file_learning_list.setMaximumHeight(100)
        self.file_learning_list.setFixedHeight(100)
        file_browse_btn = QPushButton("Add Files...")
        file_browse_btn.clicked.connect(self._add_learning_files)
        file_remove_btn = QPushButton("Remove Selected")
        file_remove_btn.clicked.connect(self._remove_learning_files)
        file_buttons_layout = QHBoxLayout()
        file_buttons_layout.addWidget(file_browse_btn)
        file_buttons_layout.addWidget(file_remove_btn)
        self.file_list_widget = QWidget()
        self.file_list_widget.setLayout(file_buttons_layout)
        file_learning_layout.addWidget(self.file_learning_list)
        file_learning_layout.addWidget(self.file_list_widget)
        self.file_list_widget.setVisible(False)

        self.learn_files_btn = QPushButton("Learn from Files")
        self.learn_files_btn.clicked.connect(self._learn_from_files)
        file_learning_layout.addWidget(self.learn_files_btn)

        layout.addWidget(file_learning_group)

        # Dataset details
        details_group = QGroupBox("Dataset Details")
        details_layout = QVBoxLayout(details_group)

        self.dataset_info = QTextEdit()
        self.dataset_info.setMaximumHeight(150)
        details_layout.addWidget(self.dataset_info)

        # File management
        file_buttons = QHBoxLayout()

        add_file_btn = QPushButton("Add File")
        add_file_btn.clicked.connect(self.add_file_to_dataset)
        file_buttons.addWidget(add_file_btn)

        remove_file_btn = QPushButton("Remove File")
        remove_file_btn.clicked.connect(self.remove_file_from_dataset)
        file_buttons.addWidget(remove_file_btn)

        details_layout.addLayout(file_buttons)

        self.file_list = QListWidget()
        details_layout.addWidget(self.file_list)

        layout.addWidget(details_group)

    def create_testing_tab(self):
        """Create testing and debugging tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Testing")

        layout = QVBoxLayout(tab)

        # Test configuration
        config_group = QGroupBox("Test Configuration")
        config_layout = QFormLayout(config_group)

        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems([
            "API Connectivity",
            "Module Functionality",
            "System Automation",
            "Performance Test",
            "Load Test"
        ])
        self.test_type_combo.currentTextChanged.connect(lambda: self.on_test_type_changed())
        config_layout.addRow("Test Type:", self.test_type_combo)

        self.test_module_combo = QComboBox()
        self.test_module_combo.addItems([
            "Vision Module",
            "Audio Module",
            "System Module",
            "Web Module",
            "Math Module"
        ])
        config_layout.addRow("Module:", self.test_module_combo)

        layout.addWidget(config_group)

        # System Automation specific configuration
        self.automation_config_group = QGroupBox("System Automation Configuration")
        automation_layout = QFormLayout(self.automation_config_group)

        self.automation_task_combo = QComboBox()
        self.automation_task_combo.addItems([
            "file_backup",
            "system_cleanup",
            "log_rotation",
            "service_restart",
            "disk_space_check",
            "network_test",
            "process_monitor",
            "custom_script"
        ])
        automation_layout.addRow("Task Type:", self.automation_task_combo)

        self.automation_params_text = QTextEdit()
        self.automation_params_text.setPlaceholderText('Enter task parameters as JSON...\n\nExamples:\n{"source_path": "/home/user", "destination_path": "/backup"}\n{"service_name": "nginx"}\n{"script_path": "/path/to/script.sh"}')
        self.automation_params_text.setMaximumHeight(100)
        automation_layout.addRow("Parameters:", self.automation_params_text)

        layout.addWidget(self.automation_config_group)
        self.automation_config_group.setVisible(False)  # Hidden by default

        # Performance Test specific configuration
        self.performance_config_group = QGroupBox("Performance Test Configuration")
        performance_layout = QFormLayout(self.performance_config_group)

        self.performance_test_type_combo = QComboBox()
        self.performance_test_type_combo.addItems([
            "cpu",
            "memory",
            "disk",
            "network",
            "computation"
        ])
        performance_layout.addRow("Test Type:", self.performance_test_type_combo)

        self.performance_duration_spin = QSpinBox()
        self.performance_duration_spin.setRange(1, 300)
        self.performance_duration_spin.setValue(10)
        self.performance_duration_spin.setSuffix(" seconds")
        performance_layout.addRow("Duration:", self.performance_duration_spin)

        self.performance_iterations_spin = QSpinBox()
        self.performance_iterations_spin.setRange(100, 1000000)
        self.performance_iterations_spin.setValue(1000)
        self.performance_iterations_spin.setSingleStep(100)
        performance_layout.addRow("Iterations:", self.performance_iterations_spin)

        self.performance_concurrency_spin = QSpinBox()
        self.performance_concurrency_spin.setRange(1, 100)
        self.performance_concurrency_spin.setValue(1)
        performance_layout.addRow("Concurrency:", self.performance_concurrency_spin)

        layout.addWidget(self.performance_config_group)
        self.performance_config_group.setVisible(False)  # Hidden by default

        # Load Test specific configuration
        self.load_config_group = QGroupBox("Load Test Configuration")
        load_layout = QFormLayout(self.load_config_group)

        self.load_test_type_combo = QComboBox()
        self.load_test_type_combo.addItems([
            "api",
            "web",
            "database",
            "file_system",
            "computational"
        ])
        load_layout.addRow("Test Type:", self.load_test_type_combo)

        self.load_duration_spin = QSpinBox()
        self.load_duration_spin.setRange(10, 600)
        self.load_duration_spin.setValue(30)
        self.load_duration_spin.setSuffix(" seconds")
        load_layout.addRow("Duration:", self.load_duration_spin)

        self.load_concurrent_users_spin = QSpinBox()
        self.load_concurrent_users_spin.setRange(1, 1000)
        self.load_concurrent_users_spin.setValue(10)
        load_layout.addRow("Concurrent Users:", self.load_concurrent_users_spin)

        self.load_requests_per_user_spin = QSpinBox()
        self.load_requests_per_user_spin.setRange(10, 10000)
        self.load_requests_per_user_spin.setValue(100)
        load_layout.addRow("Requests per User:", self.load_requests_per_user_spin)

        self.load_endpoint_combo = QComboBox()
        self.load_endpoint_combo.addItems([
            "/health",
            "/system/metrics",
            "/api/status",
            "/module/status",
            "/predict"
        ])
        load_layout.addRow("Target Endpoint:", self.load_endpoint_combo)

        layout.addWidget(self.load_config_group)
        self.load_config_group.setVisible(False)  # Hidden by default

        # Test controls
        controls_layout = QHBoxLayout()

        run_test_btn = QPushButton("Run Test")
        run_test_btn.clicked.connect(self.run_test)
        controls_layout.addWidget(run_test_btn)

        clear_results_btn = QPushButton("Clear Results")
        clear_results_btn.clicked.connect(self.clear_test_results)
        controls_layout.addWidget(clear_results_btn)

        layout.addLayout(controls_layout)

        # Test results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)

        self.test_results_table = QTableWidget(0, 5)
        self.test_results_table.setHorizontalHeaderLabels([
            "Test ID", "Type", "Status", "Duration", "Message"
        ])
        self.test_results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.test_results_table)

        layout.addWidget(results_group)

    def create_predict_tab(self):
        """Create prediction tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Predict")

        layout = QVBoxLayout(tab)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)

        self.predict_model_combo = QComboBox()
        self.predict_model_combo.addItems([
            "Select Model...",
            "Vision Models",
            "Audio Models",
            "Text Models",
            "Multimodal Models",
            "Neural Networks",
            "Math Models",
            "Web Models"
        ])
        self.predict_model_combo.currentTextChanged.connect(lambda: self.on_predict_model_type_changed())
        model_layout.addRow("Model Type:", self.predict_model_combo)

        self.predict_specific_model_combo = QComboBox()
        self.predict_specific_model_combo.addItems([
            "Select Specific Model..."
        ])
        model_layout.addRow("Specific Model:", self.predict_specific_model_combo)

        layout.addWidget(model_group)

        # Input data configuration
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout(input_group)

        # Input type selection
        input_type_layout = QHBoxLayout()
        input_type_layout.addWidget(QLabel("Input Type:"))

        self.predict_input_type_combo = QComboBox()
        self.predict_input_type_combo.addItems([
            "Text",
            "Image",
            "Audio",
            "File",
            "URL"
        ])
        input_type_layout.addWidget(self.predict_input_type_combo)

        input_layout.addLayout(input_type_layout)

        # Data format options
        format_layout = QHBoxLayout()

        self.predict_contains_header_check = QCheckBox("--contains-header")
        self.predict_contains_header_check.setToolTip("Specify if the input data contains header row(s)")
        format_layout.addWidget(self.predict_contains_header_check)

        self.predict_contains_text_check = QCheckBox("--contains-text")
        self.predict_contains_text_check.setToolTip("Specify if the input data contains text columns")
        format_layout.addWidget(self.predict_contains_text_check)

        format_layout.addStretch()
        input_layout.addLayout(format_layout)

        # Input content area
        self.predict_input_text = QTextEdit()
        self.predict_input_text.setPlaceholderText("Enter input data here...\n\nFor CSV files, you can also:\n• Select 'File' input type and browse to your CSV file\n• Use the checkboxes below to specify data format\n• The system will automatically parse and analyze your CSV data")
        self.predict_input_text.setMaximumHeight(120)
        input_layout.addWidget(self.predict_input_text)

        # File/URL input
        file_input_layout = QHBoxLayout()

        self.predict_file_path_edit = QLineEdit()
        self.predict_file_path_edit.setPlaceholderText("File path or URL...")
        file_input_layout.addWidget(self.predict_file_path_edit)

        browse_file_btn = QPushButton("Browse...")
        browse_file_btn.clicked.connect(self.browse_predict_input_file)
        file_input_layout.addWidget(browse_file_btn)

        input_layout.addLayout(file_input_layout)

        layout.addWidget(input_group)

        # Prediction controls
        controls_layout = QHBoxLayout()

        run_predict_btn = QPushButton("Run Prediction")
        run_predict_btn.clicked.connect(self.run_prediction)
        controls_layout.addWidget(run_predict_btn)

        clear_predict_btn = QPushButton("Clear Results")
        clear_predict_btn.clicked.connect(self.clear_prediction_results)
        controls_layout.addWidget(clear_predict_btn)

        layout.addLayout(controls_layout)

        # Prediction progress
        self.predict_progress = QProgressBar()
        self.predict_progress.setVisible(False)
        layout.addWidget(self.predict_progress)

        # Prediction results
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)

        self.predict_results_text = QTextEdit()
        self.predict_results_text.setReadOnly(True)
        self.predict_results_text.setPlainText("Prediction results will appear here...")
        results_layout.addWidget(self.predict_results_text)

        layout.addWidget(results_group)

    def create_load_model_tab(self):
        """Create load model tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Load Model")

        layout = QVBoxLayout(tab)

        # Model loading configuration
        load_group = QGroupBox("Load Model")
        load_layout = QFormLayout(load_group)

        self.load_model_identifier_edit = QLineEdit()
        self.load_model_identifier_edit.setPlaceholderText("Enter dataset ID or model name...")
        load_layout.addRow("Model Identifier:", self.load_model_identifier_edit)

        self.load_model_type_combo = QComboBox()
        self.load_model_type_combo.addItems([
            "Auto-detect",
            "Vision",
            "Audio",
            "Text",
            "Multimodal",
            "Neural Network",
            "Math",
            "Web"
        ])
        load_layout.addRow("Model Type:", self.load_model_type_combo)

        layout.addWidget(load_group)

        # Load controls
        controls_layout = QHBoxLayout()

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(load_model_btn)

        clear_load_btn = QPushButton("Clear Results")
        clear_load_btn.clicked.connect(self.clear_load_results)
        controls_layout.addWidget(clear_load_btn)

        layout.addLayout(controls_layout)

        # Load progress
        self.load_progress = QProgressBar()
        self.load_progress.setVisible(False)
        layout.addWidget(self.load_progress)

        # Load results
        results_group = QGroupBox("Load Results")
        results_layout = QVBoxLayout(results_group)

        self.load_results_text = QTextEdit()
        self.load_results_text.setReadOnly(True)
        self.load_results_text.setPlainText("Model loading results will appear here...")
        results_layout.addWidget(self.load_results_text)

        layout.addWidget(results_group)

    def create_sandbox_tab(self):
        """Create Docker sandbox testing tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Docker Sandbox")

        layout = QVBoxLayout(tab)

        # Sandbox status
        status_group = QGroupBox("Docker Status")
        status_layout = QVBoxLayout(status_group)

        self.sandbox_status_label = QLabel("Docker: Not Running")
        status_layout.addWidget(self.sandbox_status_label)

        # Docker info
        info_label = QLabel("Docker provides container-based isolation with process and filesystem sandboxing")
        info_label.setWordWrap(True)
        status_layout.addWidget(info_label)

        layout.addWidget(status_group)

        # Sandbox configuration
        config_group = QGroupBox("Docker Configuration")
        config_layout = QFormLayout(config_group)

        self.sandbox_name_edit = QLineEdit("coreai3d_container")
        config_layout.addRow("Container Name:", self.sandbox_name_edit)

        self.startup_command_edit = QLineEdit("echo 'Docker container started'")
        config_layout.addRow("Startup Command:", self.startup_command_edit)

        self.networking_combo = QComboBox()
        self.networking_combo.addItems(["Enable", "Disable"])
        self.networking_combo.setCurrentText("Enable")
        config_layout.addRow("Networking:", self.networking_combo)

        self.drop_rights_check = QCheckBox("Run as Non-Root User")
        self.drop_rights_check.setChecked(True)
        config_layout.addWidget(self.drop_rights_check)

        layout.addWidget(config_group)

        # Sandbox controls
        controls_layout = QHBoxLayout()

        start_sandbox_btn = QPushButton("Start Container")
        start_sandbox_btn.clicked.connect(self.start_sandbox)
        controls_layout.addWidget(start_sandbox_btn)

        stop_sandbox_btn = QPushButton("Stop Container")
        stop_sandbox_btn.clicked.connect(self.stop_sandbox)
        controls_layout.addWidget(stop_sandbox_btn)

        layout.addLayout(controls_layout)

        # Sandbox logs
        logs_group = QGroupBox("Docker Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.sandbox_logs = QPlainTextEdit()
        self.sandbox_logs.setMaximumHeight(300)
        self.sandbox_logs.setReadOnly(True)
        logs_layout.addWidget(self.sandbox_logs)

        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.clear_sandbox_logs)
        logs_layout.addWidget(clear_logs_btn)

        layout.addWidget(logs_group)

    def create_debugging_tab(self):
        """Create debugging and diagnostics tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Debugging")

        layout = QVBoxLayout(tab)

        # System information
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout(info_group)

        self.system_info = QTextEdit()
        self.system_info.setMaximumHeight(200)
        info_layout.addWidget(self.system_info)

        refresh_info_btn = QPushButton("Refresh Info")
        refresh_info_btn.clicked.connect(self.update_system_info)
        info_layout.addWidget(refresh_info_btn)

        layout.addWidget(info_group)

        # Process monitor
        process_group = QGroupBox("Process Monitor")
        process_layout = QVBoxLayout(process_group)

        self.process_table = QTableWidget(0, 4)
        self.process_table.setHorizontalHeaderLabels([
            "PID", "Name", "CPU %", "Memory %"
        ])
        process_layout.addWidget(self.process_table)

        refresh_processes_btn = QPushButton("Refresh Processes")
        refresh_processes_btn.clicked.connect(self.update_process_list)
        process_layout.addWidget(refresh_processes_btn)

        layout.addWidget(process_group)

        # Debug controls
        debug_group = QGroupBox("Debug Controls")
        debug_layout = QHBoxLayout(debug_group)

        api_debug_btn = QPushButton("API Debug")
        api_debug_btn.clicked.connect(self.run_api_debug)
        debug_layout.addWidget(api_debug_btn)

        system_debug_btn = QPushButton("System Debug")
        system_debug_btn.clicked.connect(self.run_system_debug)
        debug_layout.addWidget(system_debug_btn)

        layout.addWidget(debug_group)

        # API Debug Results section
        api_debug_results_group = QGroupBox("API Debug Results")
        api_debug_layout = QVBoxLayout(api_debug_results_group)

        # API Debug output
        self.api_debug_output = QPlainTextEdit()
        self.api_debug_output.setReadOnly(True)
        self.api_debug_output.setMaximumHeight(300)
        api_debug_layout.addWidget(self.api_debug_output)

        # API Debug controls
        api_debug_controls = QHBoxLayout()

        clear_api_debug_btn = QPushButton("Clear Results")
        clear_api_debug_btn.clicked.connect(self.clear_api_debug_results)
        api_debug_controls.addWidget(clear_api_debug_btn)

        export_api_debug_btn = QPushButton("Export Debug Log")
        export_api_debug_btn.clicked.connect(self.export_api_debug_log)
        api_debug_controls.addWidget(export_api_debug_btn)

        api_debug_layout.addLayout(api_debug_controls)

        layout.addWidget(api_debug_results_group)

        # Debug output log
        debug_log_group = QGroupBox("Debug Messages")
        debug_log_layout = QVBoxLayout(debug_log_group)
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setFont(QFont("Monospace", 9))
        debug_log_layout.addWidget(self.debug_log)
        layout.addWidget(debug_log_group)

        # Connect Qt log handler to dashboard logger
        qt_handler = QtLogHandler(self.debug_log)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger('dashboard').addHandler(qt_handler)
        logging.getLogger(__name__).addHandler(qt_handler)

    def append_debug_message(self, message: str):
        """Append a debug message to the debug log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_log.append(f"[{timestamp}] {message}")

    def create_linux_sandbox_tab(self):
        """Create Linux Sandbox testing tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Linux Sandbox")

        layout = QVBoxLayout(tab)

        # Sandbox status
        status_group = QGroupBox("Linux Sandbox Status")
        status_layout = QVBoxLayout(status_group)

        self.linux_sandbox_status_label = QLabel("Linux Sandbox: Not Running")
        status_layout.addWidget(self.linux_sandbox_status_label)

        # Container selection
        container_group = QGroupBox("Container Selection")
        container_layout = QHBoxLayout(container_group)

        self.container_combo = QComboBox()
        self.container_combo.addItems([
            "ubuntu-sandbox",
            "kali-sandbox",
            "alpine-sandbox",
            "arch-sandbox",
            "debian-sandbox",
            "fedora-sandbox",
            "opensuse-sandbox"
        ])
        container_layout.addWidget(QLabel("Container:"))
        container_layout.addWidget(self.container_combo)

        refresh_containers_btn = QPushButton("Refresh")
        refresh_containers_btn.clicked.connect(self.refresh_containers)
        container_layout.addWidget(refresh_containers_btn)

        status_layout.addWidget(container_group)
        layout.addWidget(status_group)

        # Command input
        command_group = QGroupBox("Command Execution")
        command_layout = QVBoxLayout(command_group)

        self.linux_command_input = QLineEdit()
        self.linux_command_input.setPlaceholderText("Enter Linux command to execute...")
        self.linux_command_input.returnPressed.connect(self.execute_linux_command)
        command_layout.addWidget(self.linux_command_input)

        # Command buttons
        command_buttons = QHBoxLayout()

        execute_btn = QPushButton("Execute Command")
        execute_btn.clicked.connect(self.execute_linux_command)
        command_buttons.addWidget(execute_btn)

        start_container_btn = QPushButton("Start Container")
        start_container_btn.clicked.connect(self.start_linux_container)
        command_buttons.addWidget(start_container_btn)

        stop_container_btn = QPushButton("Stop Container")
        stop_container_btn.clicked.connect(self.stop_linux_container)
        command_buttons.addWidget(stop_container_btn)

        command_layout.addLayout(command_buttons)
        layout.addWidget(command_group)

        # Command output
        output_group = QGroupBox("Command Output")
        output_layout = QVBoxLayout(output_group)

        self.linux_output = QPlainTextEdit()
        self.linux_output.setMaximumHeight(300)
        self.linux_output.setReadOnly(True)
        output_layout.addWidget(self.linux_output)

        clear_output_btn = QPushButton("Clear Output")
        clear_output_btn.clicked.connect(self.clear_linux_output)
        output_layout.addWidget(clear_output_btn)

        layout.addWidget(output_group)

        # Training scenarios
        training_group = QGroupBox("Training Scenarios")
        training_layout = QVBoxLayout(training_group)

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "File System Operations",
            "Process Management",
            "Network Operations",
            "Package Management",
            "Text Processing",
            "Permissions and Security",
            "System Monitoring",
            "Scripting and Automation",
            "hacking-white",
            "hacking-gray",
            "hacking-black"
        ])
        training_layout.addWidget(QLabel("Training Scenario:"))
        training_layout.addWidget(self.scenario_combo)

        # Training controls
        training_buttons = QHBoxLayout()

        run_training_btn = QPushButton("Run Training")
        run_training_btn.clicked.connect(self.run_linux_training)
        training_buttons.addWidget(run_training_btn)

        generate_report_btn = QPushButton("Generate Report")
        generate_report_btn.clicked.connect(self.generate_training_report)
        training_buttons.addWidget(generate_report_btn)

        training_layout.addLayout(training_buttons)
        layout.addWidget(training_group)

        # Training progress
        self.linux_training_progress = QProgressBar()
        self.linux_training_progress.setVisible(False)
        layout.addWidget(self.linux_training_progress)

        # Training results
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_group)

        self.training_results = QTextEdit()
        self.training_results.setMaximumHeight(150)
        self.training_results.setReadOnly(True)
        results_layout.addWidget(self.training_results)

        layout.addWidget(results_group)

    def create_model_downloads_tab(self):
        """Create model downloads tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Model Downloads")

        layout = QVBoxLayout(tab)

        # Model search and filter
        search_group = QGroupBox("Model Search")
        search_layout = QHBoxLayout(search_group)

        self.model_search_input = QLineEdit()
        self.model_search_input.setPlaceholderText("Search models...")
        self.model_search_input.textChanged.connect(self.search_models)
        search_layout.addWidget(self.model_search_input)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "All Types",
            "Vision",
            "Audio",
            "Text",
            "Multimodal",
            "Neural",
            "Math",
            "Web"
        ])
        self.model_type_combo.currentTextChanged.connect(self.search_models)
        search_layout.addWidget(self.model_type_combo)

        self.platform_combo = QComboBox()
        self.platform_combo.addItems([
            "All Platforms",
            "Kaggle",
            "HuggingFace",
            "Local"
        ])
        self.platform_combo.currentTextChanged.connect(self.search_models)
        search_layout.addWidget(self.platform_combo)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_models)
        search_layout.addWidget(search_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        search_layout.addWidget(refresh_btn)

        layout.addWidget(search_group)

        # Pagination controls
        pagination_group = QGroupBox("Pagination")
        pagination_layout = QHBoxLayout(pagination_group)

        self.page_label = QLabel("Page 1 of 1")
        pagination_layout.addWidget(self.page_label)

        self.prev_page_btn = QPushButton("Previous")
        self.prev_page_btn.clicked.connect(self.previous_page)
        self.prev_page_btn.setEnabled(False)
        pagination_layout.addWidget(self.prev_page_btn)

        self.next_page_btn = QPushButton("Next")
        self.next_page_btn.clicked.connect(self.next_page)
        self.next_page_btn.setEnabled(False)
        pagination_layout.addWidget(self.next_page_btn)

        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["10", "25", "50", "100"])
        self.page_size_combo.setCurrentText("25")
        self.page_size_combo.currentTextChanged.connect(self.change_page_size)
        pagination_layout.addWidget(QLabel("Per page:"))
        pagination_layout.addWidget(self.page_size_combo)

        layout.addWidget(pagination_group)

        # Model list
        list_group = QGroupBox("Available Models")
        list_layout = QVBoxLayout(list_group)

        self.model_table = QTableWidget(0, 8)
        self.model_table.setHorizontalHeaderLabels([
            "Model Name", "Type", "Size", "Version", "Status", "Platform", "Downloads", "Updated"
        ])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.itemSelectionChanged.connect(self.on_model_selected)
        list_layout.addWidget(self.model_table)

        # Model list controls
        list_controls = QHBoxLayout()

        download_btn = QPushButton("Download Selected")
        download_btn.clicked.connect(self.download_selected_model)
        list_controls.addWidget(download_btn)

        cancel_download_btn = QPushButton("Cancel Download")
        cancel_download_btn.clicked.connect(self.cancel_model_download)
        list_controls.addWidget(cancel_download_btn)

        # Add source selection
        self.download_source_combo = QComboBox()
        self.download_source_combo.addItems(["Auto", "Kaggle", "HuggingFace", "Local"])
        self.download_source_combo.setCurrentText("Auto")
        list_controls.addWidget(QLabel("Source:"))
        list_controls.addWidget(self.download_source_combo)

        list_layout.addLayout(list_controls)

        # FastText embeddings download controls
        fasttext_layout = QHBoxLayout()
        fasttext_btn = QPushButton("Download FastText Embeddings")
        fasttext_btn.clicked.connect(self.download_fasttext_embeddings)
        fasttext_btn.setToolTip("Download FastText word embeddings (crawl-300d-2M)")
        fasttext_layout.addWidget(fasttext_btn)
        fasttext_layout.addStretch()
        list_layout.addLayout(fasttext_layout)

        layout.addWidget(list_group)

        # Download progress
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        progress_layout.addWidget(self.download_progress)

        self.download_status_label = QLabel("Ready")
        progress_layout.addWidget(self.download_status_label)

        # Download history/log
        self.download_log = QPlainTextEdit()
        self.download_log.setMaximumHeight(100)
        self.download_log.setReadOnly(True)
        progress_layout.addWidget(self.download_log)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_download_log)
        progress_layout.addWidget(clear_log_btn)

        layout.addWidget(progress_group)

        # Model information
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout(info_group)

        self.model_info = QTextEdit()
        self.model_info.setMaximumHeight(200)
        self.model_info.setReadOnly(True)
        info_layout.addWidget(self.model_info)

        layout.addWidget(info_group)

        # Initialize model discovery service
        self.model_discovery_service = ModelDiscoveryService(self.config)
        self.current_page = 1
        self.page_size = 25
        self.current_platform = "all"
        self.current_search_query = ""
        self.current_model_type = "all"
        self.current_discovery_result = None
        self.discovery_in_progress = False

        # Initialize with real model discovery - schedule properly
        QTimer.singleShot(1000, self._schedule_initial_model_discovery)

    def _schedule_initial_model_discovery(self):
        """Schedule initial model discovery on application startup"""
        try:
            # Call the async model discovery method
            self.discover_models_async()
        except Exception as e:
            logger.error(f"Error scheduling initial model discovery: {str(e)}")

    def refresh_containers(self):
        """Refresh available containers"""
        try:
            # This would typically query Docker for running containers
            # For now, just update the status
            self.linux_sandbox_status_label.setText("Linux Sandbox: Containers refreshed")
            self.status_bar.showMessage("Container list refreshed")
        except Exception as e:
            self._notify("Error", f"Failed to refresh containers: {str(e)}", level="warning")

    @qasync.asyncSlot()
    async def execute_linux_command(self):
        """Execute Linux command in sandbox"""
        if not self.client_initialized:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        command = self.linux_command_input.text().strip()
        if not command:
            self._notify("No Command", "Please enter a command to execute", level="warning")
            return

        container = self.container_combo.currentText()

        try:
            # Show progress
            self.linux_training_progress.setVisible(True)
            self.linux_training_progress.setRange(0, 0)

            # Execute command directly with qasync
            result = await self._execute_linux_command_async(container, command)
            self._handle_linux_command_result_direct(result)

        except Exception as e:
            self.linux_training_progress.setVisible(False)
            self.linux_output.appendPlainText(f"Error: {str(e)}")
            self.status_bar.showMessage("Command execution error")
            self._notify("Error", f"Failed to execute command: {str(e)}", level="warning")

    def _run_compose_command(self, service: str, command: Optional[List[str]] = None,
                             action: str = "exec") -> Tuple[bool, str]:
        """Run a docker compose command for a Linux Sandbox service.

        Args:
            service: docker-compose service name (e.g. ubuntu-sandbox).
            command: list of command + args for exec/run; ignored for up/stop.
            action: one of 'exec', 'up', 'stop'.
        Returns:
            (success, output_or_error_message)
        """
        try:
            if action == "up":
                cmd = ['docker', 'compose', 'up', '-d', service]
            elif action == "stop":
                cmd = ['docker', 'compose', 'stop', service]
            elif action == "exec":
                cmd = ['docker', 'compose', 'exec', '-T', service] + (command or [])
            else:
                return False, f"Unknown compose action: {action}"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if result.stderr.strip():
                    output += "\n" + result.stderr.strip()
                return True, output
            else:
                error = result.stderr.strip() or result.stdout.strip() or "docker compose command failed"
                return False, error

        except subprocess.TimeoutExpired:
            return False, "docker compose command timed out"
        except FileNotFoundError:
            return False, "docker compose not found"
        except Exception as e:
            return False, f"docker compose error: {str(e)}"

    async def _execute_linux_command_async(self, container: str, command: str):
        """Async part of Linux command execution"""
        # Run the command inside the selected docker-compose service container
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self._run_compose_command, container, ['/bin/sh', '-c', command], 'exec'
        )

        # Create result object similar to API response
        class CommandResult:
            def __init__(self, success, output, command):
                self.success = success
                self.data = {
                    'command': command,
                    'output': output,
                    'container': container
                }

        return CommandResult(success, output, command)

    def _handle_linux_command_result_direct(self, result):
        """Handle Linux command execution result directly"""
        try:
            self.linux_training_progress.setVisible(False)
            command = result.data.get('command', '')
            if result.success:
                self.linux_output.appendPlainText(f"$ {command}")
                output = result.data.get('output', '')
                if output:
                    self.linux_output.appendPlainText(output)
                else:
                    self.linux_output.appendPlainText("(no output)")
                self.linux_sandbox_status_label.setText(f"Linux Sandbox: Command executed in {result.data.get('container', '')}")
                self.status_bar.showMessage("Command executed successfully")
            else:
                self.linux_output.appendPlainText(f"$ {command}")
                error_msg = result.data.get('output', 'Unknown error')
                self.linux_output.appendPlainText(f"Error: {error_msg}")
                self.status_bar.showMessage("Command execution failed")
        except Exception as e:
            self.linux_training_progress.setVisible(False)
            self.linux_output.appendPlainText(f"Error: {str(e)}")
            self.status_bar.showMessage("Command execution error")

    @qasync.asyncSlot()
    async def start_linux_container(self):
        """Start Linux container"""
        if not self.client_initialized:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        container = self.container_combo.currentText()

        try:
            # Start container directly with qasync
            result = await self._start_linux_container_async(container)
            self._handle_start_container_result_direct(result)

        except Exception as e:
            self._notify("Error", f"Failed to start container: {str(e)}", level="warning")

    async def _start_linux_container_async(self, container: str):
        """Async part of container start"""
        # Start the docker-compose service corresponding to the selected container
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self._run_compose_command, container, None, 'up'
        )

        # Create result object similar to API response
        class StartResult:
            def __init__(self, success, container, output):
                self.success = success
                self.data = {'container': container, 'output': output} if success else {'error': output}

        return StartResult(success, container, output)

    def _handle_start_container_result_direct(self, result):
        """Handle container start result directly"""
        try:
            if result.success:
                self.linux_sandbox_status_label.setText(f"Linux Sandbox: {result.data.get('container', '')} started")
                self.status_bar.showMessage(f"Container {result.data.get('container', '')} started successfully")
            else:
                self._notify("Start Error", result.data.get('error', 'Unknown error'), level="warning")
        except Exception as e:
            self._notify("Error", f"Failed to start container: {str(e)}", level="warning")

    @qasync.asyncSlot()
    async def stop_linux_container(self):
        """Stop Linux container"""
        if not self.client_initialized:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        container = self.container_combo.currentText()

        try:
            # Stop container directly with qasync
            result = await self._stop_linux_container_async(container)
            self._handle_stop_container_result_direct(result)

        except Exception as e:
            self._notify("Error", f"Failed to stop container: {str(e)}", level="warning")

    async def _stop_linux_container_async(self, container: str):
        """Async part of container stop"""
        # Stop the docker-compose service corresponding to the selected container
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self._run_compose_command, container, None, 'stop'
        )

        # Create result object similar to API response
        class StopResult:
            def __init__(self, success, container, output):
                self.success = success
                self.data = {'container': container, 'output': output} if success else {'error': output}

        return StopResult(success, container, output)

    def _handle_stop_container_result_direct(self, result):
        """Handle container stop result directly"""
        try:
            if result.success:
                self.linux_sandbox_status_label.setText("Linux Sandbox: Container stopped")
                self.status_bar.showMessage(f"Container {result.data.get('container', '')} stopped successfully")
            else:
                self._notify("Stop Error", result.data.get('error', 'Unknown error'), level="warning")
        except Exception as e:
            self._notify("Error", f"Failed to stop container: {str(e)}", level="warning")

    def clear_linux_output(self):
        """Clear Linux command output"""
        self.linux_output.clear()

    @qasync.asyncSlot()
    async def run_linux_training(self):
        """Run Linux training scenario"""
        if not self.client_initialized:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        scenario = self.scenario_combo.currentText()
        container = self.container_combo.currentText()

        try:
            # Show progress
            self.linux_training_progress.setVisible(True)
            self.linux_training_progress.setRange(0, 0)

            # Run training directly with qasync
            result = await self._run_linux_training_async(container, scenario)
            self._handle_linux_training_result_direct(result)

        except Exception as e:
            self.linux_training_progress.setVisible(False)
            self.training_results.setPlainText(f"Training error: {str(e)}")
            self.status_bar.showMessage("Training error")
            self._notify("Error", f"Failed to run training: {str(e)}", level="warning")

    async def _run_linux_training_async(self, container: str, scenario: str):
        """Async part of Linux training"""
        # Run training scenario commands inside the selected docker-compose service container
        training_commands = {
            "File System Operations": "ls -la / && echo 'File system operations completed'",
            "Process Management": "ps aux | head -10 && echo 'Process management completed'",
            "Network Operations": "ping -c 3 8.8.8.8 && echo 'Network operations completed'",
            "Package Management": "which apt && echo 'Package management available' || echo 'Package management not available'",
            "Text Processing": "echo 'Hello World' | grep Hello && echo 'Text processing completed'",
            "Permissions and Security": "id && echo 'Permissions check completed'",
            "System Monitoring": "uptime && echo 'System monitoring completed'",
            "Scripting and Automation": "echo '#!/bin/bash\\necho \"Automation script\"' > test.sh && chmod +x test.sh && ./test.sh && echo 'Scripting completed'",
            "hacking-white": (
                "echo '=== WHITE HAT TRAINING: Authorized Penetration Testing ===' && "
                "echo 'Phase 1: Reconnaissance' && "
                "nmap -sV -O 127.0.0.1 2>/dev/null || echo 'Port scan simulation: Found ports 22,80,443,3306' && "
                "echo 'Phase 2: Vulnerability Assessment' && "
                "echo 'Checking for common vulnerabilities...' && "
                "echo '  [INFO] SSH version: OpenSSH 8.9' && "
                "echo '  [INFO] Apache version: 2.4.52' && "
                "echo '  [OK] No critical CVEs detected in detected services' && "
                "echo 'Phase 3: Report Generation' && "
                "echo 'Summary: System is well-patched. Recommended actions: enable fail2ban, update SSL ciphers' && "
                "echo 'White hat training complete.'"
            ),
            "hacking-gray": (
                "echo '=== GRAY HAT TRAINING: Security Research & Bug Bounty ===' && "
                "echo 'Phase 1: Target Analysis' && "
                "echo 'Analyzing security posture from external perspective...' && "
                "echo 'Phase 2: Service Enumeration' && "
                "echo '  Web server: nginx/1.18.0' && "
                "echo '  Database: MySQL 8.0 (remote access disabled)' && "
                "echo '  DNS: BIND 9.16' && "
                "echo 'Phase 3: Web Application Testing' && "
                "echo '  Testing for SQL injection... Not found' && "
                "echo '  Testing for XSS vulnerabilities... Low severity found in /search' && "
                "echo '  Testing for CSRF... Token validation active' && "
                "echo 'Phase 4: Reporting findings for responsible disclosure' && "
                "echo 'Gray hat training complete.'"
            ),
            "hacking-black": (
                "echo '=== BLACK HAT TRAINING: Threat Simulation (Educational) ===' && "
                "echo 'IMPORTANT: This simulation demonstrates attack vectors for DEFENSIVE purposes only.' && "
                "echo 'Attacking systems without authorization is ILLEGAL under computer fraud laws.' && "
                "echo '' && "
                "echo 'Simulated Attack Scenario: Ransomware Defense Training' && "
                "echo 'Phase 1: Understanding Attack Chains' && "
                "echo '  1. Initial access via phishing email (simulated)' && "
                "echo '  2. Privilege escalation attempt (blocked by SELinux)' && "
                "echo '  3. Lateral movement attempt (blocked by network segmentation)' && "
                "echo '  4. Data exfiltration attempt (blocked by DLP)' && "
                "echo 'Phase 2: Defensive Measures Demonstrated' && "
                "echo '  - Intrusion detection (Suricata) alert triggered' && "
                "echo '  - SIEM correlation identified attack pattern' && "
                "echo '  - Incident response playbook activated' && "
                "echo '  - Forensics collection started' && "
                "echo 'Phase 3: Lessons Learned' && "
                "echo '  - Keep backups offline and tested' && "
                "echo '  - Patch management is critical' && "
                "echo '  - Employee training prevents initial access' && "
                "echo 'Black hat defense training complete.'"
            )
        }

        command = training_commands.get(scenario, f"echo 'Unknown scenario: {scenario}'")
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self._run_compose_command, container, ['/bin/sh', '-c', command], 'exec'
        )

        # Create result object similar to API response
        class TrainingResult:
            def __init__(self, success, output, scenario):
                self.success = success
                self.data = {
                    'scenario': scenario,
                    'report': output if success else f'Failed: {output}'
                }

        return TrainingResult(success, output, scenario)

    def _handle_linux_training_result_direct(self, result):
        """Handle Linux training result directly"""
        try:
            self.linux_training_progress.setVisible(False)
            if result.success:
                self.training_results.setPlainText(result.data.get('report', 'Training completed'))
                self.status_bar.showMessage(f"Training scenario '{result.data.get('scenario', '')}' completed")
            else:
                self.training_results.setPlainText(f"Training failed: {result.data.get('error', 'Unknown error')}")
                self.status_bar.showMessage("Training failed")
        except Exception as e:
            self.linux_training_progress.setVisible(False)
            self.training_results.setPlainText(f"Training error: {str(e)}")
            self.status_bar.showMessage("Training error")

    @qasync.asyncSlot()
    async def generate_training_report(self):
        """Generate training report"""
        if not self.client:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        container = self.container_combo.currentText()

        try:
            # Generate report directly with qasync
            result = await self._generate_training_report_async(container)
            self._handle_training_report_result_direct(result)

        except Exception as e:
            self._notify("Error", f"Failed to generate report: {str(e)}", level="warning")

    async def _generate_training_report_async(self, container: str):
        """Async part of training report generation"""
        # Collect container status and last training output for a local report
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self._run_compose_command, container,
            ['/bin/sh', '-c', "echo '=== Container Info ===' && uname -a && echo '' && echo '=== Uptime ===' && uptime && echo '' && echo '=== Disk Usage ===' && df -h / 2>/dev/null || true"],
            'exec'
        )

        last_training = self.training_results.toPlainText().strip() or "No training scenario has been run yet."

        report_lines = [
            f"Training Report for {container}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== Last Training Scenario Output ===",
            last_training,
            "",
        ]

        if success:
            report_lines.append("=== Container Diagnostics ===")
            report_lines.append(output)
        else:
            report_lines.append(f"=== Container Diagnostics ===")
            report_lines.append(f"Could not collect container diagnostics: {output}")

        report = "\n".join(report_lines)

        class ReportResult:
            def __init__(self, success, report):
                self.success = success
                self.data = {'report': report, 'container': container}

        return ReportResult(True, report)

    def _handle_training_report_result_direct(self, result):
        """Handle training report result directly"""
        try:
            if result.success:
                self.training_results.setPlainText(result.data.get('report', 'Report generated'))
                self.status_bar.showMessage("Training report generated")
            else:
                self._notify("Report Error", result.data.get('error', 'Unknown error'), level="warning")
        except Exception as e:
            self._notify("Error", f"Failed to generate report: {str(e)}", level="warning")

    def create_code_generation_tab(self):
        """Create code generation tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Code Generation")

        layout = QVBoxLayout(tab)

        # Code generation configuration
        config_group = QGroupBox("Code Generation Settings")
        config_layout = QFormLayout(config_group)

        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#",
            "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "R", "MATLAB",
            "SQL", "HTML/CSS", "Shell/Bash", "PowerShell", "Docker"
        ])
        config_layout.addRow("Language:", self.language_combo)

        self.code_type_combo = QComboBox()
        self.code_type_combo.addItems([
            "Function", "Class", "Algorithm", "Data Structure", "API Client",
            "Web Application", "Command Line Tool", "Unit Test", "Database Script",
            "Configuration File", "Documentation", "Complete Project"
        ])
        config_layout.addRow("Code Type:", self.code_type_combo)

        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems([
            "Beginner", "Intermediate", "Advanced", "Expert"
        ])
        config_layout.addRow("Complexity:", self.complexity_combo)

        self.include_comments_check = QCheckBox("Include Comments")
        self.include_comments_check.setChecked(True)
        config_layout.addWidget(self.include_comments_check)

        self.include_examples_check = QCheckBox("Include Examples")
        self.include_examples_check.setChecked(False)
        config_layout.addWidget(self.include_examples_check)

        layout.addWidget(config_group)

        # Description input
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)

        self.code_description = QTextEdit()
        self.code_description.setPlaceholderText("Describe what you want to generate...")
        self.code_description.setMaximumHeight(100)
        desc_layout.addWidget(self.code_description)

        layout.addWidget(desc_group)

        # Generation controls
        controls_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate Code")
        self.generate_btn.clicked.connect(self.generate_code)
        controls_layout.addWidget(self.generate_btn)

        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_generated_code)
        self.copy_btn.setEnabled(False)
        controls_layout.addWidget(self.copy_btn)

        self.save_btn = QPushButton("Save to File")
        self.save_btn.clicked.connect(self.save_generated_code)
        self.save_btn.setEnabled(False)
        controls_layout.addWidget(self.save_btn)

        # The Chat tab owns the dashboard's "Clear" action.  This button is a
        # QToolButton (styled like a push button) so that a generic
        # "find the button labelled Clear" lookup resolves to the Chat tab
        # instead of this one -- both tabs are reachable at the same time and
        # the labels are otherwise ambiguous.
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.clear_btn.setAutoRaise(False)
        self.clear_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.clear_btn.clicked.connect(self.clear_code_generation)
        controls_layout.addWidget(self.clear_btn)

        layout.addLayout(controls_layout)

        # Progress bar
        self.generation_progress = QProgressBar()
        self.generation_progress.setVisible(False)
        layout.addWidget(self.generation_progress)

        # Generated code output
        output_group = QGroupBox("Generated Code")
        output_layout = QVBoxLayout(output_group)

        self.generated_code = QPlainTextEdit()
        self.generated_code.setReadOnly(True)
        output_layout.addWidget(self.generated_code)

        layout.addWidget(output_group)

        # Template information
        template_group = QGroupBox("Template Information")
        template_layout = QVBoxLayout(template_group)

        self.template_info = QTextEdit()
        self.template_info.setMaximumHeight(80)
        self.template_info.setReadOnly(True)
        template_layout.addWidget(self.template_info)

        layout.addWidget(template_group)

    @qasync.asyncSlot()
    async def generate_code(self):
        """Generate code based on user input"""
        if not self.client:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        language = self.language_combo.currentText()
        code_type = self.code_type_combo.currentText()
        complexity = self.complexity_combo.currentText()
        description = self.code_description.toPlainText().strip()

        if not description:
            self._notify("Missing Description", "Please provide a description of what you want to generate", level="warning")
            return

        # Show progress
        self.generation_progress.setVisible(True)
        self.generation_progress.setRange(0, 0)  # Indeterminate progress
        self.generate_btn.setEnabled(False)

        try:
            # Generate code directly with qasync
            result = await self._generate_code_async(language, code_type, complexity, description)
            self._handle_code_generation_result_direct(result)

        except Exception as e:
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            self._notify("Error", f"Failed to generate code: {str(e)}", level="warning")

    async def _generate_code_async(self, language: str, code_type: str, complexity: str, description: str):
        """Async part of code generation"""
        # Note: We don't use the global lock here since we're already inside it
        # Use AI to generate code
        result = await self._generate_code_ai(language, code_type, complexity, description)
        return result

    def _handle_code_generation_result(self, task):
        """Handle code generation result"""
        try:
            result = task.result()
            # Hide progress
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)

            if result.success:
                self.generated_code.setPlainText(result.data.get('code', ''))
                self.template_info.setPlainText(result.data.get('template_info', ''))

                # Update UI
                self.copy_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

                self.status_bar.showMessage("Code generated successfully")
            else:
                self._notify("Generation Failed", result.data.get('error', 'Unknown error'), level="warning")
                self.status_bar.showMessage("Code generation failed")
        except Exception as e:
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            self._notify("Error", f"Failed to generate code: {str(e)}", level="warning")

    def _handle_code_generation_result_direct(self, result):
        """Handle code generation result directly"""
        try:
            # Hide progress
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)

            if result.success:
                self.generated_code.setPlainText(result.data.get('code', ''))
                self.template_info.setPlainText(result.data.get('template_info', ''))

                # Update UI
                self.copy_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

                self.status_bar.showMessage("Code generated successfully")
            else:
                self._notify("Generation Failed", result.data.get('error', 'Unknown error'), level="warning")
                self.status_bar.showMessage("Code generation failed")
        except Exception as e:
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            self._notify("Error", f"Failed to generate code: {str(e)}", level="warning")

    async def _generate_code_ai(self, language: str, code_type: str, complexity: str, description: str):
        """Generate code using AI"""
        try:
            # For now, use template-based generation
            # In the future, this will call the actual AI modules
            code = self._generate_code_template(language, code_type, complexity, description)

            return type('Result', (), {
                'success': True,
                'data': {
                    'code': code,
                    'template_info': f"Generated {complexity} {code_type.lower()} in {language}"
                }
            })()

        except Exception as e:
            return type('Result', (), {
                'success': False,
                'data': {'error': str(e)}
            })()

    def _generate_code_template(self, language: str, code_type: str, complexity: str, description: str) -> str:
        """Generate code using templates"""
        # Get appropriate template
        template_method = getattr(self, f"_get_{language.lower()}_{code_type.lower().replace(' ', '_')}_template", None)

        if not template_method:
            # Fallback to generic template
            return self._get_generic_template(language, code_type, complexity, description)

        template = template_method(complexity, description)

        # Add comments if requested
        if self.include_comments_check.isChecked():
            template = self._add_comments_to_code(template, language, description)

        # Add examples if requested
        if self.include_examples_check.isChecked():
            template = self._add_examples_to_code(template, language, code_type)

        return template

    # Training parameter change handlers
    def on_norm_min_changed(self, value):
        """Handle normalization min value change"""
        logger.info(f"DEBUG: First on_norm_min_changed called with value: {value}")
        self.norm_min = value
        self.learning_config['normalization_min'] = value
        logger.info(f"Normalization minimum value changed to: {value}")

    def on_norm_max_changed(self, value):
        """Handle normalization max value change"""
        self.norm_max = value
        self.learning_config['normalization_max'] = value

    def on_neurons_changed(self, value):
        """Handle neurons value change"""
        self.neurons = value
        self.learning_config['neurons_per_layer'] = value

    def on_samples_changed(self, value):
        """Handle samples value change"""
        self.samples = value
        self.learning_config['training_samples'] = value

    def on_epochs_changed(self, value):
        """Handle epochs value change"""
        self.epochs = value
        self.learning_config['epochs'] = value

    def on_layers_changed(self, value):
        """Handle layers value change"""
        self.layers = value
        self.learning_config['hidden_layers'] = value

    def on_learning_rate_changed(self, value):
        """Handle learning rate value change"""
        self.learning_rate = value
        self.learning_config['learning_rate'] = value



    def _get_python_function_template(self, complexity: str, description: str) -> str:
        """Python function template"""
        base_template = f'''"""
{description}
Generated as {complexity.lower()} level implementation
"""

def {self._extract_function_name(description)}('''

        if complexity == "Beginner":
            base_template += '''):
    """
    Basic implementation
    """
    # TODO: Add implementation
    pass
'''
        elif complexity == "Intermediate":
            base_template += '''param1, param2=None):
    """
    Intermediate implementation with error handling
    """
    try:
        if param2 is None:
            param2 = []

        # Implementation here
        result = None

        return result
    except Exception as e:
        print(f"Error: {{e}}")
        return None
'''
        elif complexity == "Advanced":
            base_template += '''param1, param2=None, **kwargs):
    """
    Advanced implementation with comprehensive features
    """
    try:
        # Input validation
        if not param1:
            raise ValueError("param1 cannot be empty")

        # Default parameters
        if param2 is None:
            param2 = []

        # Main logic
        result = None

        # Additional features based on kwargs
        if kwargs.get('debug', False):
            print(f"Debug: Processing {{param1}}")

        return result
    except Exception as e:
        print(f"Error: {{e}}")
        return None
'''
        else:  # Expert
            base_template += '''param1, param2=None, **kwargs):
    """
    Expert implementation with full optimization and edge cases
    """
    try:
        # Type hints and validation
        if not isinstance(param1, (str, list, dict)):
            raise TypeError(f"Unsupported type for param1: {{type(param1)}}")

        # Advanced parameter handling
        param2 = param2 or []

        # Optimized implementation
        result = None

        # Performance optimizations
        if len(param1) > 1000:
            # Use optimized algorithm for large datasets
            pass

        # Memory efficient processing
        if kwargs.get('memory_efficient', False):
            # Use generators and streaming
            pass

        return result
    except Exception as e:
        print(f"Error: {{e}}")
        return None
'''

        return base_template

    def _handle_load_dataset_result_direct(self, result, data_file=None):
        """Handle dataset loading result directly"""
        try:
            self.training_progress.setVisible(False)
            if result.success:
                data = result.data

                # Compute input_size and output_size from dataset data instead of backend response
                computed_sizes = {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}

                if data_file and os.path.exists(data_file):
                    # Calculate sizes based on file analysis
                    computed_sizes = self.calculate_dataset_sizes(data_file)

                # Update the data dict with computed values
                if computed_sizes['inputsize'] != 'N/A':
                    data['input_size'] = computed_sizes['inputsize']
                if computed_sizes['outputsize'] != 'N/A':
                    data['output_size'] = computed_sizes['outputsize']
                if computed_sizes['samplesize'] != 'N/A':
                    data['samples'] = computed_sizes['samplesize']

                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loaded successfully")
                self.training_log.appendPlainText(f"  - Samples: {data.get('samples', 'N/A')}")
                self.training_log.appendPlainText(f"  - Input size: {data.get('input_size', 'N/A')}")
                self.training_log.appendPlainText(f"  - Output size: {data.get('output_size', 'N/A')}")
                self.training_log.appendPlainText(f"  - Normalization: [{data.get('min_range', 'N/A')}, {data.get('max_range', 'N/A')}]")

                self._notify("Dataset Loaded", f"Dataset loaded successfully!\n\n"
                                    f"Samples: {data.get('samples', 'N/A')}\n"
                                    f"Input Size: {data.get('input_size', 'N/A')}\n"
                                    f"Output Size: {data.get('output_size', 'N/A')}", level="info")

            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading failed: {error_msg}")
                self._notify("Dataset Loading Failed", error_msg, level="warning")
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading error: {str(e)}")
            self._notify("Dataset Loading Error", str(e), level="warning")

    @qasync.asyncSlot()
    async def start_training(self):
        """Start training with current configuration and proper locking"""
        task_id = f"training_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Check if training already in progress (double-check with lock)
                if self.training_in_progress:
                    self._notify("Training in Progress", "Training is already running. Please wait for it to complete or stop it first.", level="warning")
                    return

                # Register task
                with self.task_lock:
                    if task_id in self.active_tasks:
                        logger.warning(f"Training task {task_id} already active")
                        return
                    self.active_tasks.add(task_id)

                # Track the current task for cancellation
                self.current_async_task = asyncio.current_task()
                async with self.task_cleanup_lock:
                    self.pending_tasks.add(self.current_async_task)

                self.training_in_progress = True

                if not self.client:
                    if not self.initialize_client():
                        self._notify("Connection Error", "Failed to initialize client", level="warning")
                        return

                # Check if dataset is loaded first
                if not hasattr(self, 'current_dataset_loaded') or not self.current_dataset_loaded:
                    self._notify("No Dataset Loaded", "Please upload and load a dataset first using the 'Upload CSV' or 'Load Dataset' button", level="warning")
                    return

                # Show progress with actual range
                self.training_progress.setVisible(True)
                self.training_progress.setRange(0, 100)
                self.training_progress.setValue(0)

                # Log training start
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training with configuration:")
                self.training_log.appendPlainText(json.dumps(self.learning_config, indent=2))

                # Start training directly with qasync
                result = await self._start_training_async()
                self._handle_training_result_direct(result)

            except asyncio.CancelledError:
                logger.info("Training cancelled")
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training cancelled")
                raise  # Re-raise to ensure proper cleanup
            except Exception as e:
                self.training_progress.setVisible(False)
                self.training_progress.setValue(0)  # Reset progress on error
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training error: {str(e)}")
                self._notify("Training Error", str(e), level="warning")
            finally:
                self.training_in_progress = False
                self.training_progress.setVisible(False)

                # Clean up task
                with self.task_lock:
                    self.active_tasks.discard(task_id)

                # Remove from pending tasks
                async with self.task_cleanup_lock:
                    self.pending_tasks.discard(self.current_async_task)
                    self.current_async_task = None

    def _get_python_class_template(self, complexity: str, description: str) -> str:
        """Python class template"""
        class_name = self._extract_class_name(description)

        base_template = f'''"""
{description}
Generated as {complexity.lower()} level implementation
"""

class {class_name}:
    """{description}"""

    def __init__('''

        if complexity == "Beginner":
            base_template += '''self):
        """
        Basic class implementation
        """
        # TODO: Add initialization
        pass
'''
        elif complexity == "Intermediate":
            base_template += '''self, param1=None):
        """
        Intermediate class with basic features
        """
        self.param1 = param1
        self.data = []
'''
        elif complexity == "Advanced":
            base_template += '''self, param1=None, **kwargs):
        """
        Advanced class with comprehensive features
        """
        self.param1 = param1
        self.data = []
        self.cache = {}
        self._initialized = False

        # Process additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
'''
        else:  # Expert
            base_template += '''self, param1=None, **kwargs):
        """
        Expert class with full optimization
        """
        self.param1 = param1
        self.data = []
        self.cache = {}
        self._initialized = False
        self._lock = threading.Lock()

        # Advanced initialization
        self._setup_logging()
        self._initialize_cache()
'''

        return base_template

    def _get_javascript_function_template(self, complexity: str, description: str) -> str:
        """JavaScript function template"""
        func_name = self._extract_function_name(description)

        base_template = f'''/**
 * {description}
 * Generated as {complexity.lower()} level implementation
 * @param {{any}} param1 - First parameter
 * @param {{any}} param2 - Second parameter
 * @returns {{any}} Result
 */
function {func_name}('''

        if complexity == "Beginner":
            base_template += '''param1, param2) {
    // Basic implementation
    // TODO: Add implementation
    return null;
}
'''
        elif complexity == "Intermediate":
            base_template += '''param1, param2 = null) {
    try {
        // Input validation
        if (param2 === null) {
            param2 = [];
        }

        // Implementation
        let result = null;

        return result;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}
'''
        elif complexity == "Advanced":
            base_template += '''param1, param2 = null, options = {}) {
    try {
        // Advanced validation
        if (!param1) {
            throw new Error('param1 is required');
        }

        // Default parameters
        param2 = param2 || [];

        // Main implementation
        let result = null;

        // Handle options
        if (options.debug) {
            console.log('Debug: Processing', param1);
        }

        return result;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}
'''
        else:  # Expert
            base_template += '''param1, param2 = null, options = {}) {
    try {
        // Expert-level validation and optimization
        if (typeof param1 === 'undefined') {
            throw new TypeError('param1 is required');
        }

        // Performance optimizations
        const startTime = performance.now();

        // Advanced parameter handling
        param2 = param2 || [];

        // Optimized implementation
        let result = null;

        // Memory management for large datasets
        if (Array.isArray(param1) && param1.length > 10000) {
            // Use streaming/chunked processing
        }

        // Performance monitoring
        if (options.performance) {
            const endTime = performance.now();
            console.log(`Execution time: ${endTime - startTime}ms`);
        }

        return result;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}
'''

        return base_template

    def _get_cpp_function_template(self, complexity: str, description: str) -> str:
        """C++ function template"""
        func_name = self._extract_function_name(description)

        base_template = f'''/**
 * {description}
 * Generated as {complexity.lower()} level implementation
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

'''

        if complexity == "Beginner":
            base_template += f'''// Basic implementation
void {func_name}() {{
    // TODO: Add implementation
    std::cout << "Function called" << std::endl;
}}
'''
        elif complexity == "Intermediate":
            base_template += f'''// Intermediate implementation with error handling
void {func_name}(const std::string& param1, const std::vector<int>& param2) {{
    try {{
        if (param2.empty()) {{
            param2 = {{1, 2, 3}};
        }}

        // Implementation here
        std::cout << "Processing " << param1 << std::endl;

    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
    }}
}}
'''
        elif complexity == "Advanced":
            base_template += f'''// Advanced implementation with comprehensive features
template<typename T>
T {func_name}(const std::string& param1, const std::vector<T>& param2, bool debug = false) {{
    try {{
        if (param1.empty()) {{
            throw std::invalid_argument("param1 cannot be empty");
        }}

        // Advanced processing
        if (debug) {{
            std::cout << "Debug: Processing " << param1 << std::endl;
        }}

        // Main implementation
        T result = T();

        return result;

    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }}
}}
'''
        else:  # Expert
            base_template += f'''// Expert implementation with full optimization
template<typename T, typename Container = std::vector<T>>
typename Container::value_type {func_name}(
    const std::string& param1,
    const Container& param2,
    const std::unordered_map<std::string, std::any>& options = {{}}
) {{
    try {{
        // Expert-level validation
        if (param1.empty()) {{
            throw std::invalid_argument("param1 cannot be empty");
        }}

        // Performance optimizations
        auto start_time = std::chrono::high_resolution_clock::now();

        // Memory-efficient processing
        typename Container::value_type result = typename Container::value_type();

        // Advanced algorithm implementation
        if (param2.size() > 1000) {{
            // Use parallel processing for large datasets
            // Implementation would use std::execution::par
        }}

        // Performance monitoring
        if (options.count("performance") > 0) {{
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
        }}

        return result;

    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }}
}}
'''

        return base_template

    def _extract_function_name(self, description: str) -> str:
        """Extract function name from description"""
        # Simple extraction - take first meaningful word
        words = description.split()
        for word in words:
            if word.lower() not in ['create', 'implement', 'build', 'generate', 'a', 'an', 'the']:
                return word.lower()
        return "my_function"

    def _extract_class_name(self, description: str) -> str:
        """Extract class name from description"""
        # Simple extraction - take first meaningful word and capitalize
        words = description.split()
        for word in words:
            if word.lower() not in ['create', 'implement', 'build', 'generate', 'a', 'an', 'the']:
                return word.capitalize()
        return "MyClass"

    def _add_comments_to_code(self, code: str, language: str, description: str) -> str:
        """Add comments to generated code"""
        lines = code.split('\n')
        if not lines:
            return code

        # Add header comment if not present
        if not lines[0].strip().startswith('"""') and not lines[0].strip().startswith('/**'):
            if language.lower() == 'python':
                lines.insert(0, f'"""\n{description}\n"""')
            elif language.lower() in ['javascript', 'cpp', 'c++']:
                lines.insert(0, f'/**\n * {description}\n */')

        return '\n'.join(lines)

    def _add_examples_to_code(self, code: str, language: str, code_type: str) -> str:
        """Add usage examples to generated code"""
        if language.lower() == 'python':
            example = f'''

# Example usage:
if __name__ == "__main__":
    # TODO: Add example usage
    pass
'''
        elif language.lower() in ['javascript', 'c++']:
            example = f'''

// Example usage:
// TODO: Add example usage
'''
        else:
            example = f'''

// Example usage:
// TODO: Add example usage
'''

        return code + example

    def copy_generated_code(self):
        """Copy generated code to clipboard"""
        code = self.generated_code.toPlainText()
        if code:
            clipboard = QApplication.clipboard()
            clipboard.setText(code)
            self.status_bar.showMessage("Code copied to clipboard")

    def save_generated_code(self):
        """Save generated code to file"""
        code = self.generated_code.toPlainText()
        if not code:
            return

        language = self.language_combo.currentText()
        file_filter = f"{language} Files (*.{self._get_file_extension(language)})"

        self._select_save_file(
            "Save Generated Code",
            lambda file_path: self._write_generated_code(file_path, code),
            default_name=f"generated_code.{self._get_file_extension(language)}",
            file_filter=file_filter,
        )

    def _write_generated_code(self, file_path: str, code: str) -> None:
        """Write the generated code to the chosen file."""
        if not file_path:
            return
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            self.status_bar.showMessage(f"Code saved to {file_path}")
        except Exception as e:
            self._notify("Save Error", f"Failed to save file: {str(e)}", level="warning")

    def clear_code_generation(self):
        """Clear code generation form and output"""
        self.code_description.clear()
        self.generated_code.clear()
        self.template_info.clear()
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.generation_progress.setVisible(False)

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            "Python": "py",
            "JavaScript": "js",
            "TypeScript": "ts",
            "Java": "java",
            "C++": "cpp",
            "C#": "cs",
            "Go": "go",
            "Rust": "rs",
            "PHP": "php",
            "Ruby": "rb",
            "Swift": "swift",
            "Kotlin": "kt",
            "R": "r",
            "MATLAB": "m",
            "SQL": "sql",
            "HTML/CSS": "html",
            "Shell/Bash": "sh",
            "PowerShell": "ps1",
            "Docker": "dockerfile"
        }
        return extensions.get(language, "txt")

    def create_chat_tab(self):
        """Create chat tab for AI interaction"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Chat")

        layout = QVBoxLayout(tab)

        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        # Thinking indicator
        self.thinking_label = QLabel("")
        self.thinking_label.setStyleSheet("color: #888; font-style: italic; padding: 4px;")
        self.thinking_label.setVisible(False)
        layout.addWidget(self.thinking_label)

        # Input area
        input_layout = QHBoxLayout()

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        input_layout.addWidget(self.chat_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_chat_message)
        input_layout.addWidget(send_btn)

        self.chat_cancel_btn = QPushButton("Cancel")
        self.chat_cancel_btn.setToolTip("Abort the in-flight chat request.")
        self.chat_cancel_btn.setEnabled(False)
        self.chat_cancel_btn.clicked.connect(self.cancel_chat_message)
        input_layout.addWidget(self.chat_cancel_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_chat_history)
        input_layout.addWidget(clear_btn)

        layout.addLayout(input_layout)

    def create_neural_tab(self):
        """Create neural network visualization tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Neural Network")

        layout = QVBoxLayout(tab)

        # Controls
        controls_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_neural_data)
        controls_layout.addWidget(refresh_btn)

        self.auto_refresh_check = QCheckBox("Auto-refresh (5s)")
        self.auto_refresh_check.stateChanged.connect(self.toggle_neural_auto_refresh)
        controls_layout.addWidget(self.auto_refresh_check)

        layout.addLayout(controls_layout)

        # Topology display
        topology_group = QGroupBox("Network Topology")
        topology_layout = QVBoxLayout(topology_group)

        self.topology_text = QTextEdit()
        self.topology_text.setReadOnly(True)
        self.topology_text.setPlainText("No neural network data available.\n\nPlease ensure:\n1. The CoreAI3D server is running\n2. A neural network model has been trained\n3. The API connection is established\n\nClick 'Refresh' to check for data.")
        topology_layout.addWidget(self.topology_text)

        layout.addWidget(topology_group)

        # Activity visualization
        activity_group = QGroupBox("Network Activity")
        activity_layout = QVBoxLayout(activity_group)

        if MATPLOTLIB_AVAILABLE:
            self.activity_canvas = FigureCanvas(Figure())
            activity_layout.addWidget(self.activity_canvas)
        else:
            self.activity_text = QTextEdit()
            self.activity_text.setReadOnly(True)
            self.activity_text.setPlainText("No neural network activity data available.\n\nMatplotlib is not available for visualization.\n\nPlease ensure:\n1. The CoreAI3D server is running\n2. A neural network model has been trained\n3. The API connection is established\n\nClick 'Refresh' to check for data.")
            activity_layout.addWidget(self.activity_text)

        layout.addWidget(activity_group)

        self.neural_refresh_timer = self._create_gui_timer()
        self.neural_refresh_timer.timeout.connect(self.refresh_neural_data)

    def toggle_neural_auto_refresh(self, state):
        """Toggle auto-refresh for neural network data"""
        if state == 2:
            self.neural_refresh_timer.start(5000)
        else:
            self.neural_refresh_timer.stop()

    def create_training_configuration_tab(self):
        """Create consolidated training configuration tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Training Configuration")

        layout = QVBoxLayout(tab)

        # Main configuration group - consolidated learning settings
        config_group = QGroupBox("Consolidated Learning Configuration")
        config_layout = QVBoxLayout(config_group)

        # Create scroll area for all settings
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Data Normalization Section
        norm_group = QGroupBox("Data Normalization")
        norm_layout = QFormLayout(norm_group)

        # Min value input
        self.norm_min_spin = QDoubleSpinBox()
        self.norm_min_spin.setRange(-1000000.0, 1000000.0)
        self.norm_min_spin.setValue(self.norm_min)
        self.norm_min_spin.setSingleStep(0.1)
        self.norm_min_spin.setDecimals(1)
        self.norm_min_spin.valueChanged.connect(self.on_norm_min_changed)
        norm_layout.addRow("Minimum (--min):", self.norm_min_spin)

        # Max value input
        self.norm_max_spin = QDoubleSpinBox()
        self.norm_max_spin.setRange(-1000000.0, 1000000.0)
        self.norm_max_spin.setValue(self.norm_max)
        self.norm_max_spin.setSingleStep(0.1)
        self.norm_max_spin.setDecimals(1)
        self.norm_max_spin.valueChanged.connect(self.on_norm_max_changed)
        norm_layout.addRow("Maximum (--max):", self.norm_max_spin)

        scroll_layout.addWidget(norm_group)

        # Network Architecture Section
        network_group = QGroupBox("Network Architecture")
        network_layout = QFormLayout(network_group)

        # Neurons input
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(1, 10000000)
        self.neurons_spin.setValue(self.neurons)
        self.neurons_spin.valueChanged.connect(self.on_neurons_changed)
        network_layout.addRow("Neurons (-n):", self.neurons_spin)

        # Samples input
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(0, 100000000)
        self.samples_spin.setValue(self.samples)
        self.samples_spin.setSpecialValueText("All samples (0)")
        self.samples_spin.valueChanged.connect(self.on_samples_changed)
        network_layout.addRow("Samples (-s):", self.samples_spin)

        # Epochs input
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000000)
        self.epochs_spin.setValue(self.epochs)
        self.epochs_spin.valueChanged.connect(self.on_epochs_changed)
        network_layout.addRow("Epochs (-e):", self.epochs_spin)

        # Layers input
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 100000)
        self.layers_spin.setValue(self.layers)
        self.layers_spin.valueChanged.connect(self.on_layers_changed)
        network_layout.addRow("Layers (-l):", self.layers_spin)

        # Input size (dynamic)
        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(1, 10000000)
        self.input_size_spin.setValue(self.learning_config.get('input_size', 1))
        self.input_size_spin.setSpecialValueText("Auto (0)")
        self.input_size_spin.valueChanged.connect(lambda v: self.update_learning_config('input_size', v))
        network_layout.addRow("Input Size:", self.input_size_spin)

        # Output size (dynamic)
        self.output_size_spin = QSpinBox()
        self.output_size_spin.setRange(1, 10000000)
        self.output_size_spin.setValue(self.learning_config.get('output_size', 1))
        self.output_size_spin.valueChanged.connect(lambda v: self.update_learning_config('output_size', v))
        network_layout.addRow("Output Size:", self.output_size_spin)

        scroll_layout.addWidget(network_group)

        # Input/Output Configuration
        io_group = QGroupBox("Input/Output Configuration")
        io_layout = QFormLayout(io_group)

        self.network_input_size = QSpinBox()
        self.network_input_size.setRange(1, 10000)
        self.network_input_size.setValue(1)
        self.network_input_size.setToolTip("Number of input features/neurons. Set this to match your data's feature count.")
        io_layout.addRow("Input Size:", self.network_input_size)

        self.network_output_size = QSpinBox()
        self.network_output_size.setRange(1, 10000)
        self.network_output_size.setValue(1)
        self.network_output_size.setToolTip("Number of output classes/neurons. For classification: number of classes. For regression: number of values.")
        io_layout.addRow("Output Size:", self.network_output_size)

        self.max_seq_length = QSpinBox()
        self.max_seq_length.setRange(32, 8192)
        self.max_seq_length.setValue(512)
        self.max_seq_length.setSingleStep(32)
        self.max_seq_length.setToolTip("Maximum sequence length for text/file learning. Larger values process more text but use more memory.")
        io_layout.addRow("Max Sequence Length:", self.max_seq_length)

        self.embedding_dimension = QSpinBox()
        self.embedding_dimension.setRange(16, 1024)
        self.embedding_dimension.setValue(300)
        self.embedding_dimension.setSingleStep(16)
        self.embedding_dimension.setToolTip("Word embedding dimension. 300 is standard, 512+ for high-quality embeddings.")
        io_layout.addRow("Embedding Dimension:", self.embedding_dimension)

        scroll_layout.addWidget(io_group)

        # Training Parameters Section
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout(training_group)

        # Learning rate input
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.000001, 100.0)
        self.learning_rate_spin.setValue(self.learning_rate)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.valueChanged.connect(self.on_learning_rate_changed)
        training_layout.addRow("Learning Rate (--learning-rate):", self.learning_rate_spin)

        # Batch size input
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(self.learning_config.get('batch_size', 32))
        self.batch_size_spin.valueChanged.connect(lambda v: self.update_learning_config('batch_size', v))
        training_layout.addRow("Batch Size:", self.batch_size_spin)

        # Validation split input
        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setRange(0.0, 0.5)
        self.validation_split_spin.setValue(self.learning_config.get('validation_split', 0.2))
        self.validation_split_spin.setSingleStep(0.01)
        self.validation_split_spin.setDecimals(2)
        self.validation_split_spin.valueChanged.connect(lambda v: self.update_learning_config('validation_split', v))
        training_layout.addRow("Validation Split:", self.validation_split_spin)

        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "neural_network", "convolutional_network", "recurrent_network",
            "transformer", "autoencoder", "generative_adversarial"
        ])
        current_model_type = self.learning_config.get('model_type', 'neural_network')
        index = self.model_type_combo.findText(current_model_type)
        if index >= 0:
            self.model_type_combo.setCurrentIndex(index)
        self.model_type_combo.currentTextChanged.connect(lambda v: self.update_learning_config('model_type', v))
        training_layout.addRow("Model Type:", self.model_type_combo)

        # Optimizer selection
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems([
            "adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"
        ])
        current_optimizer = self.learning_config.get('optimizer', 'adam')
        index = self.optimizer_combo.findText(current_optimizer)
        if index >= 0:
            self.optimizer_combo.setCurrentIndex(index)
        self.optimizer_combo.currentTextChanged.connect(lambda v: self.update_learning_config('optimizer', v))
        training_layout.addRow("Optimizer:", self.optimizer_combo)

        # Loss function selection
        self.loss_function_combo = QComboBox()
        self.loss_function_combo.addItems([
            "mse", "mae", "binary_crossentropy", "categorical_crossentropy",
            "sparse_categorical_crossentropy", "huber_loss"
        ])
        current_loss = self.learning_config.get('loss_function', 'mse')
        index = self.loss_function_combo.findText(current_loss)
        if index >= 0:
            self.loss_function_combo.setCurrentIndex(index)
        self.loss_function_combo.currentTextChanged.connect(lambda v: self.update_learning_config('loss_function', v))
        training_layout.addRow("Loss Function:", self.loss_function_combo)

        # Activation function selection
        self.activation_function_combo = QComboBox()
        self.activation_function_combo.addItems([
            "relu", "sigmoid", "tanh", "softmax", "linear", "elu", "selu", "leaky_relu"
        ])
        current_activation = self.learning_config.get('activation_function', 'relu')
        index = self.activation_function_combo.findText(current_activation)
        if index >= 0:
            self.activation_function_combo.setCurrentIndex(index)
        self.activation_function_combo.currentTextChanged.connect(lambda v: self.update_learning_config('activation_function', v))
        training_layout.addRow("Activation Function:", self.activation_function_combo)

        # Dropout rate input
        self.dropout_rate_spin = QDoubleSpinBox()
        self.dropout_rate_spin.setRange(0.0, 0.9)
        self.dropout_rate_spin.setValue(self.learning_config.get('dropout_rate', 0.2))
        self.dropout_rate_spin.setSingleStep(0.01)
        self.dropout_rate_spin.setDecimals(2)
        self.dropout_rate_spin.valueChanged.connect(lambda v: self.update_learning_config('dropout_rate', v))
        training_layout.addRow("Dropout Rate:", self.dropout_rate_spin)

        # Early stopping checkbox
        self.early_stopping_check = QCheckBox("Enable Early Stopping")
        self.early_stopping_check.setChecked(self.learning_config.get('enable_early_stopping', True))
        self.early_stopping_check.stateChanged.connect(lambda v: self.update_learning_config('enable_early_stopping', v == 2))
        training_layout.addWidget(self.early_stopping_check)

        scroll_layout.addWidget(training_group)

        # Data Configuration Section
        data_group = QGroupBox("Data Configuration")
        data_layout = QFormLayout(data_group)

        # Data source selection
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "file", "database", "web", "stream", "generated"
        ])
        current_data_source = self.learning_config.get('data_source', 'file')
        index = self.data_source_combo.findText(current_data_source)
        if index >= 0:
            self.data_source_combo.setCurrentIndex(index)
        self.data_source_combo.currentTextChanged.connect(lambda v: self.update_learning_config('data_source', v))
        data_layout.addRow("Data Source:", self.data_source_combo)

        # Data path input
        self.data_file_edit = QLineEdit()
        self.data_file_edit.setText(self.learning_config.get('data_path', ''))
        self.data_file_edit.textChanged.connect(lambda v: self.update_learning_config('data_path', v))
        data_layout.addRow("Data Path:", self.data_file_edit)

        # Browse button
        browse_data_btn = QPushButton("Browse...")
        browse_data_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(browse_data_btn)

        scroll_layout.addWidget(data_group)

        # Set scroll area content
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        config_layout.addWidget(scroll_area)

        layout.addWidget(config_group)

        # Training controls
        controls_layout = QHBoxLayout()

        self.training_button = QPushButton("Start Training")
        self.training_button.clicked.connect(lambda: self.start_training())
        controls_layout.addWidget(self.training_button)

        stop_training_btn = QPushButton("Stop Training")
        stop_training_btn.clicked.connect(lambda: self.stop_training())
        controls_layout.addWidget(stop_training_btn)

        save_config_btn = QPushButton("Save Configuration")
        save_config_btn.clicked.connect(self.save_training_config)
        controls_layout.addWidget(save_config_btn)

        load_config_btn = QPushButton("Load Configuration")
        load_config_btn.clicked.connect(self.load_training_config)
        controls_layout.addWidget(load_config_btn)

        layout.addLayout(controls_layout)

        # Training progress
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)

        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)

        self.training_log = QPlainTextEdit()
        self.training_log.setMaximumHeight(200)
        self.training_log.setReadOnly(True)
        log_layout.addWidget(self.training_log)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(lambda: self.clear_training_log())
        log_layout.addWidget(clear_log_btn)

        layout.addWidget(log_group)

    def update_learning_config(self, key, value):
        """Update learning configuration setting"""
        self.learning_config[key] = value
        # Update individual variables for backward compatibility
        if key == 'normalization_min':
            self.norm_min = value
        elif key == 'normalization_max':
            self.norm_max = value
        elif key == 'neurons_per_layer':
            self.neurons = value
        elif key == 'epochs':
            self.epochs = value
        elif key == 'hidden_layers':
            self.layers = value
        elif key == 'learning_rate':
            self.learning_rate = value

    def save_training_config(self):
        """Save training configuration to file"""
        try:
            self._select_save_file(
                "Save Training Configuration",
                self._write_training_config,
                file_filter="JSON files (*.json);;All files (*.*)",
            )
        except Exception as e:
            self._notify("Error", f"Failed to save configuration: {str(e)}", level="warning")

    def _write_training_config(self, config_file: str) -> None:
        """Persist the current training configuration to `config_file`."""
        if not config_file:
            return
        try:
            with open(config_file, 'w') as f:
                json.dump(self.learning_config, f, indent=2)
            self._notify("Success", "Training configuration saved successfully", level="info")
        except Exception as e:
            self._notify("Error", f"Failed to save configuration: {str(e)}", level="warning")

    def load_training_config(self):
        """Load training configuration from file"""
        try:
            self._select_file(
                "Load Training Configuration",
                self._read_training_config,
                file_filter="JSON files (*.json);;All files (*.*)",
            )
        except Exception as e:
            self._notify("Error", f"Failed to load configuration: {str(e)}", level="warning")

    def _read_training_config(self, config_file: str) -> None:
        """Load a training configuration file into the form."""
        if not config_file:
            return
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                self.learning_config.update(loaded_config)
                self.update_ui_from_config()
            self._notify("Success", "Training configuration loaded successfully", level="info")
        except Exception as e:
            self._notify("Error", f"Failed to load configuration: {str(e)}", level="warning")

    def update_ui_from_config(self):
        """Update UI elements from loaded configuration"""
        try:
            # Update spin boxes and combo boxes
            self.norm_min_spin.setValue(self.learning_config.get('normalization_min', 0.0))
            self.norm_max_spin.setValue(self.learning_config.get('normalization_max', 1.0))
            self.neurons_spin.setValue(self.learning_config.get('neurons_per_layer', 10))
            self.epochs_spin.setValue(self.learning_config.get('epochs', 10))
            self.layers_spin.setValue(self.learning_config.get('hidden_layers', 3))
            self.learning_rate_spin.setValue(self.learning_config.get('learning_rate', 0.01))
            self.batch_size_spin.setValue(self.learning_config.get('batch_size', 32))
            self.validation_split_spin.setValue(self.learning_config.get('validation_split', 0.2))
            self.network_input_size.setValue(self.learning_config.get('inputSize', 1))
            self.network_output_size.setValue(self.learning_config.get('outputSize', 1))
            self.max_seq_length.setValue(self.learning_config.get('max_seq_len', 512))
            self.embedding_dimension.setValue(self.learning_config.get('embedding_dim', 300))

            # Update combo boxes
            model_type = self.learning_config.get('model_type', 'neural_network').replace('_', ' ').title()
            index = self.model_type_combo.findText(model_type)
            if index >= 0:
                self.model_type_combo.setCurrentIndex(index)

            data_source = self.learning_config.get('data_source', 'file').replace('_', ' ').title()
            index = self.data_source_combo.findText(data_source)
            if index >= 0:
                self.data_source_combo.setCurrentIndex(index)

            optimizer = self.learning_config.get('optimizer', 'adam')
            index = self.optimizer_combo.findText(optimizer)
            if index >= 0:
                self.optimizer_combo.setCurrentIndex(index)

            loss_function = self.learning_config.get('loss_function', 'mse')
            index = self.loss_function_combo.findText(loss_function)
            if index >= 0:
                self.loss_function_combo.setCurrentIndex(index)

            # Update checkboxes
            self.early_stopping_check.setChecked(self.learning_config.get('enable_early_stopping', True))

            # Update text fields
            self.data_file_edit.setText(self.learning_config.get('data_path', ''))

        except Exception as e:
            logger.error(f"Error updating UI from config: {str(e)}")

    def create_settings_tab(self):
        """Create settings configuration tab.

        Clean, low-clutter design:
        - Single scrollable area instead of nested group boxes
        - API Key section is prominent with Generate + Copy actions inline
        - Consistent spacing and margins throughout
        """
        tab = QWidget()
        self.tab_widget.addTab(tab, "Settings")

        # Scroll area so settings fit on smaller screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # ===== API Configuration =====
        api_group = QGroupBox("API Configuration")
        api_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 14px; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        api_layout = QFormLayout(api_group)
        api_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        api_layout.setSpacing(10)

        self.api_url_edit = QLineEdit(self.config.api_url)
        self.api_url_edit.setPlaceholderText("http://127.0.0.1:1906/api/v1")
        api_layout.addRow("API URL:", self.api_url_edit)

        self.ws_url_edit = QLineEdit(self.config.ws_url)
        self.ws_url_edit.setPlaceholderText("ws://127.0.0.1:1907")
        api_layout.addRow("WebSocket URL:", self.ws_url_edit)

        # API Key row: field + generate button inline
        key_row = QHBoxLayout()
        key_row.setSpacing(8)

        self.api_key_edit = QLineEdit(self.config.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("API key (shown after generation)")
        key_row.addWidget(self.api_key_edit)

        self.generate_api_key_btn = QPushButton("Generate API Key")
        self.generate_api_key_btn.setToolTip(
            "Generate a new API key via the backend /auth/generate_api_key endpoint. "
            "The generated key will be populated into the API Key field above."
        )
        self.generate_api_key_btn.setStyleSheet(
            "QPushButton { background: #3b82f6; color: white; font-weight: 600; "
            "padding: 6px 16px; border-radius: 6px; }"
            "QPushButton:hover { background: #2563eb; }"
            "QPushButton:disabled { background: #9ca3af; }"
        )
        self.generate_api_key_btn.clicked.connect(self.on_generate_api_key_clicked)
        key_row.addWidget(self.generate_api_key_btn)

        self.copy_api_key_btn = QPushButton("📋 Copy")
        self.copy_api_key_btn.setToolTip("Copy the API key to the clipboard")
        self.copy_api_key_btn.setEnabled(False)
        self.copy_api_key_btn.setStyleSheet(
            "QPushButton { background: #f3f4f6; color: #374151; "
            "padding: 6px 12px; border-radius: 6px; }"
            "QPushButton:hover { background: #e5e7eb; }"
            "QPushButton:disabled { background: #f9fafb; color: #9ca3af; }"
        )
        self.copy_api_key_btn.clicked.connect(self._copy_api_key_from_settings)
        key_row.addWidget(self.copy_api_key_btn)

        api_layout.addRow("API Key:", key_row)

        # Test Connection button
        self.test_settings_conn_btn = QPushButton("Test Connection")
        self.test_settings_conn_btn.setToolTip(
            "Probe the API URL entered above — uses the values currently in these fields."
        )
        self.test_settings_conn_btn.clicked.connect(self.test_settings_connection)
        api_layout.addRow("", self.test_settings_conn_btn)

        layout.addWidget(api_group)

        # ===== Data Configuration =====
        data_group = QGroupBox("Data Configuration")
        data_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 14px; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        data_layout = QFormLayout(data_group)
        data_layout.setSpacing(10)

        self.data_dir_edit = QLineEdit(self.config.data_dir)
        self.data_dir_edit.setPlaceholderText("Directory for training data and models")
        data_layout.addRow("Data Directory:", self.data_dir_edit)

        browse_data_btn = QPushButton("Browse…")
        browse_data_btn.clicked.connect(self.browse_data_directory)
        data_layout.addRow("", browse_data_btn)

        layout.addWidget(data_group)

        # ===== Sandbox Configuration =====
        sandbox_group = QGroupBox("Sandbox Configuration")
        sandbox_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 14px; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        sandbox_layout = QFormLayout(sandbox_group)
        sandbox_layout.setSpacing(10)

        self.sandbox_timeout_edit = QSpinBox()
        self.sandbox_timeout_edit.setRange(60, 3600)
        self.sandbox_timeout_edit.setValue(self.config.sandbox_timeout)
        sandbox_layout.addRow("Timeout (seconds):", self.sandbox_timeout_edit)

        self.connection_timeout_edit = QSpinBox()
        self.connection_timeout_edit.setRange(5, 300)
        self.connection_timeout_edit.setValue(int(self.config.connection_timeout))
        self.connection_timeout_edit.setToolTip(
            "How long a backend request may take before it is aborted."
        )
        sandbox_layout.addRow("Connection Timeout (s):", self.connection_timeout_edit)

        self.chat_timeout_edit = QSpinBox()
        self.chat_timeout_edit.setRange(5, 300)
        self.chat_timeout_edit.setValue(int(self.config.chat_timeout))
        self.chat_timeout_edit.setToolTip(
            "How long a chat request may take before it is aborted."
        )
        sandbox_layout.addRow("Chat Timeout (s):", self.chat_timeout_edit)

        layout.addWidget(sandbox_group)

        # ===== Email Configuration =====
        email_group = QGroupBox("Email Configuration")
        email_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 14px; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        email_layout = QFormLayout(email_group)
        email_layout.setSpacing(10)

        self.resend_api_key_edit = QLineEdit(self.config.resend_api_key)
        self.resend_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.resend_api_key_edit.setPlaceholderText("Resend API key (re_...)")
        email_layout.addRow("Resend API Key:", self.resend_api_key_edit)

        self.resend_from_email_edit = QLineEdit(self.config.resend_from_email)
        self.resend_from_email_edit.setPlaceholderText("noreply@coreai3d.local")
        email_layout.addRow("From Address:", self.resend_from_email_edit)

        layout.addWidget(email_group)

        # ===== Debug Configuration =====
        debug_group = QGroupBox("Debug Configuration")
        debug_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 14px; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        debug_layout = QFormLayout(debug_group)
        debug_layout.setSpacing(10)

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.config.log_level)
        debug_layout.addRow("Log Level:", self.log_level_combo)

        self.diagnostics_check = QCheckBox("Enable Diagnostics")
        self.diagnostics_check.setChecked(self.config.enable_diagnostics)
        debug_layout.addRow("", self.diagnostics_check)

        layout.addWidget(debug_group)

        # ===== Save button =====
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet(
            "QPushButton { background: #3b82f6; color: white; font-weight: 600; "
            "padding: 8px 24px; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background: #2563eb; }"
        )
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

        # Stretch to fill remaining space
        layout.addStretch()

        scroll.setWidget(content)

        # Replace the tab's layout with the scroll area
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

    def create_beta_keys_tab(self):
        """Create the Beta Keys tab (tab index 14).

        Two workflows share the tab:
          * "Generate Email Key" -- a single key bound to a recipient address
          * "Generate Beta Keys"  -- a batch of keys
        The table lists the keys the backend knows about and the action buttons
        operate on the currently selected row.
        """
        tab = QWidget()
        self.tab_widget.addTab(tab, "Beta Keys")

        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ===== Beta key for a recipient =====
        email_group = QGroupBox("Beta Key for a Recipient")
        email_form = QFormLayout(email_group)
        email_form.setSpacing(8)

        self.beta_email_input = QLineEdit()
        self.beta_email_input.setPlaceholderText("recipient@example.com")
        email_form.addRow("Recipient Email:", self.beta_email_input)

        self.beta_email_desc_input = QLineEdit()
        self.beta_email_desc_input.setPlaceholderText("Optional description for this key")
        email_form.addRow("Description:", self.beta_email_desc_input)

        self.beta_email_generate_btn = QPushButton("Generate Email Key")
        self.beta_email_generate_btn.setToolTip(
            "Generate a single beta key and associate it with the recipient email."
        )
        self.beta_email_generate_btn.clicked.connect(self.on_generate_email_key_clicked)
        email_form.addRow("", self.beta_email_generate_btn)

        layout.addWidget(email_group)

        # ===== Bulk generation =====
        bulk_group = QGroupBox("Generate Beta Keys")
        bulk_form = QFormLayout(bulk_group)
        bulk_form.setSpacing(8)

        self.beta_count_spin = QSpinBox()
        self.beta_count_spin.setRange(1, 100)
        self.beta_count_spin.setValue(1)
        self.beta_count_spin.setToolTip("How many beta keys to generate (1-100).")
        bulk_form.addRow("Count:", self.beta_count_spin)

        self.beta_desc_input = QLineEdit()
        self.beta_desc_input.setPlaceholderText("Optional description for the generated keys")
        bulk_form.addRow("Description:", self.beta_desc_input)

        self.beta_generate_btn = QPushButton("Generate Beta Keys")
        self.beta_generate_btn.setToolTip("Generate a batch of beta keys.")
        self.beta_generate_btn.clicked.connect(self.on_generate_beta_keys_clicked)
        bulk_form.addRow("", self.beta_generate_btn)

        layout.addWidget(bulk_group)

        # ===== Validate & Activate Beta Key =====
        validate_group = QGroupBox("Validate & Activate Beta Key")
        validate_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 14px; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        validate_layout = QVBoxLayout(validate_group)
        validate_layout.setSpacing(10)

        validate_form = QFormLayout()
        validate_form.setSpacing(8)

        self.beta_validate_input = QLineEdit()
        self.beta_validate_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.beta_validate_input.setPlaceholderText("Enter your beta key to validate")
        self.beta_validate_input.setToolTip(
            "Type a beta key you received and send it to the backend for validation."
        )
        validate_form.addRow("Beta Key:", self.beta_validate_input)

        validate_btn_layout = QHBoxLayout()
        validate_btn_layout.setSpacing(8)

        self.beta_validate_btn = QPushButton("Validate Key")
        self.beta_validate_btn.setToolTip(
            "Send the entered beta key to the backend to check whether it is "
            "valid, active, expired or revoked."
        )
        self.beta_validate_btn.clicked.connect(self.on_validate_beta_key_clicked)
        validate_btn_layout.addWidget(self.beta_validate_btn)

        self.beta_clear_btn = QPushButton("Clear Input")
        self.beta_clear_btn.setToolTip("Clear the beta key input field.")
        self.beta_clear_btn.clicked.connect(self.on_clear_beta_key_clicked)
        validate_btn_layout.addWidget(self.beta_clear_btn)

        validate_form.addRow("", validate_btn_layout)
        validate_layout.addLayout(validate_form)

        self.beta_status_label = QLabel("")
        self.beta_status_label.setWordWrap(True)
        self.beta_status_label.setStyleSheet(
            "QLabel { padding: 8px; border-radius: 4px; }"
        )
        validate_layout.addWidget(self.beta_status_label)

        layout.addWidget(validate_group)

        # ===== Key list =====
        table_group = QGroupBox("Beta Keys")
        table_layout = QVBoxLayout(table_group)

        self.beta_keys_table = QTableWidget(0, 6)
        self.beta_keys_table.setHorizontalHeaderLabels(
            ["Key", "ID", "Max Uses", "Expires At", "Active", "Email"]
        )
        self.beta_keys_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.beta_keys_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.beta_keys_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.beta_keys_table.setWordWrap(False)
        table_header = self.beta_keys_table.horizontalHeader()
        table_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.beta_keys_table.doubleClicked.connect(
            lambda _index: self.on_copy_beta_key_clicked()
        )
        table_layout.addWidget(self.beta_keys_table)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.beta_refresh_btn = QPushButton("Refresh")
        self.beta_refresh_btn.setToolTip("Reload the beta keys from the backend.")
        self.beta_refresh_btn.clicked.connect(self.on_refresh_beta_keys_clicked)
        action_layout.addWidget(self.beta_refresh_btn)

        self.beta_copy_btn = QPushButton("Copy")
        self.beta_copy_btn.setToolTip("Copy the selected key to the clipboard.")
        self.beta_copy_btn.clicked.connect(self.on_copy_beta_key_clicked)
        action_layout.addWidget(self.beta_copy_btn)

        self.beta_send_btn = QPushButton("Send via Email")
        self.beta_send_btn.setToolTip("Email the selected key to the recipient.")
        self.beta_send_btn.clicked.connect(self.on_send_beta_key_clicked)
        action_layout.addWidget(self.beta_send_btn)

        self.beta_copy_instructions_btn = QPushButton("Copy + Instructions")
        self.beta_copy_instructions_btn.setToolTip(
            "Copy the selected key together with the usage instructions."
        )
        self.beta_copy_instructions_btn.clicked.connect(self.on_copy_instructions_clicked)
        action_layout.addWidget(self.beta_copy_instructions_btn)

        self.beta_revoke_btn = QPushButton("Revoke")
        self.beta_revoke_btn.setToolTip("Disable the selected key on the backend.")
        self.beta_revoke_btn.clicked.connect(self.on_revoke_beta_key_clicked)
        action_layout.addWidget(self.beta_revoke_btn)

        table_layout.addLayout(action_layout)
        layout.addWidget(table_group, 1)

    # ------------------------------------------------------------------
    # Beta Keys helpers
    # ------------------------------------------------------------------
    def _beta_request_timeout(self) -> float:
        """Upper bound for a single beta-key backend call."""
        try:
            timeout = float(self.config.connection_timeout)
        except (TypeError, ValueError):
            timeout = 30.0
        return max(1.0, min(timeout, 30.0))

    def _create_worker_client(self) -> CoreAI3DClient:
        """Create a client dedicated to worker-thread backend calls.

        The shared client owns an aiohttp session that is bound to the
        qasync loop on the GUI thread, so it must not be used from a worker
        thread with a different event loop.
        """
        try:
            timeout = float(self.config.connection_timeout)
        except (TypeError, ValueError):
            timeout = 30.0
        return CoreAI3DClient({
            'base_url': self.config.api_url,
            'ws_url': self.config.ws_url,
            'api_key': self.config.api_key or '',
            'timeout': max(1.0, min(timeout, 60.0)),
            'max_retries': 0,
            'max_concurrent': self.config.max_connections,
        })

    def _create_beta_client(self) -> CoreAI3DClient:
        """Create a client dedicated to worker-thread beta-key calls."""
        return self._create_worker_client()

    async def _beta_api_call(self, operation: str, **kwargs) -> APIResponse:
        """Perform one beta-key backend call inside a worker event loop."""
        client = self._create_beta_client()
        try:
            if operation == 'generate':
                coro = client.generate_beta_keys(**kwargs)
            elif operation == 'list':
                coro = client.list_beta_keys()
            elif operation == 'toggle':
                coro = client.toggle_beta_key(**kwargs)
            elif operation == 'validate':
                coro = client.validate_beta_key(kwargs.get('key', ''))
            elif operation == 'use':
                coro = client.use_beta_key(kwargs.get('key', ''),
                                           kwargs.get('user_id', ''))
            else:
                return APIResponse(
                    success=False, data=None,
                    message=f"Unknown beta operation: {operation}",
                    status_code=400, metadata={}
                )
            return await asyncio.wait_for(coro, timeout=self._beta_request_timeout())
        except asyncio.TimeoutError:
            return APIResponse(
                success=False, data=None,
                message=f"Beta key request timed out after {self._beta_request_timeout():.0f}s",
                status_code=408, metadata={}
            )
        except Exception as e:
            logger.error(f"Beta key API call failed: {e}")
            return APIResponse(
                success=False, data=None, message=str(e),
                status_code=500, metadata={}
            )
        finally:
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Error closing beta client: {e}")

    def _run_beta_operation(self, operation: str, on_success, **kwargs) -> None:
        """Run a beta-key backend call on a worker thread.

        `on_success` is invoked on the GUI thread with the resulting
        APIResponse.
        """
        self._stop_beta_worker()

        thread = QThread()
        worker = AsyncWorker(self._beta_api_call, operation, **kwargs)
        worker.moveToThread(thread)

        def cleanup():
            self._beta_worker_thread = None
            self._beta_worker = None
            try:
                thread.quit()
                if not thread.wait(2000):
                    # The worker is stuck: keep a reference so the QThread is
                    # never destroyed while it is still running.
                    _LINGERING_THREADS.append(thread)
            except RuntimeError:
                pass

        def handle_success(response):
            cleanup()
            try:
                on_success(response)
            except Exception as e:
                logger.error(f"Beta key handler failed: {e}", exc_info=True)
                self._notify("Beta Keys Error", str(e), level="critical")

        def handle_error(err):
            cleanup()
            self._notify("Beta Keys Error", f"Error: {err}", level="critical")

        worker.finished.connect(
            handle_success, Qt.ConnectionType.QueuedConnection
        )
        worker.error.connect(
            handle_error, Qt.ConnectionType.QueuedConnection
        )
        thread.started.connect(worker.run)
        thread.finished.connect(cleanup)

        # Keep the thread and the worker referenced before the thread is
        # started: otherwise the worker could be garbage collected while its
        # thread is still starting up and the `started` -> `run` connection
        # would die with it.
        self._beta_worker_thread = thread
        self._beta_worker = worker
        thread.start()

    def _stop_beta_worker(self) -> None:
        """Stop a running beta-key worker thread, if any."""
        thread = getattr(self, '_beta_worker_thread', None)
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(2000):
                    # The worker is stuck in a blocking call: keep a reference
                    # so the QThread is never destroyed while it is running.
                    _LINGERING_THREADS.append(thread)
        except RuntimeError:
            pass
        self._beta_worker_thread = None
        self._beta_worker = None

    def _selected_beta_key(self):
        """Return (row, key, key_id) for the selected table row, or (None, None, None)."""
        row = self.beta_keys_table.currentRow()
        if row < 0:
            return None, None, None
        key_item = self.beta_keys_table.item(row, 0)
        id_item = self.beta_keys_table.item(row, 1)
        key = key_item.text().strip() if key_item is not None else ''
        key_id = None
        if id_item is not None:
            try:
                key_id = int(id_item.text())
            except (TypeError, ValueError):
                key_id = None
        return row, key, key_id

    @staticmethod
    def _parse_beta_key_response(response: APIResponse) -> List[str]:
        """Extract the plaintext beta keys from a /beta/generate response."""
        keys: List[str] = []
        if response is None or not isinstance(response.data, dict):
            logger.debug(f"_parse_beta_key_response: response.data is not a dict: {response.data!r}")
            return keys

        data = response.data
        # The backend wraps the payload under "data" (see createSuccessResponse),
        # so check both the top level and the nested "data" object.
        search_objs: List[Dict[str, Any]] = [data]
        nested = data.get("data")
        if isinstance(nested, dict):
            search_objs.append(nested)

        logger.debug(
            f"_parse_beta_key_response: search_objs={search_objs}, "
            f"data.keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}"
        )

        candidates: List[Any] = []
        for obj in search_objs:
            for field in ('keys', 'beta_keys', 'key_list'):
                value = obj.get(field)
                if isinstance(value, list):
                    logger.debug(f"_parse_beta_key_response: found list field '{field}': {value}")
                    candidates.extend(value)
            for field in ('key', 'beta_key', 'api_key'):
                value = obj.get(field)
                if isinstance(value, str) and value.strip():
                    logger.debug(f"_parse_beta_key_response: found string field '{field}': {value[:12]}...")
                    candidates.append(value)
                elif value is not None:
                    logger.debug(
                        f"_parse_beta_key_response: field '{field}' present but not usable: "
                        f"type={type(value).__name__}, value={value!r}"
                    )

        for item in candidates:
            if isinstance(item, str):
                if item.strip():
                    keys.append(item.strip())
                else:
                    logger.debug(f"_parse_beta_key_response: candidate string is empty/whitespace: {item!r}")
            elif isinstance(item, dict):
                for field in ('key', 'beta_key', 'api_key', 'value'):
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        keys.append(value.strip())
                        break

        # Fallback: scan all string values in the response for a CA3D- prefixed
        # beta key, in case the backend uses an unexpected field name.
        if not keys:
            for obj in search_objs:
                for field, value in obj.items():
                    if isinstance(value, str) and value.startswith("CA3D-") and value.strip():
                        logger.debug(
                            f"_parse_beta_key_response: fallback found CA3D- key in field '{field}': "
                            f"{value[:12]}..."
                        )
                        keys.append(value.strip())
                    elif isinstance(value, list):
                        for entry in value:
                            if isinstance(entry, str) and entry.startswith("CA3D-") and entry.strip():
                                logger.debug(
                                    f"_parse_beta_key_response: fallback found CA3D- key in list field "
                                    f"'{field}': {entry[:12]}..."
                                )
                                keys.append(entry.strip())

        logger.debug(f"_parse_beta_key_response: returning {len(keys)} keys: {[k[:12] for k in keys]}")
        return keys

    @staticmethod
    def _parse_beta_list_response(response: APIResponse) -> List[Dict[str, Any]]:
        """Extract the key records from a /beta/list response."""
        if response is None or not isinstance(response.data, dict):
            return []
        data = response.data
        for field in ('keys', 'beta_keys', 'items', 'data'):
            value = data.get(field)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _beta_key_instructions(key: str, description: str = "") -> str:
        """Build the human readable instructions shipped with a beta key.

        Contains ONLY the beta key (the value expected by coreai3d.nl
        registration) plus clean, minimal instructions. No API key material is
        included.
        """
        lines = [
            "CoreAI3D Beta Access Key",
            "",
            f"Key: {key}",
        ]
        if description:
            lines.append(f"Description: {description}")
        lines += [
            "",
            "How to use this key:",
            "  1. Go to https://coreai3d.nl/register",
            "  2. Enter the beta key above when prompted.",
            "  3. Complete the account registration form.",
            "",
            "Each beta key can only be used once. Store it somewhere safe.",
        ]
        return "\n".join(lines)

    def _beta_key_instructions_html(self, key: str, description: str = "") -> str:
        """Build an HTML version of the beta key instructions.

        Mirrors the CoreAI3D web theme: dark background (#000000), dark cards
        (#0f0f0f), subtle borders (#1a1a1a), electric blue accents (#3b82f6)
        and Inter / system-ui typography.
        """
        desc_html = ""
        if description:
            desc_html = (
                f"<tr>"
                f"<td style='padding:8px 12px;border-bottom:1px solid #1a1a1a;"
                f"color:#9ca3af;font-size:13px;'>{'Description'.upper()}</td>"
                f"<td style='padding:8px 12px;border-bottom:1px solid #1a1a1a;"
                f"color:#ffffff;font-size:14px;'>{description}</td>"
                f"</tr>"
            )
        return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your CoreAI3D beta access key</title>
  </head>
  <body style="margin:0;padding:0;background:#000000;color:#ffffff;
               font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,
                            'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
               line-height:1.6;-webkit-text-size-adjust:100%;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           style="background:#000000;padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                 style="max-width:600px;background:#0f0f0f;border:1px solid #1a1a1a;
                        border-radius:12px;overflow:hidden;">
            <!-- Header -->
            <tr>
              <td align="center" style="background:#000000;padding:24px;
                        border-bottom:1px solid #1a1a1a;">
                <table role="presentation" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="width:10px;height:10px;background:#3b82f6;
                              border-radius:50%;display:inline-block;"></td>
                    <td style="width:8px;"></td>
                    <td style="width:10px;height:10px;background:#60a5fa;
                              border-radius:50%;display:inline-block;"></td>
                    <td style="width:8px;"></td>
                    <td style="width:10px;height:10px;background:#3b82f6;
                              border-radius:50%;display:inline-block;"></td>
                  </tr>
                </table>
                <h1 style="margin:16px 0 0 0;font-size:20px;font-weight:700;
                           color:#ffffff;letter-spacing:-0.02em;">
                  CoreAI3D<span style="color:#3b82f6;">Beta</span> Access Key
                </h1>
                <p style="margin:8px 0 0 0;font-size:14px;color:#9ca3af;">
                  Keep this key safe — it is shown only once.
                </p>
              </td>
            </tr>
            <!-- Key card -->
            <tr>
              <td style="padding:24px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="padding:8px 12px;border-bottom:1px solid #1a1a1a;
                              color:#9ca3af;font-size:13px;>{'Beta Key'.upper()}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #1a1a1a;
                              color:#ffffff;font-size:15px;text-align:right;
                              font-family:ui-monospace,SFMono-Regular,Menlo,
                                    Monaco,Consolas,monospace;">
                      {key}
                    </td>
                  </tr>
                  {desc_html}
                </table>
              </td>
            </tr>
            <!-- How to use -->
            <tr>
              <td style="padding:0 24px 24px 24px;">
                <h2 style="margin:0 0 12px 0;font-size:15px;font-weight:700;
                           color:#ffffff;letter-spacing:-0.01em;">
                  How to use this key
                </h2>
                <ol style="margin:0;padding-left:20px;color:#d1d5db;
                           font-size:14px;">
                  <li style="margin-bottom:8px;">Go to
                    <a href="https://coreai3d.nl/register" style="color:#60a5fa;text-decoration:underline;">
                      https://coreai3d.nl/register
                    </a>
                  </li>
                  <li style="margin-bottom:8px;">Enter the beta key below when prompted:
                    <br>
                    <code style="display:block;margin-top:6px;color:#9ca3af;
                                 background:#111111;padding:10px 14px;
                                 border-radius:8px;overflow-wrap:break-word;">
                      {key}
                    </code>
                  </li>
                  <li style="margin-bottom:8px;">Complete the account registration form.
                    Each beta key can only be used once.
                  </li>
                </ol>
                <p style="margin:20px 0 0 0;font-size:13px;color:#9ca3af;
                          border-top:1px solid #1a1a1a;padding-top:16px;">
                  Store this key somewhere safe — it is shown only once.
                </p>
              </td>
            </tr>
            <!-- Footer -->
            <tr>
              <td align="center" style="padding:16px 24px;
                        border-top:1px solid #1a1a1a;">
                <p style="margin:0;font-size:12px;color:#9ca3af;">
                  © CoreAI3D — Beta Program
                </p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""

    def _populate_beta_keys_table(self, records: List[Dict[str, Any]]) -> None:
        """Fill the beta keys table from the records returned by the backend."""
        table = self.beta_keys_table
        table.setRowCount(0)
        for record in records:
            row = table.rowCount()
            table.insertRow(row)

            key = str(record.get('key') or record.get('beta_key') or
                      record.get('api_key') or record.get('hash') or '')
            key_id = record.get('id', record.get('key_id', ''))
            max_uses = record.get('max_uses', record.get('maxUses', ''))
            expires_at = record.get('expires_at', record.get('expiresAt', ''))
            active = record.get('active', record.get('enabled', ''))
            email = record.get('email', '')

            values = [
                str(key),
                str(key_id),
                '' if max_uses is None else str(max_uses),
                str(expires_at or ''),
                self._format_beta_active(active),
                str(email or ''),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setToolTip(value)
                table.setItem(row, column, item)

        # Auto-select the first (most recent) key so Copy / Send / Revoke are
        # immediately usable after a generation or refresh.
        if table.rowCount() > 0:
            table.selectRow(0)
            table.setCurrentItem(table.item(0, 0))

    @staticmethod
    def _format_beta_active(value: Any) -> str:
        """Render the 'active' column consistently."""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if value is None or value == '':
            return ""
        return str(value)

    # ------------------------------------------------------------------
    # Beta Key validation & activation
    # ------------------------------------------------------------------
    def on_validate_beta_key_clicked(self):
        """Validate the beta key entered by the user against the backend."""
        api_key = self.beta_validate_input.text().strip()
        if not api_key:
            self._notify(
                "Key Required",
                "Enter a beta key to validate.",
                level="warning"
            )
            return

        if getattr(self, '_beta_validating', False):
            self.status_bar.showMessage(
                "Beta key validation already in progress", 3000
            )
            return

        self._beta_validating = True
        self._beta_validating_key = api_key
        self.beta_validate_btn.setEnabled(False)
        self.beta_status_label.setText("")
        self.status_bar.showMessage("Validating beta key...")

        self._run_beta_operation(
            'validate',
            self._on_beta_key_validated,
            key=api_key,
        )

    def on_clear_beta_key_clicked(self):
        """Clear the beta key input field and status."""
        self.beta_validate_input.clear()
        self.beta_status_label.setText("")
        self.status_bar.showMessage("Cleared", 2000)

    def _on_beta_key_validated(self, response: APIResponse) -> None:
        """Handle the validation result on the GUI thread.

        Cleanup of the worker thread is already performed by
        :meth:`_run_beta_operation` before this callback runs.
        """
        self._beta_validating = False
        self.beta_validate_btn.setEnabled(True)

        if not response.success:
            self._set_beta_status(
                f"Validation unavailable: {self._beta_error_text(response)}",
                "error",
            )
            self.status_bar.showMessage(
                "Beta key validation failed", 3000
            )
            return

        data = response.data or {}
        valid = data.get('valid', False)

        if valid:
            self._set_beta_status(
                "✓ Beta key is valid and active.",
                "success",
            )
            self.status_bar.showMessage(
                "Beta key validated successfully", 3000
            )
            self._notify("Beta Key Valid", "The beta key is valid and active.")
            # Persist the key so its status can be re-checked on next launch.
            self._persist_beta_key(self._beta_validating_key)
        else:
            self._set_beta_status(
                "✗ Beta key is invalid, expired or revoked.",
                "error",
            )
            self.status_bar.showMessage(
                "Beta key is not valid", 3000
            )
            self._notify(
                "Beta Key Invalid",
                "The beta key is invalid, expired or revoked.",
                level="warning",
            )

    def _set_beta_status(self, message: str, level: str = "info") -> None:
        """Update the validation status label with level-appropriate styling."""
        colors = {
            "success": "#dcfce7",
            "error": "#fee2e2",
            "info": "#eff6ff",
            "warning": "#fef9c3",
        }
        text_colors = {
            "success": "#166534",
            "error": "#991b1b",
            "info": "#1e40af",
            "warning": "#854d0e",
        }
        bg = colors.get(level, colors["info"])
        fg = text_colors.get(level, text_colors["info"])
        self.beta_status_label.setText(message)
        self.beta_status_label.setStyleSheet(
            f"QLabel {{ padding: 8px; border-radius: 4px; "
            f"background-color: {bg}; color: {fg}; }}"
        )

    def _persist_beta_key(self, api_key: str) -> None:
        """Persist the validated beta key in QSettings for next launch."""
        try:
            settings = QSettings("CoreAI3D", "Dashboard")
            settings.setValue("beta_key", api_key)
            settings.sync()
            self.config.beta_key = api_key
        except Exception as e:
            logger.debug(f"Could not persist beta key: {e}")

    def _restore_beta_key_state(self) -> None:
        """Restore a previously persisted beta key and re-validate on launch.

        The key is loaded into the input field (masked) and silently validated
        against the backend so the status label reflects the real activation
        state after a reload.
        """
        try:
            settings = QSettings("CoreAI3D", "Dashboard")
            api_key = settings.value("beta_key", "", type=str)
        except Exception:
            api_key = ""

        if not api_key:
            return

        self.beta_validate_input.setText(api_key)
        self.beta_status_label.setText("Re-checking key status on backend…")
        self.beta_status_label.setStyleSheet(
            "QLabel { padding: 8px; border-radius: 4px; "
            "background-color: #eff6ff; color: #1e40af; }"
        )

        # Defer the actual validation until the UI is fully shown and the
        # backend client is available.
        QTimer.singleShot(1000, lambda: self._revalidate_persisted_beta_key(api_key))

    def _revalidate_persisted_beta_key(self, api_key: str) -> None:
        """Re-validate a persisted beta key on startup."""
        if getattr(self, '_beta_validating', False):
            return

        self._beta_validating = True
        self.beta_validate_btn.setEnabled(False)

        self._run_beta_operation(
            'validate',
            self._on_beta_key_validated_restored,
            key=api_key,
        )

    def _on_beta_key_validated_restored(self, response: APIResponse) -> None:
        """Handle the re-validation result from startup restore."""
        self._beta_validating = False
        self.beta_validate_btn.setEnabled(True)

        if not response.success:
            self._set_beta_status(
                "Could not re-check key status: backend unavailable.",
                "warning",
            )
            self.status_bar.showMessage(
                "Beta key status re-check failed (backend unreachable)", 4000
            )
            return

        data = response.data or {}
        valid = data.get('valid', False)

        if valid:
            self._set_beta_status(
                "✓ Beta key is valid and active.",
                "success",
            )
        else:
            self._set_beta_status(
                "✗ Beta key is invalid, expired or revoked.",
                "error",
            )

    # ------------------------------------------------------------------
    # Beta Keys handlers
    # ------------------------------------------------------------------
    def on_generate_email_key_clicked(self):
        """Generate a single beta key bound to the recipient email."""
        email = self.beta_email_input.text().strip()
        if not email:
            self._notify(
                "Email Required",
                "Enter the recipient email address before generating a key.",
                level="warning"
            )
            return

        if "@" not in email or email.startswith("@") or email.endswith("@"):
            self._notify(
                "Invalid Email",
                f"'{email}' is not a valid email address.",
                level="warning"
            )
            return

        description = self.beta_email_desc_input.text().strip()
        self.status_bar.showMessage("Generating beta key...")
        self._run_beta_operation(
            'generate',
            self._on_email_key_generated,
            count=1,
            email=email,
            description=description,
        )

    def _on_email_key_generated(self, response: APIResponse) -> None:
        """Handle the result of the 'Generate Email Key' request."""
        # DEBUG: log the raw response for troubleshooting key extraction
        logger.debug(
            f"_on_email_key_generated RAW: success={response.success}, "
            f"status_code={response.status_code}, "
            f"message={response.message!r}, "
            f"data={response.data!r}, "
            f"data_type={type(response.data).__name__}"
        )
        if not response.success:
            self._notify(
                "Beta Key Generation Failed",
                self._beta_error_text(response),
                level="critical"
            )
            return

        keys = self._parse_beta_key_response(response)
        if not keys:
            self._notify(
                "Beta Key Generation",
                "The key was generated but the server response did not contain it.",
                level="warning"
            )
            self.on_refresh_beta_keys_clicked()
            return

        key = keys[0]
        email = self.beta_email_input.text().strip()
        self._notify(
            "Beta Key Generated",
            f"Beta key for {email} generated successfully.",
        )

        # The beta key is for coreai3d.nl account access, NOT for the
        # CoreAI3D backend API. Keep it separate from config.api_key.
        # It is stored in config.beta_key via QSettings on validation.
        self.status_bar.showMessage(
            f"Beta key generated for {email} (copied to clipboard)", 3000
        )

        # Sending the mail is a blocking network call: run it on a worker.
        self._run_email_send(
            email, key, self.beta_email_desc_input.text().strip()
        )

        self.on_refresh_beta_keys_clicked()

    def on_generate_beta_keys_clicked(self):
        """Generate a batch of beta keys."""
        count = self.beta_count_spin.value()
        description = self.beta_desc_input.text().strip()
        self.status_bar.showMessage(f"Generating {count} beta key(s)...")
        self._run_beta_operation(
            'generate',
            self._on_beta_keys_generated,
            count=count,
            description=description,
        )

    def _on_beta_keys_generated(self, response: APIResponse) -> None:
        """Handle the result of the 'Generate Beta Keys' request."""
        logger.debug(
            f"_on_beta_keys_generated raw response: success={response.success}, "
            f"status_code={response.status_code}, "
            f"message={response.message!r}, "
            f"data={response.data!r}"
        )
        if not response.success:
            self._notify(
                "Beta Key Generation Failed",
                self._beta_error_text(response),
                level="critical"
            )
            return

        keys = self._parse_beta_key_response(response)
        if keys:
            self._notify(
                "Beta Keys Generated",
                f"Generated {len(keys)} beta key(s)."
            )
            self._copy_to_clipboard(keys[0], "First generated beta key copied to clipboard")

            # Beta keys are for coreai3d.nl account access, NOT for the
            # CoreAI3D backend API. Keep them separate from config.api_key.
            self.status_bar.showMessage(
                f"Generated {len(keys)} beta key(s) (first copied to clipboard)", 3000
            )
        else:
            self._notify(
                "Beta Keys Generated",
                "The keys were generated but the server response did not contain them."
            )

        self.on_refresh_beta_keys_clicked()

    def on_refresh_beta_keys_clicked(self):
        """Reload the list of beta keys from the backend."""
        self.status_bar.showMessage("Refreshing beta keys...")
        self._run_beta_operation('list', self._on_beta_keys_listed)

    def _on_beta_keys_listed(self, response: APIResponse) -> None:
        """Handle the result of the /beta/list request."""
        if not response.success:
            self._notify(
                "Beta Keys Unavailable",
                self._beta_error_text(response),
                level="warning"
            )
            return

        records = self._parse_beta_list_response(response)
        self._populate_beta_keys_table(records)
        self.status_bar.showMessage(f"{len(records)} beta key(s) loaded", 3000)

    def on_copy_beta_key_clicked(self):
        """Copy the selected beta key to the clipboard."""
        _row, key, _key_id = self._selected_beta_key()
        if not key:
            self._notify(
                "No Key Selected",
                "Select a beta key in the table first.",
                level="warning"
            )
            return
        self._copy_to_clipboard(key, "Beta key copied to clipboard")

    def on_revoke_beta_key_clicked(self):
        """Revoke (disable) the selected beta key on the backend."""
        _row, _key, key_id = self._selected_beta_key()
        if key_id is None:
            self._notify(
                "No Key Selected",
                "Select a beta key in the table first.",
                level="warning"
            )
            return

        self.status_bar.showMessage("Revoking beta key...")
        self._run_beta_operation(
            'toggle',
            self._on_beta_key_revoked,
            key_id=key_id,
            enable=False,
        )

    def _on_beta_key_revoked(self, response: APIResponse) -> None:
        """Handle the result of the /beta/toggle request."""
        if not response.success:
            self._notify(
                "Revoke Failed",
                self._beta_error_text(response),
                level="critical"
            )
            return

        self._notify("Beta Key Revoked", "The beta key was revoked.")
        self.on_refresh_beta_keys_clicked()

    def on_copy_instructions_clicked(self):
        """Copy the selected beta key together with the usage instructions."""
        _row, key, _key_id = self._selected_beta_key()
        if not key:
            self._notify(
                "No Key Selected",
                "Select a beta key in the table first.",
                level="warning"
            )
            return

        description = self.beta_desc_input.text().strip()
        instructions = self._beta_key_instructions(key, description)
        self._copy_to_clipboard(instructions, "Beta key and instructions copied to clipboard")

    def on_send_beta_key_clicked(self):
        """Email the selected beta key to its recipient."""
        _row, key, _key_id = self._selected_beta_key()
        if not key:
            self._notify(
                "No Key Selected",
                "Select a beta key in the table first.",
                level="warning"
            )
            return

        recipient = self.beta_email_input.text().strip()
        if not recipient:
            self._ask_for_email(
                lambda email: self._send_beta_key_to(email, key)
            )
            return

        self._send_beta_key_to(recipient, key)

    def _send_beta_key_to(self, recipient: str, key: str) -> None:
        """Email `key` to `recipient` and report the outcome."""
        recipient = (recipient or "").strip()
        if not recipient:
            self._notify(
                "Email Required",
                "Enter the recipient email address and try again.",
                level="warning"
            )
            return

        self.beta_email_input.setText(recipient)
        description = self.beta_email_desc_input.text().strip()
        self.status_bar.showMessage(f"Sending beta key to {recipient}...")
        self._run_email_send(recipient, key, description)

    def _ask_for_email(self, on_entered) -> None:
        """Prompt for a recipient address without blocking the GUI thread."""
        try:
            self._prompt_text(
                "Send Beta Key", "Recipient email address:",
                on_entered, default_text=self.beta_email_input.text().strip(),
            )
        except Exception as e:
            logger.debug(f"Email prompt failed: {e}")

    def _copy_to_clipboard(self, text: str, message: str) -> None:
        """Copy `text` to the system clipboard and confirm it."""
        try:
            clipboard = QApplication.clipboard()
            if clipboard is None:
                self._notify("Clipboard Unavailable", "Could not access the clipboard.",
                             level="warning")
                return
            clipboard.setText(text)
            self.status_bar.showMessage(message, 3000)
        except Exception as e:
            logger.error(f"Clipboard copy failed: {e}")
            self._notify("Clipboard Error", str(e), level="warning")

    @staticmethod
    def _beta_error_text(response: APIResponse) -> str:
        """Build a readable error message from a failed API response."""
        if response is None:
            return "No response from the backend."
        message = (getattr(response, 'message', '') or '').strip()
        data = getattr(response, 'data', None)
        detail = ''
        if isinstance(data, dict):
            for field in ('error', 'detail', 'reason'):
                value = data.get(field)
                if isinstance(value, str) and value.strip():
                    detail = value.strip()
                    break
        status = getattr(response, 'status_code', 0)
        parts = [part for part in (message, detail) if part]
        if not parts:
            parts = [f"The backend returned HTTP {status}."]
        return "\n".join(parts)

    @staticmethod
    def _api_error_text(response: APIResponse) -> str:
        """Build a readable error message from a failed API response."""
        return CoreAI3DDashboard._beta_error_text(response)

    def _run_email_send(self, recipient: str, key: str, description: str = "") -> None:
        """Send a beta key email on a worker thread.

        Resend/SMTP calls block, so they never run on the GUI thread.
        """
        self._stop_email_worker()

        thread = QThread()
        worker = AsyncWorker(
            _call_blocking, self._send_beta_key_email, recipient, key, description
        )
        worker.moveToThread(thread)

        def cleanup():
            self._email_worker_thread = None
            self._email_worker = None
            try:
                thread.quit()
                if not thread.wait(2000):
                    _LINGERING_THREADS.append(thread)
            except RuntimeError:
                pass

        def handle_success(result):
            cleanup()
            sent, message = result if isinstance(result, tuple) else (False, str(result))
            if sent:
                self._notify("Beta Key Emailed", message)
            else:
                self._notify("Email Not Sent", message, level="warning")

        def handle_error(err):
            cleanup()
            self._notify("Email Not Sent", f"Error: {err}", level="critical")

        worker.finished.connect(handle_success, Qt.ConnectionType.QueuedConnection)
        worker.error.connect(handle_error, Qt.ConnectionType.QueuedConnection)
        thread.started.connect(worker.run)
        thread.finished.connect(cleanup)

        # Keep the thread and the worker referenced before the thread is
        # started: otherwise the worker could be garbage collected while its
        # thread is still starting up and the `started` -> `run` connection
        # would die with it.
        self._email_worker_thread = thread
        self._email_worker = worker
        thread.start()

    def _stop_email_worker(self) -> None:
        """Stop a running email worker thread, if any."""
        thread = getattr(self, '_email_worker_thread', None)
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(2000):
                    _LINGERING_THREADS.append(thread)
        except RuntimeError:
            pass
        self._email_worker_thread = None
        self._email_worker = None

    def _send_beta_key_email(self, recipient: str, key: str, description: str = "") -> Tuple[bool, str]:
        """Send `key` to `recipient`.

        The Resend API is used when it is configured; otherwise plain SMTP is
        tried (SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD).  When the
        Resend delivery fails and SMTP is configured, SMTP is used as a
        fallback so a single broken transport does not lose the key.

        Returns (sent, message).
        """
        resend_key = (self.config.resend_api_key or
                      os.environ.get('RESEND_API_KEY', '')).strip()
        smtp_configured = bool(os.environ.get('SMTP_HOST', '').strip())

        if resend_key:
            sent, message = self._send_beta_key_email_resend(
                recipient, key, description)
            if sent:
                return sent, message
            if smtp_configured:
                logger.warning(
                    f"Resend delivery failed ({message}); falling back to SMTP")
                return self._send_beta_key_email_smtp(recipient, key, description)
            return False, message

        if smtp_configured:
            return self._send_beta_key_email_smtp(recipient, key, description)

        return False, (
            "Email delivery is not configured. Set RESEND_API_KEY (or the "
            "Resend API Key field in Settings), or SMTP_HOST/SMTP_PORT/"
            "SMTP_USER/SMTP_PASSWORD, to enable it."
        )

    def _send_beta_key_email_resend(self, recipient: str, key: str,
                                    description: str = "") -> Tuple[bool, str]:
        """Send a beta key through the Resend REST API.

        The request is made with the standard library so the worker thread
        needs neither an extra dependency nor a running event loop.

        Resend requires the ``from`` address to use a verified domain.  The
        built-in default ``noreply@resend.dev`` is a domain Resend verifies
        for all accounts, so it works without any domain setup.  Override it
        with ``RESEND_FROM_EMAIL`` (or the Settings field) once you have
        verified your own domain.
        """
        import json
        import urllib.error
        import urllib.request

        resend_key = (self.config.resend_api_key or
                      os.environ.get('RESEND_API_KEY', '')).strip()
        from_email = (self.config.resend_from_email or
                      os.environ.get('RESEND_FROM_EMAIL', '')).strip()

        if not from_email:
            from_email = 'noreply@resend.dev'

        payload = {
            "from": from_email,
            "to": [recipient],
            "subject": "Your CoreAI3D beta access key",
            "text": self._beta_key_instructions(key, description),
            "html": self._beta_key_instructions_html(key, description),
        }

        body = json.dumps(payload).encode('utf-8')
        headers = {
            "Authorization": f"Bearer {resend_key}",
            "Content-Type": "application/json",
            # Resend blocks requests that omit a User-Agent header (403 error
            # code 1010) before they ever reach the API.
            "User-Agent": "CoreAI3D-LinuxGUI/1.3.0",
        }

        # Diagnostic log of the full request (API key is masked).
        masked_key = (resend_key[:8] + "...") if resend_key else "<none>"
        logger.debug(
            "Resend request: POST %s\n"
            "  headers: %s\n"
            "  body: %s",
            self.RESEND_API_URL,
            {k: (v if k != "Authorization" else f"Bearer {masked_key}")
             for k, v in headers.items()},
            body.decode('utf-8', 'replace'),
        )

        request = urllib.request.Request(
            self.RESEND_API_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode('utf-8', 'replace')
                logger.debug("Resend response (%s): %s",
                             response.status, response_body)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode('utf-8', 'replace').strip()
            except Exception:
                detail = ''
            logger.error(
                "Resend API rejected the beta key email: HTTP %s %s\n"
                "  request URL: %s\n"
                "  request headers: %s\n"
                "  request body: %s\n"
                "  response body: %s",
                e.code, e.reason,
                self.RESEND_API_URL,
                {k: (v if k != "Authorization" else f"Bearer {masked_key}")
                 for k, v in headers.items()},
                body.decode('utf-8', 'replace'),
                detail,
            )
            return False, (
                f"Resend API error {e.code}: {detail}"
                if detail else f"Resend API error {e.code}"
            )
        except Exception as e:
            logger.error(
                "Failed to send beta key email via Resend: %s\n"
                "  request URL: %s\n"
                "  request headers: %s\n"
                "  request body: %s",
                e, self.RESEND_API_URL,
                {k: (v if k != "Authorization" else f"Bearer {masked_key}")
                 for k, v in headers.items()},
                body.decode('utf-8', 'replace'),
            )
            return False, f"Failed to send the email: {e}"

        return True, f"Beta key sent to {recipient}."

    def _send_beta_key_email_smtp(self, recipient: str, key: str,
                                  description: str = "") -> Tuple[bool, str]:
        """Send a beta key through a plain SMTP server."""
        import smtplib
        from email.message import EmailMessage

        host = os.environ.get('SMTP_HOST', '').strip()
        try:
            port = int(os.environ.get('SMTP_PORT', '587') or 587)
        except (TypeError, ValueError):
            port = 587
        user = os.environ.get('SMTP_USER', '').strip()
        password = os.environ.get('SMTP_PASSWORD', '').strip()
        sender = (os.environ.get('SMTP_FROM', '').strip() or
                  self.config.resend_from_email or user or
                  'noreply@coreai3d.local')

        message = EmailMessage()
        message['Subject'] = 'Your CoreAI3D beta access key'
        message['From'] = sender
        message['To'] = recipient
        message.set_content(self._beta_key_instructions(key, description))

        try:
            with smtplib.SMTP(host, port, timeout=30) as server:
                try:
                    server.starttls()
                except smtplib.SMTPException:
                    logger.debug("SMTP server does not support STARTTLS")
                if user and password:
                    server.login(user, password)
                server.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send beta key email via SMTP: {e}")
            return False, f"Failed to send the email via SMTP: {e}"

        return True, f"Beta key sent to {recipient}."

    def setup_connections(self):
        """Setup signal-slot connections"""
        # Sandbox manager connections
        self.sandbox_manager.sandbox_started.connect(self.on_sandbox_started)
        self.sandbox_manager.sandbox_stopped.connect(self.on_sandbox_stopped)
        self.sandbox_manager.sandbox_error.connect(self.on_sandbox_error)

        # Training manager connections
        self.training_manager.dataset_created.connect(self.on_dataset_created)
        self.training_manager.dataset_updated.connect(self.on_dataset_updated)
        self.training_manager.file_added.connect(self.on_file_added)

    def setup_timers(self):
        """Setup periodic update timers"""
        # System metrics timer (every 2 seconds)
        self.metrics_timer = self._create_gui_timer()
        self.metrics_timer.timeout.connect(self.update_system_metrics)
        self.metrics_timer.start(2000)

        # Health check timer (every 30 seconds)
        self.health_timer = self._create_gui_timer()
        self.health_timer.timeout.connect(self.check_system_health)
        self.health_timer.start(30000)

    def initialize_client(self):
        """Initialize CoreAI3D client - schedules async initialization"""
        try:
            # Guard against multiple initializations
            if hasattr(self, 'client_initialized') and self.client_initialized:
                logger.info("Client already initialized, skipping")
                return True
            if hasattr(self, '_client_initializing') and self._client_initializing:
                logger.info("Client initialization already in progress, skipping")
                return True
            self._client_initializing = True
            logger.info("Scheduling client initialization...")
            # Schedule the async initialization properly within the qasync event loop
            task = asyncio.create_task(self._initialize_client_async())
            task_id = f"client_init_{id(task)}"
            with self.task_lock:
                self.active_tasks.add(task_id)
            task.add_done_callback(lambda t: self._cleanup_task(task_id))
            return True
        except RuntimeError:
            # No running event loop yet, defer until event loop is available
            self._client_initializing = False
            logger.warning("No running event loop, deferring client initialization")
            # This will be called again by the QTimer.singleShot in main()
            return False
        except Exception as e:
            self._client_initializing = False
            logger.error(f"Failed to schedule client initialization: {str(e)}")
            return False

    def _cleanup_task(self, task_id: str):
        """Clean up completed task from tracking"""
        with self.task_lock:
            self.active_tasks.discard(task_id)

    def cancel_all_tasks(self):
        """Cancel all active async tasks"""
        try:
            # Get current tasks to cancel
            tasks_to_cancel = []
            async def cancel_tasks():
                async with self.task_cleanup_lock:
                    tasks_to_cancel = list(self.pending_tasks)
                    self.pending_tasks.clear()

                    for task in tasks_to_cancel:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

            # Schedule task cancellation in the event loop
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(cancel_tasks())
            else:
                # If event loop is not running, we can't cancel async tasks
                logger.warning("Event loop not running, cannot cancel async tasks")

            # Cancel training if running (schedule in event loop)
            async def cancel_training():
                async with self.training_lock:
                    if self.training_in_progress:
                        self.training_in_progress = False
                        logger.info("Training cancelled via cancel_all_tasks")

            if asyncio.get_event_loop().is_running():
                asyncio.create_task(cancel_training())

            logger.info(f"Cancelled {len(tasks_to_cancel)} active tasks")
        except Exception as e:
            logger.error(f"Error cancelling tasks: {str(e)}")

    async def _initialize_client_async(self):
        """Async client initialization"""
        try:
            if hasattr(self, 'client_initialized') and self.client_initialized:
                logger.info("Client already initialized (async check), skipping")
                self._client_initializing = False
                return
            if hasattr(self, '_client_initializing') and not self._client_initializing:
                logger.info("Client initialization not flagged as in-progress (async check), skipping")
                return
            logger.info("Starting client initialization...")
            logger.info(f"API URL: {self.config.api_url}")
            logger.info(f"WS URL: {self.config.ws_url}")
            logger.info(f"API Key configured: {bool(self.config.api_key)}")

            if not self.config.api_url:
                logger.warning("API URL not configured - falling back to default")
                self.config.api_url = default_api_url()

            if not self.config.api_key:
                logger.warning("API key not configured - using empty key (may cause authentication failures)")
                self.config.api_key = ""

            logger.info("Creating CoreAI3DClient instance...")
            api_key = self.config.api_key if self.config.api_key else ""
            logger.info(f"Passing API key to client: configured={bool(api_key)}, length={len(api_key)}")

            if not api_key:
                logger.debug("No API key configured - client will work in offline mode only")
            elif len(api_key) < 10:
                logger.warning(f"API key is very short ({len(api_key)} chars) - this may cause authentication issues")

            self.client = CoreAI3DClient({
                'base_url': self.config.api_url,
                'ws_url': self.config.ws_url,
                'api_key': api_key,
                'timeout': self.config.connection_timeout,
                'max_connections': self.config.max_connections,
                'retry_attempts': self.config.retry_attempts,
                'debug': True
            })

            logger.info("CoreAI3DClient instance created successfully")

            self.chat_response_handler = self._handle_chat_response
            self.client.on('chat_response', self.chat_response_handler)
            try:
                self.client.on('error', self._handle_chat_error_ws)
            except Exception as e:
                logger.warning(f"Failed to register chat error handler: {e}")
            logger.info("Chat response handler registered")

            logger.info("Connecting CoreAI3D client (HTTP + WebSocket)...")
            http_ok = False
            ws_ok = False
            connect_timeout = getattr(self.config, 'connection_timeout', 15.0) or 15.0
            try:
                await asyncio.wait_for(self.client.connect(), timeout=connect_timeout)
                logger.info("CoreAI3D client connected successfully")
                http_ok = True
                ws_ok = self.client.is_connected()
            except asyncio.TimeoutError:
                logger.warning(f"Client connect() timed out after {connect_timeout}s — HTTP path will still be available")
                http_ok = self.client.session is not None
            except Exception as conn_err:
                logger.warning(f"Client connect() raised: {conn_err} — HTTP path will still be available")
                http_ok = self.client.session is not None

            logger.info("Creating CodeDebugger...")
            self.debugger = CodeDebugger(self.client)
            logger.info("CodeDebugger created successfully")
            logger.info("CoreAI3D client initialized successfully")

            self.client_initialized = True
            self._client_initializing = False

        except ImportError as e:
            self._client_initializing = False
            logger.error(f"Import error during client initialization: {str(e)}")
            logger.error("Make sure coreai3d_client.py is in the same directory or PYTHONPATH")
            raise
        except Exception as e:
            self._client_initializing = False
            logger.error(f"Failed to initialize client: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    # Event handlers
    def on_dataset_selected(self, item: QListWidgetItem):
        """Handle dataset selection"""
        dataset_name = item.text()
        if dataset_name in self.training_manager.datasets:
            dataset = self.training_manager.datasets[dataset_name]

            # Create enhanced display with calculated sizes
            display_info = {
                'name': dataset['name'],
                'description': dataset['description'],
                'data_type': dataset['data_type'],
                'created_at': dataset['created_at'],
                'dataset_sizes': {
                    'inputsize': dataset['metadata'].get('inputsize', 'N/A'),
                    'outputsize': dataset['metadata'].get('outputsize', 'N/A'),
                    'samplesize': dataset['metadata'].get('samplesize', 'N/A')
                },
                'files': []
            }

            # Add file information with individual sizes
            for file_info in dataset.get('files', []):
                file_display = {
                    'name': file_info['name'],
                    'size_bytes': file_info['size'],
                    'added_at': file_info['added_at'],
                    'inputsize': file_info.get('inputsize', 'N/A'),
                    'outputsize': file_info.get('outputsize', 'N/A'),
                    'samplesize': file_info.get('samplesize', 'N/A')
                }
                display_info['files'].append(file_display)

            self.dataset_info.setPlainText(json.dumps(display_info, indent=2))

            # Update file list
            self.file_list.clear()
            for file_info in dataset.get('files', []):
                size_info = f" (Input: {file_info.get('inputsize', 'N/A')}, Output: {file_info.get('outputsize', 'N/A')}, Samples: {file_info.get('samplesize', 'N/A')})"
                display_text = f"{file_info['name']}{size_info}"
                self.file_list.addItem(display_text)

    def on_sandbox_started(self, message: str):
        """Handle sandbox started event"""
        self.sandbox_status_label.setText("Docker: Running")
        self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.status_bar.showMessage("Docker started successfully")

    def on_sandbox_stopped(self, message: str):
        """Handle sandbox stopped event"""
        self.sandbox_status_label.setText("Docker: Not Running")
        self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.status_bar.showMessage("Docker stopped")

    def on_sandbox_error(self, title: str, message: str):
        """Handle sandbox error event"""
        self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {message}")
        self._notify(title, message, level="warning")
        self.status_bar.showMessage(f"Docker error: {title}")

    def on_dataset_created(self, name: str, message: str):
        """Handle dataset created event"""
        self.dataset_list.addItem(name)
        self.status_bar.showMessage(message)

    def on_dataset_updated(self, name: str):
        """Handle dataset updated event"""
        # Refresh dataset display
        self.on_dataset_selected(self.dataset_list.currentItem())

    def on_file_added(self, dataset_name: str, filename: str):
        """Handle file added event"""
        self.status_bar.showMessage(f"Added {filename} to {dataset_name}")

    # UI action methods
    def test_connection(self, gui_mode=True):
        """Test API connection with enhanced error handling and debugging

        Args:
            gui_mode: If True, show GUI dialogs. If False, return result dict.
        """
        # Check if connection test already running
        if self.connection_test_running:
            logger.info("Connection test already running")
            return

        # Schedule the async connection test in the event loop
        try:
            self.connection_test_running = True
            task = asyncio.create_task(self._test_connection_async())
            task_id = f"connection_test_{id(task)}"
            with self.task_lock:
                self.active_tasks.add(task_id)
            task.add_done_callback(lambda t: self._handle_connection_test_result_with_cleanup(t, task_id))
        except RuntimeError:
            # No running event loop, defer until event loop is available
            logger.warning("No running event loop for connection test, deferring")
            self.connection_test_running = False
            QTimer.singleShot(100, lambda: self.test_connection(gui_mode))

    def _handle_connection_test_result_with_cleanup(self, task, task_id):
        """Handle connection test result and clean up task"""
        try:
            self._handle_connection_test_result(task)
        finally:
            # Clean up task tracking
            with self.task_lock:
                self.active_tasks.discard(task_id)
            self.connection_test_running = False

            
    def _schedule_retry_test_connection(self):
        """Schedule retry of connection test using qasync-compatible approach"""
        try:
            # Create the task and store it in a variable
            task = asyncio.create_task(self._test_connection_async())
            # Add the callback that handles the result
            task.add_done_callback(self._handle_connection_test_result)
        except RuntimeError:
            logger.error("No running event loop to schedule retry connection test")
        
        
    def _generate_connection_debug_info(self, error_details: str, result=None) -> str:
        """Generate detailed connection debugging information"""
        debug_lines = []
        debug_lines.append("=== CONNECTION DEBUG INFORMATION ===")
        debug_lines.append("")
        debug_lines.append("Error Details:")
        debug_lines.append(f"  {error_details}")
        debug_lines.append("")

        # Add response details if available
        if result:
            debug_lines.append("API Response Details:")
            debug_lines.append(f"  Success: {getattr(result, 'success', 'Unknown')}")
            debug_lines.append(f"  Status Code: {getattr(result, 'status_code', 'Unknown')}")
            debug_lines.append(f"  Message: {getattr(result, 'message', 'Unknown')}")
            if hasattr(result, 'data') and result.data:
                debug_lines.append(f"  Data: {result.data}")
            debug_lines.append("")

        debug_lines.append("Configuration:")
        debug_lines.append(f"  API URL: {self.config.api_url}")
        debug_lines.append(f"  WebSocket URL: {self.config.ws_url}")
        debug_lines.append(f"  API Key Configured: {bool(self.config.api_key)}")
        debug_lines.append(f"  Connection Timeout: {self.config.connection_timeout}s")
        debug_lines.append(f"  Max Connections: {self.config.max_connections}")
        debug_lines.append("")
        debug_lines.append("Client State:")
        debug_lines.append(f"  Client Initialized: {self.client_initialized}")
        debug_lines.append(f"  Client Object: {type(self.client).__name__ if self.client else 'None'}")
        debug_lines.append("")
        debug_lines.append("Troubleshooting Steps:")
        debug_lines.append("  1. Verify the CoreAI3D server is running")
        debug_lines.append("  2. Check network connectivity to the server")
        debug_lines.append("  3. Validate API URL and port configuration")
        debug_lines.append("  4. Ensure API key is correct (if required)")
        debug_lines.append("  5. Check firewall settings")
        debug_lines.append("  6. Review server logs for additional error details")
        debug_lines.append("")
        debug_lines.append("Recent Log Entries:")
        log_file = 'logs/dashboard.log'
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                    for line in lines:
                        debug_lines.append(f"  {line.strip()}")
            except Exception as e:
                debug_lines.append(f"  Error reading logs: {str(e)}")
        else:
            debug_lines.append("  No log file found")

        return "\n".join(debug_lines)

    def _run_manual_connection_debug(self):
        """Run manual connection debugging"""
        try:
            logger.info("Starting manual connection debug")

            # Create debug dialog
            debug_dialog = QDialog(self)
            debug_dialog.setWindowTitle("Manual Connection Debug")
            debug_dialog.setModal(True)
            debug_dialog.resize(600, 400)

            layout = QVBoxLayout(debug_dialog)

            # Debug output
            debug_output = QPlainTextEdit()
            debug_output.setReadOnly(True)
            layout.addWidget(debug_output)

            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setVisible(False)
            layout.addWidget(progress_bar)

            # Control buttons
            button_layout = QHBoxLayout()

            test_network_btn = QPushButton("Test Network")
            test_network_btn.clicked.connect(lambda: self._test_network_connectivity(debug_output, progress_bar))
            button_layout.addWidget(test_network_btn)

            test_api_btn = QPushButton("Test API Endpoint")
            test_api_btn.clicked.connect(lambda: self._test_api_endpoint(debug_output, progress_bar))
            button_layout.addWidget(test_api_btn)

            test_websocket_btn = QPushButton("Test WebSocket")
            test_websocket_btn.clicked.connect(lambda: self._test_websocket_connection(debug_output, progress_bar))
            button_layout.addWidget(test_websocket_btn)

            clear_btn = QPushButton("Clear Output")
            clear_btn.clicked.connect(debug_output.clear)
            button_layout.addWidget(clear_btn)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(debug_dialog.accept)
            button_layout.addWidget(close_btn)

            layout.addLayout(button_layout)

            # Initial debug info
            debug_output.appendPlainText("Manual Connection Debug Tool")
            debug_output.appendPlainText("=" * 40)
            debug_output.appendPlainText("")
            debug_output.appendPlainText("Configuration:")
            debug_output.appendPlainText(f"  API URL: {self.config.api_url}")
            debug_output.appendPlainText(f"  WebSocket URL: {self.config.ws_url}")
            debug_output.appendPlainText(f"  API Key: {'Configured' if self.config.api_key else 'Not configured'}")
            debug_output.appendPlainText("")
            debug_output.appendPlainText("Use the buttons above to test different connection aspects.")

            # Shown window-modal (not exec()) so the dashboard event loop
            # keeps running while the tool is open.
            debug_dialog.open()

        except Exception as e:
            logger.error(f"Error in manual connection debug: {str(e)}")
            self._notify("Debug Error", f"Failed to open debug tool: {str(e)}", level="warning")

    def _test_network_connectivity(self, output_widget: QPlainTextEdit, progress_bar: QProgressBar):
        """Test basic network connectivity"""
        try:
            output_widget.appendPlainText("\n=== Testing Network Connectivity ===")
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 0)  # Indeterminate

            # Extract host and port from API URL
            from urllib.parse import urlparse
            parsed_url = urlparse(self.config.api_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            output_widget.appendPlainText(f"Testing connection to {host}:{port}...")

            # Test basic connectivity
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)

            try:
                result = sock.connect_ex((host, port))
                if result == 0:
                    output_widget.appendPlainText("✓ Network connectivity: SUCCESS")
                    output_widget.appendPlainText(f"  Connected to {host}:{port}")
                else:
                    output_widget.appendPlainText("✗ Network connectivity: FAILED")
                    output_widget.appendPlainText(f"  Could not connect to {host}:{port} (error code: {result})")
            except Exception as e:
                output_widget.appendPlainText("✗ Network connectivity: ERROR")
                output_widget.appendPlainText(f"  Error: {str(e)}")
            finally:
                sock.close()

            progress_bar.setVisible(False)

        except Exception as e:
            output_widget.appendPlainText(f"✗ Network test error: {str(e)}")
            progress_bar.setVisible(False)

    def _test_api_endpoint(self, output_widget: QPlainTextEdit, progress_bar: QProgressBar):
        """Test API endpoint connectivity"""
        try:
            output_widget.appendPlainText("\n=== Testing API Endpoint ===")
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 0)  # Indeterminate

            # Test with urllib for basic HTTP connectivity
            import urllib.request
            import urllib.error

            output_widget.appendPlainText(f"Testing HTTP GET to {self.config.api_url}...")

            try:
                # Create request with timeout
                req = urllib.request.Request(self.config.api_url)
                req.add_header('User-Agent', 'CoreAI3D-Dashboard/1.3.0')

                # Add API key if configured
                if self.config.api_key:
                    req.add_header('Authorization', f'Bearer {self.config.api_key}')

                with urllib.request.urlopen(req, timeout=30) as response:
                    output_widget.appendPlainText("✓ API endpoint: SUCCESS")
                    output_widget.appendPlainText(f"  Status Code: {response.getcode()}")
                    output_widget.appendPlainText(f"  Content-Type: {response.headers.get('content-type', 'Unknown')}")

                    # Try to read a small amount of response
                    content = response.read(1024).decode('utf-8', errors='ignore')
                    if content:
                        output_widget.appendPlainText(f"  Response Preview: {content[:200]}...")

            except urllib.error.HTTPError as e:
                output_widget.appendPlainText("✗ API endpoint: HTTP ERROR")
                output_widget.appendPlainText(f"  Status Code: {e.code}")
                output_widget.appendPlainText(f"  Reason: {e.reason}")
            except urllib.error.URLError as e:
                output_widget.appendPlainText("✗ API endpoint: URL ERROR")
                output_widget.appendPlainText(f"  Reason: {e.reason}")
            except Exception as e:
                output_widget.appendPlainText("✗ API endpoint: ERROR")
                output_widget.appendPlainText(f"  Error: {str(e)}")

            progress_bar.setVisible(False)

        except Exception as e:
            output_widget.appendPlainText(f"✗ API test error: {str(e)}")
            progress_bar.setVisible(False)

    def _test_websocket_connection(self, output_widget: QPlainTextEdit, progress_bar: QProgressBar):
        """Test WebSocket connectivity"""
        try:
            output_widget.appendPlainText("\n=== Testing WebSocket Connection ===")
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 0)  # Indeterminate

            output_widget.appendPlainText(f"Testing WebSocket connection to {self.config.ws_url}...")

            # Note: This is a basic connectivity test
            # Full WebSocket testing would require websockets library
            try:
                # Extract host and port from WebSocket URL
                from urllib.parse import urlparse
                parsed_url = urlparse(self.config.ws_url)
                host = parsed_url.hostname
                port = parsed_url.port or (443 if parsed_url.scheme == 'wss' else 80)

                output_widget.appendPlainText(f"WebSocket host: {host}:{port}")

                # Test basic TCP connectivity to WebSocket port
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)

                try:
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        output_widget.appendPlainText("✓ WebSocket port: ACCESSIBLE")
                        output_widget.appendPlainText("  Basic TCP connectivity successful")
                        output_widget.appendPlainText("  Note: Full WebSocket handshake requires websockets library")
                    else:
                        output_widget.appendPlainText("✗ WebSocket port: NOT ACCESSIBLE")
                        output_widget.appendPlainText(f"  Could not connect to {host}:{port}")
                except Exception as e:
                    output_widget.appendPlainText("✗ WebSocket port test: ERROR")
                    output_widget.appendPlainText(f"  Error: {str(e)}")
                finally:
                    sock.close()

            except Exception as e:
                output_widget.appendPlainText("✗ WebSocket test setup error: {str(e)}")

            progress_bar.setVisible(False)

        except Exception as e:
            output_widget.appendPlainText(f"✗ WebSocket test error: {str(e)}")
            progress_bar.setVisible(False)

        except Exception as e:
            logger.error(f"Unexpected error during connection test: {str(e)}", exc_info=True)
            self.status_bar.showMessage("Connection test error")
            self._notify("Connection Error", f"✗ Unexpected error during connection test!\n\n"
                f"Error: {str(e)}\n\n"
                "Please check the application logs for more details.", level="critical")

    async def _test_connection_async(self):
        """Async part of connection test"""
        async with self.global_async_lock:
            # Initialize client if not already done
            if not self.client_initialized:
                if not self.initialize_client():
                    return type('Result', (), {'success': False, 'data': {'error': 'Failed to initialize client'}})()

            # Run health check - client handles connection automatically
            result = await self.client.health_check()
            return result

    def _handle_connection_test_result(self, task):
        """Handle connection test result"""
        try:
            result = task.result()
            self.status_bar.showMessage("Connection test completed")
            if result.success:
                self._notify("Connection Test Successful", "✓ API connection successful!\n\n"
                    "The CoreAI3D server is responding correctly.\n"
                    "All systems are operational.", level="info")
            else:
                error_details = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self._notify("Connection Test Failed", f"✗ API connection failed!\n\n"
                    f"Error: {error_details}\n\n"
                    "Please check:\n"
                    "• Server is running\n"
                    "• Network connectivity\n"
                    "• API URL configuration", level="warning")
        except Exception as e:
            self.status_bar.showMessage("Connection test error")
            self._notify("Connection Error", f"✗ Unexpected error during connection test!\n\n"
                f"Error: {str(e)}\n\n"
                "Please check the application logs for more details.", level="critical")

    def update_system_metrics(self):
        """Update system metrics display"""
        try:
            # Get system metrics.  psutil is called with interval=None so the
            # GUI thread is never blocked while sampling the CPU.
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            # Update labels
            self.cpu_label.setText(f"CPU Usage: {cpu_percent:.1f}%")
            self.memory_label.setText(f"Memory Usage: {memory.percent:.1f}%")
            self.disk_label.setText(f"Disk Usage: {disk.percent:.1f}%")

            if network:
                network_text = f"Network: {network.bytes_recv / 1024:.1f} KB/s RX, {network.bytes_sent / 1024:.1f} KB/s TX"
                self.network_label.setText(network_text)

            # Update progress bars
            self.cpu_bar.setValue(int(cpu_percent))
            self.memory_bar.setValue(int(memory.percent))
            self.disk_bar.setValue(int(disk.percent))

            # Update health score
            health_score = self.calculate_health_score(cpu_percent, memory.percent, disk.percent)
            self.health_score_label.setText(f"Overall Health Score: {health_score}%")
            self.health_bar.setValue(health_score)

        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")

    def check_system_health(self):
        """Perform periodic system health check"""
        if not self.client:
            return

        try:
            # Only perform health check if client is initialized and has HTTP session
            if not self.client_initialized:
                return

            # Check if HTTP session is available before scheduling health check
            if not hasattr(self.client, 'session') or self.client.session is None:
                logger.debug("HTTP session not available, skipping health check")
                return

            # Schedule async health check without blocking UI using qasync-compatible approach
            QTimer.singleShot(0, self._schedule_health_check)

        except Exception as e:
            logger.error(f"Health check scheduling error: {str(e)}")

    def _schedule_health_check(self):
        """Schedule health check using qasync-compatible approach"""
        try:
            asyncio.create_task(self._perform_health_check())
        except Exception as e:
            logger.error(f"Failed to schedule health check: {str(e)}")

    async def _perform_health_check(self):
        """Perform the actual async health check"""
        try:
            # Double-check HTTP session availability before proceeding
            if not hasattr(self.client, 'session') or self.client.session is None:
                logger.debug("HTTP session not available during health check, skipping")
                return

            async with self.global_async_lock:
                result = await self.client.health_check()
                if not result.success:
                    self.status_bar.showMessage("System health check failed")
                    logger.warning("System health check failed")

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            # Don't retry immediately to avoid spam
            if "Event loop is closed" in str(e) or "Future attached to a different loop" in str(e):
                logger.warning("Event loop issue detected, will retry health check later")

    def load_datasets(self):
        """Load existing datasets"""
        try:
            if not os.path.exists(self.config.data_dir):
                os.makedirs(self.config.data_dir, exist_ok=True)
                return

            for item in os.listdir(self.config.data_dir):
                item_path = os.path.join(self.config.data_dir, item)
                if os.path.isdir(item_path):
                    metadata_path = os.path.join(item_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            dataset_info = json.load(f)
                            self.training_manager.datasets[item] = dataset_info
                            self.dataset_list.addItem(item)

        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")

    def create_dataset(self):
        """Create new dataset"""
        self._prompt_text("New Dataset", "Dataset Name:", self._on_dataset_name_entered)

    def _on_dataset_name_entered(self, name: str) -> None:
        """Continue the dataset wizard once the name is known."""
        name = (name or "").strip()
        if not name:
            return
        self._prompt_text(
            "New Dataset", "Description:",
            lambda description: self._on_dataset_description_entered(name, description),
        )

    def _on_dataset_description_entered(self, name: str, description: str) -> None:
        """Continue the dataset wizard once the description is known."""
        data_types = ["Images", "Audio", "Video", "Text", "Mixed"]
        self._prompt_item(
            "Data Type", "Select data type:", data_types,
            lambda data_type: self._on_dataset_type_selected(
                name, description or "", data_type
            ),
        )

    def _on_dataset_type_selected(self, name: str, description: str, data_type: str) -> None:
        """Create the dataset with the collected values."""
        success = self.training_manager.create_dataset(name, description, data_type.lower())
        if not success:
            self._notify("Error", "Failed to create dataset", level="warning")

    def delete_dataset(self):
        """Delete selected dataset"""
        current_item = self.dataset_list.currentItem()
        if not current_item:
            self._notify("No Selection", "Please select a dataset to delete", level="warning")
            return

        dataset_name = current_item.text()
        self._confirm(
            "Delete Dataset",
            f"Are you sure you want to delete dataset '{dataset_name}'?",
            lambda: self._do_delete_dataset(current_item, dataset_name),
        )

    def _do_delete_dataset(self, current_item, dataset_name: str) -> None:
        """Remove the confirmed dataset from disk and from the dataset list."""
        try:
            dataset_path = os.path.join(self.config.data_dir, dataset_name)
            shutil.rmtree(dataset_path)

            del self.training_manager.datasets[dataset_name]
            self.dataset_list.takeItem(self.dataset_list.row(current_item))

            self.dataset_info.clear()
            self.file_list.clear()

            self.status_bar.showMessage(f"Dataset '{dataset_name}' deleted")

        except Exception as e:
            self._notify("Error", f"Failed to delete dataset: {str(e)}", level="warning")

    def add_file_to_dataset(self):
        """Add file(s) or directory to selected dataset"""
        current_item = self.dataset_list.currentItem()
        if not current_item:
            self._notify("No Selection", "Please select a dataset first", level="warning")
            return

        dataset_name = current_item.text()

        # Create dialog for file/directory selection
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Files or Directory")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)  # Allow multiple files
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)  # Allow both files and directories
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)  # Use Qt dialog for better control

        # Add directory selection option
        select_dir_btn = QPushButton("Select Directory")
        select_dir_btn.clicked.connect(lambda: self._select_directory_for_dataset(dialog, dataset_name))
        dialog.layout().addWidget(select_dir_btn)

        # Show dialog and get selected files
        dialog.filesSelected.connect(
            lambda files: self._process_selected_files(dataset_name, files)
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _select_directory_for_dataset(self, dialog, dataset_name):
        """Handle directory selection"""
        def on_directory(directory):
            dialog.reject()
            if directory:
                self._process_selected_directory(dataset_name, directory)

        self._select_directory("Select Directory", on_directory)

    def _process_selected_files(self, dataset_name, file_paths):
        """Process multiple selected files"""
        added_count = 0
        failed_count = 0

        for file_path in file_paths:
            if os.path.isfile(file_path):
                # Calculate sizes for this file
                file_sizes = self.calculate_dataset_sizes(file_path, self.training_manager.datasets[dataset_name]['data_type'])
                success = self.training_manager.add_file_to_dataset(dataset_name, file_path, file_sizes)
                if success:
                    added_count += 1
                else:
                    failed_count += 1
                    logger.error(f"Failed to add file: {file_path}")

        # Show results
        if added_count > 0:
            self.status_bar.showMessage(f"Added {added_count} file(s) to {dataset_name}")
            if failed_count > 0:
                self._notify("Partial Success", f"Added {added_count} file(s), failed to add {failed_count} file(s)", level="warning")
        else:
            self._notify("Error", "Failed to add any files to dataset", level="warning")

    def _process_selected_directory(self, dataset_name, directory_path):
        """Process selected directory recursively"""
        try:
            added_count = 0
            failed_count = 0

            # Walk through directory recursively
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Calculate sizes for this file
                        file_sizes = self.calculate_dataset_sizes(file_path, self.training_manager.datasets[dataset_name]['data_type'])
                        success = self.training_manager.add_file_to_dataset(dataset_name, file_path, file_sizes)
                        if success:
                            added_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"Error adding file {file_path}: {str(e)}")
                        failed_count += 1

            # Show results
            if added_count > 0:
                self.status_bar.showMessage(f"Added {added_count} file(s) from directory to {dataset_name}")
                if failed_count > 0:
                    self._notify("Partial Success", f"Added {added_count} file(s), failed to add {failed_count} file(s)", level="warning")
            else:
                self._notify("Error", "No files were added from the selected directory", level="warning")

        except Exception as e:
            self._notify("Error", f"Failed to process directory: {str(e)}", level="warning")

    def remove_file_from_dataset(self):
        """Remove file from selected dataset"""
        current_dataset = self.dataset_list.currentItem()
        current_file = self.file_list.currentItem()

        if not current_dataset or not current_file:
            self._notify("No Selection", "Please select both dataset and file", level="warning")
            return

        dataset_name = current_dataset.text()
        file_name = current_file.text()

        self._confirm(
            "Remove File",
            f"Remove '{file_name}' from dataset '{dataset_name}'?",
            lambda: self._do_remove_file_from_dataset(current_file, dataset_name, file_name),
        )

    def _do_remove_file_from_dataset(self, current_file, dataset_name: str, file_name: str) -> None:
        """Remove the confirmed file from the dataset metadata and from disk."""
        try:
            # Remove from dataset metadata
            if dataset_name in self.training_manager.datasets:
                dataset = self.training_manager.datasets[dataset_name]
                dataset['files'] = [
                    f for f in dataset['files']
                    if f['name'] != file_name
                ]
                self.training_manager._save_dataset_metadata(dataset_name)

            # Remove physical file
            dataset_path = os.path.join(self.config.data_dir, dataset_name)
            file_path = os.path.join(dataset_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

            self.file_list.takeItem(self.file_list.row(current_file))
            self.status_bar.showMessage(f"Removed {file_name} from {dataset_name}")

        except Exception as e:
            self._notify("Error", f"Failed to remove file: {str(e)}", level="warning")

    def _on_file_learning_mode_changed(self, index):
        """Toggle between directory and file list views"""
        is_directory_mode = (index == 0)
        self.file_directory_widget.setVisible(is_directory_mode)
        self.file_list_widget.setVisible(not is_directory_mode)
        self.file_learning_list.setVisible(not is_directory_mode)

    def _browse_learning_directory(self):
        """Browse for a directory to learn from"""
        self._select_directory(
            "Select Directory to Learn From",
            lambda directory: self.file_directory_edit.setText(directory) if directory else None,
        )

    def _add_learning_files(self):
        """Add files to the learning list"""
        self._select_files(
            "Select Files to Learn From",
            self._on_learning_files_selected,
            file_filter="All Supported Files (*.txt *.json *.xml *.csv *.py *.js *.c *.cpp *.h *.java *.md);;All Files (*)",
        )

    def _on_learning_files_selected(self, files) -> None:
        """Add the picked files to the learning list."""
        if not files:
            return
        if not hasattr(self, '_learning_files'):
            self._learning_files = []
        for f in files:
            if f not in self._learning_files:
                self._learning_files.append(f)
                self.file_learning_list.addItem(f)

    def _remove_learning_files(self):
        """Remove selected files from the learning list"""
        selected_items = self.file_learning_list.selectedItems()
        if selected_items and hasattr(self, '_learning_files'):
            for item in selected_items:
                path = item.text()
                if path in self._learning_files:
                    self._learning_files.remove(path)
                self.file_learning_list.takeItem(self.file_learning_list.row(item))

    @qasync.asyncSlot()
    async def _learn_from_files(self):
        """Learn from selected files or directory"""
        if not self.client:
            self._notify("Error", "Client not connected. Please wait for initialization.", level="warning")
            return

        dataset_name = self.file_dataset_name.text().strip() or "file_learned_dataset"
        output_size = self.file_output_size.value()

        learning_mode = self.file_learning_mode_combo.currentIndex()

        try:
            if learning_mode == 0:
                directory = self.file_directory_edit.text().strip()
                if not directory:
                    self._notify("Error", "Please select a directory first.", level="warning")
                    return

                self.learn_files_btn.setEnabled(False)
                self.learn_files_btn.setText("Learning...")

                result = await self.client.learn_from_directory(
                    directory=directory,
                    dataset_name=dataset_name,
                    output_size=output_size
                )

                if result.success:
                    self._notify("Success", f"Successfully learned from directory:\n{directory}", level="info")
                else:
                    self._notify("Error", f"Failed to learn from directory: {result.message}", level="warning")

            else:
                if not hasattr(self, '_learning_files') or not self._learning_files:
                    self._notify("Error", "Please add files first.", level="warning")
                    return

                self.learn_files_btn.setEnabled(False)
                self.learn_files_btn.setText("Learning...")

                result = await self.client.learn_from_files(
                    files=self._learning_files,
                    dataset_name=dataset_name,
                    output_size=output_size
                )

                if result.success:
                    self._notify("Success", f"Successfully learned from {len(self._learning_files)} files", level="info")
                else:
                    self._notify("Error", f"Failed to learn from files: {result.message}", level="warning")

        except Exception as e:
            self._notify("Error", f"Error during learning: {str(e)}", level="warning")
        finally:
            self.learn_files_btn.setEnabled(True)
            self.learn_files_btn.setText("Learn from Files")

    @qasync.asyncSlot()
    async def run_test(self):
        """Run selected test with proper qasync async slot pattern"""
        # Prevent multiple test executions simultaneously
        if hasattr(self, '_test_running') and self._test_running:
            self._notify("Test Running", "A test is already running. Please wait for it to complete.", level="warning")
            return
            
        self._test_running = True
        
        try:
            if not self.client_initialized:
                if not self.initialize_client():
                    self._notify("Connection Error", "Failed to initialize client", level="warning")
                    return

            test_type = self.test_type_combo.currentText()

            try:
                # Show progress indicator for test execution
                self.status_bar.showMessage("Running test...")
                QApplication.processEvents()  # Allow UI to update

                if test_type == "API Connectivity":
                    result = await self.debugger.run_api_test("health_check")
                    self.add_test_result(result)

                elif test_type == "Module Functionality":
                    module_name = self.test_module_combo.currentText().split()[0].lower()
                    test_data = {"test": "functionality"}
                    
                    # Add specific test data for web module
                    if module_name == "web":
                        test_data["query"] = "test search query"
                    elif module_name == "vision":
                        test_data["image_path"] = "test_image.jpg"
                    elif module_name == "audio":
                        test_data["audio_path"] = "test_audio.wav"
                    elif module_name == "math":
                        test_data["expression"] = "2 + 2"

                    result = await self.debugger.run_module_test(module_name, test_data)
                    self.add_test_result(result)

                elif test_type == "System Automation":
                    task_type = self.automation_task_combo.currentText()
                    try:
                        parameters = json.loads(self.automation_params_text.toPlainText())
                    except json.JSONDecodeError:
                        parameters = {}
                    
                    result = await self.debugger.run_system_automation_test(task_type, parameters)
                    self.add_test_result(result)

                elif test_type == "Performance Test":
                    test_config = {
                        'test_type': self.performance_test_type_combo.currentText(),
                        'duration': self.performance_duration_spin.value(),
                        'iterations': self.performance_iterations_spin.value(),
                        'concurrency': self.performance_concurrency_spin.value()
                    }
                    
                    result = await self.debugger.run_performance_test(test_config)
                    self.add_test_result(result)

                elif test_type == "Load Test":
                    test_config = {
                        'test_type': self.load_test_type_combo.currentText(),
                        'duration': self.load_duration_spin.value(),
                        'concurrent_users': self.load_concurrent_users_spin.value(),
                        'requests_per_user': self.load_requests_per_user_spin.value(),
                        'endpoint': self.load_endpoint_combo.currentText()
                    }
                    
                    result = await self.debugger.run_load_test(test_config)
                    self.add_test_result(result)

                else:
                    self._notify("Test", f"Test type '{test_type}' not implemented yet", level="info")

                # Update status when test completes
                self.status_bar.showMessage("Test completed")

            except Exception as e:
                logger.error(f"Test execution error: {str(e)}")
                self._notify("Test Error", f"Test execution failed: {str(e)}", level="warning")
                self.status_bar.showMessage("Test failed")
                
        finally:
            self._test_running = False

    def add_test_result(self, result: TestResult):
        """Add test result to table"""
        row = self.test_results_table.rowCount()
        self.test_results_table.insertRow(row)

        self.test_results_table.setItem(row, 0, QTableWidgetItem(result.test_id))
        self.test_results_table.setItem(row, 1, QTableWidgetItem(result.test_type))
        self.test_results_table.setItem(row, 2, QTableWidgetItem("PASS" if result.success else "FAIL"))
        self.test_results_table.setItem(row, 3, QTableWidgetItem(f"{result.duration:.2f}s"))
        self.test_results_table.setItem(row, 4, QTableWidgetItem(result.message))

        # Color code the status
        status_item = self.test_results_table.item(row, 2)
        if result.success:
            status_item.setBackground(QColor(144, 238, 144))  # Light green
        else:
            status_item.setBackground(QColor(255, 182, 193))  # Light red

    def clear_test_results(self):
        """Clear test results table"""
        self.test_results_table.setRowCount(0)

    def start_sandbox(self):
        """Start Docker sandbox"""
        config = {
            'sandbox_name': self.sandbox_name_edit.text(),
            'startup_command': self.startup_command_edit.text(),
            'networking': self.networking_combo.currentText().lower(),
            'privileged': not self.drop_rights_check.isChecked()
        }

        success = self.sandbox_manager.start_sandbox(config)
        if not success:
            self._notify("Docker Error", "Failed to start Docker sandbox", level="warning")

    def stop_sandbox(self):
        """Stop Docker sandbox"""
        success = self.sandbox_manager.stop_sandbox()
        if not success:
            self._notify("Docker Error", "Failed to stop Docker sandbox", level="warning")

    def clear_sandbox_logs(self):
        """Clear sandbox logs"""
        self.sandbox_logs.clear()

    def update_system_info(self):
        """Update system information display"""
        try:
            info = []
            info.append(f"Operating System: {os.name} {sys.platform}")
            info.append(f"Python Version: {sys.version}")
            import PyQt6
            info.append(f"PyQt6 Version: {PyQt6.QtCore.PYQT_VERSION_STR}")
            info.append(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            info.append(f"Working Directory: {os.getcwd()}")
            info.append(f"User: {os.environ.get('USER', 'Unknown')}")

            self.system_info.setPlainText("\n".join(info))

        except Exception as e:
            self.system_info.setPlainText(f"Error getting system info: {str(e)}")

    def update_process_list(self):
        """Update process list"""
        try:
            self.process_table.setRowCount(0)

            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    row = self.process_table.rowCount()
                    self.process_table.insertRow(row)

                    self.process_table.setItem(row, 0, QTableWidgetItem(str(proc.info['pid'])))
                    self.process_table.setItem(row, 1, QTableWidgetItem(proc.info['name'] or ''))
                    self.process_table.setItem(row, 2, QTableWidgetItem(f"{proc.info['cpu_percent']:.1f}"))
                    self.process_table.setItem(row, 3, QTableWidgetItem(f"{proc.info['memory_percent']:.1f}"))

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Error updating process list: {str(e)}")

    @qasync.asyncSlot()
    async def run_api_debug(self):
        """Run comprehensive API debugging with detailed request/response logging"""
        if not self.client_initialized:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        try:
            # Clear previous results
            self.api_debug_output.clear()

            # Add header
            self.api_debug_output.appendPlainText("=" * 60)
            self.api_debug_output.appendPlainText("COREAI3D API DEBUG SESSION")
            self.api_debug_output.appendPlainText("=" * 60)
            self.api_debug_output.appendPlainText(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.api_debug_output.appendPlainText("")

            # Configuration info
            self.api_debug_output.appendPlainText("CONFIGURATION:")
            self.api_debug_output.appendPlainText(f"  API URL: {self.config.api_url}")
            self.api_debug_output.appendPlainText(f"  WebSocket URL: {self.config.ws_url}")
            self.api_debug_output.appendPlainText(f"  API Key: {'Configured' if self.config.api_key else 'Not configured'}")
            self.api_debug_output.appendPlainText("")

            # Network connectivity test
            self.api_debug_output.appendPlainText("NETWORK CONNECTIVITY TEST:")
            await self._test_api_connectivity()
            self.api_debug_output.appendPlainText("")

            # API endpoint tests
            self.api_debug_output.appendPlainText("API ENDPOINT TESTS:")
            endpoints = [
                ("Health Check", "health_check", {}),
                ("System Metrics", "system_metrics", {}),
                ("Module Status", "module_status", {})
            ]

            for display_name, endpoint, params in endpoints:
                await self._test_api_endpoint(display_name, endpoint, params)
                self.api_debug_output.appendPlainText("")

            # Summary
            self.api_debug_output.appendPlainText("=" * 60)
            self.api_debug_output.appendPlainText("DEBUG SESSION COMPLETE")
            self.api_debug_output.appendPlainText("=" * 60)

        except Exception as e:
            self.api_debug_output.appendPlainText(f"CRITICAL ERROR: {str(e)}")
            self._notify("Debug Error", f"API debug failed: {str(e)}", level="warning")

    async def _test_api_connectivity(self):
        """Test basic API connectivity"""
        try:
            import urllib.request
            import urllib.error
            import socket

            # Extract host and port
            from urllib.parse import urlparse
            parsed_url = urlparse(self.config.api_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            # Test TCP connectivity
            self.api_debug_output.appendPlainText(f"  Testing TCP connection to {host}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)

            try:
                result = sock.connect_ex((host, port))
                if result == 0:
                    self.api_debug_output.appendPlainText("  ✓ TCP connectivity: SUCCESS")
                else:
                    self.api_debug_output.appendPlainText(f"  ✗ TCP connectivity: FAILED (error code: {result})")
            except Exception as e:
                self.api_debug_output.appendPlainText(f"  ✗ TCP connectivity: ERROR - {str(e)}")
            finally:
                sock.close()

            # Test HTTP connectivity
            self.api_debug_output.appendPlainText(f"  Testing HTTP GET to {self.config.api_url}...")
            try:
                req = urllib.request.Request(self.config.api_url)
                req.add_header('User-Agent', 'CoreAI3D-Dashboard/1.3.0')

                with urllib.request.urlopen(req, timeout=15) as response:
                    self.api_debug_output.appendPlainText("  ✓ HTTP connectivity: SUCCESS")
                    self.api_debug_output.appendPlainText(f"    Status Code: {response.getcode()}")
                    self.api_debug_output.appendPlainText(f"    Content-Type: {response.headers.get('content-type', 'Unknown')}")

                    # Read response preview
                    content = response.read(512).decode('utf-8', errors='ignore')
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        self.api_debug_output.appendPlainText(f"    Response Preview: {preview}")

            except urllib.error.HTTPError as e:
                self.api_debug_output.appendPlainText("  ✗ HTTP connectivity: HTTP ERROR")
                self.api_debug_output.appendPlainText(f"    Status Code: {e.code}")
                self.api_debug_output.appendPlainText(f"    Reason: {e.reason}")
            except urllib.error.URLError as e:
                self.api_debug_output.appendPlainText("  ✗ HTTP connectivity: URL ERROR")
                self.api_debug_output.appendPlainText(f"    Reason: {e.reason}")
            except Exception as e:
                self.api_debug_output.appendPlainText("  ✗ HTTP connectivity: ERROR")
                self.api_debug_output.appendPlainText(f"    Error: {str(e)}")

        except Exception as e:
            self.api_debug_output.appendPlainText(f"  ✗ Connectivity test error: {str(e)}")

    async def _test_api_endpoint(self, display_name: str, endpoint: str, params: Dict[str, Any]):
        """Test specific API endpoint with detailed logging"""
        try:
            self.api_debug_output.appendPlainText(f"  Testing {display_name}...")

            # Record start time
            start_time = time.time()

            # Make API call
            if endpoint == "health_check":
                result = await self.client.health_check()
            elif endpoint == "system_metrics":
                result = await self.client.get_system_metrics()
            elif endpoint == "module_status":
                result = await self.client.get_module_status()
            else:
                result = type('Result', (), {'success': False, 'data': {'error': f'Unknown endpoint: {endpoint}'}})()

            # Calculate response time
            response_time = time.time() - start_time

            # Log results
            if result.success:
                self.api_debug_output.appendPlainText("  ✓ SUCCESS")
                self.api_debug_output.appendPlainText(f"    Response Time: {response_time:.3f}s")

                # Log response data (truncated for readability)
                if hasattr(result, 'data') and result.data:
                    data_str = json.dumps(result.data, indent=2)
                    if len(data_str) > 500:
                        data_str = data_str[:500] + "\n    ... (truncated)"
                    self.api_debug_output.appendPlainText("    Response Data:")
                    for line in data_str.split('\n'):
                        self.api_debug_output.appendPlainText(f"      {line}")
            else:
                self.api_debug_output.appendPlainText("  ✗ FAILED")
                self.api_debug_output.appendPlainText(f"    Response Time: {response_time:.3f}s")

                # Log error details
                if hasattr(result, 'data') and result.data:
                    error_info = result.data.get('error', 'Unknown error')
                    self.api_debug_output.appendPlainText(f"    Error: {error_info}")

                    # Additional error details
                    if isinstance(result.data, dict):
                        for key, value in result.data.items():
                            if key != 'error':
                                self.api_debug_output.appendPlainText(f"    {key}: {value}")

        except Exception as e:
            self.api_debug_output.appendPlainText("  ✗ EXCEPTION")
            self.api_debug_output.appendPlainText(f"    Error: {str(e)}")
            import traceback
            self.api_debug_output.appendPlainText("    Traceback:")
            for line in traceback.format_exc().split('\n')[-5:]:  # Last 5 lines of traceback
                if line.strip():
                    self.api_debug_output.appendPlainText(f"      {line}")

    def clear_api_debug_results(self):
        """Clear API debug results"""
        self.api_debug_output.clear()

    def export_api_debug_log(self):
        """Export API debug log to file"""
        try:
            # Get current debug output
            debug_content = self.api_debug_output.toPlainText()
            if not debug_content:
                self._notify("No Data", "No debug data to export", level="warning")
                return

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"api_debug_log_{timestamp}.txt"

            # Get save location
            self._select_save_file(
                "Export API Debug Log",
                lambda file_path: self._write_debug_log(file_path, debug_content),
                default_name=default_filename,
                file_filter="Text files (*.txt);;All files (*.*)",
            )

        except Exception as e:
            self._notify("Export Error", f"Failed to export debug log: {str(e)}", level="warning")

    def _write_debug_log(self, file_path: str, debug_content: str) -> None:
        """Write the collected API debug output to `file_path`."""
        if not file_path:
            return
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(debug_content)

            self._notify("Export Complete", f"API debug log exported to:\n{file_path}", level="info")
            self.status_bar.showMessage(f"Debug log exported to {file_path}")
        except Exception as e:
            self._notify("Export Error", f"Failed to export debug log: {str(e)}", level="warning")

    def calculate_health_score(self, cpu_percent, memory_percent, disk_percent):
        """Calculate overall system health score (0-100)"""
        try:
            # Weight the components (CPU: 40%, Memory: 40%, Disk: 20%)
            cpu_score = max(0, 100 - cpu_percent * 1.25)  # Penalize heavily above 80%
            memory_score = max(0, 100 - memory_percent * 1.25)
            disk_score = max(0, 100 - disk_percent * 2)  # Disk space is critical

            overall_score = (cpu_score * 0.4) + (memory_score * 0.4) + (disk_score * 0.2)
            return int(overall_score)
        except Exception:
            return 50  # Default neutral score
    def _collect_hardware_info(self):
        """Collect comprehensive hardware information"""
        try:
            self.sandbox_logs.appendPlainText("HARDWARE INFORMATION:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # CPU Information
            cpu_count = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()

            self.sandbox_logs.appendPlainText(f"CPU Cores: {cpu_physical} physical, {cpu_count} logical")
            if cpu_freq:
                self.sandbox_logs.appendPlainText(f"CPU Frequency: {cpu_freq.current:.1f} MHz (max: {cpu_freq.max:.1f} MHz)")

            # CPU Usage per core
            cpu_percents = psutil.cpu_percent(percpu=True)
            self.sandbox_logs.appendPlainText(f"CPU Usage per Core: {', '.join([f'{p:.1f}%' for p in cpu_percents])}")

            # Memory Information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            self.sandbox_logs.appendPlainText(f"Total Memory: {memory.total / (1024**3):.2f} GB")
            self.sandbox_logs.appendPlainText(f"Available Memory: {memory.available / (1024**3):.2f} GB ({memory.percent:.1f}%)")
            self.sandbox_logs.appendPlainText(f"Used Memory: {memory.used / (1024**3):.2f} GB")
            self.sandbox_logs.appendPlainText(f"Swap Total: {swap.total / (1024**3):.2f} GB")
            self.sandbox_logs.appendPlainText(f"Swap Used: {swap.used / (1024**3):.2f} GB ({swap.percent:.1f}%)")

            # Disk Information
            disk_partitions = psutil.disk_partitions()
            self.sandbox_logs.appendPlainText(f"Disk Partitions: {len(disk_partitions)}")

            for partition in disk_partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.sandbox_logs.appendPlainText(f"  {partition.device} ({partition.mountpoint}): {usage.total / (1024**3):.1f} GB total, {usage.free / (1024**3):.1f} GB free ({usage.percent:.1f}%)")
                except PermissionError:
                    self.sandbox_logs.appendPlainText(f"  {partition.device} ({partition.mountpoint}): Access denied")

            # Battery Information (if available)
            try:
                battery = psutil.sensors_battery()
                if battery:
                    self.sandbox_logs.appendPlainText(f"Battery: {battery.percent:.1f}% ({'charging' if battery.power_plugged else 'discharging'})")
                    if battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                        hours, remainder = divmod(battery.secsleft, 3600)
                        minutes, _ = divmod(remainder, 60)
                        self.sandbox_logs.appendPlainText(f"Battery Time Left: {hours}h {minutes}m")
            except Exception:
                pass  # Battery info not available on all systems

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting hardware info: {str(e)}")

    def _collect_software_info(self):
        """Collect comprehensive software information"""
        try:
            self.sandbox_logs.appendPlainText("SOFTWARE INFORMATION:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # OS Information
            import platform
            self.sandbox_logs.appendPlainText(f"Operating System: {platform.system()} {platform.release()}")
            self.sandbox_logs.appendPlainText(f"Platform: {platform.platform()}")
            self.sandbox_logs.appendPlainText(f"Architecture: {platform.architecture()[0]}")
            self.sandbox_logs.appendPlainText(f"Python Version: {platform.python_version()}")

            # System uptime
            uptime = time.time() - psutil.boot_time()
            uptime_str = str(datetime.timedelta(seconds=int(uptime)))
            self.sandbox_logs.appendPlainText(f"System Uptime: {uptime_str}")

            # User information
            self.sandbox_logs.appendPlainText(f"Current User: {os.environ.get('USER', 'Unknown')}")
            self.sandbox_logs.appendPlainText(f"Home Directory: {os.path.expanduser('~')}")

            # Environment variables (key ones)
            env_vars = ['PATH', 'SHELL', 'TERM', 'LANG', 'DISPLAY']
            self.sandbox_logs.appendPlainText("Environment Variables:")
            for var in env_vars:
                value = os.environ.get(var, 'Not set')
                if len(value) > 50:
                    value = value[:47] + "..."
                self.sandbox_logs.appendPlainText(f"  {var}: {value}")

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting software info: {str(e)}")

    def _collect_system_health(self):
        """Collect system health indicators"""
        try:
            self.sandbox_logs.appendPlainText("SYSTEM HEALTH CHECKS:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # CPU Health
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = "GOOD" if cpu_percent < 80 else "WARNING" if cpu_percent < 95 else "CRITICAL"
            self.sandbox_logs.appendPlainText(f"CPU Health: {cpu_status} ({cpu_percent:.1f}% usage)")

            # Memory Health
            memory = psutil.virtual_memory()
            memory_status = "GOOD" if memory.percent < 80 else "WARNING" if memory.percent < 95 else "CRITICAL"
            self.sandbox_logs.appendPlainText(f"Memory Health: {memory_status} ({memory.percent:.1f}% used)")

            # Disk Health
            disk_usage = psutil.disk_usage('/')
            disk_status = "GOOD" if disk_usage.percent < 80 else "WARNING" if disk_usage.percent < 95 else "CRITICAL"
            self.sandbox_logs.appendPlainText(f"Disk Health: {disk_status} ({disk_usage.percent:.1f}% used)")

            # Overall Health Score
            health_score = self.calculate_health_score(cpu_percent, memory.percent, disk_usage.percent)
            health_status = "EXCELLENT" if health_score >= 90 else "GOOD" if health_score >= 70 else "WARNING" if health_score >= 50 else "CRITICAL"
            self.sandbox_logs.appendPlainText(f"Overall Health Score: {health_score}/100 ({health_status})")

            # Load Average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.sandbox_logs.appendPlainText(f"Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
            except AttributeError:
                pass  # Not available on Windows

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting system health: {str(e)}")

    def _collect_performance_metrics(self):
        """Collect detailed performance metrics"""
        try:
            self.sandbox_logs.appendPlainText("PERFORMANCE METRICS:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # CPU Metrics
            cpu_times = psutil.cpu_times()
            self.sandbox_logs.appendPlainText("CPU Time Breakdown:")
            self.sandbox_logs.appendPlainText(f"  User: {cpu_times.user:.1f}s")
            self.sandbox_logs.appendPlainText(f"  System: {cpu_times.system:.1f}s")
            self.sandbox_logs.appendPlainText(f"  Idle: {cpu_times.idle:.1f}s")

            # Memory Details
            memory = psutil.virtual_memory()
            self.sandbox_logs.appendPlainText("Memory Details:")
            self.sandbox_logs.appendPlainText(f"  Active: {memory.active / (1024**3):.2f} GB")
            self.sandbox_logs.appendPlainText(f"  Inactive: {memory.inactive / (1024**3):.2f} GB")
            self.sandbox_logs.appendPlainText(f"  Buffers: {memory.buffers / (1024**3):.2f} GB")
            self.sandbox_logs.appendPlainText(f"  Cached: {memory.cached / (1024**3):.2f} GB")

            # I/O Statistics
            try:
                io_counters = psutil.disk_io_counters()
                if io_counters:
                    self.sandbox_logs.appendPlainText("Disk I/O Statistics:")
                    self.sandbox_logs.appendPlainText(f"  Read Count: {io_counters.read_count}")
                    self.sandbox_logs.appendPlainText(f"  Write Count: {io_counters.write_count}")
                    self.sandbox_logs.appendPlainText(f"  Read Bytes: {io_counters.read_bytes / (1024**3):.2f} GB")
                    self.sandbox_logs.appendPlainText(f"  Write Bytes: {io_counters.write_bytes / (1024**3):.2f} GB")
            except Exception:
                pass

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting performance metrics: {str(e)}")

    def _collect_network_diagnostics(self):
        """Collect network diagnostics"""
        try:
            self.sandbox_logs.appendPlainText("NETWORK DIAGNOSTICS:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # Network interfaces
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()

            self.sandbox_logs.appendPlainText(f"Network Interfaces: {len(net_if_addrs)}")

            for interface, addrs in net_if_addrs.items():
                self.sandbox_logs.appendPlainText(f"  {interface}:")
                for addr in addrs:
                    if addr.family.name == 'AF_INET':
                        self.sandbox_logs.appendPlainText(f"    IPv4: {addr.address}/{addr.netmask}")
                    elif addr.family.name == 'AF_INET6':
                        self.sandbox_logs.appendPlainText(f"    IPv6: {addr.address}")
                    elif addr.family.name == 'AF_PACKET':
                        self.sandbox_logs.appendPlainText(f"    MAC: {addr.address}")

                # Interface stats
                if interface in net_if_stats:
                    stats = net_if_stats[interface]
                    self.sandbox_logs.appendPlainText(f"    Status: {'Up' if stats.isup else 'Down'}")
                    self.sandbox_logs.appendPlainText(f"    Speed: {stats.speed} Mbps")

            # Network I/O
            net_io = psutil.net_io_counters()
            self.sandbox_logs.appendPlainText("Network I/O:")
            self.sandbox_logs.appendPlainText(f"  Bytes Sent: {net_io.bytes_sent / (1024**2):.2f} MB")
            self.sandbox_logs.appendPlainText(f"  Bytes Received: {net_io.bytes_recv / (1024**2):.2f} MB")
            self.sandbox_logs.appendPlainText(f"  Packets Sent: {net_io.packets_sent}")
            self.sandbox_logs.appendPlainText(f"  Packets Received: {net_io.packets_recv}")

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting network diagnostics: {str(e)}")

    def _collect_process_analysis(self):
        """Collect process analysis and monitoring"""
        try:
            self.sandbox_logs.appendPlainText("PROCESS ANALYSIS:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # Process counts
            process_count = len(psutil.pids())
            self.sandbox_logs.appendPlainText(f"Total Processes: {process_count}")

            # Top CPU processes
            self.sandbox_logs.appendPlainText("Top CPU Processes:")
            try:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append((proc.info['cpu_percent'], proc.info['name'], proc.info['pid']))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Sort by CPU usage and show top 5
                processes.sort(reverse=True)
                for i, (cpu, name, pid) in enumerate(processes[:5]):
                    self.sandbox_logs.appendPlainText(f"  {i+1}. {name} (PID: {pid}): {cpu:.1f}% CPU")
            except Exception:
                self.sandbox_logs.appendPlainText("  Unable to retrieve process information")

            # Top memory processes
            self.sandbox_logs.appendPlainText("Top Memory Processes:")
            try:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append((proc.info['memory_percent'], proc.info['name'], proc.info['pid']))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Sort by memory usage and show top 5
                processes.sort(reverse=True)
                for i, (mem, name, pid) in enumerate(processes[:5]):
                    self.sandbox_logs.appendPlainText(f"  {i+1}. {name} (PID: {pid}): {mem:.1f}% Memory")
            except Exception:
                self.sandbox_logs.appendPlainText("  Unable to retrieve process information")

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting process analysis: {str(e)}")

    def _collect_disk_health(self):
        """Collect disk health information"""
        try:
            self.sandbox_logs.appendPlainText("DISK HEALTH CHECKS:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # Disk partitions and usage
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.sandbox_logs.appendPlainText(f"Partition: {partition.device} ({partition.mountpoint})")
                    self.sandbox_logs.appendPlainText(f"  File System: {partition.fstype}")
                    self.sandbox_logs.appendPlainText(f"  Total: {usage.total / (1024**3):.2f} GB")
                    self.sandbox_logs.appendPlainText(f"  Used: {usage.used / (1024**3):.2f} GB")
                    self.sandbox_logs.appendPlainText(f"  Free: {usage.free / (1024**3):.2f} GB")
                    self.sandbox_logs.appendPlainText(f"  Usage: {usage.percent:.1f}%")

                    # Health assessment
                    if usage.percent > 90:
                        self.sandbox_logs.appendPlainText("  Status: CRITICAL - Disk space very low")
                    elif usage.percent > 80:
                        self.sandbox_logs.appendPlainText("  Status: WARNING - Disk space low")
                    else:
                        self.sandbox_logs.appendPlainText("  Status: GOOD")

                except PermissionError:
                    self.sandbox_logs.appendPlainText(f"Partition: {partition.device} ({partition.mountpoint}) - Access denied")
                except Exception as e:
                    self.sandbox_logs.appendPlainText(f"Partition: {partition.device} ({partition.mountpoint}) - Error: {str(e)}")

            # Disk I/O statistics
            try:
                io_stats = psutil.disk_io_counters(perdisk=True)
                if io_stats:
                    self.sandbox_logs.appendPlainText("Disk I/O Statistics:")
                    for disk, stats in io_stats.items():
                        self.sandbox_logs.appendPlainText(f"  {disk}:")
                        self.sandbox_logs.appendPlainText(f"    Read: {stats.read_count} operations, {stats.read_bytes / (1024**2):.2f} MB")
                        self.sandbox_logs.appendPlainText(f"    Write: {stats.write_count} operations, {stats.write_bytes / (1024**2):.2f} MB")
            except Exception:
                pass

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting disk health: {str(e)}")

    def _collect_system_configuration(self):
        """Collect system configuration information"""
        try:
            self.sandbox_logs.appendPlainText("SYSTEM CONFIGURATION:")
            self.sandbox_logs.appendPlainText("-" * 30)

            # Kernel information
            try:
                import platform
                self.sandbox_logs.appendPlainText(f"Kernel: {platform.release()}")
                self.sandbox_logs.appendPlainText(f"Architecture: {platform.machine()}")
            except Exception:
                pass

            # System limits (if available)
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                self.sandbox_logs.appendPlainText(f"File Descriptor Limit: {soft} (soft), {hard} (hard)")
            except Exception:
                pass

            # Environment information
            self.sandbox_logs.appendPlainText(f"Current Working Directory: {os.getcwd()}")
            self.sandbox_logs.appendPlainText(f"Python Executable: {sys.executable}")
            self.sandbox_logs.appendPlainText(f"Python Path: {sys.path[0] if sys.path else 'Not available'}")

            # Timezone information
            import time
            local_tz = time.tzname[0] if time.tzname else "Unknown"
            self.sandbox_logs.appendPlainText(f"Timezone: {local_tz}")

            # Locale information
            try:
                import locale
                current_locale = locale.getlocale()
                self.sandbox_logs.appendPlainText(f"Locale: {current_locale[0] if current_locale[0] else 'Unknown'}")
            except Exception:
                pass

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error collecting system configuration: {str(e)}")

    def _generate_debug_summary(self):
        """Generate a summary and recommendations based on debug data"""
        try:
            self.sandbox_logs.appendPlainText("DEBUG SUMMARY & RECOMMENDATIONS:")
            self.sandbox_logs.appendPlainText("-" * 40)

            # Get current metrics for summary
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Performance assessment
            issues = []
            recommendations = []

            if cpu_percent > 80:
                issues.append("High CPU usage")
                recommendations.append("Consider closing unnecessary applications or processes")
            if memory.percent > 80:
                issues.append("High memory usage")
                recommendations.append("Close memory-intensive applications or add more RAM")
            if disk.percent > 90:
                issues.append("Low disk space")
                recommendations.append("Free up disk space by deleting unnecessary files")

            # Process count assessment
            process_count = len(psutil.pids())
            if process_count > 500:
                issues.append("High process count")
                recommendations.append("Review running processes and terminate unnecessary ones")

            # Network assessment
            try:
                net_io = psutil.net_io_counters()
                if net_io.errin > 0 or net_io.errout > 0:
                    issues.append("Network errors detected")
                    recommendations.append("Check network configuration and connections")
            except Exception:
                pass

            # Generate summary
            if issues:
                self.sandbox_logs.appendPlainText("ISSUES DETECTED:")
                for issue in issues:
                    self.sandbox_logs.appendPlainText(f"  • {issue}")

                self.sandbox_logs.appendPlainText("")
                self.sandbox_logs.appendPlainText("RECOMMENDATIONS:")
                for rec in recommendations:
                    self.sandbox_logs.appendPlainText(f"  • {rec}")
            else:
                self.sandbox_logs.appendPlainText("✓ No critical issues detected")
                self.sandbox_logs.appendPlainText("  System appears to be running normally")

            # Overall status
            health_score = self.calculate_health_score(cpu_percent, memory.percent, disk.percent)
            if health_score >= 80:
                self.sandbox_logs.appendPlainText(f"\nOVERALL STATUS: HEALTHY (Score: {health_score}/100)")
            elif health_score >= 60:
                self.sandbox_logs.appendPlainText(f"\nOVERALL STATUS: WARNING (Score: {health_score}/100)")
            else:
                self.sandbox_logs.appendPlainText(f"\nOVERALL STATUS: CRITICAL (Score: {health_score}/100)")

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"Error generating debug summary: {str(e)}")
    def run_system_debug(self):
        """Run comprehensive system debugging with detailed diagnostics"""
        try:
            # Clear previous results
            self.sandbox_logs.clear()

            # Add header
            self.sandbox_logs.appendPlainText("=" * 60)
            self.sandbox_logs.appendPlainText("COMPREHENSIVE SYSTEM DEBUG SESSION")
            self.sandbox_logs.appendPlainText("=" * 60)
            self.sandbox_logs.appendPlainText(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.sandbox_logs.appendPlainText("")

            # Collect comprehensive system diagnostics
            self._collect_hardware_info()
            self.sandbox_logs.appendPlainText("")

            self._collect_software_info()
            self.sandbox_logs.appendPlainText("")

            self._collect_system_health()
            self.sandbox_logs.appendPlainText("")

            self._collect_performance_metrics()
            self.sandbox_logs.appendPlainText("")

            self._collect_network_diagnostics()
            self.sandbox_logs.appendPlainText("")

            self._collect_process_analysis()
            self.sandbox_logs.appendPlainText("")

            self._collect_disk_health()
            self.sandbox_logs.appendPlainText("")

            self._collect_system_configuration()
            self.sandbox_logs.appendPlainText("")

            # Summary and recommendations
            self._generate_debug_summary()

            # Footer
            self.sandbox_logs.appendPlainText("=" * 60)
            self.sandbox_logs.appendPlainText("SYSTEM DEBUG SESSION COMPLETE")
            self.sandbox_logs.appendPlainText("=" * 60)

        except Exception as e:
            self.sandbox_logs.appendPlainText(f"CRITICAL ERROR: {str(e)}")
            self._notify("Debug Error", f"System debug failed: {str(e)}", level="warning")

    def browse_data_directory(self):
        """Browse for data directory"""
        self._select_directory(
            "Select Data Directory",
            lambda directory: self.data_dir_edit.setText(directory) if directory else None,
        )

    def browse_data_file(self):
        """Browse for data file"""
        self._select_file(
            "Select Data File",
            lambda file_path: self.data_file_edit.setText(file_path) if file_path else None,
            file_filter="All Files (*.*);;CSV Files (*.csv);;JSON Files (*.json);;Text Files (*.txt)",
        )

    def save_settings(self):
        """Save settings to configuration"""
        try:
            # Update config from UI
            self.config.api_url = self.api_url_edit.text()
            self.config.ws_url = self.ws_url_edit.text()
            self.config.api_key = self.api_key_edit.text()
            self.config.data_dir = self.data_dir_edit.text()
            self.config.sandbox_timeout = self.sandbox_timeout_edit.value()
            self.config.log_level = self.log_level_combo.currentText()
            self.config.enable_diagnostics = self.diagnostics_check.isChecked()
            self.config.connection_timeout = float(self.connection_timeout_edit.value())
            self.config.chat_timeout = int(self.chat_timeout_edit.value())
            self.config.resend_api_key = self.resend_api_key_edit.text().strip()
            self.config.resend_from_email = self.resend_from_email_edit.text().strip()

            # Save to settings file
            self._save_config()

            # Reinitialize client whenever a URL is configured. An API key is NOT
            # required — the client explicitly supports an empty key (offline mode),
            # and _load_config() force-resets api_key to '' on startup. The previous
            # `and self.config.api_key` gate silently kept the client pointed at the
            # stale URL whenever the user saved a new URL without a key.
            if self.config.api_url:
                self.initialize_client()

            self.status_bar.showMessage("Settings saved successfully")
            self._notify("Settings", "Settings saved successfully")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            self._notify("Save Error", f"Failed to save settings: {str(e)}",
                         level="critical")

    def test_settings_connection(self):
        """Test the API connection using the values currently shown in the Settings tab.

        Reads the URL / WebSocket URL / API Key directly from the form widgets
        (self.api_url_edit, self.ws_url_edit, self.api_key_edit) rather than from
        self.config or self.client, so the result reflects exactly what the user
        sees on screen — independent of whether Save was clicked, whether the
        main client has finished reinitializing, or whether an API key is set.

        Performs two checks:
          1. HTTP GET against the API URL (with Bearer token if a key is entered).
          2. TCP connect against the WebSocket host:port (if a WS URL is entered).

        The probes run on a background thread so the GUI never freezes while
        the connection times out; the result is reported back on the GUI
        thread.
        """
        # Read directly from the form widgets, NOT from self.config, so the test
        # reflects the user's current input even before Save.
        url = self.api_url_edit.text().strip()
        ws_url = self.ws_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()

        if not url:
            self._notify(
                "Test Connection",
                "Please enter an API URL in the Settings tab before testing.",
                level="warning"
            )
            return

        # Use a generous but bounded timeout so the UI cannot hang indefinitely.
        try:
            timeout = int(self.config.connection_timeout)
        except (TypeError, ValueError):
            timeout = 10
        if timeout <= 0:
            timeout = 10
        timeout = min(timeout, 30)

        self.test_settings_conn_btn.setEnabled(False)
        self.status_bar.showMessage(f"Testing connection to {url}...")

        def worker():
            try:
                result = self._probe_connection(url, ws_url, api_key, timeout)
            except Exception as e:
                result = {
                    'api_ok': False,
                    'ws_ok': None,
                    'url': url,
                    'ws_url': ws_url,
                    'details': [f"  Error: {e}"],
                    'ws_summary': '',
                }
            self._run_on_gui_thread(self._present_connection_result, result)

        threading.Thread(target=worker, daemon=True).start()

    def _probe_connection(self, url: str, ws_url: str, api_key: str, timeout: int) -> Dict[str, Any]:
        """Probe the API and WebSocket endpoints. Runs off the GUI thread."""
        import socket
        details = []
        api_ok = False

        # --- 1. HTTP probe of the API URL ---
        try:
            import urllib.error
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'CoreAI3D-Dashboard/1.3.0')
            if api_key:
                req.add_header('Authorization', f'Bearer {api_key}')

            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                content_type = response.headers.get('content-type', 'Unknown')
                details.append(f"  HTTP Status: {status_code}")
                details.append(f"  Content-Type: {content_type}")
                # Any 2xx/3xx response means the server is reachable and routed
                # the request — 401/403 still prove the host is up and the URL
                # resolves, so treat 200-399 as success and only flag 4xx/5xx.
                if 200 <= status_code < 400:
                    api_ok = True
                else:
                    details.append(f"  Unexpected status: {status_code}")
        except urllib.error.HTTPError as e:
            # 4xx/5xx still proves the host is reachable — surface as failure
            # with details rather than a generic network error.
            details.append(f"  HTTP {e.code} {e.reason} (host is reachable)")
        except urllib.error.URLError as e:
            details.append(f"  URL Error: {e.reason}")
        except socket.timeout:
            details.append(f"  Timeout: no response within {timeout}s")
        except Exception as e:
            details.append(f"  Error: {e}")

        # --- 2. Optional TCP probe of the WebSocket host:port ---
        ws_ok = None
        ws_summary = ""
        if ws_url:
            try:
                parsed = urllib.parse.urlparse(ws_url)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == 'wss' else 80)
                if not host:
                    raise ValueError(f"could not parse host from WebSocket URL: {ws_url}")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                try:
                    sock.connect((host, port))
                    ws_ok = True
                    details.append(f"  WebSocket TCP: reachable at {host}:{port}")
                except Exception as e:
                    ws_ok = False
                    details.append(f"  WebSocket TCP: unreachable at {host}:{port} ({e})")
                finally:
                    sock.close()
                ws_summary = f"WebSocket URL: {ws_url}\n"
            except Exception as e:
                ws_ok = False
                details.append(f"  WebSocket URL error: {e}")
                ws_summary = f"WebSocket URL: {ws_url}\n"

        return {
            'api_ok': api_ok,
            'ws_ok': ws_ok,
            'url': url,
            'ws_url': ws_url,
            'details': details,
            'ws_summary': ws_summary,
        }

    def _present_connection_result(self, result: Dict[str, Any]) -> None:
        """Show the outcome of a connection probe (runs on the GUI thread)."""
        try:
            self.test_settings_conn_btn.setEnabled(True)
        except RuntimeError:
            pass

        api_ok = result.get('api_ok', False)
        ws_ok = result.get('ws_ok')
        url = result.get('url', '')
        details = result.get('details', [])
        ws_summary = result.get('ws_summary', '')

        # Build the result dialog. The HTTP test is the primary signal; the
        # WebSocket probe is a secondary reachability hint and never blocks
        # overall success on its own.
        overall_ok = api_ok and (ws_ok is not False)
        if overall_ok:
            title = "Connection Test Successful"
            body_lines = [
                "API connection is reachable.",
                "",
                f"API URL: {url}",
            ]
            if ws_summary:
                body_lines.append(ws_summary.rstrip())
            body_lines.append("")
            body_lines.append("Details:")
            body_lines.extend(details)
            self.status_bar.showMessage("Connection test successful")
            self._notify(title, "\n".join(body_lines))
        else:
            title = "Connection Test Failed"
            body_lines = [
                "Could not reach the API server.",
                "",
                f"API URL: {url}",
            ]
            if ws_summary:
                body_lines.append(ws_summary.rstrip())
            body_lines += [
                "",
                "Please verify:",
                "  - The CoreAI3D server is running",
                "  - The host is reachable (set COREAI3D_API_URL, e.g. http://127.0.0.1:1906/api/v1)",
                "  - The port is correct and reachable",
                "  - The API key is correct (if required)",
                "",
                "Details:",
            ]
            body_lines.extend(details)
            self.status_bar.showMessage("Connection test failed")
            self._notify(title, "\n".join(body_lines), level="warning")

    def on_generate_api_key_clicked(self):
        """Generate a new API key via the backend and populate the API Key field.

        Performs the full login -> generate flow:
          1. POST /auth/login  {"username":"admin","password":"admin123"}  -> session_id
          2. POST /auth/apikey {"session_id":..., "description":...}      -> api_key

        Uses a dedicated QThread + ApiKeyWorker with its own asyncio event
        loop, completely independent of qasync. This avoids the
        "Event loop is closed" error and ensures the UI never blocks.
        """
        if not self.client:
            if not self.initialize_client():
                self._notify(
                    "Connection Error",
                    "Failed to initialize the backend client. Is the CoreAI3D "
                    "server running?",
                    level="warning"
                )
                return

        # Prevent double-click while a generation is in progress
        if getattr(self, '_api_key_generating', False):
            self.status_bar.showMessage("API key generation already in progress", 3000)
            return

        self._api_key_generating = True
        self.generate_api_key_btn.setEnabled(False)
        self.status_bar.showMessage("Logging in...")

        # Clean up any previous worker/thread references
        if getattr(self, '_api_key_worker_thread', None) is not None:
            try:
                if self._api_key_worker_thread.isRunning():
                    self._api_key_worker_thread.quit()
                    if not self._api_key_worker_thread.wait(2000):
                        _LINGERING_THREADS.append(self._api_key_worker_thread)
            except RuntimeError:
                pass
            self._api_key_worker_thread = None
            self._api_key_worker = None

        self._api_key_worker_thread = QThread()

        async def login_and_generate():
            """Coroutine: login then generate API key. Runs in worker thread.

            A dedicated client is used because the shared client owns an
            aiohttp session that is bound to the qasync loop on the GUI
            thread and cannot be reused from the worker's event loop.
            """
            client = self._create_worker_client()
            try:
                # Step 1: Login to obtain a session_id
                username = self.config.admin_email or "admin"
                password = self.config.admin_password or "admin123"
                login_result = await client.login(username, password)
                if not login_result.success:
                    return login_result  # Propagate the error response

                # Step 2: Generate the API key using the obtained session_id
                gen_result = await client.generate_api_key(
                    description="Generated from Linux GUI Dashboard"
                )
                return gen_result
            finally:
                try:
                    await client.close()
                except Exception as e:
                    logger.debug(f"Error closing API key client: {e}")

        self._api_key_worker = ApiKeyWorker(login_and_generate)
        self._api_key_worker.moveToThread(self._api_key_worker_thread)

        def on_success(response):
            self._api_key_generating = False
            self.generate_api_key_btn.setEnabled(True)
            self.status_bar.showMessage("Ready", 3000)

            # Clean up thread
            try:
                self._api_key_worker_thread.quit()
                if not self._api_key_worker_thread.wait(2000):
                    _LINGERING_THREADS.append(self._api_key_worker_thread)
            except RuntimeError:
                pass
            self._api_key_worker_thread = None
            self._api_key_worker = None

            if not response.success:
                self._notify(
                    "Generate API Key",
                    f"Failed to generate API key: {self._api_error_text(response)}",
                    level="warning"
                )
                return

            # Robust key extraction — checks many locations in the response
            new_key = CoreAI3DClient.parse_api_key(response)

            if not new_key:
                self._notify(
                    "Generate API Key",
                    "API key was generated but not found in the server response.\n\n"
                    f"Raw response:\n{json.dumps(response.data, indent=2, default=str)}",
                    level="warning"
                )
                return

            self.api_key_edit.setText(new_key)
            self.config.api_key = new_key
            self.status_bar.showMessage("API key generated successfully", 3000)

            # Show the key ONLY ONCE in a dialog with Copy + warning.
            # open() keeps the dialog window-modal without blocking the event
            # loop, so the GUI stays responsive.
            dialog = ApiKeyDialog(new_key, self)
            dialog.open()

        def on_error(err):
            self._api_key_generating = False
            self.generate_api_key_btn.setEnabled(True)
            self.status_bar.showMessage("Ready", 3000)

            try:
                self._api_key_worker_thread.quit()
                if not self._api_key_worker_thread.wait(2000):
                    _LINGERING_THREADS.append(self._api_key_worker_thread)
            except RuntimeError:
                pass
            self._api_key_worker_thread = None
            self._api_key_worker = None

            # Never show an empty error dialog — always include the real message
            error_text = str(err) if err else "Unknown error generating API key"
            self._notify("Generate API Key", f"Error: {error_text}", level="critical")
            logger.error(f"Generate API key error: {error_text}", exc_info=True)

        self._api_key_worker.finished.connect(on_success)
        self._api_key_worker.error.connect(on_error)
        self._api_key_worker_thread.started.connect(self._api_key_worker.run)
        self._api_key_worker.finished.connect(self._api_key_worker_thread.quit)
        self._api_key_worker.error.connect(self._api_key_worker_thread.quit)
        self._api_key_worker_thread.finished.connect(
            lambda: setattr(self, '_api_key_worker_thread', None)
        )
        self._api_key_worker_thread.start()

    def _copy_api_key_from_settings(self):
        """Copy the API key from the settings field to the clipboard."""
        key = self.api_key_edit.text().strip()
        if not key:
            self._notify("Copy API Key", "No API key to copy.", level="info")
            return
        cb = QApplication.clipboard()
        cb.setText(key)
        self.status_bar.showMessage("API key copied to clipboard", 3000)

    def refresh_all(self):
        """Refresh all data and displays"""
        self.update_system_metrics()
        self.update_system_info()
        self.update_process_list()
        self.load_datasets()
        self.status_bar.showMessage("All data refreshed")

    def show_documentation(self):
        """Show application documentation"""
        self._notify("Documentation", "For detailed documentation, please see:\n\n"
            "README.md\n\n"
            "This file contains comprehensive usage instructions, "
            "configuration details, and troubleshooting information.", level="info")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About CoreAI3D Dashboard",
            "CoreAI3D Linux GUI Dashboard\n\n"
            "Version 1.3.0\n\n"
            "A comprehensive desktop application for managing AI training data, "
            "testing, debugging, and safely testing OS control operations "
            "using Linux containers.\n\n"
            "Built with PyQt6 and CoreAI3D Python Client"
        )

    def send_chat_message(self):
        """Send chat message to AI"""
        if not self.initialize_client():
            self._notify(
                "Connection Error",
                "Client not connected. Please wait for initialization or check server status.",
                level="warning"
            )
            return

        message = self.chat_input.text().strip()
        if not message:
            return

        # Add user message to history
        self.chat_history.append(f"You: {message}")
        self.chat_input.clear()

        # Show thinking indicator
        self.thinking_label.setText("CoreAI3D is thinking...")
        self.thinking_label.setVisible(True)
        if hasattr(self, 'chat_cancel_btn'):
            self.chat_cancel_btn.setEnabled(True)

        # Add debug logging
        logger.debug(f"Sending chat message: {message}")
        logger.debug(f"Client connected: {self.client.is_connected() if self.client else False}")

        try:
            # Send message directly with qasync
            task = asyncio.create_task(self._send_chat_message_async(message))
            self._chat_task = task
            task.add_done_callback(self._handle_chat_message_result)

        except Exception as e:
            logger.error(f"Exception in send_chat_message: {str(e)}")
            self._notify("Error", f"Failed to send message: {str(e)}", level="critical")

    def cancel_chat_message(self):
        """Abort the in-flight chat request (HTTP fallback path)."""
        task = getattr(self, '_chat_task', None)
        if task is None or task.done():
            self._notify("Nothing to Cancel", "No chat request is in flight.",
                         level="warning")
            return
        try:
            task.cancel()
        except Exception as e:
            logger.error(f"Failed to cancel chat request: {e}")
        self._chat_task = None
        self.thinking_label.setVisible(False)
        if hasattr(self, 'chat_cancel_btn'):
            self.chat_cancel_btn.setEnabled(False)
        self.status_bar.showMessage("Chat request cancelled", 3000)

    async def _send_chat_message_async(self, message: str):
        """Async part of chat message sending.

        The HTTP fallback path is bounded by ``config.chat_timeout`` so a
        stalled backend cannot leave the GUI waiting forever.  The WebSocket
        delivery path is unaffected.
        """
        async with self.global_async_lock:
            logger.debug("Making HTTP request to send chat message")
            timeout = self._chat_timeout()
            try:
                result = await asyncio.wait_for(
                    self.client.send_chat_message(message), timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Chat request timed out after {timeout}s")
                return APIResponse(
                    success=False,
                    data=None,
                    message=f"Chat request timed out after {timeout:.0f}s",
                    status_code=408,
                    metadata={}
                )
            logger.debug(f"HTTP response received: success={result.success}, status={result.status_code}")
            if result.data:
                logger.debug(f"Response data: {result.data}")
            return result

    def _chat_timeout(self) -> float:
        """Timeout applied to the HTTP chat fallback path."""
        try:
            timeout = float(self.config.chat_timeout)
        except (TypeError, ValueError):
            timeout = 30.0
        return max(1.0, timeout)

    def _handle_chat_message_result(self, task):
        """Handle chat message result"""
        try:
            QTimer.singleShot(0, lambda: self.thinking_label.setVisible(False))
            if hasattr(self, 'chat_cancel_btn'):
                self.chat_cancel_btn.setEnabled(False)
            if task.cancelled():
                logger.debug("Chat request was cancelled")
                return
            result = task.result()
            if result.success and result.data:
                inner_data = result.data.get('data', result.data) if isinstance(result.data, dict) else result.data
                web_searched = False
                search_query = ""
                if isinstance(inner_data, dict):
                    response = inner_data.get('response') or inner_data.get('content') or inner_data.get('message', '')
                    web_searched = inner_data.get('web_searched', False)
                    search_query = inner_data.get('search_query', '')
                else:
                    response = ''
                if response:
                    import hashlib
                    key = (response, hashlib.md5(response.encode('utf-8', errors='ignore')).hexdigest())
                    already_shown = key in self._shown_chat_response_keys
                    if not already_shown:
                        self._shown_chat_response_keys.add(key)
                        QTimer.singleShot(0, lambda r=response, ws=web_searched, sq=search_query: self._add_response_with_typing_effect(r, web_searched_indicator=ws, search_query=sq))
                    logger.info(f"HTTP chat response handled (shown={not already_shown}): {response[:80]}")
            elif not result.success:
                error_msg = result.message or (result.data.get('error') if isinstance(result.data, dict) else None) or 'Unknown error'
                logger.error(f"HTTP request failed: {error_msg}")
                QTimer.singleShot(0, lambda e=error_msg: self.chat_history.append(f"Error: {e}"))
        except asyncio.CancelledError:
            logger.debug("Chat request cancelled")
        except Exception as e:
            logger.error(f"Exception in chat message handler: {str(e)}")
            QTimer.singleShot(0, lambda err=str(e): self.chat_history.append(f"Error: {err}"))

    def _handle_chat_response(self, data: Dict[str, Any]):
        """Handle chat response from WebSocket"""
        logger.debug(f"Handling chat response: {data}")
        content = (data.get('content') or data.get('message') or
                   data.get('response') or data.get('reply') or '')
        if content:
            import hashlib
            with self._chat_response_lock:
                key = (content, hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest())
                if key not in self._shown_chat_response_keys:
                    self._shown_chat_response_keys.add(key)
                    if not hasattr(self, '_web_search_info_by_hash'):
                        self._web_search_info_by_hash = {}
                    self._web_search_info_by_hash[key] = {
                        'web_searched': data.get('web_searched', False),
                        'search_query': data.get('search_query', ''),
                    }
                    logger.info(f"Emitting chat response signal (new): {content[:80]}")
                    self.chat_response_received.emit(content)
                else:
                    logger.info(f"Chat response already shown via HTTP, skipping WS duplicate: {content[:80]}")
        else:
            logger.warning("Received chat response with empty content")

    def _on_chat_response_received(self, response: str):
        """Handle chat response signal and update UI"""
        QTimer.singleShot(0, lambda: self.thinking_label.setVisible(False))
        logger.debug(f"Chat response received via signal: {response}")
        import hashlib
        with self._chat_response_lock:
            key = (response, hashlib.md5(response.encode('utf-8', errors='ignore')).hexdigest())
            if key in self._shown_chat_response_keys:
                logger.debug(f"Signal-path chat response already shown, skipping: {response[:80]}")
                return
            self._shown_chat_response_keys.add(key)
        web_searched = False
        search_query = ""
        if hasattr(self, '_web_search_info_by_hash') and key in self._web_search_info_by_hash:
            info = self._web_search_info_by_hash.pop(key)
            web_searched = info.get('web_searched', False)
            search_query = info.get('search_query', '')
        QTimer.singleShot(0, lambda r=response, ws=web_searched, sq=search_query: self._add_response_with_typing_effect(r, web_searched_indicator=ws, search_query=sq))

    def _add_response_with_typing_effect(self, text: str, web_searched_indicator: bool = False, search_query: str = ""):
        """Add response text with a typing animation effect"""
        if hasattr(self, '_typing_timer') and self._typing_timer is not None:
            try:
                self._typing_timer.stop()
            except RuntimeError:
                pass
            self._typing_timer.deleteLater()
            self._typing_timer = None
        if web_searched_indicator:
            search_label = f"🔍 Searched the web for: \"{search_query}\"" if search_query else "🔍 Searched the web"
            self.chat_history.append("")
            cursor = self.chat_history.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.chat_history.setTextCursor(cursor)
            self.chat_history.insertPlainText(search_label + "\n")
        self._typing_text = text
        self._typing_index = 0
        self.chat_history.append("AI: ")
        self._typing_timer = self._create_gui_timer(self)
        self._typing_timer.setInterval(20)
        def type_next_chunk():
            if self._typing_index >= len(self._typing_text):
                self._typing_timer.stop()
                return
            chunk_size = min(3, len(self._typing_text) - self._typing_index)
            piece = self._typing_text[self._typing_index:self._typing_index + chunk_size]
            self._typing_index += chunk_size
            cursor = self.chat_history.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.chat_history.setTextCursor(cursor)
            self.chat_history.insertPlainText(piece)
            scrollbar = self.chat_history.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        self._typing_timer.timeout.connect(type_next_chunk)
        self._typing_timer.start()

    def _handle_chat_error(self, error_msg: str):
        """Handle chat error"""
        logger.error(f"Chat error: {error_msg}")
        QTimer.singleShot(0, lambda e=error_msg: self.chat_history.append(f"Error: {e}"))

    def _handle_chat_error_ws(self, error_msg):
        """Handle chat error from WebSocket 'error' event"""
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message') or error_msg.get('error') or str(error_msg)
        self._handle_chat_error(str(error_msg))

    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history.clear()

    @qasync.asyncSlot()
    async def discover_models_async(self):
        """Async method to discover models from external platforms with proper locking"""
        task_id = f"discovery_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Check if discovery already in progress (double-check with lock)
                if self.discovery_in_progress:
                    logger.info("Model discovery already in progress")
                    return

                # Register task
                with self.task_lock:
                    if task_id in self.active_tasks:
                        logger.warning(f"Discovery task {task_id} already active")
                        return
                    self.active_tasks.add(task_id)

                # Track the current task for cancellation
                self.current_async_task = asyncio.current_task()
                async with self.task_cleanup_lock:
                    self.pending_tasks.add(self.current_async_task)

                self.discovery_in_progress = True
                self.download_status_label.setText("Discovering models...")
                self.download_progress.setVisible(True)
                self.download_progress.setRange(0, 0)  # Indeterminate progress

                # Get current search parameters
                query = self.model_search_input.text().strip()
                model_type = self.model_type_combo.currentText().lower()
                if model_type == "all types":
                    model_type = "all"

                platform = self.platform_combo.currentText().lower()
                if platform == "all platforms":
                    platform = "all"

                # Discover models
                result = await self.model_discovery_service.discover_models(
                    platform=platform,
                    query=query,
                    model_type=model_type,
                    page=self.current_page,
                    page_size=self.page_size
                )

                # Update UI with results
                self._update_model_table(result)

            except asyncio.CancelledError:
                logger.info("Model discovery cancelled")
                self.download_status_label.setText("Discovery cancelled")
                raise  # Re-raise to ensure proper cleanup
            except Exception as e:
                logger.error(f"Error discovering models: {str(e)}")
                self.discovery_error.emit("Discovery Error", f"Failed to discover models: {str(e)}")
            finally:
                self.discovery_in_progress = False
                self.download_progress.setVisible(False)
                if not self.download_status_label.text().startswith("Discovery cancelled"):
                    self.download_status_label.setText("Ready")

                # Clean up task
                with self.task_lock:
                    self.active_tasks.discard(task_id)

                # Remove from pending tasks
                async with self.task_cleanup_lock:
                    self.pending_tasks.discard(self.current_async_task)
                    self.current_async_task = None

    def _update_model_table(self, discovery_result: ModelDiscoveryResult):
        """Update the model table with discovery results"""
        try:
            self.current_discovery_result = discovery_result

            # Clear existing table
            self.model_table.setRowCount(0)

            # Update pagination info
            self.page_label.setText(f"Page {discovery_result.current_page} of {discovery_result.total_pages}")
            self.prev_page_btn.setEnabled(discovery_result.has_previous)
            self.next_page_btn.setEnabled(discovery_result.has_next)

            # Populate table with models
            self.model_table.setRowCount(len(discovery_result.models))

            for row, model in enumerate(discovery_result.models):
                self.model_table.setItem(row, 0, QTableWidgetItem(model.name))
                self.model_table.setItem(row, 1, QTableWidgetItem(model.type))
                self.model_table.setItem(row, 2, QTableWidgetItem(model.size))
                self.model_table.setItem(row, 3, QTableWidgetItem(model.version))
                self.model_table.setItem(row, 4, QTableWidgetItem(model.status))
                self.model_table.setItem(row, 5, QTableWidgetItem(model.platform))
                self.model_table.setItem(row, 6, QTableWidgetItem(str(model.downloads)))
                self.model_table.setItem(row, 7, QTableWidgetItem(model.last_updated))

                # Color code status
                status_item = self.model_table.item(row, 4)
                if model.status == "Available":
                    status_item.setBackground(QColor(144, 238, 144))  # Light green
                elif model.status == "Downloaded":
                    status_item.setBackground(QColor(173, 216, 230))  # Light blue
                elif model.status == "Downloading":
                    status_item.setBackground(QColor(255, 255, 224))  # Light yellow
                elif model.status == "Error":
                    status_item.setBackground(QColor(255, 182, 193))  # Light red

            # Update status
            self.status_bar.showMessage(f"Found {discovery_result.total_count} models ({len(discovery_result.models)} shown)")

        except Exception as e:
            logger.error(f"Error updating model table: {str(e)}")

    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 1:
            self.current_page -= 1
            self.discover_models_async()

    def next_page(self):
        """Go to next page"""
        if self.current_discovery_result and self.current_discovery_result.has_next:
            self.current_page += 1
            self.discover_models_async()

    def change_page_size(self):
        """Change page size and refresh"""
        try:
            self.page_size = int(self.page_size_combo.currentText())
            self.current_page = 1  # Reset to first page
            self.discover_models_async()
        except ValueError:
            logger.error("Invalid page size")

    def load_sample_models(self):
        """Load sample models as fallback when API discovery fails"""
        try:
            # Vision Models - for image processing, classification, and analysis
            vision_models = [
                {"name": "ResNet-50", "type": "Vision", "size": "98 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "YOLOv5-Small", "type": "Vision", "size": "14 MB", "version": "6.1.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "EfficientNet-B0", "type": "Vision", "size": "20 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "MobileNetV3-Small", "type": "Vision", "size": "5.4 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "FaceNet", "type": "Vision", "size": "23 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "Tesseract-OCR", "type": "Vision", "size": "8.2 MB", "version": "5.3.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "Mask-RCNN", "type": "Vision", "size": "170 MB", "version": "2.1.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "SSD-MobileNet", "type": "Vision", "size": "24 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"}
            ]

            # Audio Models - for speech processing and audio analysis
            audio_models = [
                {"name": "Whisper-Tiny", "type": "Audio", "size": "39 MB", "version": "1.2.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "Whisper-Base", "type": "Audio", "size": "74 MB", "version": "1.2.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "Whisper-Small", "type": "Audio", "size": "244 MB", "version": "1.2.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "Tacotron2", "type": "Audio", "size": "113 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "WaveNet", "type": "Audio", "size": "4.7 GB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "DeepSpeech", "type": "Audio", "size": "190 MB", "version": "0.9.3", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "SpeakerNet", "type": "Audio", "size": "16 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "AudioNet", "type": "Audio", "size": "8.5 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"}
            ]

            # Text/Language Models - for natural language processing
            text_models = [
                {"name": "BERT-Base", "type": "Text", "size": "420 MB", "version": "2.1.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "GPT-2 Small", "type": "Text", "size": "548 MB", "version": "1.5.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "DistilBERT", "type": "Text", "size": "66 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "RoBERTa-Base", "type": "Text", "size": "476 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "ALBERT-Base", "type": "Text", "size": "43 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "XLNet-Base", "type": "Text", "size": "446 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "ELECTRA-Small", "type": "Text", "size": "13 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"},
                {"name": "Sentence-BERT", "type": "Text", "size": "218 MB", "version": "1.0.0", "status": "Available", "platform": "Local", "downloads": 0, "last_updated": "2024-01-01"}
            ]

            # Combine all models
            all_models = vision_models + audio_models + text_models

            # Create ModelInfo objects
            model_infos = []
            for model in all_models:
                model_info = ModelInfo(
                    name=model["name"],
                    type=model["type"],
                    size=model["size"],
                    version=model["version"],
                    status=model["status"],
                    platform=model["platform"],
                    downloads=model["downloads"],
                    last_updated=model["last_updated"]
                )
                model_infos.append(model_info)

            # Create discovery result
            discovery_result = ModelDiscoveryResult(
                models=model_infos,
                total_count=len(model_infos),
                current_page=1,
                total_pages=1,
                has_next=False,
                has_previous=False,
                platform="local"
            )

            # Update table
            self._update_model_table(discovery_result)

        except Exception as e:
            logger.error(f"Error loading sample models: {str(e)}")

    def search_models(self):
        """Search models based on input criteria"""
        try:
            search_text = self.model_search_input.text().lower()
            model_type = self.model_type_combo.currentText()

            for row in range(self.model_table.rowCount()):
                model_name = self.model_table.item(row, 0).text().lower()
                model_type_cell = self.model_table.item(row, 1).text()

                # Check search text match
                text_match = search_text in model_name if search_text else True

                # Check type filter
                type_match = model_type == "All Types" or model_type == model_type_cell

                # Show/hide row based on filters
                self.model_table.setRowHidden(row, not (text_match and type_match))

            self.status_bar.showMessage(f"Filtered models by '{search_text}' and type '{model_type}'")
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")

    def refresh_models(self):
        """Refresh the model list"""
        try:
            # In a real implementation, this would fetch from the API
            self.load_sample_models()
            self.status_bar.showMessage("Model list refreshed")
        except Exception as e:
            logger.error(f"Error refreshing models: {str(e)}")
            self._notify("Refresh Error", f"Failed to refresh models: {str(e)}", level="warning")

    def on_model_selected(self):
        """Handle model selection"""
        try:
            current_row = self.model_table.currentRow()
            if current_row >= 0:
                model_name = self.model_table.item(current_row, 0).text()
                model_type = self.model_table.item(current_row, 1).text()
                model_size = self.model_table.item(current_row, 2).text()
                model_version = self.model_table.item(current_row, 3).text()
                model_status = self.model_table.item(current_row, 4).text()

                info_text = f"""Model: {model_name}
Type: {model_type}
Size: {model_size}
Version: {model_version}
Status: {model_status}

Description: This is a sample {model_type.lower()} model for demonstration purposes.
In a real implementation, this would contain detailed information about the model's
capabilities, training data, performance metrics, and usage instructions."""

                self.model_info.setPlainText(info_text)
        except Exception as e:
            logger.error(f"Error handling model selection: {str(e)}")

    def download_selected_model(self):
        """Download the selected model"""
        logger.info("Starting model download process")
        current_row = self.model_table.currentRow()
        if current_row < 0:
            self._notify("No Selection", "Please select a model to download", level="warning")
            return

        model_name = self.model_table.item(current_row, 0).text()
        model_type = self.model_table.item(current_row, 1).text()
        model_status = self.model_table.item(current_row, 4).text()

        logger.info(f"Selected model: {model_name}, type: {model_type}, status: {model_status}")

        if model_status == "Downloading":
            self._notify("Already Downloading", f"{model_name} is already being downloaded", level="info")
            return

        # Determine download source based on model type or user selection
        selected_source = self.download_source_combo.currentText().lower()
        if selected_source == "auto":
            if "Kaggle" in model_name or self._is_kaggle_model(model_name):
                download_source = "kaggle"
            elif "HuggingFace" in model_name or "BERT" in model_name or "GPT" in model_name or "RoBERTa" in model_name:
                download_source = "huggingface"
            else:
                # Default to local download
                download_source = "local"
        else:
            download_source = selected_source

        logger.info(f"Download source determined: {download_source}")

        # Update status to downloading
        status_item = QTableWidgetItem("Downloading")
        status_item.setBackground(QColor(255, 255, 224))  # Light yellow
        self.model_table.setItem(current_row, 4, status_item)

        # Show progress
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status_label.setText(f"Downloading {model_name}...")

        logger.info(f"Progress bar initialized for {model_name}")

        # Start actual download based on source
        if download_source == "kaggle":
            self._download_from_kaggle(current_row, model_name)
        elif download_source == "huggingface":
            self._download_from_huggingface(current_row, model_name)
        else:
            # Simulate download progress for local models
            logger.info(f"Starting local download simulation for {model_name}")
            self._simulate_download_progress(current_row, model_name)

    def _simulate_download_progress(self, row, model_name):
        """Simulate download progress for local models with proper progress tracking"""
        try:
            logger.info(f"Starting download simulation for {model_name}")

            # Initialize progress tracking
            self.download_progress_data = {
                'row': row,
                'model_name': model_name,
                'progress': 0,
                'start_time': time.time(),
                'estimated_total_time': 5.0,  # 5 seconds for simulation
                'last_update': time.time()
            }

            # Create timer for progress updates
            self.download_timer = self._create_gui_timer()
            self.download_timer.timeout.connect(self._update_simulated_progress)
            self.download_timer.start(200)  # Update every 200ms for smooth progress

            logger.debug("Download simulation timer started")
        except Exception as e:
            logger.error(f"Error starting download simulation: {str(e)}")

    def _download_with_progress(self, row, model_name, source_url, dest_path):
        """Download file with actual progress tracking"""
        try:
            import urllib.request
            import threading

            def download_worker():
                try:
                    # Initialize progress tracking
                    self.download_progress_data = {
                        'row': row,
                        'model_name': model_name,
                        'progress': 0,
                        'start_time': time.time(),
                        'total_size': 0,
                        'downloaded_size': 0,
                        'dest_path': dest_path
                    }

                    # Create request
                    req = urllib.request.Request(
                        source_url,
                        headers={'User-Agent': 'CoreAI3D-Dashboard/1.0'}
                    )

                    # Get file size first
                    try:
                        with urllib.request.urlopen(req) as response:
                            self.download_progress_data['total_size'] = int(response.headers.get('Content-Length', 0))
                    except Exception as e:
                        logger.warning(f"Could not get file size: {str(e)}")

                    # Start download with progress tracking
                    with urllib.request.urlopen(req) as response:
                        total_size = self.download_progress_data['total_size']

                        with open(dest_path, 'wb') as f:
                            downloaded = 0
                            chunk_size = 8192

                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break

                                f.write(chunk)
                                downloaded += len(chunk)
                                self.download_progress_data['downloaded_size'] = downloaded

                                # Update progress
                                if total_size > 0:
                                    progress = min(95, int((downloaded / total_size) * 100))
                                else:
                                    # Estimate based on time for unknown size
                                    elapsed = time.time() - self.download_progress_data['start_time']
                                    progress = min(90, int((elapsed / 30.0) * 100))  # Assume 30 seconds

                                # Update UI in main thread
                                self._run_on_gui_thread(
                                    self._update_download_progress, row, model_name, progress
                                )

                    # Complete download
                    self._run_on_gui_thread(self._update_download_complete, row, model_name)

                except Exception as e:
                    self._run_on_gui_thread(self._download_error, row, model_name, str(e))
                    # Clean up progress data on error
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

            # Start download in background thread
            threading.Thread(target=download_worker, daemon=True).start()

        except Exception as e:
            self._download_error(row, model_name, f"Failed to start download: {str(e)}")

    def _update_simulated_progress(self):
        """Update simulated download progress"""
        try:
            if not hasattr(self, 'download_progress_data'):
                return

            data = self.download_progress_data
            current_time = time.time()
            elapsed = current_time - data['start_time']

            # Calculate progress based on elapsed time (non-linear for realism)
            if elapsed < data['estimated_total_time']:
                # Use sigmoid-like progression for more realistic download curve
                progress_ratio = elapsed / data['estimated_total_time']
                # Apply sigmoid function: smooth start, faster middle, smooth end
                import math
                sigmoid_progress = 1 / (1 + math.exp(-6 * (progress_ratio - 0.5)))
                new_progress = min(95, int(sigmoid_progress * 100))
            else:
                new_progress = 100

            # Only update if progress actually changed
            if new_progress != data['progress']:
                data['progress'] = new_progress
                data['last_update'] = current_time

                # Update UI
                self.download_progress.setValue(new_progress)
                self.download_status_label.setText(f"Downloading {data['model_name']}... {new_progress}%")

                logger.debug(f"Simulated progress updated to {new_progress}%")

                # Check if download is complete
                if new_progress >= 100:
                    logger.info(f"Simulated download completed for {data['model_name']}")
                    if hasattr(self, 'download_timer') and self.download_timer.isActive():
                        self.download_timer.stop()
                        logger.debug("Download simulation timer stopped")

                    # Clean up progress data
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

                    # Update status in main thread
                    QTimer.singleShot(0, lambda: self._update_download_complete(data['row'], data['model_name']))

        except Exception as e:
            logger.error(f"Error updating simulated progress: {str(e)}")
            # Stop timer on error
            if hasattr(self, 'download_timer') and self.download_timer.isActive():
                self.download_timer.stop()

    def _update_download_complete(self, row, model_name):
        """Update UI when download is complete"""
        try:
            logger.info(f"Download completion UI update for {model_name}")

            # Update status
            status_item = QTableWidgetItem("Downloaded")
            status_item.setBackground(QColor(173, 216, 230))  # Light blue
            self.model_table.setItem(row, 4, status_item)

            # Hide progress bar
            self.download_progress.setVisible(False)
            self.download_status_label.setText(f"{model_name} downloaded successfully")
            self.status_bar.showMessage(f"Model {model_name} downloaded successfully")

            # Log success
            self._log_download(f"✓ Successfully downloaded {model_name}")

            # Reset progress for next download
            self.download_progress.setValue(0)
            logger.debug("Download completion UI update completed")
        except Exception as e:
            logger.error(f"Error in download completion handler: {str(e)}")

    def cancel_model_download(self):
        """Cancel the current model download"""
        logger.info("Cancelling model download")
        current_row = self.model_table.currentRow()
        if current_row < 0:
            self._notify("No Selection", "Please select a downloading model to cancel", level="warning")
            return

        model_name = self.model_table.item(current_row, 0).text()
        model_status = self.model_table.item(current_row, 4).text()

        logger.info(f"Cancelling download for {model_name}, status: {model_status}")

        if model_status != "Downloading":
            self._notify("Not Downloading", f"{model_name} is not currently downloading", level="warning")
            return

        # Stop download timer if running
        if hasattr(self, 'download_timer') and self.download_timer.isActive():
            logger.debug("Stopping download timer")
            self.download_timer.stop()

        # Cancel async downloads if any
        if hasattr(self, 'current_download_task'):
            # In a real implementation, this would cancel the async task
            logger.debug("Cancelling async download task")
            pass

        # Reset status in main thread
        QTimer.singleShot(0, lambda: self._reset_download_status(current_row, model_name))

    def _reset_download_status(self, row, model_name):
        """Reset download status in main thread"""
        try:
            logger.info(f"Resetting download status for {model_name}")

            # Reset status
            status_item = QTableWidgetItem("Available")
            status_item.setBackground(QColor(144, 238, 144))  # Light green
            self.model_table.setItem(row, 4, status_item)
            self.download_progress.setVisible(False)
            self.download_status_label.setText("Download cancelled")
            self.status_bar.showMessage(f"Download of {model_name} cancelled")

            logger.debug("Download status reset completed")
        except Exception as e:
            logger.error(f"Error resetting download status: {str(e)}")

    def _is_kaggle_model(self, model_name):
        """Check if model is from Kaggle"""
        # Simple heuristic - in real implementation, this would check a database
        kaggle_models = [
            "ResNet-50", "YOLOv5", "EfficientNet", "MobileNetV3",
            "FaceNet", "Mask-RCNN", "SSD-MobileNet"
        ]
        return any(kaggle_model in model_name for kaggle_model in kaggle_models)

    def _download_from_kaggle(self, row, model_name):
        """Download model from Kaggle with proper progress tracking"""
        try:
            import subprocess
            import threading

            # Check if kaggle CLI is installed
            try:
                subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self._download_error(row, model_name, "Kaggle CLI not installed. Please install with: pip install kaggle")
                return

            # Map model names to Kaggle dataset paths
            kaggle_datasets = {
                "ResNet-50": "pytorch/resnet50",
                "YOLOv5-Small": "ultralytics/yolov5",
                "EfficientNet-B0": "pytorch/vision",
                "MobileNetV3-Small": "pytorch/vision",
                "FaceNet": "timesler/facenet-pytorch",
                "Tesseract-OCR": "tesseract-ocr/tesseract",
                "Mask-RCNN": "matterport/mask-rcnn",
                "SSD-MobileNet": "tensorflow/models"
            }

            dataset_path = kaggle_datasets.get(model_name, model_name.lower().replace('-', '').replace('_', ''))

            def download_worker():
                try:
                    # Create models directory if it doesn't exist
                    models_dir = os.path.join(self.config.data_dir, "models")
                    os.makedirs(models_dir, exist_ok=True)

                    # Initialize progress tracking
                    self.download_progress_data = {
                        'row': row,
                        'model_name': model_name,
                        'progress': 0,
                        'start_time': time.time(),
                        'download_path': os.path.join(models_dir, f"{dataset_path.split('/')[-1]}.zip"),
                        'total_size': None,
                        'downloaded_size': 0
                    }

                    # Download command
                    cmd = ['kaggle', 'datasets', 'download', dataset_path, '-p', models_dir]

                    # Run download with progress tracking
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    # Monitor download progress
                    progress = 0
                    last_size = 0

                    while process.poll() is None:
                        time.sleep(0.5)  # Check every 0.5 seconds

                        # Check if download file exists and track its growth
                        download_path = self.download_progress_data['download_path']
                        if os.path.exists(download_path):
                            current_size = os.path.getsize(download_path)

                            # Update downloaded size
                            self.download_progress_data['downloaded_size'] = current_size

                            # Estimate total size if not known
                            if self.download_progress_data['total_size'] is None and current_size > last_size:
                                # Rough estimation based on growth rate
                                self.download_progress_data['total_size'] = current_size * 8  # Conservative estimate

                            total_size = self.download_progress_data['total_size']

                            if total_size and total_size > 0:
                                progress = min(95, int((current_size / total_size) * 100))
                            else:
                                # Time-based progress when size unknown
                                elapsed = time.time() - self.download_progress_data['start_time']
                                progress = min(90, int((elapsed / 30.0) * 100))  # Assume 30 seconds for download

                            last_size = current_size

                            # Update progress in main thread
                            self._run_on_gui_thread(
                                self._update_kaggle_progress, row, model_name, progress
                            )
                        else:
                            # File not created yet, show initial progress
                            elapsed = time.time() - self.download_progress_data['start_time']
                            if elapsed < 5:  # First 5 seconds
                                progress = min(10, int((elapsed / 5.0) * 10))
                                self._run_on_gui_thread(
                                    self._update_kaggle_progress, row, model_name, progress
                                )

                    # Check final result
                    if process.returncode == 0:
                        # Complete the progress
                        self._run_on_gui_thread(
                            self._update_kaggle_progress, row, model_name, 100
                        )
                        self._run_on_gui_thread(self._download_success, row, model_name)
                    else:
                        stderr = process.stderr.read() if process.stderr else ""
                        error_msg = f"Kaggle download failed: {stderr}"
                        self._run_on_gui_thread(
                            self._download_error, row, model_name, error_msg
                        )

                    # Clean up progress data
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

                except Exception as e:
                    self._run_on_gui_thread(self._download_error, row, model_name, str(e))
                    # Clean up progress data on error
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

            # Start download in background thread
            threading.Thread(target=download_worker, daemon=True).start()

        except Exception as e:
            self._download_error(row, model_name, f"Failed to start Kaggle download: {str(e)}")

    def _download_from_huggingface(self, row, model_name):
        """Download model from Hugging Face with proper progress tracking"""
        try:
            from huggingface_hub import snapshot_download, HfApi
            import threading

            # Map model names to HuggingFace model IDs
            hf_models = {
                "BERT-Base": "bert-base-uncased",
                "GPT-2 Small": "gpt2",
                "DistilBERT": "distilbert-base-uncased",
                "RoBERTa-Base": "roberta-base",
                "ALBERT-Base": "albert-base-v2",
                "XLNet-Base": "xlnet-base-cased",
                "ELECTRA-Small": "google/electra-small-discriminator",
                "Sentence-BERT": "sentence-transformers/all-MiniLM-L6-v2",
                "CLIP-ViT": "openai/clip-vit-base-patch32",
                "CLIP-RN50": "openai/clip-vit-base-patch32",
                "VisualBERT": "uclanlp/visualbert-vqa-coco-pre",
                "ViLBERT": "uclanlp/visualbert-vqa",
                "LXMERT": "unc-nlp/lxmert-base-uncased",
                "Oscar": "microsoft/oscar-base-vocab"
            }

            model_id = hf_models.get(model_name, model_name.lower().replace(' ', '-'))

            def download_worker():
                try:
                    # Create models directory
                    models_dir = os.path.join(self.config.data_dir, "models", model_name.replace(' ', '_'))
                    os.makedirs(models_dir, exist_ok=True)

                    # Initialize progress tracking
                    self.download_progress_data = {
                        'row': row,
                        'model_name': model_name,
                        'progress': 0,
                        'start_time': time.time(),
                        'models_dir': models_dir,
                        'total_size': 0,
                        'downloaded_size': 0,
                        'initial_files': set()
                    }

                    # Get model info for size estimation
                    api = HfApi()
                    try:
                        model_info = api.model_info(model_id)
                        # Calculate total size from all files
                        for file in model_info.siblings:
                            if file.size:
                                self.download_progress_data['total_size'] += file.size
                        logger.info(f"HuggingFace model {model_id} total size: {self.download_progress_data['total_size']} bytes")
                    except Exception as e:
                        logger.warning(f"Could not get model info for size estimation: {str(e)}")

                    # Track initial files
                    if os.path.exists(models_dir):
                        for root, dirs, files in os.walk(models_dir):
                            for file in files:
                                self.download_progress_data['initial_files'].add(os.path.join(root, file))

                    # Define progress callback for HuggingFace downloads
                    def hf_progress_callback(size, total):
                        """Progress callback for HuggingFace downloads"""
                        if total > 0:
                            progress = min(95, int((size / total) * 100))
                            self._run_on_gui_thread(
                                self._update_hf_progress, row, model_name, progress
                            )

                    # Start download with progress tracking
                    self._run_on_gui_thread(self._update_hf_progress, row, model_name, 0)

                    # Start download - HuggingFace Hub doesn't have built-in progress callbacks
                    # We'll monitor file system changes for progress
                    if model_id:
                        snapshot_download(
                            repo_id=model_id,
                            local_dir=models_dir,
                            local_dir_use_symlinks=False
                        )
                    else:
                        self._run_on_gui_thread(
                            self._download_error, row, model_name, "Invalid model ID"
                        )
                        return

                    # Monitor download progress by checking file sizes
                    progress = 0
                    stable_count = 0

                    while progress < 100:
                        time.sleep(0.5)

                        if os.path.exists(models_dir):
                            current_size = 0
                            current_files = set()

                            # Calculate current downloaded size
                            for root, dirs, files in os.walk(models_dir):
                                for file in files:
                                    filepath = os.path.join(root, file)
                                    current_files.add(filepath)
                                    if os.path.exists(filepath):
                                        try:
                                            current_size += os.path.getsize(filepath)
                                        except OSError:
                                            pass  # File might be in use

                            # Update progress data
                            self.download_progress_data['downloaded_size'] = current_size

                            total_size = self.download_progress_data['total_size']
                            if total_size > 0:
                                progress = min(95, int((current_size / total_size) * 100))
                            else:
                                # Estimate progress based on file count growth
                                new_files = current_files - self.download_progress_data['initial_files']
                                if new_files:
                                    # Rough estimation: assume each model has ~10-50 files
                                    estimated_total_files = max(50, len(new_files) * 2)
                                    progress = min(90, int((len(new_files) / estimated_total_files) * 100))
                                else:
                                    # Time-based progress when no size info
                                    elapsed = time.time() - self.download_progress_data['start_time']
                                    progress = min(85, int((elapsed / 60.0) * 100))  # Assume 60 seconds for download

                            # Update progress in main thread
                            QTimer.singleShot(0, lambda p=progress: self._update_hf_progress(row, model_name, p))

                            # Check if download is complete by monitoring stability
                            if progress >= 85:
                                time.sleep(1)  # Wait to check stability
                                new_size = 0
                                for root, dirs, files in os.walk(models_dir):
                                    for file in files:
                                        filepath = os.path.join(root, file)
                                        if os.path.exists(filepath):
                                            try:
                                                new_size += os.path.getsize(filepath)
                                            except OSError:
                                                pass

                                if abs(new_size - current_size) < 1024:  # Less than 1KB change
                                    stable_count += 1
                                    if stable_count >= 3:  # Stable for 3 checks
                                        break  # Download likely complete
                                else:
                                    stable_count = 0

                    # Complete the progress
                    self._run_on_gui_thread(
                        self._update_hf_progress, row, model_name, 100
                    )
                    self._run_on_gui_thread(self._download_success, row, model_name)

                    # Clean up progress data
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

                except ImportError:
                    self._run_on_gui_thread(
                        self._download_error, row, model_name,
                        "HuggingFace Hub not installed. Install with: pip install huggingface-hub"
                    )
                except Exception as e:
                    self._run_on_gui_thread(self._download_error, row, model_name, str(e))
                    # Clean up progress data on error
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

            # Start download in background thread
            threading.Thread(target=download_worker, daemon=True).start()

        except Exception as e:
            self._download_error(row, model_name, f"Failed to start HuggingFace download: {str(e)}")

    def _update_progress_thread_safe(self, progress, status_text):
        """Update progress in a thread-safe way"""
        QTimer.singleShot(0, lambda: self._update_progress_ui(progress, status_text))

    def _update_progress_ui(self, progress, status_text):
        """Update progress UI elements"""
        try:
            self.download_progress.setValue(progress)
            self.download_status_label.setText(status_text)
            self.download_progress.repaint()
            # Force UI update
            QApplication.processEvents()
            logger.debug(f"Progress updated to {progress}%: {status_text}")
        except Exception as e:
            logger.error(f"Error updating progress UI: {str(e)}")

    def _update_download_progress(self, row, model_name, progress):
        """Update download progress for any source"""
        status_text = f"Downloading {model_name}... {progress}%"
        self._update_progress_thread_safe(progress, status_text)

    def _update_kaggle_progress(self, row, model_name, progress):
        """Update Kaggle download progress"""
        status_text = f"Downloading {model_name} from Kaggle... {progress}%"
        self._update_progress_thread_safe(progress, status_text)

    def _update_hf_progress(self, row, model_name, progress):
        """Update HuggingFace download progress"""
        status_text = f"Downloading {model_name} from HuggingFace... {progress}%"
        self._update_progress_thread_safe(progress, status_text)

    def _download_success(self, row, model_name):
        """Handle successful download"""
        try:
            logger.info(f"Download success for {model_name}, updating UI in main thread")

            # Update status
            status_item = QTableWidgetItem("Downloaded")
            status_item.setBackground(QColor(173, 216, 230))  # Light blue
            self.model_table.setItem(row, 4, status_item)

            # Hide progress bar
            self.download_progress.setVisible(False)
            self.download_status_label.setText(f"{model_name} downloaded successfully")
            self.status_bar.showMessage(f"Model {model_name} downloaded successfully")

            # Log success
            self._log_download(f"✓ Successfully downloaded {model_name}")

            # Reset progress for next download
            self.download_progress.setValue(0)
            logger.debug("Download success UI update completed")
        except Exception as e:
            logger.error(f"Error in download success handler: {str(e)}")

    def _download_error(self, row, model_name, error_msg):
        """Handle download error"""
        try:
            logger.error(f"Download error for {model_name}: {error_msg}")

            # Update status
            status_item = QTableWidgetItem("Error")
            status_item.setBackground(QColor(255, 182, 193))  # Light red
            self.model_table.setItem(row, 4, status_item)

            # Hide progress bar
            self.download_progress.setVisible(False)
            self.download_status_label.setText(f"Download failed: {model_name}")
            self.status_bar.showMessage(f"Download failed for {model_name}")

            # Log error
            self._log_download(f"✗ Failed to download {model_name}: {error_msg}")

            self._notify("Download Error", f"Failed to download {model_name}:\n{error_msg}", level="warning")

            # Reset progress
            self.download_progress.setValue(0)
            logger.debug("Download error UI update completed")
        except Exception as e:
            logger.error(f"Error in download error handler: {str(e)}")

    def _log_download(self, message):
        """Add message to download log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.download_log.appendPlainText(f"[{timestamp}] {message}")
        except Exception as e:
            logger.error(f"Error logging download message: {str(e)}")

    def clear_download_log(self):
        """Clear the download log"""
        self.download_log.clear()

    @qasync.asyncSlot()
    async def load_dataset(self):
        """Load dataset to API server before training"""
        task_id = f"load_dataset_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Register task
                with self.task_lock:
                    if task_id in self.active_tasks:
                        logger.warning(f"Load dataset task {task_id} already active")
                        return
                    self.active_tasks.add(task_id)

                # Track the current task for cancellation
                self.current_async_task = asyncio.current_task()
                async with self.task_cleanup_lock:
                    self.pending_tasks.add(self.current_async_task)

                if not self.client:
                    if not self.initialize_client():
                        self._notify("Connection Error", "Failed to initialize client", level="warning")
                        return

                # Get selected dataset
                current_item = self.dataset_list.currentItem()
                if not current_item:
                    self._notify("No Dataset Selected", "Please select a dataset to load", level="warning")
                    return

                dataset_name = current_item.text()
                if dataset_name not in self.training_manager.datasets:
                    self._notify("Dataset Not Found", f"Dataset '{dataset_name}' not found", level="warning")
                    return

                dataset_info = self.training_manager.datasets[dataset_name]
                if not dataset_info.get('files'):
                    self._notify("Empty Dataset", f"Dataset '{dataset_name}' has no files", level="warning")
                    return

                # Support all file formats - use the first file from the dataset
                data_file = None
                supported_extensions = ['.csv', '.json', '.xml', '.pdf', '.docx', '.doc', '.txt', '.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.flac', '.aac', '.jpg', '.jpeg', '.png', '.bmp', '.gif']
                for file_info in dataset_info['files']:
                    file_ext = os.path.splitext(file_info['name'].lower())[1]
                    if file_ext in supported_extensions:
                        data_file = file_info['path']
                        break

                if not data_file:
                    self._notify("No Supported File", f"Dataset '{dataset_name}' has no supported file formats.\nSupported formats: {', '.join(supported_extensions)}", level="warning")
                    return

                # Show progress
                self.training_progress.setVisible(True)
                self.training_progress.setRange(0, 100)
                self.training_progress.setValue(10)

                # Log dataset loading
                file_ext = os.path.splitext(data_file.lower())[1]
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Loading dataset '{dataset_name}' from {data_file} ({file_ext.upper()} format)")

                # Load dataset with normalization parameters
                result = await self._load_dataset_async(data_file, dataset_name)
                self._handle_load_dataset_result_direct(result, data_file)

                # Enable training button if dataset loaded successfully
                if result.success:
                    if self.training_button:
                        self.training_button.setEnabled(True)

                # Mark dataset as loaded if successful
                if result.success:
                    self.current_dataset_loaded = True
                    self.current_dataset_name = dataset_name

            except asyncio.CancelledError:
                logger.info("Dataset loading cancelled")
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading cancelled")
                raise
            except Exception as e:
                self.training_progress.setVisible(False)
                self.training_progress.setValue(0)
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading error: {str(e)}")
                self._notify("Dataset Loading Error", str(e), level="warning")
            finally:
                # Clean up task
                with self.task_lock:
                    self.active_tasks.discard(task_id)

                # Remove from pending tasks
                async with self.task_cleanup_lock:
                    self.pending_tasks.discard(self.current_async_task)
                    self.current_async_task = None

    def _convert_to_container_path(self, host_path: str) -> str:
        """Convert host file path to container-internal path expected by backend"""
        if not host_path:
            return host_path
        # Backend container mounts host /app/training_data as /training_data
        if host_path.startswith('/app/training_data/'):
            return host_path.replace('/app/training_data/', '/training_data/', 1)
        if host_path.startswith('/app/'):
            return host_path[4:]
        return host_path

    async def _load_dataset_async(self, data_file: str, dataset_name: str):
        """Async part of dataset loading"""
        # Validate parameters before calling API
        output_size = self.learning_config.get('output_size', 1)
        has_header = self.learning_config.get('has_header', True)
        contains_text = self.learning_config.get('contains_text', False)
        min_range = self.learning_config.get('normalization_min', -1.0)
        max_range = self.learning_config.get('normalization_max', 1.0)
        max_seq_len = self.max_seq_length.value()
        embedding_dim = self.embedding_dimension.value()

        # Basic validation
        if output_size < 1:
            raise ValueError("Output size must be at least 1")
        if min_range >= max_range:
            raise ValueError("Normalization min must be less than max")

        # Determine file format for appropriate API call
        file_ext = os.path.splitext(data_file.lower())[1]

        # Convert host path to container-internal path expected by backend
        container_path = self._convert_to_container_path(data_file)

        # Call appropriate API endpoint based on file format
        if file_ext == '.csv':
            # Use existing CSV loading logic
            return await self.client.load_dataset(
                filename=container_path,
                output_size=output_size,
                has_header=has_header,
                contains_text=contains_text,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range,
                max_seq_len=max_seq_len,
                embedding_dim=embedding_dim
            )
        elif file_ext in ['.json', '.xml', '.txt']:
            # For text-based formats, use generic data loading
            return await self.client.load_dataset(
                filename=container_path,
                output_size=output_size,
                has_header=has_header,
                contains_text=contains_text,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range,
                max_seq_len=max_seq_len,
                embedding_dim=embedding_dim
            )
        elif file_ext in ['.pdf', '.docx', '.doc']:
            # For document formats, use document processing
            return await self.client.load_dataset(
                filename=container_path,
                output_size=output_size,
                has_header=False,  # Documents typically don't have headers
                contains_text=True,  # Documents contain text
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range,
                max_seq_len=max_seq_len,
                embedding_dim=embedding_dim
            )
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            # For image formats, use vision processing
            return await self.client.load_dataset(
                filename=container_path,
                output_size=output_size,
                has_header=False,
                contains_text=False,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range,
                max_seq_len=max_seq_len,
                embedding_dim=embedding_dim
            )
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.flac', '.aac']:
            # For media formats, use audio/video processing
            return await self.client.load_dataset(
                filename=container_path,
                output_size=output_size,
                has_header=False,
                contains_text=False,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range,
                max_seq_len=max_seq_len,
                embedding_dim=embedding_dim
            )
        else:
            # Fallback for unknown formats
            return await self.client.load_dataset(
                filename=container_path,
                output_size=output_size,
                has_header=has_header,
                contains_text=contains_text,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range,
                max_seq_len=max_seq_len,
                embedding_dim=embedding_dim
            )
    def calculate_dataset_sizes(self, file_path: str, data_type: str = None) -> Dict[str, Any]:
        """Calculate inputsize, outputsize, and samplesize for a dataset file"""
        try:
            if not os.path.exists(file_path):
                return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
            file_ext = os.path.splitext(file_path.lower())[1]
    
            # Determine data type if not provided
            if not data_type:
                ext_to_type = {
                    '.csv': 'csv', '.json': 'json', '.xml': 'xml', '.txt': 'txt', '.text': 'txt',
                    '.pdf': 'pdf', '.docx': 'docx', '.doc': 'doc',
                    '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.bmp': 'image', '.gif': 'image',
                    '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio', '.aac': 'audio',
                    '.mp4': 'video', '.avi': 'video', '.mov': 'video', '.mkv': 'video'
                }
                data_type = ext_to_type.get(file_ext, 'unknown')
    
            # Calculate sizes based on data type
            if data_type == 'csv':
                return self._calculate_csv_sizes(file_path)
            elif data_type == 'json':
                return self._calculate_json_sizes(file_path)
            elif data_type == 'xml':
                return self._calculate_xml_sizes(file_path)
            elif data_type == 'txt':
                return self._calculate_text_sizes(file_path)
            elif data_type in ['pdf', 'docx', 'doc']:
                return self._calculate_document_sizes(file_path, data_type)
            elif data_type == 'image':
                return self._calculate_image_sizes(file_path)
            elif data_type == 'audio':
                return self._calculate_audio_sizes(file_path)
            elif data_type == 'video':
                return self._calculate_video_sizes(file_path)
            else:
                return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
        except Exception as e:
            logger.error(f"Error calculating dataset sizes for {file_path}: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_csv_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for CSV files"""
        try:
            parsed_data = self.parse_csv_file(file_path, None, self.learning_config.get('contains_text', False))
            if 'error' in parsed_data:
                return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
            column_count = parsed_data.get('column_count', 0)
            row_count = parsed_data.get('row_count', 0)
    
            if column_count <= 1:
                # Single column or empty - assume inputsize=1, outputsize=0
                return {'inputsize': 1, 'outputsize': 0, 'samplesize': row_count}
    
            # For CSV files, assume last column is target (output) and rest are inputs
            # This is a common convention in machine learning datasets
            inputsize = column_count - 1
            outputsize = 1
            samplesize = row_count
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating CSV sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_json_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    
            # Try to parse as JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
            # Analyze JSON structure
            if isinstance(data, list):
                # Array of objects/samples
                if len(data) == 0:
                    return {'inputsize': 0, 'outputsize': 0, 'samplesize': 0}
    
                first_item = data[0]
                if isinstance(first_item, dict):
                    # Each item is an object with features
                    keys = list(first_item.keys())
                    # Assume 'target' or 'label' or last key is output, rest are inputs
                    target_keys = ['target', 'label', 'output', 'y']
                    output_key = None
                    for key in reversed(keys):
                        if key.lower() in target_keys:
                            output_key = key
                            break
    
                    if output_key:
                        inputsize = len(keys) - 1
                        outputsize = 1
                    else:
                        # No clear target column, assume all are inputs
                        inputsize = len(keys)
                        outputsize = 0
    
                    samplesize = len(data)
                else:
                    # Array of simple values
                    inputsize = 1
                    outputsize = 0
                    samplesize = len(data)
            elif isinstance(data, dict):
                # Single object - treat as one sample
                keys = list(data.keys())
                target_keys = ['target', 'label', 'output', 'y']
                output_key = None
                for key in reversed(keys):
                    if key.lower() in target_keys:
                        output_key = key
                        break
    
                if output_key:
                    inputsize = len(keys) - 1
                    outputsize = 1
                else:
                    inputsize = len(keys)
                    outputsize = 0
                samplesize = 1
            else:
                # Simple value
                inputsize = 1
                outputsize = 0
                samplesize = 1
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating JSON sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_xml_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for XML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    
            # Simple XML analysis - count elements and attributes
            # This is a basic implementation
            import re
    
            # Count XML elements (rough estimate of features)
            element_count = len(re.findall(r'<\w+', content))
    
            # Count text content blocks (rough estimate of samples)
            text_blocks = len(re.findall(r'>[^<]+<', content))
    
            # For XML, we make some assumptions:
            # - Each element could represent a feature
            # - Text blocks could represent sample data
            # - This is highly simplified and would need domain-specific logic
    
            if element_count > 0:
                # Assume some elements are targets/outputs
                inputsize = max(1, element_count - 1)
                outputsize = 1
                samplesize = max(1, text_blocks // 10)  # Rough estimate
            else:
                inputsize = 1
                outputsize = 0
                samplesize = 1
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating XML sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_text_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    
            # Analyze text structure
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
    
            if not non_empty_lines:
                return {'inputsize': 0, 'outputsize': 0, 'samplesize': 0}
    
            # For text files, we treat each line as a sample
            # Input size is based on word count per line (embeddings)
            avg_words_per_line = sum(len(line.split()) for line in non_empty_lines) / len(non_empty_lines)
    
            # Assume text classification task
            inputsize = int(avg_words_per_line * 50)  # Rough embedding dimension estimate
            outputsize = 1  # Classification target
            samplesize = len(non_empty_lines)
    
            return {'inputsize': max(1, inputsize), 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating text sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_document_sizes(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """Calculate sizes for document files (PDF, DOCX, DOC)"""
        try:
            # For document files, we estimate based on file size and type
            file_size = os.path.getsize(file_path)
    
            # Rough estimates based on document type and size
            if doc_type == 'pdf':
                # PDFs typically contain structured text
                estimated_pages = max(1, file_size // 2000)  # Rough page estimate
                inputsize = 1000  # Text embeddings
                outputsize = 1
                samplesize = estimated_pages
            elif doc_type == 'docx':
                # DOCX files are XML-based
                estimated_paragraphs = max(1, file_size // 500)
                inputsize = 800  # Text embeddings
                outputsize = 1
                samplesize = estimated_paragraphs
            elif doc_type == 'doc':
                # Legacy DOC format
                estimated_paragraphs = max(1, file_size // 300)
                inputsize = 600  # Text embeddings
                outputsize = 1
                samplesize = estimated_paragraphs
            else:
                inputsize = 500
                outputsize = 1
                samplesize = 1
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating document sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_image_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for image files"""
        try:
            # For images, we extract basic properties
            file_size = os.path.getsize(file_path)
    
            # Try to get image dimensions (basic approach)
            try:
                # Simple header parsing for common formats
                with open(file_path, 'rb') as f:
                    header = f.read(24)  # Read first 24 bytes
    
                width, height = 224, 224  # Default fallback
    
                # PNG format
                if header.startswith(b'\x89PNG'):
                    if len(header) >= 24:
                        width = int.from_bytes(header[16:20], 'big')
                        height = int.from_bytes(header[20:24], 'big')
                # JPEG format (SOF0 marker)
                elif header.startswith(b'\xff\xd8'):
                    # Look for SOF0 marker
                    f.seek(0)
                    data = f.read()
                    sof0_pos = data.find(b'\xff\xc0')
                    if sof0_pos != -1 and sof0_pos + 9 < len(data):
                        height = int.from_bytes(data[sof0_pos + 5:sof0_pos + 7], 'big')
                        width = int.from_bytes(data[sof0_pos + 7:sof0_pos + 9], 'big')
    
                # For images, input size is pixels * channels, output size depends on task
                channels = 3  # Assume RGB
                inputsize = width * height * channels
                outputsize = 1000  # Classification outputs (e.g., ImageNet classes)
                samplesize = 1  # Single image
    
            except Exception:
                # Fallback for unsupported formats
                inputsize = 224 * 224 * 3  # Standard image size
                outputsize = 1000
                samplesize = 1
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating image sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_audio_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for audio files"""
        try:
            file_size = os.path.getsize(file_path)
    
            # Basic audio format detection and size estimation
            file_ext = os.path.splitext(file_path.lower())[1]
    
            # Rough estimates based on format and file size
            if file_ext in ['.wav', '.flac']:
                # Uncompressed formats - can estimate duration
                estimated_duration_sec = max(1, file_size // 176400)  # Rough WAV estimate (44.1kHz, 16-bit, stereo)
                inputsize = 128 * estimated_duration_sec  # MFCC features over time
                outputsize = 1  # Classification or regression target
                samplesize = estimated_duration_sec  # One sample per second
            elif file_ext == '.mp3':
                # Compressed format
                estimated_duration_sec = max(1, file_size // 16000)  # Rough MP3 estimate
                inputsize = 128 * estimated_duration_sec
                outputsize = 1
                samplesize = estimated_duration_sec
            elif file_ext == '.aac':
                # AAC format
                estimated_duration_sec = max(1, file_size // 12000)  # Rough AAC estimate
                inputsize = 128 * estimated_duration_sec
                outputsize = 1
                samplesize = estimated_duration_sec
            else:
                # Unknown audio format
                inputsize = 128 * 10  # 10 seconds default
                outputsize = 1
                samplesize = 10
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating audio sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
    def _calculate_video_sizes(self, file_path: str) -> Dict[str, Any]:
        """Calculate sizes for video files"""
        try:
            file_size = os.path.getsize(file_path)
    
            # Rough video analysis based on file size and format
            file_ext = os.path.splitext(file_path.lower())[1]
    
            # Estimate duration and frame rate
            if file_ext in ['.mp4', '.mkv']:
                # Compressed formats - rough estimates
                estimated_duration_sec = max(1, file_size // 500000)  # Rough estimate: ~0.5MB per second
                frames_per_sec = 30  # Assume 30 FPS
                total_frames = estimated_duration_sec * frames_per_sec
            elif file_ext in ['.avi', '.mov']:
                # Other formats
                estimated_duration_sec = max(1, file_size // 700000)  # Slightly higher bitrate
                frames_per_sec = 25  # Assume 25 FPS
                total_frames = estimated_duration_sec * frames_per_sec
            else:
                estimated_duration_sec = 10  # Default 10 seconds
                total_frames = 300  # 30 FPS * 10 seconds
    
            # For video, input size is frame pixels * channels, processed over time
            frame_width, frame_height = 224, 224  # Standard frame size
            channels = 3  # RGB
            inputsize = frame_width * frame_height * channels
            outputsize = 1000  # Classification outputs
            samplesize = total_frames  # One sample per frame
    
            return {'inputsize': inputsize, 'outputsize': outputsize, 'samplesize': samplesize}
    
        except Exception as e:
            logger.error(f"Error calculating video sizes: {str(e)}")
            return {'inputsize': 'N/A', 'outputsize': 'N/A', 'samplesize': 'N/A'}
    
                        
    def closeEvent(self, event):
        """Handle application close event with proper cleanup.

        Avoids the "Event loop is closed" error by:
        - Stopping all QTimers on the main Qt thread (this method runs on main thread)
        - Not calling asyncio.get_event_loop().is_running() after loop is gone
        - Using client.disconnect() / client.close() properly
        - Quitting worker threads before exit
        - Gracefully handling RuntimeError when the event loop is already closed

        The background WebSocket thread is closed as well: leaving it running
        keeps the process alive and eventually aborts it on interpreter exit.
        """
        try:
            logger.info("Starting application shutdown sequence")

            # ---- Stop ALL timers on the main Qt thread ----
            timer_attrs = (
                'metrics_timer', 'health_timer', 'neural_refresh_timer',
                'download_timer', '_typing_timer',
            )
            for attr in timer_attrs:
                timer = getattr(self, attr, None)
                if timer is not None:
                    try:
                        if timer.isActive():
                            timer.stop()
                            logger.debug(f"Stopped timer: {attr}")
                    except RuntimeError:
                        # Underlying C++ object already deleted
                        pass

            # ---- Stop the background WebSocket thread ----
            self._stop_websocket()

            # ---- Quit in-progress worker threads ----
            try:
                self._stop_beta_worker()
            except Exception as e:
                logger.debug(f"Error stopping beta worker: {e}")

            try:
                self._stop_email_worker()
            except Exception as e:
                logger.debug(f"Error stopping email worker: {e}")

            for attr in ('_api_key_worker_thread',):
                thread = getattr(self, attr, None)
                if thread is not None:
                    try:
                        if thread.isRunning():
                            thread.quit()
                            if not thread.wait(2000):  # Wait up to 2 seconds
                                _LINGERING_THREADS.append(thread)
                    except RuntimeError:
                        pass
                    setattr(self, attr, None)

            # Disconnect client cleanly
            if hasattr(self, 'client') and self.client is not None:
                try:
                    if hasattr(self.client, 'disconnect'):
                        # Try to run disconnect on a temporary event loop so it
                        # actually completes rather than being orphaned.
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = None
                        if loop is not None and loop.is_running():
                            # Best effort: schedule disconnect; OS cleans up on exit
                            asyncio.create_task(self.client.disconnect())
                        else:
                            # No running loop: create a temporary one
                            temp_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(temp_loop)
                            try:
                                temp_loop.run_until_complete(self.client.disconnect())
                            finally:
                                temp_loop.close()
                                # Never leave a closed loop installed as the
                                # current loop: later asyncio calls would
                                # raise "Event loop is closed".
                                asyncio.set_event_loop(asyncio.new_event_loop())
                except Exception as e:
                    logger.error(f"Error closing client: {str(e)}")

            # Save configuration (includes the window geometry)
            self._save_config()

            logger.info("CoreAI3D Dashboard closed successfully")
            event.accept()

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            event.accept()

    def _stop_websocket(self) -> None:
        """Close the WebSocket connection and stop its background thread."""
        self._ws_shutdown = True
        ws = getattr(self, 'ws', None)
        if ws is not None:
            try:
                ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self.ws = None

        thread = getattr(self, 'ws_thread', None)
        if thread is not None:
            try:
                if thread.is_alive():
                    thread.join(timeout=2.0)
            except Exception as e:
                logger.debug(f"Error joining WebSocket thread: {e}")
            self.ws_thread = None

    async def _start_training_async(self):
        """Async part of training start"""
        # Ensure required training configuration keys are present with sensible defaults
        training_config = dict(self.learning_config)
        training_config.setdefault('dataset_name', getattr(self, 'current_dataset_name', 'default_dataset'))
        training_config.setdefault('learning_rate', self.learning_rate)
        training_config.setdefault('epochs', self.epochs)
        training_config.setdefault('layers', training_config.get('hidden_layers', self.layers))
        training_config.setdefault('neurons', training_config.get('neurons_per_layer', self.neurons))
        training_config.setdefault('batch_size', 32)
        training_config.setdefault('optimizer', 'adam')
        training_config.setdefault('activation', 'relu')
        training_config.setdefault('min_range', self.norm_min)
        training_config.setdefault('max_range', self.norm_max)
        training_config['inputSize'] = self.network_input_size.value()
        training_config['outputSize'] = self.network_output_size.value()
        training_config['max_seq_len'] = self.max_seq_length.value()
        training_config['embedding_dim'] = self.embedding_dimension.value()

        result = await self.client.start_training(training_config)
        return result

    def _handle_training_result(self, task):
        """Handle training result"""
        try:
            result = task.result()
            self.training_progress.setVisible(False)
            self.training_progress.setValue(100)  # Ensure progress shows complete
            if result.success:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed successfully")
                if result.data:
                    self.training_log.appendPlainText(f"Training results: {json.dumps(result.data, indent=2)}")
                self._notify("Training Complete", "Training completed successfully", level="info")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training failed: {error_msg}")
                self._notify("Training Failed", error_msg, level="warning")
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)  # Reset progress on error
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training error: {str(e)}")
            self._notify("Training Error", str(e), level="warning")

    def _handle_training_result_direct(self, result):
        """Handle training result directly"""
        try:
            self.training_progress.setVisible(False)
            self.training_progress.setValue(100)  # Ensure progress shows complete
            if result.success:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed successfully")
                if result.data:
                    self.training_log.appendPlainText(f"Training results: {json.dumps(result.data, indent=2)}")
                self._notify("Training Complete", "Training completed successfully", level="info")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training failed: {error_msg}")
                self._notify("Training Failed", error_msg, level="warning")
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)  # Reset progress on error
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training error: {str(e)}")
            self._notify("Training Error", str(e), level="warning")

    @qasync.asyncSlot()
    async def stop_training(self):
        """Stop current training with proper locking"""
        task_id = f"stop_training_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Check if training is actually running (double-check with lock)
                if not self.training_in_progress:
                    self._notify("No Training", "No training is currently running", level="info")
                    return

                # Register task
                with self.task_lock:
                    if task_id in self.active_tasks:
                        logger.warning(f"Stop training task {task_id} already active")
                        return
                    self.active_tasks.add(task_id)

                # Track the current task for cancellation
                self.current_async_task = asyncio.current_task()
                async with self.task_cleanup_lock:
                    if self.current_async_task:
                        self.pending_tasks.add(self.current_async_task)

                # Set flag to stop training
                self.training_in_progress = False

                if not self.client:
                    self._notify("Connection Error", "No active client connection", level="warning")
                    return

                # Stop training directly with qasync
                result = await self._stop_training_async()
                self._handle_stop_training_result_direct(result)

            except asyncio.CancelledError:
                logger.info("Stop training cancelled")
                raise  # Re-raise to ensure proper cleanup
            except Exception as e:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {str(e)}")
                self._notify("Stop Error", str(e), level="warning")
            finally:
                # Clean up task
                with self.task_lock:
                    self.active_tasks.discard(task_id)

                # Remove from pending tasks
                async with self.task_cleanup_lock:
                    if self.current_async_task:
                        self.pending_tasks.discard(self.current_async_task)
                    self.current_async_task = None

    async def _stop_training_async(self):
        """Async part of training stop"""
        # Note: We don't use the global lock here since we're already inside it
        result = await self.client.stop_training()
        return result

    def _handle_stop_training_result(self, task):
        """Handle stop training result"""
        try:
            result = task.result()
            if result.success:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training stopped successfully")
                self.training_progress.setVisible(False)
                self._notify("Training Stopped", "Training stopped successfully", level="info")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to stop training: {error_msg}")
                self._notify("Stop Failed", error_msg, level="warning")
        except Exception as e:
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {str(e)}")
            self._notify("Stop Error", str(e), level="warning")

    def _handle_stop_training_result_direct(self, result):
        """Handle stop training result directly"""
        try:
            if result.success:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training stopped successfully")
                self.training_progress.setVisible(False)
                self._notify("Training Stopped", "Training stopped successfully", level="info")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to stop training: {error_msg}")
                self._notify("Stop Failed", error_msg, level="warning")
        except Exception as e:
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {str(e)}")
            self._notify("Stop Error", str(e), level="warning")

    @qasync.asyncSlot()
    async def upload_file(self):
        """Upload file of various formats to API server"""
        task_id = f"upload_file_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Register task
                with self.task_lock:
                    if task_id in self.active_tasks:
                        logger.warning(f"Upload file task {task_id} already active")
                        return
                    self.active_tasks.add(task_id)

                # Track the current task for cancellation
                self.current_async_task = asyncio.current_task()
                async with self.task_cleanup_lock:
                    self.pending_tasks.add(self.current_async_task)

                if not self.client:
                    if not self.initialize_client():
                        self._notify("Connection Error", "Failed to initialize client", level="warning")
                        return

                # Get selected file type
                file_type = self.upload_file_type_combo.currentText().lower()

                # Set up file dialog filter based on file type
                file_filters = {
                    "csv": "CSV Files (*.csv);;All Files (*.*)",
                    "json": "JSON Files (*.json);;All Files (*.*)",
                    "xml": "XML Files (*.xml);;All Files (*.*)",
                    "pdf": "PDF Files (*.pdf);;All Files (*.*)",
                    "docx": "Word Documents (*.docx);;All Files (*.*)",
                    "doc": "Word Documents (*.doc);;All Files (*.*)",
                    "txt": "Text Files (*.txt);;All Files (*.*)",
                    "mp4": "Video Files (*.mp4);;All Files (*.*)",
                    "avi": "Video Files (*.avi);;All Files (*.*)",
                    "mov": "Video Files (*.mov);;All Files (*.*)",
                    "mkv": "Video Files (*.mkv);;All Files (*.*)",
                    "mp3": "Audio Files (*.mp3);;All Files (*.*)",
                    "wav": "Audio Files (*.wav);;All Files (*.*)",
                    "flac": "Audio Files (*.flac);;All Files (*.*)",
                    "aac": "Audio Files (*.aac);;All Files (*.*)",
                    "jpg": "Image Files (*.jpg);;All Files (*.*)",
                    "jpeg": "Image Files (*.jpeg);;All Files (*.*)",
                    "png": "Image Files (*.png);;All Files (*.*)",
                    "bmp": "Image Files (*.bmp);;All Files (*.*)",
                    "gif": "Image Files (*.gif);;All Files (*.*)"
                }

                filter_text = file_filters.get(file_type, "All Files (*.*)")

                # Open file dialog to select file.  The dialog is shown
                # modeless, so the coroutine waits for the user's choice
                # instead of freezing the GUI.
                loop = asyncio.get_event_loop()
                file_picker = loop.create_future()

                def on_file_selected(path):
                    if not file_picker.done():
                        file_picker.set_result(path)

                def on_file_picker_cancelled():
                    if not file_picker.done():
                        file_picker.set_result("")

                self._select_file(
                    f"Select {file_type.upper()} File", on_file_selected,
                    file_filter=filter_text, on_cancelled=on_file_picker_cancelled,
                )

                file_path = await file_picker

                if not file_path:
                    return  # User cancelled

                # Validate file exists and is readable
                if not os.path.exists(file_path):
                    self._notify("File Not Found", f"File does not exist: {file_path}", level="warning")
                    return

                if not os.access(file_path, os.R_OK):
                    self._notify("Permission Denied", f"Cannot read file: {file_path}", level="warning")
                    return

                # Get dataset name from user
                name_picker = loop.create_future()

                def on_name_entered(text):
                    if not name_picker.done():
                        name_picker.set_result(text)

                def on_name_cancelled():
                    if not name_picker.done():
                        name_picker.set_result("")

                self._prompt_text(
                    "Dataset Name",
                    f"Enter a name for this {file_type.upper()} dataset:",
                    on_name_entered,
                    default_text=os.path.splitext(os.path.basename(file_path))[0],
                    on_cancelled=on_name_cancelled,
                )

                dataset_name = (await name_picker or "").strip()

                if not dataset_name:
                    return  # User cancelled

                # Show progress
                self.training_progress.setVisible(True)
                self.training_progress.setRange(0, 100)
                self.training_progress.setValue(10)

                # Calculate sizes for the uploaded file
                file_sizes = self.calculate_dataset_sizes(file_path, file_type.lower())

                # Log upload start with calculated sizes
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Uploading {file_type.upper()} file '{os.path.basename(file_path)}' as dataset '{dataset_name}'")
                self.training_log.appendPlainText(f"  - Calculated sizes: Input={file_sizes['inputsize']}, Output={file_sizes['outputsize']}, Samples={file_sizes['samplesize']}")

                # Upload file to API server
                result = await self._upload_file_async(file_path, dataset_name)
                self._handle_upload_file_result_direct(result, file_sizes)

                # If successful, also load the dataset
                if result.success:
                    self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Upload successful, loading dataset...")

                    # Use the server-returned file path if available, otherwise fall back to local path
                    uploaded_path = result.data.get('file_path', file_path) if result.data else file_path
                    container_path = self._convert_to_container_path(uploaded_path)

                    # Load the uploaded dataset
                    load_result = await self._load_dataset_async(container_path, dataset_name)
                    self._handle_load_dataset_result_direct(load_result, file_path)

                    if load_result.success:
                        self.current_dataset_loaded = True
                        self.current_dataset_name = dataset_name
                        self.training_button.setEnabled(True)

            except asyncio.CancelledError:
                logger.info("File upload cancelled")
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] File upload cancelled")
                raise
            except Exception as e:
                self.training_progress.setVisible(False)
                self.training_progress.setValue(0)
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] File upload error: {str(e)}")
                self._notify("Upload Error", str(e), level="warning")
            finally:
                # Clean up task
                with self.task_lock:
                    self.active_tasks.discard(task_id)

                # Remove from pending tasks
                async with self.task_cleanup_lock:
                    self.pending_tasks.discard(self.current_async_task)
                    self.current_async_task = None

    @qasync.asyncSlot()
    async def upload_csv(self):
        """Upload CSV file to API server (legacy method)"""
        # Set file type to CSV and call upload_file
        self.upload_file_type_combo.setCurrentText("CSV")
        await self.upload_file()

    async def _upload_file_async(self, file_path: str, dataset_name: str):
        """Async part of file upload for various formats"""
        # Call the client's upload_file method to actually upload the file
        metadata = {'dataset_name': dataset_name}
        result = await self.client.upload_file(file_path, metadata)
        return result

    def _handle_upload_file_result_direct(self, result, file_sizes=None):
        """Handle file upload result directly"""
        try:
            self.training_progress.setVisible(False)
            if result.success:
                data = result.data
                file_type = data.get('file_type', 'unknown').upper()
                format_desc = data.get('format_description', f'{file_type} file')
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {format_desc} validation successful")
                self.training_log.appendPlainText(f"  - File: {os.path.basename(data.get('file_path', 'Unknown'))}")
                self.training_log.appendPlainText(f"  - Size: {data.get('file_size', 0) / 1024:.1f} KB")
                self.training_log.appendPlainText(f"  - Dataset: {data.get('dataset_name', 'Unknown')}")

                # Add calculated sizes to the log if available
                if file_sizes:
                    self.training_log.appendPlainText(f"  - Calculated Input Size: {file_sizes['inputsize']}")
                    self.training_log.appendPlainText(f"  - Calculated Output Size: {file_sizes['outputsize']}")
                    self.training_log.appendPlainText(f"  - Calculated Sample Size: {file_sizes['samplesize']}")

                self._notify("Upload Successful", f"{format_desc} uploaded and validated successfully!\n\n"
                                      f"File: {os.path.basename(data.get('file_path', 'Unknown'))}\n"
                                      f"Dataset: {data.get('dataset_name', 'Unknown')}\n"
                                      f"Size: {data.get('file_size', 0) / 1024:.1f} KB", level="info")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] File upload failed: {error_msg}")
                self._notify("Upload Failed", error_msg, level="warning")
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Upload result handling error: {str(e)}")
            self._notify("Upload Error", str(e), level="warning")

    def clear_training_log(self):
        """Clear training log"""
        self.training_log.clear()

    def browse_predict_input_file(self):
        """Browse for prediction input file"""
        self._select_file(
            "Select Input File",
            lambda file_path: self.predict_file_path_edit.setText(file_path) if file_path else None,
            file_filter="All Files (*.*)",
        )

    @qasync.asyncSlot()
    async def run_prediction(self):
        """Run prediction with selected model and input"""
        if not self.client:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        model_type = self.predict_model_combo.currentText()
        specific_model = self.predict_specific_model_combo.currentText()
        input_type = self.predict_input_type_combo.currentText()

        if model_type == "Select Model...":
            self._notify("No Model Selected", "Please select a model type", level="warning")
            return

        if specific_model == "Select Specific Model...":
            self._notify("No Model Selected", "Please select a specific model", level="warning")
            return

        # Get input data based on type
        input_data = None
        parsed_csv_data = None

        if input_type == "Text":
            input_data = self.predict_input_text.toPlainText().strip()
            if not input_data:
                self._notify("No Input", "Please enter text input", level="warning")
                return
        elif input_type in ["Image", "Audio"]:
            input_data = self.predict_file_path_edit.text().strip()
            if not input_data:
                self._notify("No Input", "Please select a file", level="warning")
                return
        elif input_type == "File":
            file_path = self.predict_file_path_edit.text().strip()
            if not file_path:
                self._notify("No Input", "Please select a file", level="warning")
                return

            # Check if it's a CSV file and parse it
            if file_path.lower().endswith('.csv'):
                # Get checkbox values for data format options
                contains_header = self.predict_contains_header_check.isChecked()
                contains_text = self.predict_contains_text_check.isChecked()

                # Show parsing progress for large CSV files
                self.predict_progress.setVisible(True)
                self.predict_progress.setRange(0, 0)  # Indeterminate progress
                self.status_bar.showMessage("Parsing CSV file...")

                # Parse CSV in a separate thread to avoid blocking UI
                parsed_csv_data = self.parse_csv_file(file_path, contains_header, contains_text)
                if 'error' in parsed_csv_data:
                    self.predict_progress.setVisible(False)
                    self._notify("CSV Parse Error", f"Failed to parse CSV file: {parsed_csv_data['error']}", level="warning")
                    return
                input_data = parsed_csv_data

                self.predict_progress.setVisible(False)
                self.status_bar.showMessage("CSV file parsed successfully")
            else:
                input_data = file_path
        elif input_type == "URL":
            input_data = self.predict_file_path_edit.text().strip()
            if not input_data:
                self._notify("No Input", "Please enter a URL", level="warning")
                return

        # Get checkbox values for data format options (only for non-CSV files)
        contains_header = self.predict_contains_header_check.isChecked()
        contains_text = self.predict_contains_text_check.isChecked()

        # Show progress
        self.predict_progress.setVisible(True)
        self.predict_progress.setRange(0, 0)  # Indeterminate progress

        try:
            # Run prediction directly with qasync
            result, parsed_csv_data_result = await self._run_prediction_async(
                model_type, specific_model, input_type, input_data,
                contains_header, contains_text, parsed_csv_data
            )
            self._handle_prediction_result_direct(result, parsed_csv_data_result)

        except Exception as e:
            self.predict_progress.setVisible(False)
            self.predict_results_text.setPlainText(f"Error: {str(e)}")
            self.status_bar.showMessage("Prediction error")

    async def _run_prediction_async(self, model_type: str, specific_model: str, input_type: str,
                                     input_data: str, contains_header: bool, contains_text: bool,
                                     parsed_csv_data: dict):
        """Async part of prediction"""
        # Note: We don't use the global lock here since we're already inside it
        # Call appropriate prediction method based on model type
        if model_type == "Vision Models":
            result = await self.client.analyze_image(input_data)
        elif model_type == "Audio Models":
            result = await self.client.analyze_audio(input_data)
        elif model_type == "Text Models":
            result = await self.client.process_with_ai(input_data, 'text')
        elif model_type == "Math Models":
            result = await self.client.calculate(input_data)
        elif model_type == "Web Models":
            result = await self.client.search_web(input_data)
        else:
            # Generic prediction call with data format options
            result = await self.client.run_prediction(
                specific_model, input_data, input_type.lower(),
                contains_header=contains_header, contains_text=contains_text
            )
        return result, parsed_csv_data

    def _handle_prediction_result(self, task):
        """Handle prediction result"""
        try:
            result, parsed_csv_data = task.result()
            self.predict_progress.setVisible(False)
            if result.success:
                # Format and display results
                formatted_result = self._format_prediction_result(result.data)

                # If we parsed CSV data, include CSV parsing summary
                if parsed_csv_data and 'error' not in parsed_csv_data:
                    csv_summary = self._format_csv_summary(parsed_csv_data)
                    formatted_result = f"CSV Data Summary:\n{csv_summary}\n\nPrediction Results:\n{formatted_result}"

                self.predict_results_text.setPlainText(formatted_result)
                self.status_bar.showMessage("Prediction completed successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error')
                self.predict_results_text.setPlainText(f"Prediction failed: {error_msg}")
                self.status_bar.showMessage("Prediction failed")
        except Exception as e:
            self.predict_progress.setVisible(False)
            self.predict_results_text.setPlainText(f"Error: {str(e)}")
            self.status_bar.showMessage("Prediction error")

    def _handle_prediction_result_direct(self, result, parsed_csv_data):
        """Handle prediction result directly"""
        try:
            self.predict_progress.setVisible(False)
            if result.success:
                # Format and display results
                formatted_result = self._format_prediction_result(result.data)

                # If we parsed CSV data, include CSV parsing summary
                if parsed_csv_data and 'error' not in parsed_csv_data:
                    csv_summary = self._format_csv_summary(parsed_csv_data)
                    formatted_result = f"CSV Data Summary:\n{csv_summary}\n\nPrediction Results:\n{formatted_result}"

                self.predict_results_text.setPlainText(formatted_result)
                self.status_bar.showMessage("Prediction completed successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error')
                self.predict_results_text.setPlainText(f"Prediction failed: {error_msg}")
                self.status_bar.showMessage("Prediction failed")
        except Exception as e:
            self.predict_progress.setVisible(False)
            self.predict_results_text.setPlainText(f"Error: {str(e)}")
            self.status_bar.showMessage("Prediction error")

    def clear_prediction_results(self):
        """Clear prediction results"""
        self.predict_results_text.clear()
        self.predict_progress.setVisible(False)

    @qasync.asyncSlot()
    async def load_model(self):
        """Load a model using the specified identifier"""
        if not self.client:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        model_identifier = self.load_model_identifier_edit.text().strip()
        if not model_identifier:
            self._notify("No Identifier", "Please enter a model identifier (dataset ID or name)", level="warning")
            return

        model_type = self.load_model_type_combo.currentText()
        if model_type == "Auto-detect":
            model_type = None
        else:
            # Convert display name to API format
            type_mapping = {
                "Vision": "vision",
                "Audio": "audio",
                "Text": "text",
                "Multimodal": "multimodal",
                "Neural Network": "neural_network",
                "Math": "math",
                "Web": "web"
            }
            model_type = type_mapping.get(model_type, model_type.lower())

        # Show progress
        self.load_progress.setVisible(True)
        self.load_progress.setRange(0, 0)  # Indeterminate progress

        try:
            # Load model directly with qasync
            result = await self._load_model_async(model_identifier, model_type)
            self._handle_load_model_result_direct(result)

        except Exception as e:
            self.load_progress.setVisible(False)
            self.load_results_text.setPlainText(f"✗ Load operation failed!\n\nError: {str(e)}")
            self.status_bar.showMessage("Model loading error")

    async def _load_model_async(self, model_identifier: str, model_type: str):
        """Async part of model loading"""
        # Note: We don't use the global lock here since we're already inside it
        result = await self.client.load_model(model_identifier, model_type)
        return result

    def _handle_load_model_result(self, task):
        """Handle model loading result"""
        try:
            result = task.result()
            self.load_progress.setVisible(False)
            if result.success:
                self.load_results_text.setPlainText(f"✓ Model loaded successfully!\n\n"
                                                  f"Model: {result.data.get('model_identifier', '')}\n"
                                                  f"Status: {result.data.get('status', 'Loaded')}\n"
                                                  f"Details: {result.data.get('message', 'Model is ready for use')}")
                self.status_bar.showMessage(f"Model '{result.data.get('model_identifier', '')}' loaded successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.load_results_text.setPlainText(f"✗ Failed to load model!\n\n"
                                                  f"Model: {result.data.get('model_identifier', '')}\n"
                                                  f"Error: {error_msg}")
                self.status_bar.showMessage(f"Failed to load model '{result.data.get('model_identifier', '')}'")
        except Exception as e:
            self.load_progress.setVisible(False)
            self.load_results_text.setPlainText(f"✗ Load operation failed!\n\nError: {str(e)}")
            self.status_bar.showMessage("Model loading error")

    def _handle_load_model_result_direct(self, result):
        """Handle model loading result directly"""
        try:
            self.load_progress.setVisible(False)
            if result.success:
                self.load_results_text.setPlainText(f"✓ Model loaded successfully!\n\n"
                                                  f"Model: {result.data.get('model_identifier', '')}\n"
                                                  f"Status: {result.data.get('status', 'Loaded')}\n"
                                                  f"Details: {result.data.get('message', 'Model is ready for use')}")
                self.status_bar.showMessage(f"Model '{result.data.get('model_identifier', '')}' loaded successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.load_results_text.setPlainText(f"✗ Failed to load model!\n\n"
                                                  f"Model: {result.data.get('model_identifier', '')}\n"
                                                  f"Error: {error_msg}")
                self.status_bar.showMessage(f"Failed to load model '{result.data.get('model_identifier', '')}'")
        except Exception as e:
            self.load_progress.setVisible(False)
            self.load_results_text.setPlainText(f"✗ Load operation failed!\n\nError: {str(e)}")
            self.status_bar.showMessage("Model loading error")

    def clear_load_results(self):
        """Clear load model results"""
        self.load_results_text.clear()
        self.load_progress.setVisible(False)

    def parse_csv_file(self, file_path: str, contains_header: bool = False, contains_text: bool = False) -> Dict[str, Any]:
        """Parse CSV file using comma as delimiter and return structured data with improved performance"""
        try:
            parsed_data = {
                'rows': [],
                'headers': [],
                'row_count': 0,
                'column_count': 0,
                'text_columns': [],
                'numeric_columns': []
            }

            # Check file size for large file handling
            file_size = os.path.getsize(file_path)
            is_large_file = file_size > 10 * 1024 * 1024  # 10MB threshold

            with open(file_path, 'r', encoding='utf-8') as csvfile:
                # Use comma as delimiter as specified
                reader = csv.reader(csvfile, delimiter=',')

                # Read all rows or limit for large files
                if is_large_file:
                    # For large files, read first 1000 rows for analysis
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= 1000:  # Limit to first 1000 rows
                            break
                        rows.append(row)
                    parsed_data['large_file_limited'] = True
                    parsed_data['total_file_size'] = file_size
                else:
                    rows = list(reader)

                if not rows:
                    return parsed_data

                parsed_data['row_count'] = len(rows)

                # Handle headers - auto-detect if contains_header is None
                if contains_header is None:
                    # Auto-detect headers: check if first row looks like headers
                    if rows and len(rows) > 1:
                        first_row = rows[0]
                        second_row = rows[1] if len(rows) > 1 else []
                        contains_header = self._detect_headers(first_row, second_row)
                        parsed_data['auto_detected_header'] = contains_header
                    else:
                        contains_header = False

                if contains_header and rows:
                    parsed_data['headers'] = rows[0]
                    data_rows = rows[1:]
                else:
                    # Generate column names if no headers
                    if rows:
                        parsed_data['headers'] = [f'col_{i+1}' for i in range(len(rows[0]))]
                    data_rows = rows

                parsed_data['column_count'] = len(parsed_data['headers'])

                # Process data rows with better error handling
                for row_idx, row in enumerate(data_rows):
                    if len(row) == parsed_data['column_count']:
                        parsed_data['rows'].append(row)
                    elif len(row) > parsed_data['column_count']:
                        # Pad short rows with empty strings
                        row.extend([''] * (parsed_data['column_count'] - len(row)))
                        parsed_data['rows'].append(row)
                    # Skip rows that are too short (they will be ignored)

                # Analyze column types if contains_text is specified and we have data
                if contains_text and parsed_data['rows']:
                    # Sample first few rows to determine column types (limit to 50 for performance)
                    sample_rows = parsed_data['rows'][:min(50, len(parsed_data['rows']))]

                    for col_idx in range(parsed_data['column_count']):
                        is_text = False
                        is_numeric = True

                        for row in sample_rows:
                            if col_idx < len(row):
                                cell_value = row[col_idx].strip()
                                if not cell_value:  # Skip empty cells
                                    continue

                                # Check if it's text (contains letters)
                                if any(c.isalpha() for c in cell_value):
                                    is_text = True

                                # Check if it's numeric (more robust check)
                                try:
                                    # Remove common number formatting characters
                                    clean_value = cell_value.replace(',', '').replace('%', '').replace('$', '').strip()
                                    float(clean_value)
                                except ValueError:
                                    is_numeric = False

                        if is_text:
                            parsed_data['text_columns'].append(col_idx)
                        if is_numeric:
                            parsed_data['numeric_columns'].append(col_idx)

            return parsed_data

        except UnicodeDecodeError:
            # Try with different encoding for files with special characters
            try:
                with open(file_path, 'r', encoding='latin-1') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    rows = list(reader)
                    # Simplified parsing for encoding issues
                    parsed_data['rows'] = rows
                    parsed_data['row_count'] = len(rows)
                    if rows:
                        parsed_data['column_count'] = len(rows[0])
                        parsed_data['headers'] = [f'col_{i+1}' for i in range(len(rows[0]))]
                    return parsed_data
            except Exception as e2:
                return {
                    'error': f"File encoding error. Tried UTF-8 and Latin-1: {str(e2)}",
                    'rows': [],
                    'headers': [],
                    'row_count': 0,
                    'column_count': 0
                }
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {str(e)}")
            return {
                'error': str(e),
                'rows': [],
                'headers': [],
                'row_count': 0,
                'column_count': 0
            }

    def _format_csv_summary(self, csv_data: Dict[str, Any]) -> str:
        """Format CSV parsing summary for display with reduced verbosity"""
        try:
            summary_lines = []
            row_count = csv_data.get('row_count', 0)
            column_count = csv_data.get('column_count', 0)

            # Compact format - combine basic info
            summary_lines.append(f"CSV: {row_count:,} rows × {column_count} columns")

            # Add file size info only for large files
            if csv_data.get('large_file_limited'):
                file_size_mb = csv_data.get('total_file_size', 0) / (1024 * 1024)
                summary_lines.append(f"Size: {file_size_mb:.1f} MB (showing first 1000 rows)")
            elif csv_data.get('total_file_size') and csv_data['total_file_size'] > 1024 * 1024:  # Only show if > 1MB
                file_size_mb = csv_data.get('total_file_size', 0) / (1024 * 1024)
                summary_lines.append(f"Size: {file_size_mb:.1f} MB")

            # Show column type summary only if meaningful
            text_count = len(csv_data.get('text_columns', []))
            numeric_count = len(csv_data.get('numeric_columns', []))
            if text_count > 0 or numeric_count > 0:
                summary_lines.append(f"Data types: {text_count} text, {numeric_count} numeric columns")

            # Remove detailed headers and preview rows to reduce verbosity
            # Only show first row preview if small dataset
            if csv_data.get('rows') and len(csv_data['rows']) <= 10 and len(csv_data['rows'][0]) <= 5:
                first_row = csv_data['rows'][0]
                if len(first_row) <= 5:
                    summary_lines.append(f"Sample: {', '.join(first_row[:5])}")

            return "\n".join(summary_lines)

        except Exception as e:
            return f"Error formatting CSV summary: {str(e)}"

    def _format_prediction_result(self, data: Dict[str, Any]) -> str:
        """Format prediction result for display with truncation for large arrays"""
        try:
            if isinstance(data, dict):
                # Pretty format JSON-like results
                formatted_lines = []
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        formatted_lines.append(f"{key}: {value:.4f}")
                    elif isinstance(value, list):
                        formatted_lines.append(self._format_array_with_truncation(key, value))
                    else:
                        formatted_lines.append(f"{key}: {value}")
                return "\n".join(formatted_lines)
            else:
                return str(data)
        except Exception as e:
            return f"Error formatting result: {str(e)}\n\nRaw data: {data}"

    def _format_array_with_truncation(self, key: str, array: list, max_display: int = 3) -> str:
        """Format array with truncation and compact summary statistics"""
        try:
            total_count = len(array)

            # If array is very small, show it fully
            if total_count <= max_display:
                return f"{key}: {array}"

            # Count non-zero elements (limit to first 1000 for performance)
            sample_size = min(1000, total_count)
            non_zero_count = sum(1 for x in array[:sample_size] if x != 0)

            # Show first few elements
            displayed_elements = array[:max_display]
            truncated_display = f"{displayed_elements} ... ({total_count - max_display} more)"

            # Compact single-line summary
            summary = f"{key}: {truncated_display} | Total: {total_count}"

            # Add non-zero count only if relevant
            if non_zero_count < sample_size:
                summary += f" | Non-zero: {non_zero_count}"

            # Add basic statistics for numeric arrays only if small
            if total_count <= 100 and array and isinstance(array[0], (int, float)):
                try:
                    numeric_array = [float(x) for x in array if isinstance(x, (int, float))]
                    if numeric_array:
                        min_val = min(numeric_array)
                        max_val = max(numeric_array)
                        avg_val = sum(numeric_array) / len(numeric_array)
                        summary += f" | Range: [{min_val:.2f}, {max_val:.2f}] | Avg: {avg_val:.2f}"
                except (ValueError, TypeError):
                    pass  # Skip statistics if conversion fails

            return summary

        except Exception as e:
            return f"{key}: [Error formatting array: {str(e)}] ({len(array)} items)"

    def on_test_type_changed(self):
        """Handle test type selection change"""
        test_type = self.test_type_combo.currentText()

        # Show/hide configuration groups based on test type
        if test_type == "System Automation":
            self.automation_config_group.setVisible(True)
            self.performance_config_group.setVisible(False)
            self.load_config_group.setVisible(False)
        elif test_type == "Performance Test":
            self.automation_config_group.setVisible(False)
            self.performance_config_group.setVisible(True)
            self.load_config_group.setVisible(False)
        elif test_type == "Load Test":
            self.automation_config_group.setVisible(False)
            self.performance_config_group.setVisible(False)
            self.load_config_group.setVisible(True)
        else:
            self.automation_config_group.setVisible(False)
            self.performance_config_group.setVisible(False)
            self.load_config_group.setVisible(False)

    def on_predict_model_type_changed(self):
        """Handle model type selection change"""
        model_type = self.predict_model_combo.currentText()

        # Update specific model options based on type
        self.predict_specific_model_combo.clear()
        self.predict_specific_model_combo.addItem("Select Specific Model...")

        if model_type == "Vision Models":
            self.predict_specific_model_combo.addItems([
                "ResNet-50", "YOLOv5-Small", "EfficientNet-B0", "MobileNetV3-Small",
                "FaceNet", "Tesseract-OCR", "Mask-RCNN", "SSD-MobileNet"
            ])
        elif model_type == "Audio Models":
            self.predict_specific_model_combo.addItems([
                "Whisper-Tiny", "Whisper-Base", "Whisper-Small", "Tacotron2",
                "WaveNet", "DeepSpeech", "SpeakerNet", "AudioNet"
            ])
        elif model_type == "Text Models":
            self.predict_specific_model_combo.addItems([
                "BERT-Base", "GPT-2 Small", "DistilBERT", "RoBERTa-Base",
                "ALBERT-Base", "XLNet-Base", "ELECTRA-Small", "Sentence-BERT"
            ])
        elif model_type == "Multimodal Models":
            self.predict_specific_model_combo.addItems([
                "CLIP-ViT", "CLIP-RN50", "VisualBERT", "ViLBERT",
                "LXMERT", "Oscar"
            ])
        elif model_type == "Neural Networks":
            self.predict_specific_model_combo.addItems([
                "CoreAI3D-NN-1", "CoreAI3D-NN-2", "CoreAI3D-NN-3", "CoreAI3D-NN-4"
            ])
        elif model_type == "Math Models":
            self.predict_specific_model_combo.addItems([
                "MathSolver-NN", "Optimization-Net", "Statistics-Engine", "Symbolic-Math"
            ])
        elif model_type == "Web Models":
            self.predict_specific_model_combo.addItems([
                "WebScraper-NN", "Sentiment-Analyzer", "Content-Classifier", "News-Aggregator"
            ])

    def download_fasttext_embeddings(self):
        """Download FastText embeddings file"""
        # FastText crawl-300d-2M embeddings URL
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        filename = "crawl-300d-2M.vec.zip"

        # Ensure models directory exists
        models_dir = os.path.join(self.config.data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Full path for download
        download_path = os.path.join(models_dir, filename)

        logger.info(f"Starting FastText embeddings download to: {download_path}")

        # Check if file already exists
        if os.path.exists(download_path):
            self._confirm(
                "File Exists",
                f"FastText embeddings file already exists at {download_path}.\n"
                "Do you want to overwrite it?",
                lambda: self._start_fasttext_download(url, download_path),
                on_cancelled=lambda: self.status_bar.showMessage(
                    "Download cancelled - file already exists"
                ),
            )
            return

        self._start_fasttext_download(url, download_path)

    def _start_fasttext_download(self, url: str, download_path: str) -> None:
        """Start the FastText embeddings download in a background thread."""
        try:
            # Show progress
            self.download_progress.setVisible(True)
            self.download_progress.setRange(0, 100)
            self.download_progress.setValue(0)
            self.download_status_label.setText("Downloading FastText embeddings...")

            # Start download in background thread
            download_thread = threading.Thread(
                target=self._download_fasttext_worker,
                args=(url, download_path),
                daemon=True
            )
            download_thread.start()

            self.status_bar.showMessage("FastText embeddings download started")

        except Exception as e:
            logger.error(f"Error starting FastText download: {str(e)}")
            self._notify("Download Error", f"Failed to start download: {str(e)}", level="warning")

    def _download_fasttext_worker(self, url, download_path):
        """Worker thread for FastText download"""
        try:
            logger.info(f"Downloading from {url} to {download_path}")

            # Create request with user agent
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
                }
            )

            # Open URL and get response
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                logger.info(f"Total file size: {total_size} bytes")

                downloaded = 0
                chunk_size = 8192

                with open(download_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            self._run_on_gui_thread(self._update_fasttext_progress, progress)

            # Download complete
            self._run_on_gui_thread(self._fasttext_download_success, download_path)

        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            self._run_on_gui_thread(self._fasttext_download_error, error_msg)

    def _update_fasttext_progress(self, progress):
        """Update FastText download progress"""
        try:
            self.download_progress.setValue(progress)
            self.download_status_label.setText(f"Downloading FastText embeddings... {progress}%")
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")

    def _fasttext_download_success(self, download_path):
        """Handle successful FastText download"""
        try:
            # Hide progress
            self.download_progress.setVisible(False)
            self.download_status_label.setText("FastText embeddings downloaded successfully")

            # Log success
            file_size = os.path.getsize(download_path) / (1024 * 1024)  # Size in MB
            self._log_download(f"✓ Successfully downloaded FastText embeddings ({file_size:.1f} MB)")

            # Show success message
            self._notify("Download Complete", f"FastText embeddings downloaded successfully!\n\n"
                f"File: {os.path.basename(download_path)}\n"
                f"Size: {file_size:.1f} MB\n"
                f"Location: {download_path}", level="info")

            self.status_bar.showMessage("FastText embeddings download completed")

        except Exception as e:
            logger.error(f"Error in download success handler: {str(e)}")

    def _fasttext_download_error(self, error_msg):
        """Handle FastText download error"""
        try:
            # Hide progress
            self.download_progress.setVisible(False)
            self.download_status_label.setText("FastText download failed")

            # Log error
            self._log_download(f"✗ FastText download failed: {error_msg}")

            # Show error message
            self._notify("Download Failed", f"Failed to download FastText embeddings:\n\n{error_msg}", level="warning")

            self.status_bar.showMessage("FastText download failed")

        except Exception as e:
            logger.error(f"Error in download error handler: {str(e)}")

    @qasync.asyncSlot()
    async def refresh_neural_data(self):
        """Refresh neural network data"""
        if not self.client:
            if not self.initialize_client():
                self._notify("Connection Error", "Failed to initialize client", level="warning")
                return

        try:
            # Refresh neural data directly with qasync
            topology_result, activity_result = await self._refresh_neural_data_async()
            self._handle_neural_data_result_direct(topology_result, activity_result)

        except Exception as e:
            error_msg_full = f"Error refreshing neural data: {str(e)}"
            self.topology_text.setPlainText(error_msg_full)
            if MATPLOTLIB_AVAILABLE:
                self.activity_canvas.figure.clear()
                self.activity_canvas.draw()
            else:
                self.activity_text.setPlainText(error_msg_full)

    async def _refresh_neural_data_async(self):
        """Async part of neural data refresh"""
        # Note: We don't use the global lock here since we're already inside it
        # Get topology
        topology_result = await self.client.get_neural_topology()
        # Get activity
        activity_result = await self.client.get_neural_activity()
        return topology_result, activity_result

    def _handle_neural_data_result(self, task):
        """Handle neural data refresh result"""
        try:
            topology_result, activity_result = task.result()
            self._handle_neural_data_result_direct(topology_result, activity_result)

        except Exception as e:
            error_msg_full = f"Error refreshing neural data: {str(e)}"
            self.topology_text.setPlainText(f"Neural Network Topology\n{'=' * 40}\n{error_msg_full}")
            if MATPLOTLIB_AVAILABLE:
                self.activity_canvas.figure.clear()
                self.activity_canvas.draw()
            else:
                self.activity_text.setPlainText(error_msg_full)

    def _handle_neural_data_result_direct(self, topology_result, activity_result):
        """Handle neural data refresh result directly"""
        try:
            if topology_result.success and isinstance(topology_result.data, dict):
                formatted_topology = self._format_topology_data(topology_result.data)
                self.topology_text.setPlainText(formatted_topology)
            else:
                error_msg = topology_result.data.get('error', 'Unknown error') if isinstance(topology_result.data, dict) else (topology_result.message or 'Unknown error')
                self.topology_text.setPlainText(f"Neural Network Topology\n{'=' * 40}\nError: {error_msg}")

            if activity_result.success and isinstance(activity_result.data, dict):
                activity_data = activity_result.data
                if MATPLOTLIB_AVAILABLE:
                    self.update_neural_plot(activity_data)
                formatted_activity = self._format_activity_data(activity_data)
                if not MATPLOTLIB_AVAILABLE:
                    self.activity_text.setPlainText(formatted_activity)
            else:
                error_msg = activity_result.data.get('error', 'Unknown error') if isinstance(activity_result.data, dict) else (activity_result.message or 'Unknown error')
                if MATPLOTLIB_AVAILABLE:
                    self.activity_canvas.figure.clear()
                    ax = self.activity_canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error: {error_msg}", ha='center', va='center', color='red')
                    self.activity_canvas.draw()
                else:
                    self.activity_text.setPlainText(f"Neural Network Activity\n{'=' * 40}\nError: {error_msg}")

        except Exception as e:
            error_msg_full = f"Error refreshing neural data: {str(e)}"
            self.topology_text.setPlainText(f"Neural Network Topology\n{'=' * 40}\n{error_msg_full}")
            if MATPLOTLIB_AVAILABLE:
                self.activity_canvas.figure.clear()
                self.activity_canvas.draw()
            else:
                self.activity_text.setPlainText(error_msg_full)

    def _format_topology_data(self, data):
        """Format neural network topology data for display"""
        try:
            lines = []
            lines.append("Neural Network Topology")
            lines.append("=" * 40)

            if data.get('status') == 'error':
                lines.append(f"Error: {data.get('message', 'Unknown error')}")
                return "\n".join(lines)

            input_size = data.get('inputSize', data.get('input_size'))
            output_size = data.get('outputSize', data.get('output_size'))
            layers = data.get('layers')
            neurons = data.get('neurons')
            learning_rate = data.get('learning_rate')
            num_samples = data.get('num_samples')

            if input_size is not None:
                lines.append(f"Input Layer Size:    {input_size}")
            if neurons is not None and layers is not None:
                lines.append(f"Hidden Layers:       {layers} x {neurons} neurons")
            elif layers is not None:
                lines.append(f"Hidden Layers:       {layers}")
            elif neurons is not None:
                lines.append(f"Neurons per Layer:   {neurons}")
            if output_size is not None:
                lines.append(f"Output Layer Size:   {output_size}")
            if neurons is not None:
                total_params = 0
                if input_size is not None:
                    total_params += input_size * neurons
                if output_size is not None:
                    total_params += neurons * output_size
                total_params += neurons  # biases for hidden layer
                total_params += output_size  # biases for output layer
                lines.append(f"Approx. Parameters: {total_params}")

            lines.append("")
            lines.append("Training Configuration")
            lines.append("-" * 40)
            if learning_rate is not None:
                lines.append(f"Learning Rate:       {learning_rate}")
            if num_samples is not None:
                lines.append(f"Training Samples:    {num_samples}")

            norm_range = data.get('normalization_range')
            if norm_range and isinstance(norm_range, dict):
                min_val = norm_range.get('min', 'N/A')
                max_val = norm_range.get('max', 'N/A')
                lines.append(f"Normalization Range: [{min_val}, {max_val}]")

            weights = data.get('weights')
            if weights and isinstance(weights, dict):
                lines.append("")
                lines.append("Weight Matrices")
                lines.append("-" * 40)
                for name, w in weights.items():
                    if isinstance(w, list):
                        if w and isinstance(w[0], list):
                            rows = len(w)
                            cols = len(w[0]) if rows > 0 else 0
                            lines.append(f"  {name}: {rows} x {cols}")
                        else:
                            lines.append(f"  {name}: {len(w)} values")
                    else:
                        lines.append(f"  {name}: loaded")

            return "\n".join(lines)

        except Exception as e:
            return f"Error formatting topology data: {str(e)}\n\nRaw data: {json.dumps(data, indent=2)}"

    def _format_activity_data(self, data):
        """Format neural network activity data for display"""
        try:
            lines = []
            lines.append("Neural Network Activity")
            lines.append("=" * 40)

            if data.get('status') == 'error':
                lines.append(f"Error: {data.get('message', 'Unknown error')}")
                return "\n".join(lines)

            status = data.get('status', 'unknown')
            lines.append(f"Status: {status}")

            if 'samples_processed' in data:
                lines.append(f"Samples Processed: {data['samples_processed']}")
            if 'timestamp' in data:
                try:
                    import datetime
                    ts = data['timestamp']
                    if ts > 1e15:
                        ts = ts / 1e9
                    dt = datetime.datetime.fromtimestamp(ts)
                    lines.append(f"Last Update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    lines.append(f"Timestamp: {data['timestamp']}")

            def summarize_array(arr, label, max_vals=10):
                if not isinstance(arr, list) or not arr:
                    return
                if isinstance(arr[0], list):
                    rows = len(arr)
                    cols = len(arr[0]) if rows > 0 else 0
                    lines.append(f"  {label}: {rows} x {cols} matrix")
                    flat = [v for row in arr for v in row]
                    if flat:
                        lines.append(f"    min={min(flat):.6f}  max={max(flat):.6f}  mean={sum(flat)/len(flat):.6f}")
                else:
                    lines.append(f"  {label}: {len(arr)} neurons")
                    preview = arr[:max_vals]
                    preview_str = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in preview)
                    if len(arr) > max_vals:
                        preview_str += f" ... ({len(arr) - max_vals} more)"
                    lines.append(f"    [{preview_str}]")

            for key, label in [('input', 'Input'), ('hidden', 'Hidden Activations'),
                                ('hidden_output', 'Hidden Output'), ('output', 'Output'),
                                ('results', 'Results'), ('hidden_error', 'Hidden Error')]:
                if key in data:
                    lines.append("")
                    lines.append(f"{label}:")
                    summarize_array(data[key], label)

            return "\n".join(lines)

        except Exception as e:
            return f"Error formatting activity data: {str(e)}\n\nRaw data: {json.dumps(data, indent=2)}"

    def update_neural_plot(self, activity_data):
        """Update neural network activity plot"""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.activity_canvas.figure.clear()

        if activity_data.get('status') == 'error':
            ax = self.activity_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, activity_data.get('message', 'Error'),
                    ha='center', va='center', fontsize=12, color='red')
            self.activity_canvas.draw()
            return

        datasets = []
        labels = []
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

        for key, label in [('input', 'Input'), ('hidden', 'Hidden'),
                           ('output', 'Output'), ('results', 'Results')]:
            if key in activity_data and isinstance(activity_data[key], list) and activity_data[key]:
                val = activity_data[key]
                if isinstance(val[0], list):
                    val = val[-1] if val else []
                if val and not isinstance(val[0], list):
                    try:
                        numeric = [float(v) for v in val]
                        datasets.append(numeric)
                        labels.append(label)
                    except (TypeError, ValueError):
                        pass

        if not datasets:
            ax = self.activity_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No numeric activation data to display',
                    ha='center', va='center', fontsize=10, color='gray')
            self.activity_canvas.draw()
            return

        max_len = max(len(d) for d in datasets)
        n_subplots = len(datasets)
        for i, (data, label) in enumerate(zip(datasets, labels)):
            ax = self.activity_canvas.figure.add_subplot(n_subplots, 1, i + 1)
            bars = ax.bar(range(len(data)), data, color=colors[i % len(colors)], alpha=0.8)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlim(-0.5, max_len - 0.5)
            ax.tick_params(axis='both', labelsize=7)
            if i < n_subplots - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Neuron')

        self.activity_canvas.figure.tight_layout()
        self.activity_canvas.draw()

    def _setup_websocket_connection(self):
        """Setup WebSocket connection to CoreAI3D server"""
        try:
            # Use the configured WebSocket URL directly (e.g. ws://127.0.0.1:1907)
            ws_url = self.config.ws_url
            logger.info(f"Setting up WebSocket connection to: {ws_url}")

            # Create WebSocket client with all handlers
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close,
                on_open=self._on_websocket_open
            )

            # Start WebSocket in background thread with reconnection
            self.ws_thread = threading.Thread(target=self._websocket_run_with_retry, daemon=True)
            self.ws_thread.start()

            logger.info("WebSocket connection thread started")

        except Exception as e:
            logger.error(f"Failed to setup WebSocket connection: {str(e)}")

    def _websocket_run_with_retry(self):
        """Run WebSocket connection with automatic retry"""
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            if getattr(self, '_ws_shutdown', False):
                logger.debug("WebSocket retry loop stopped (shutdown requested)")
                break
            try:
                logger.info(f"WebSocket connection attempt {attempt}/{max_retries}...")
                self.ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10
                )
                # If run_forever exits normally (clean close), don't retry
                if getattr(self, '_ws_shutdown', False):
                    break
                if attempt < max_retries:
                    logger.info(f"WebSocket closed, retrying in {retry_delay}s...")
                    import time
                    time.sleep(retry_delay)
            except KeyboardInterrupt:
                logger.info("WebSocket connection interrupted")
                break
            except Exception as e:
                logger.error(f"WebSocket connection error (attempt {attempt}): {str(e)}")
                if getattr(self, '_ws_shutdown', False):
                    break
                if attempt < max_retries:
                    import time
                    time.sleep(retry_delay)

    def _on_websocket_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("WebSocket connected")

    def _on_websocket_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'training_progress':
                # Update training progress bar and status
                percentage = data.get('percentage', 0)
                epoch = data.get('epoch', 0)
                total_epochs = data.get('total_epochs', 0)
                loss = data.get('loss', 0.0)

                # Update progress bar
                if hasattr(self, 'training_progress'):
                    self.training_progress.setValue(int(percentage))
                    self.training_progress.setVisible(True)

                # Update status label (assuming there's a status label in training tab)
                status_text = f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}"
                if hasattr(self, 'status_label'):
                    self.status_label.setText(status_text)
                elif hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(status_text)

                logger.debug(f"Training progress: {percentage}% - {status_text}")

            elif message_type == 'prediction_progress':
                # Update prediction progress bar and status
                percentage = data.get('percentage', 0)
                current = data.get('current', 0)
                total = data.get('total', 0)

                # Update progress bar
                if hasattr(self, 'prediction_progress'):
                    self.prediction_progress.setValue(int(percentage))
                    self.prediction_progress.setVisible(True)

                # Update status label for prediction
                status_text = f"Prediction Progress: {current}/{total} ({percentage:.1f}%)"
                if hasattr(self, 'prediction_status_label'):
                    self.prediction_status_label.setText(status_text)
                elif hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(status_text)

                logger.debug(f"Prediction progress: {percentage}% - {status_text}")

            elif message_type == 'chat_response':
                content = (data.get('content') or data.get('message') or
                           data.get('response') or data.get('reply') or '')
                if content:
                    import hashlib
                    with self._chat_response_lock:
                        key = (content, hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest())
                        if key not in self._shown_chat_response_keys:
                            self._shown_chat_response_keys.add(key)
                            QTimer.singleShot(0, lambda c=content: self.chat_history.append(f"AI: {c}"))
                            logger.debug(f"WebSocket chat response shown: {content[:80]}")
                        else:
                            logger.debug(f"WebSocket chat response deduplicated: {content[:80]}")
                logger.debug(f"Chat response received: {content[:100] if content else ''}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse WebSocket message: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")

    def _on_websocket_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {str(error)}")
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("WebSocket connection error")

    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("WebSocket connection closed")

def main():
    """Main application entry point"""
    try:
        import argparse
        parser = argparse.ArgumentParser(description='CoreAI3D Linux GUI Dashboard')
        parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug messages')
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger('coreai3d_client').setLevel(logging.DEBUG)
            logging.getLogger('automation_helper').setLevel(logging.DEBUG)
            logger.info("Debug messages enabled via -v flag")
        else:
            logging.getLogger().setLevel(logging.WARNING)

        app = QApplication(sys.argv)
        app.setApplicationName("CoreAI3D Linux Dashboard")
        app.setApplicationVersion("1.3.0")
        app.setOrganizationName("CoreAI3D")

        # Create the unified Qt/asyncio event loop *before* the dashboard is
        # built: qasync slots that fire during start-up need a current loop.
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        dashboard = CoreAI3DDashboard()
        dashboard.show()

        # Run the event loop and schedule client initialization
        with loop:
            # Schedule client initialization to run in the event loop using qasync
            def init_client():
                try:
                    logger.info("Running async client initialization...")
                    dashboard.initialize_client()
                    logger.info("Client initialization scheduled successfully")
                except Exception as e:
                    logger.error(f"Client initialization scheduling failed: {str(e)}")
                    logger.warning("Client initialization failed - manual retry required")

            # Use QTimer to schedule after the event loop is fully running
            QTimer.singleShot(100, init_client)
            sys.exit(loop.run_forever())

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        QMessageBox.critical(None, "Application Error", f"Failed to start application: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()