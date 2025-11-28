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

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QProgressBar, QComboBox, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QInputDialog, QMenuBar, QMenu, QStatusBar,
    QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem, QPlainTextEdit
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
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QRect,
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

# Reduce verbosity of third-party libraries
logging.getLogger('websockets.client').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    api_url: str = "http://localhost:8080/api/v1"
    ws_url: str = "ws://localhost:8081"
    api_key: str = ""
    data_dir: str = "training_data"
    sandbox_timeout: int = 300
    log_level: str = "INFO"
    enable_diagnostics: bool = True
    max_connections: int = 10
    retry_attempts: int = 3
    connection_timeout: float = 30.0

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
                'container_name': self.sandbox_name,
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

            # Pull the image if needed
            self._pull_image()

            # Start the container
            cmd = [
                'docker', 'run',
                '--name', self.sandbox_config['container_name'],
                '--rm'  # Remove container when stopped
            ]

            # Add networking option
            if self.sandbox_config.get('networking') == 'disable':
                cmd.append('--network')
                cmd.append('none')

            # Add privileged option (opposite of drop_rights)
            if not self.sandbox_config.get('privileged', False):
                cmd.append('--user')
                cmd.append('1000:1000')  # Non-root user
            elif self.sandbox_config.get('drop_rights', True):
                cmd.append('--user')
                cmd.append('1000:1000')  # Non-root user

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
                ['docker', 'ps', '--filter', f'name={self.sandbox_config["container_name"]}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True
            )

            return self.sandbox_config['container_name'] in result.stdout

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

            # Stop the container
            try:
                subprocess.run(
                    ['docker', 'stop', self.sandbox_config['container_name']],
                    capture_output=True,
                    timeout=30,
                    text=True
                )
            except subprocess.TimeoutExpired:
                logger.warning("Docker container stop timed out")
                # Force kill if needed
                try:
                    subprocess.run(
                        ['docker', 'kill', self.sandbox_config['container_name']],
                        capture_output=True,
                        timeout=10,
                        text=True
                    )
                except subprocess.TimeoutExpired:
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
                file_sizes = self.calculate_dataset_sizes(dest_path, self.datasets[dataset_name]['data_type'])

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
                    'direction': -1
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
                result = await self.client.web_search(test_data.get('query', ''))
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




class CoreAI3DDashboard(QMainWindow):
    """Main dashboard window"""

    # Chat signals for thread-safe UI updates
    chat_response_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Initialize configuration
        self.config = self._load_config()

        # Initialize qasync event loop manager
        self.async_manager = None

        # Initialize components
        self.client: Optional[CoreAI3DClient] = None
        self.sandbox_manager = SandboxManager(self.config)
        self.training_manager = TrainingDataManager(self.config)
        self.debugger: Optional[CodeDebugger] = None

        # Chat response handler
        self.chat_response_handler = None

        # Connect chat signal to handler
        self.chat_response_received.connect(self._on_chat_response_received)

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
            'loss_function': 'mse'
        }

        # Initialize UI
        self.setup_ui()
        self.setup_connections()
        self.setup_timers()

        # Load initial data
        self.load_datasets()
        self.update_system_metrics()

        # Defer client initialization until event loop is running
        self.client_initialized = False
        # Set the API key from user input
        self.config.api_key = "2956a13167"

        logger.info("CoreAI3D Dashboard initialized")

    def _load_config(self) -> DashboardConfig:
        """Load configuration from settings"""
        settings = QSettings("CoreAI3D", "Dashboard")

        config = DashboardConfig()
        config.api_url = settings.value("api_url", config.api_url)
        config.ws_url = settings.value("ws_url", config.ws_url)
        config.api_key = settings.value("api_key", config.api_key)
        config.data_dir = settings.value("data_dir", config.data_dir)
        config.sandbox_timeout = settings.value("sandbox_timeout", config.sandbox_timeout, type=int)
        config.log_level = settings.value("log_level", config.log_level)
        config.enable_diagnostics = settings.value("enable_diagnostics", config.enable_diagnostics, type=bool)

        # Force reset API and WebSocket URLs to localhost, overriding any saved QSettings
        config.api_url = "http://localhost:8080/api/v1"
        config.ws_url = "ws://localhost:8081"

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

    async def _initialize_client_async(self):
        """Async method to initialize the CoreAI3D client"""
        try:
            logger.info("Initializing CoreAI3D client...")

            # Create client instance
            self.client = CoreAI3DClient(
                api_url=self.config.api_url,
                ws_url=self.config.ws_url,
                api_key=self.config.api_key
            )

            # Register chat response handler
            self.chat_response_handler = self._handle_chat_response
            self.client.register_chat_handler(self.chat_response_handler)
            logger.info("Chat response handler registered successfully")

            # Create debugger
            self.debugger = CodeDebugger(self.client)

            # Register error handler
            self.client.register_error_handler(self._handle_chat_error)
            logger.info("Chat error handler registered successfully")

            logger.info("CoreAI3D client initialized successfully")
            self.client_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            self.client_initialized = False
            raise

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

    def setup_ui(self):
        """Setup main user interface"""
        self.setWindowTitle("CoreAI3D Linux Dashboard")
        self.setGeometry(100, 100, 1400, 900)

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

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create menu bar
        self.create_menu_bar()

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
            "centos-sandbox",
            "alpine-sandbox"
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
            QMessageBox.warning(self, "Error", f"Failed to refresh containers: {str(e)}")

    @qasync.asyncSlot()
    async def execute_linux_command(self):
        """Execute Linux command in sandbox"""
        if not self.client_initialized:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        command = self.linux_command_input.text().strip()
        if not command:
            QMessageBox.warning(self, "No Command", "Please enter a command to execute")
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
            QMessageBox.warning(self, "Error", f"Failed to execute command: {str(e)}")

    async def _execute_linux_command_async(self, container: str, command: str):
        """Async part of Linux command execution"""
        # Execute command directly in sandbox manager instead of API call
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self.sandbox_manager.execute_command, command
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
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        container = self.container_combo.currentText()

        try:
            # Start container directly with qasync
            result = await self._start_linux_container_async(container)
            self._handle_start_container_result_direct(result)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start container: {str(e)}")

    async def _start_linux_container_async(self, container: str):
        """Async part of container start"""
        # Use sandbox manager directly instead of API call
        config = {
            'image': container,  # Use container name as image
            'startup_command': 'tail -f /dev/null',
            'networking': 'enable',
            'privileged': False
        }
        success = await asyncio.get_event_loop().run_in_executor(
            None, self.sandbox_manager.start_sandbox, config
        )

        # Create result object similar to API response
        class StartResult:
            def __init__(self, success, container):
                self.success = success
                self.data = {'container': container} if success else {'error': 'Failed to start container'}

        return StartResult(success, container)

    def _handle_start_container_result_direct(self, result):
        """Handle container start result directly"""
        try:
            if result.success:
                self.linux_sandbox_status_label.setText(f"Linux Sandbox: {result.data.get('container', '')} started")
                self.status_bar.showMessage(f"Container {result.data.get('container', '')} started successfully")
            else:
                QMessageBox.warning(self, "Start Error", result.data.get('error', 'Unknown error'))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start container: {str(e)}")

    @qasync.asyncSlot()
    async def stop_linux_container(self):
        """Stop Linux container"""
        if not self.client_initialized:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        container = self.container_combo.currentText()

        try:
            # Stop container directly with qasync
            result = await self._stop_linux_container_async(container)
            self._handle_stop_container_result_direct(result)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop container: {str(e)}")

    async def _stop_linux_container_async(self, container: str):
        """Async part of container stop"""
        # Use sandbox manager directly instead of API call
        success = await asyncio.get_event_loop().run_in_executor(
            None, self.sandbox_manager.stop_sandbox
        )

        # Create result object similar to API response
        class StopResult:
            def __init__(self, success, container):
                self.success = success
                self.data = {'container': container} if success else {'error': 'Failed to stop container'}

        return StopResult(success, container)

    def _handle_stop_container_result_direct(self, result):
        """Handle container stop result directly"""
        try:
            if result.success:
                self.linux_sandbox_status_label.setText("Linux Sandbox: Container stopped")
                self.status_bar.showMessage(f"Container {result.data.get('container', '')} stopped successfully")
            else:
                QMessageBox.warning(self, "Stop Error", result.data.get('error', 'Unknown error'))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop container: {str(e)}")

    def clear_linux_output(self):
        """Clear Linux command output"""
        self.linux_output.clear()

    @qasync.asyncSlot()
    async def run_linux_training(self):
        """Run Linux training scenario"""
        if not self.client_initialized:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
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
            QMessageBox.warning(self, "Error", f"Failed to run training: {str(e)}")

    async def _run_linux_training_async(self, container: str, scenario: str):
        """Async part of Linux training"""
        # Use sandbox manager directly instead of API call
        # For training scenarios, we'll execute appropriate commands in the container
        training_commands = {
            "File System Operations": "ls -la / && echo 'File system operations completed'",
            "Process Management": "ps aux | head -10 && echo 'Process management completed'",
            "Network Operations": "ping -c 3 8.8.8.8 && echo 'Network operations completed'",
            "Package Management": "which apt && echo 'Package management available' || echo 'Package management not available'",
            "Text Processing": "echo 'Hello World' | grep Hello && echo 'Text processing completed'",
            "Permissions and Security": "id && echo 'Permissions check completed'",
            "System Monitoring": "uptime && echo 'System monitoring completed'",
            "Scripting and Automation": "echo '#!/bin/bash\necho \"Automation script\"' > test.sh && chmod +x test.sh && ./test.sh && echo 'Scripting completed'",
            "hacking-white": "echo 'White hat hacking simulation - authorized testing only' && echo 'Simulation completed'",
            "hacking-gray": "echo 'Gray hat hacking simulation - educational purposes only' && echo 'Simulation completed'",
            "hacking-black": "echo 'Black hat hacking simulation - illegal activities not permitted' && echo 'Simulation cancelled'"
        }

        command = training_commands.get(scenario, f"echo 'Unknown scenario: {scenario}'")
        success, output = await asyncio.get_event_loop().run_in_executor(
            None, self.sandbox_manager.execute_command, command
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
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        container = self.container_combo.currentText()

        try:
            # Generate report directly with qasync
            result = await self._generate_training_report_async(container)
            self._handle_training_report_result_direct(result)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate report: {str(e)}")

    async def _generate_training_report_async(self, container: str):
        """Async part of training report generation"""
        async with self.global_async_lock:
            result = await self.client.generate_training_report(container)
            return result

    def _handle_training_report_result(self, task):
        """Handle training report result"""
        try:
            result = task.result()
            if result.success:
                self.training_results.setPlainText(result.data.get('report', 'Report generated'))
                self.status_bar.showMessage("Training report generated")
            else:
                QMessageBox.warning(self, "Report Error", result.data.get('error', 'Unknown error'))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate report: {str(e)}")

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

        self.clear_btn = QPushButton("Clear")
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
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        language = self.language_combo.currentText()
        code_type = self.code_type_combo.currentText()
        complexity = self.complexity_combo.currentText()
        description = self.code_description.toPlainText().strip()

        if not description:
            QMessageBox.warning(self, "Missing Description", "Please provide a description of what you want to generate")
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
            QMessageBox.warning(self, "Error", f"Failed to generate code: {str(e)}")

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
                QMessageBox.warning(self, "Generation Failed", result.data.get('error', 'Unknown error'))
                self.status_bar.showMessage("Code generation failed")
        except Exception as e:
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            QMessageBox.warning(self, "Error", f"Failed to generate code: {str(e)}")

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
                QMessageBox.warning(self, "Generation Failed", result.data.get('error', 'Unknown error'))
                self.status_bar.showMessage("Code generation failed")
        except Exception as e:
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            QMessageBox.warning(self, "Error", f"Failed to generate code: {str(e)}")

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

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Generated Code",
            f"generated_code.{self._get_file_extension(language)}",
            file_filter
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                self.status_bar.showMessage(f"Code saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save file: {str(e)}")

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

        # Input area
        input_layout = QHBoxLayout()

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        input_layout.addWidget(self.chat_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_chat_message)
        input_layout.addWidget(send_btn)

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

        scroll_layout.addWidget(network_group)

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

        start_training_btn = QPushButton("Start Training")
        start_training_btn.clicked.connect(lambda: self.start_training())
        controls_layout.addWidget(start_training_btn)

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
            config_file, _ = QFileDialog.getSaveFileName(
                self, "Save Training Configuration",
                "", "JSON files (*.json);;All files (*.*)"
            )
            if config_file:
                with open(config_file, 'w') as f:
                    json.dump(self.learning_config, f, indent=2)
                QMessageBox.information(self, "Success", "Training configuration saved successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_training_config(self):
        """Load training configuration from file"""
        try:
            config_file, _ = QFileDialog.getOpenFileName(
                self, "Load Training Configuration",
                "", "JSON files (*.json);;All files (*.*)"
            )
            if config_file:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.learning_config.update(loaded_config)
                    self.update_ui_from_config()
                QMessageBox.information(self, "Success", "Training configuration loaded successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load configuration: {str(e)}")

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
        """Create settings configuration tab"""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Settings")

        layout = QVBoxLayout(tab)

        # API Settings
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout(api_group)

        self.api_url_edit = QLineEdit(self.config.api_url)
        api_layout.addRow("API URL:", self.api_url_edit)

        self.ws_url_edit = QLineEdit(self.config.ws_url)
        api_layout.addRow("WebSocket URL:", self.ws_url_edit)

        self.api_key_edit = QLineEdit(self.config.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addRow("API Key:", self.api_key_edit)

        layout.addWidget(api_group)

        # Data Settings
        data_group = QGroupBox("Data Configuration")
        data_layout = QFormLayout(data_group)

        self.data_dir_edit = QLineEdit(self.config.data_dir)
        data_layout.addRow("Data Directory:", self.data_dir_edit)

        browse_data_btn = QPushButton("Browse...")
        browse_data_btn.clicked.connect(self.browse_data_directory)
        data_layout.addWidget(browse_data_btn)

        layout.addWidget(data_group)

        # Sandbox Settings
        sandbox_group = QGroupBox("Sandbox Configuration")
        sandbox_layout = QFormLayout(sandbox_group)

        self.sandbox_timeout_edit = QSpinBox()
        self.sandbox_timeout_edit.setRange(60, 3600)
        self.sandbox_timeout_edit.setValue(self.config.sandbox_timeout)
        sandbox_layout.addRow("Timeout (seconds):", self.sandbox_timeout_edit)

        layout.addWidget(sandbox_group)

        # Debug Settings
        debug_group = QGroupBox("Debug Configuration")
        debug_layout = QFormLayout(debug_group)

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.config.log_level)
        debug_layout.addRow("Log Level:", self.log_level_combo)

        self.diagnostics_check = QCheckBox("Enable Diagnostics")
        self.diagnostics_check.setChecked(self.config.enable_diagnostics)
        debug_layout.addWidget(self.diagnostics_check)

        layout.addWidget(debug_group)

        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

        # Stretch to fill space
        layout.addStretch()

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
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_system_metrics)
        self.metrics_timer.start(2000)

        # Health check timer (every 30 seconds)
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.check_system_health)
        self.health_timer.start(30000)

    def initialize_client(self):
        """Initialize CoreAI3D client - schedules async initialization"""
        try:
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
            logger.warning("No running event loop, deferring client initialization")
            # This will be called again by the QTimer.singleShot in main()
            return False
        except Exception as e:
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
            logger.info("Starting client initialization...")
            logger.info(f"API URL: {self.config.api_url}")
            logger.info(f"WS URL: {self.config.ws_url}")
            logger.info(f"API Key configured: {bool(self.config.api_key)}")

            # Allow initialization even without API key for offline mode
            if not self.config.api_url:
                logger.warning("API URL not configured - client will work in offline mode")
                self.config.api_url = "http://localhost:8080/api/v1"  # Set default

            if not self.config.api_key:
                logger.warning("API key not configured - using empty key (may cause authentication failures)")
                self.config.api_key = ""  # Allow empty key

            logger.info("Creating CoreAI3DClient instance...")
            # Ensure API key is properly validated before passing to client
            api_key = self.config.api_key if self.config.api_key else ""
            logger.info(f"Passing API key to client: configured={bool(api_key)}, length={len(api_key)}")

            # Additional validation before client creation
            if not api_key:
                logger.warning("No API key configured - client will work in offline mode only")
            elif len(api_key) < 10:
                logger.warning(f"API key is very short ({len(api_key)} chars) - this may cause authentication issues")

            self.client = CoreAI3DClient({
                'base_url': self.config.api_url,
                'ws_url': self.config.ws_url,
                'api_key': api_key,
                'timeout': self.config.connection_timeout,
                'max_connections': self.config.max_connections,
                'retry_attempts': self.config.retry_attempts,
                'debug': True  # Enable debug logging
            })

            logger.info("CoreAI3DClient instance created successfully")

            # Register chat response handler
            self.chat_response_handler = self._handle_chat_response
            self.client.on('chat_response', self.chat_response_handler)
            logger.info("Chat response handler registered")

            logger.info("Creating CodeDebugger...")
            self.debugger = CodeDebugger(self.client)
            logger.info("CodeDebugger created successfully")
            logger.info("CoreAI3D client initialized successfully")

            self.client_initialized = True

        except ImportError as e:
            logger.error(f"Import error during client initialization: {str(e)}")
            logger.error("Make sure coreai3d_client.py is in the same directory or PYTHONPATH")
            raise  # Re-raise to let caller handle
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to let caller handle

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
        QMessageBox.warning(self, title, message)
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

            debug_dialog.exec()

        except Exception as e:
            logger.error(f"Error in manual connection debug: {str(e)}")
            QMessageBox.warning(self, "Debug Error", f"Failed to open debug tool: {str(e)}")

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
            QMessageBox.critical(
                self, "Connection Error",
                f"✗ Unexpected error during connection test!\n\n"
                f"Error: {str(e)}\n\n"
                "Please check the application logs for more details."
            )

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
                QMessageBox.information(
                    self, "Connection Test Successful",
                    "✓ API connection successful!\n\n"
                    "The CoreAI3D server is responding correctly.\n"
                    "All systems are operational."
                )
            else:
                error_details = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                QMessageBox.warning(
                    self, "Connection Test Failed",
                    f"✗ API connection failed!\n\n"
                    f"Error: {error_details}\n\n"
                    "Please check:\n"
                    "• Server is running\n"
                    "• Network connectivity\n"
                    "• API URL configuration"
                )
        except Exception as e:
            self.status_bar.showMessage("Connection test error")
            QMessageBox.critical(
                self, "Connection Error",
                f"✗ Unexpected error during connection test!\n\n"
                f"Error: {str(e)}\n\n"
                "Please check the application logs for more details."
            )

    def update_system_metrics(self):
        """Update system metrics display"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
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

    def calculate_health_score(self, cpu: float, memory: float, disk: float) -> int:
        """Calculate overall system health score"""
        # Simple health calculation (can be made more sophisticated)
        cpu_score = max(0, 100 - cpu)
        memory_score = max(0, 100 - memory)
        disk_score = max(0, 100 - disk)

        return int((cpu_score + memory_score + disk_score) / 3)

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
        name, ok = QInputDialog.getText(self, "New Dataset", "Dataset Name:")
        if not ok or not name:
            return

        description, ok = QInputDialog.getText(self, "New Dataset", "Description:")
        if not ok:
            description = ""

        data_types = ["Images", "Audio", "Video", "Text", "Mixed"]
        data_type, ok = QInputDialog.getItem(self, "Data Type", "Select data type:", data_types, 0, False)

        if ok:
            success = self.training_manager.create_dataset(name, description, data_type.lower())
            if not success:
                QMessageBox.warning(self, "Error", "Failed to create dataset")

    def delete_dataset(self):
        """Delete selected dataset"""
        current_item = self.dataset_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a dataset to delete")
            return

        dataset_name = current_item.text()

        reply = QMessageBox.question(
            self, "Delete Dataset",
            f"Are you sure you want to delete dataset '{dataset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                dataset_path = os.path.join(self.config.data_dir, dataset_name)
                shutil.rmtree(dataset_path)

                del self.training_manager.datasets[dataset_name]
                self.dataset_list.takeItem(self.dataset_list.row(current_item))

                self.dataset_info.clear()
                self.file_list.clear()

                self.status_bar.showMessage(f"Dataset '{dataset_name}' deleted")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete dataset: {str(e)}")

    def add_file_to_dataset(self):
        """Add file(s) or directory to selected dataset"""
        current_item = self.dataset_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a dataset first")
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
        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                self._process_selected_files(dataset_name, selected_files)

    def _select_directory_for_dataset(self, dialog, dataset_name):
        """Handle directory selection"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            # Close the file dialog
            dialog.reject()
            # Process the directory
            self._process_selected_directory(dataset_name, directory)

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
                QMessageBox.warning(self, "Partial Success",
                                  f"Added {added_count} file(s), failed to add {failed_count} file(s)")
        else:
            QMessageBox.warning(self, "Error", "Failed to add any files to dataset")

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
                    QMessageBox.warning(self, "Partial Success",
                                      f"Added {added_count} file(s), failed to add {failed_count} file(s)")
            else:
                QMessageBox.warning(self, "Error", "No files were added from the selected directory")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process directory: {str(e)}")

    def remove_file_from_dataset(self):
        """Remove file from selected dataset"""
        current_dataset = self.dataset_list.currentItem()
        current_file = self.file_list.currentItem()

        if not current_dataset or not current_file:
            QMessageBox.warning(self, "No Selection", "Please select both dataset and file")
            return

        dataset_name = current_dataset.text()
        file_name = current_file.text()

        reply = QMessageBox.question(
            self, "Remove File",
            f"Remove '{file_name}' from dataset '{dataset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
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
                QMessageBox.warning(self, "Error", f"Failed to remove file: {str(e)}")

    @qasync.asyncSlot()
    async def run_test(self):
        """Run selected test with proper qasync async slot pattern"""
        if not self.client_initialized:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
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

                result = await self.debugger.run_module_test(module_name, test_data)
                self.add_test_result(result)

            else:
                QMessageBox.information(self, "Test", f"Test type '{test_type}' not implemented yet")

            # Update status when test completes
            self.status_bar.showMessage("Test completed")

        except Exception as e:
            logger.error(f"Test execution error: {str(e)}")
            QMessageBox.warning(self, "Test Error", f"Test execution failed: {str(e)}")
            self.status_bar.showMessage("Test failed")

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
            QMessageBox.warning(self, "Docker Error", "Failed to start Docker sandbox")

    def stop_sandbox(self):
        """Stop Docker sandbox"""
        success = self.sandbox_manager.stop_sandbox()
        if not success:
            QMessageBox.warning(self, "Docker Error", "Failed to stop Docker sandbox")

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
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
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
            QMessageBox.warning(self, "Debug Error", f"API debug failed: {str(e)}")

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
                QMessageBox.warning(self, "No Data", "No debug data to export")
                return

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"api_debug_log_{timestamp}.txt"

            # Get save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export API Debug Log",
                default_filename, "Text files (*.txt);;All files (*.*)"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(debug_content)

                QMessageBox.information(self, "Export Complete",
                                      f"API debug log exported to:\n{file_path}")
                self.status_bar.showMessage(f"Debug log exported to {file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export debug log: {str(e)}")

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
            QMessageBox.warning(self, "Debug Error", f"System debug failed: {str(e)}")

    def browse_data_directory(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.data_dir_edit.setText(directory)

    def browse_data_file(self):
        """Browse for data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File",
            "", "All Files (*.*);;CSV Files (*.csv);;JSON Files (*.json);;Text Files (*.txt)"
        )
        if file_path:
            self.data_file_edit.setText(file_path)

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

            # Save to settings file
            self._save_config()

            # Reinitialize client if needed
            if self.config.api_url and self.config.api_key:
                self.initialize_client()

            self.status_bar.showMessage("Settings saved successfully")
            QMessageBox.information(self, "Settings", "Settings saved successfully")

        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save settings: {str(e)}")

    def refresh_all(self):
        """Refresh all data and displays"""
        self.update_system_metrics()
        self.update_system_info()
        self.update_process_list()
        self.load_datasets()
        self.status_bar.showMessage("All data refreshed")

    def show_documentation(self):
        """Show application documentation"""
        QMessageBox.information(
            self, "Documentation",
            "For detailed documentation, please see:\n\n"
            "README.md\n\n"
            "This file contains comprehensive usage instructions, "
            "configuration details, and troubleshooting information."
        )

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
            QMessageBox.warning(self, "Connection Error", "Client not connected. Please wait for initialization or check server status.")
            return

        message = self.chat_input.text().strip()
        if not message:
            return

        # Add user message to history
        self.chat_history.append(f"You: {message}")
        self.chat_input.clear()

        # Add debug logging
        logger.info(f"Sending chat message: {message}")
        logger.info(f"Client connected: {self.client.is_connected() if self.client else False}")

        try:
            # Send message directly with qasync
            asyncio.create_task(self._send_chat_message_async(message))

        except Exception as e:
            logger.error(f"Exception in send_chat_message: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to send message: {str(e)}")

    async def _send_chat_message_async(self, message: str):
        """Async part of chat message sending"""
        async with self.global_async_lock:
            logger.info("Making HTTP request to send chat message")
            result = await self.client.send_chat_message(message)
            logger.info(f"HTTP response received: success={result.success}, status={result.status_code}")
            if result.data:
                logger.info(f"Response data: {result.data}")
            return result

    def _handle_chat_message_result(self, task):
        """Handle chat message result"""
        try:
            result = task.result()
            if result.success:
                # HTTP request succeeded, but actual response comes via WebSocket
                # The response will be handled by the registered chat response handler
                pass
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                logger.error(f"HTTP request failed: {error_msg}")
                self.chat_history.append(f"Error: {error_msg}")
        except Exception as e:
            logger.error(f"Exception in chat message handler: {str(e)}")
            self.chat_history.append(f"Error: {str(e)}")

    def _handle_chat_response(self, data: Dict[str, Any]):
        """Handle chat response from WebSocket"""
        logger.info(f"Handling chat response: {data}")
        content = data.get('content', '')
        if content:
            logger.info(f"Emitting chat response signal: {content}")
            self.chat_response_received.emit(content)
        else:
            logger.warning("Received chat response with empty content")

    def _on_chat_response_received(self, response: str):
        """Handle chat response signal and update UI"""
        logger.info(f"Chat response received via signal: {response}")
        # Use QTimer to ensure UI update happens in main thread
        QTimer.singleShot(0, lambda: self.chat_history.append(f"AI: {response}"))

    def _handle_chat_error(self, error_msg: str):
        """Handle chat error"""
        logger.error(f"Chat error: {error_msg}")
        self.chat_history.append(f"Error: {error_msg}")

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

    def search_models(self):
        """Search models based on input criteria"""
        try:
            # Reset to first page when searching
            self.current_page = 1
            self.discover_models_async()
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")

    def refresh_models(self):
        """Refresh the model list"""
        try:
            self.current_page = 1
            self.discover_models_async()
        except Exception as e:
            logger.error(f"Error refreshing models: {str(e)}")
            QMessageBox.warning(self, "Refresh Error", f"Failed to refresh models: {str(e)}")

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
            QMessageBox.warning(self, "Refresh Error", f"Failed to refresh models: {str(e)}")

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
            QMessageBox.warning(self, "No Selection", "Please select a model to download")
            return

        model_name = self.model_table.item(current_row, 0).text()
        model_type = self.model_table.item(current_row, 1).text()
        model_status = self.model_table.item(current_row, 4).text()

        logger.info(f"Selected model: {model_name}, type: {model_type}, status: {model_status}")

        if model_status == "Downloading":
            QMessageBox.information(self, "Already Downloading", f"{model_name} is already being downloaded")
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
            self.download_timer = QTimer()
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
                                QTimer.singleShot(0, lambda p=progress: self._update_download_progress(row, model_name, p))

                    # Complete download
                    QTimer.singleShot(0, lambda: self._update_download_complete(row, model_name))

                except Exception as e:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name, str(e)))
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
            QMessageBox.warning(self, "No Selection", "Please select a downloading model to cancel")
            return

        model_name = self.model_table.item(current_row, 0).text()
        model_status = self.model_table.item(current_row, 4).text()

        logger.info(f"Cancelling download for {model_name}, status: {model_status}")

        if model_status != "Downloading":
            QMessageBox.warning(self, "Not Downloading", f"{model_name} is not currently downloading")
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
                            QTimer.singleShot(0, lambda p=progress: self._update_kaggle_progress(row, model_name, p))
                        else:
                            # File not created yet, show initial progress
                            elapsed = time.time() - self.download_progress_data['start_time']
                            if elapsed < 5:  # First 5 seconds
                                progress = min(10, int((elapsed / 5.0) * 10))
                                QTimer.singleShot(0, lambda p=progress: self._update_kaggle_progress(row, model_name, p))

                    # Check final result
                    if process.returncode == 0:
                        # Complete the progress
                        QTimer.singleShot(0, lambda: self._update_kaggle_progress(row, model_name, 100))
                        QTimer.singleShot(0, lambda: self._download_success(row, model_name))
                    else:
                        stderr = process.stderr.read() if process.stderr else ""
                        error_msg = f"Kaggle download failed: {stderr}"
                        QTimer.singleShot(0, lambda: self._download_error(row, model_name, error_msg))

                    # Clean up progress data
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

                except Exception as e:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name, str(e)))
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
                            QTimer.singleShot(0, lambda p=progress: self._update_hf_progress(row, model_name, p))

                    # Start download with progress tracking
                    QTimer.singleShot(0, lambda: self._update_hf_progress(row, model_name, 0))

                    # Start download - HuggingFace Hub doesn't have built-in progress callbacks
                    # We'll monitor file system changes for progress
                    if model_id:
                        snapshot_download(
                            repo_id=model_id,
                            local_dir=models_dir,
                            local_dir_use_symlinks=False
                        )
                    else:
                        QTimer.singleShot(0, lambda: self._download_error(row, model_name, "Invalid model ID"))
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
                    QTimer.singleShot(0, lambda: self._update_hf_progress(row, model_name, 100))
                    QTimer.singleShot(0, lambda: self._download_success(row, model_name))

                    # Clean up progress data
                    if hasattr(self, 'download_progress_data'):
                        delattr(self, 'download_progress_data')

                except ImportError:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name,
                        "HuggingFace Hub not installed. Install with: pip install huggingface-hub"))
                except Exception as e:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name, str(e)))
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

            QMessageBox.warning(self, "Download Error", f"Failed to download {model_name}:\n{error_msg}")

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
                        QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                        return

                # Get selected dataset
                current_item = self.dataset_list.currentItem()
                if not current_item:
                    QMessageBox.warning(self, "No Dataset Selected", "Please select a dataset to load")
                    return

                dataset_name = current_item.text()
                if dataset_name not in self.training_manager.datasets:
                    QMessageBox.warning(self, "Dataset Not Found", f"Dataset '{dataset_name}' not found")
                    return

                dataset_info = self.training_manager.datasets[dataset_name]
                if not dataset_info.get('files'):
                    QMessageBox.warning(self, "Empty Dataset", f"Dataset '{dataset_name}' has no files")
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
                    QMessageBox.warning(self, "No Supported File", f"Dataset '{dataset_name}' has no supported file formats.\nSupported formats: {', '.join(supported_extensions)}")
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
                    self.training_button.setEnabled(True)

                # Mark dataset as loaded if successful
                if result.success:
                    self.current_dataset_loaded = True

            except asyncio.CancelledError:
                logger.info("Dataset loading cancelled")
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading cancelled")
                raise
            except Exception as e:
                self.training_progress.setVisible(False)
                self.training_progress.setValue(0)
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading error: {str(e)}")
                QMessageBox.warning(self, "Dataset Loading Error", str(e))
            finally:
                # Clean up task
                with self.task_lock:
                    self.active_tasks.discard(task_id)

                # Remove from pending tasks
                async with self.task_cleanup_lock:
                    self.pending_tasks.discard(self.current_async_task)
                    self.current_async_task = None

    async def _load_dataset_async(self, data_file: str, dataset_name: str):
        """Async part of dataset loading"""
        # Validate parameters before calling API
        output_size = self.learning_config.get('output_size', 1)
        has_header = self.learning_config.get('has_header', True)
        contains_text = self.learning_config.get('contains_text', False)
        min_range = self.learning_config.get('normalization_min', -1.0)
        max_range = self.learning_config.get('normalization_max', 1.0)

        # Basic validation
        if output_size < 1:
            raise ValueError("Output size must be at least 1")
        if min_range >= max_range:
            raise ValueError("Normalization min must be less than max")

        # Determine file format for appropriate API call
        file_ext = os.path.splitext(data_file.lower())[1]

        # Call appropriate API endpoint based on file format
        if file_ext == '.csv':
            # Use existing CSV loading logic
            return await self.client.load_dataset(
                filename=data_file,
                output_size=output_size,
                has_header=has_header,
                contains_text=contains_text,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range
            )
        elif file_ext in ['.json', '.xml', '.txt']:
            # For text-based formats, use generic data loading
            return await self.client.load_dataset(
                filename=data_file,
                output_size=output_size,
                has_header=has_header,
                contains_text=contains_text,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range
            )
        elif file_ext in ['.pdf', '.docx', '.doc']:
            # For document formats, use document processing
            return await self.client.load_dataset(
                filename=data_file,
                output_size=output_size,
                has_header=False,  # Documents typically don't have headers
                contains_text=True,  # Documents contain text
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range
            )
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            # For image formats, use vision processing
            return await self.client.load_dataset(
                filename=data_file,
                output_size=output_size,
                has_header=False,
                contains_text=False,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range
            )
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.flac', '.aac']:
            # For media formats, use audio/video processing
            return await self.client.load_dataset(
                filename=data_file,
                output_size=output_size,
                has_header=False,
                contains_text=False,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range
            )
        else:
            # Fallback for unknown formats
            return await self.client.load_dataset(
                filename=data_file,
                output_size=output_size,
                has_header=has_header,
                contains_text=contains_text,
                dataset_name=dataset_name,
                min_range=min_range,
                max_range=max_range
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

                QMessageBox.information(self, "Dataset Loaded",
                                    f"Dataset loaded successfully!\n\n"
                                    f"Samples: {data.get('samples', 'N/A')}\n"
                                    f"Input Size: {data.get('input_size', 'N/A')}\n"
                                    f"Output Size: {data.get('output_size', 'N/A')}")

            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading failed: {error_msg}")
                QMessageBox.warning(self, "Dataset Loading Failed", error_msg)
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loading error: {str(e)}")
            QMessageBox.warning(self, "Dataset Loading Error", str(e))

    @qasync.asyncSlot()
    async def start_training(self):
        """Start training with current configuration and proper locking"""
        task_id = f"training_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Check if training already in progress (double-check with lock)
                if self.training_in_progress:
                    QMessageBox.warning(self, "Training in Progress",
                                      "Training is already running. Please wait for it to complete or stop it first.")
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
                        QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                        return

                # Check if dataset is loaded first
                if not hasattr(self, 'current_dataset_loaded') or not self.current_dataset_loaded:
                    QMessageBox.warning(self, "No Dataset Loaded",
                                      "Please upload and load a dataset first using the 'Upload CSV' or 'Load Dataset' button")
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
                QMessageBox.warning(self, "Training Error", str(e))
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
                    
    def closeEvent(self, event):
        """Handle application close event with proper async cleanup"""
        try:
            logger.info("Starting application shutdown sequence")

            # Cancel all active tasks first
            self.cancel_all_tasks()

            # Stop health check timer to prevent new tasks during shutdown
            if hasattr(self, 'health_timer') and self.health_timer.isActive():
                self.health_timer.stop()
                logger.debug("Health check timer stopped during shutdown")

            # Stop training if running (using global async lock)
            async def stop_training_on_shutdown():
                async with self.global_async_lock:
                    if self.training_in_progress:
                        logger.info("Stopping training during shutdown")
                        self.training_in_progress = False

            # Schedule training stop in event loop
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(stop_training_on_shutdown())

            # Stop sandbox if running
            if hasattr(self, 'sandbox_manager') and self.sandbox_manager.is_running:
                logger.info("Stopping sandbox manager")
                self.sandbox_manager.stop_sandbox()

            # Close client connection using the existing qasync event loop
            if hasattr(self, 'client_initialized') and self.client_initialized and self.client:
                logger.info("Closing client connection")
                try:
                    # Use the existing qasync event loop instead of creating a new one
                    if hasattr(self.client, 'close'):
                        # Schedule the close operation in the current event loop
                        async def close_client():
                            await self.client.close()

                        # Use qasync-compatible approach for closing client
                        # Schedule the close operation using asyncio.create_task() for qasync compatibility
                        asyncio.create_task(close_client())
                except Exception as e:
                    logger.error(f"Error closing client: {str(e)}")

            # Save configuration
            self._save_config()

            logger.info("CoreAI3D Dashboard closed successfully")
            event.accept()

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            event.accept()

    async def _start_training_async(self):
        """Async part of training start"""
        # Note: We don't use the global lock here since we're already inside it
        # Call training with consolidated configuration
        result = await self.client.start_training(self.learning_config)
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
                QMessageBox.information(self, "Training Complete", "Training completed successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training failed: {error_msg}")
                QMessageBox.warning(self, "Training Failed", error_msg)
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)  # Reset progress on error
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training error: {str(e)}")
            QMessageBox.warning(self, "Training Error", str(e))

    def _handle_training_result_direct(self, result):
        """Handle training result directly"""
        try:
            self.training_progress.setVisible(False)
            self.training_progress.setValue(100)  # Ensure progress shows complete
            if result.success:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed successfully")
                if result.data:
                    self.training_log.appendPlainText(f"Training results: {json.dumps(result.data, indent=2)}")
                QMessageBox.information(self, "Training Complete", "Training completed successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training failed: {error_msg}")
                QMessageBox.warning(self, "Training Failed", error_msg)
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)  # Reset progress on error
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training error: {str(e)}")
            QMessageBox.warning(self, "Training Error", str(e))

    @qasync.asyncSlot()
    async def stop_training(self):
        """Stop current training with proper locking"""
        task_id = f"stop_training_{id(asyncio.current_task())}"

        # Use global async lock to prevent concurrent operations
        async with self.global_async_lock:
            try:
                # Check if training is actually running (double-check with lock)
                if not self.training_in_progress:
                    QMessageBox.information(self, "No Training", "No training is currently running")
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
                    QMessageBox.warning(self, "Connection Error", "No active client connection")
                    return

                # Stop training directly with qasync
                result = await self._stop_training_async()
                self._handle_stop_training_result_direct(result)

            except asyncio.CancelledError:
                logger.info("Stop training cancelled")
                raise  # Re-raise to ensure proper cleanup
            except Exception as e:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {str(e)}")
                QMessageBox.warning(self, "Stop Error", str(e))
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
                QMessageBox.information(self, "Training Stopped", "Training stopped successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to stop training: {error_msg}")
                QMessageBox.warning(self, "Stop Failed", error_msg)
        except Exception as e:
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {str(e)}")
            QMessageBox.warning(self, "Stop Error", str(e))

    def _handle_stop_training_result_direct(self, result):
        """Handle stop training result directly"""
        try:
            if result.success:
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training stopped successfully")
                self.training_progress.setVisible(False)
                QMessageBox.information(self, "Training Stopped", "Training stopped successfully")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to stop training: {error_msg}")
                QMessageBox.warning(self, "Stop Failed", error_msg)
        except Exception as e:
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {str(e)}")
            QMessageBox.warning(self, "Stop Error", str(e))

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
                        QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
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

                # Open file dialog to select file
                file_path, _ = QFileDialog.getOpenFileName(
                    self, f"Select {file_type.upper()} File",
                    "", filter_text
                )

                if not file_path:
                    return  # User cancelled

                # Validate file exists and is readable
                if not os.path.exists(file_path):
                    QMessageBox.warning(self, "File Not Found", f"File does not exist: {file_path}")
                    return

                if not os.access(file_path, os.R_OK):
                    QMessageBox.warning(self, "Permission Denied", f"Cannot read file: {file_path}")
                    return

                # Get dataset name from user
                dataset_name, ok = QInputDialog.getText(
                    self, "Dataset Name",
                    f"Enter a name for this {file_type.upper()} dataset:",
                    text=os.path.splitext(os.path.basename(file_path))[0]
                )

                if not ok or not dataset_name.strip():
                    return  # User cancelled

                dataset_name = dataset_name.strip()

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

                    # Load the uploaded dataset
                    load_result = await self._load_dataset_async(file_path, dataset_name)
                    self._handle_load_dataset_result_direct(load_result, file_path)

                    if load_result.success:
                        self.current_dataset_loaded = True
                        self.training_button.setEnabled(True)

            except asyncio.CancelledError:
                logger.info("File upload cancelled")
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] File upload cancelled")
                raise
            except Exception as e:
                self.training_progress.setVisible(False)
                self.training_progress.setValue(0)
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] File upload error: {str(e)}")
                QMessageBox.warning(self, "Upload Error", str(e))
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

                QMessageBox.information(self, "Upload Successful",
                                      f"{format_desc} uploaded and validated successfully!\n\n"
                                      f"File: {os.path.basename(data.get('file_path', 'Unknown'))}\n"
                                      f"Dataset: {data.get('dataset_name', 'Unknown')}\n"
                                      f"Size: {data.get('file_size', 0) / 1024:.1f} KB")
            else:
                error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] File upload failed: {error_msg}")
                QMessageBox.warning(self, "Upload Failed", error_msg)
        except Exception as e:
            self.training_progress.setVisible(False)
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Upload result handling error: {str(e)}")
            QMessageBox.warning(self, "Upload Error", str(e))

    def clear_training_log(self):
        """Clear training log"""
        self.training_log.clear()

    def browse_predict_input_file(self):
        """Browse for prediction input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File",
            "", "All Files (*.*)"
        )
        if file_path:
            self.predict_file_path_edit.setText(file_path)

    @qasync.asyncSlot()
    async def run_prediction(self):
        """Run prediction with selected model and input"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        model_type = self.predict_model_combo.currentText()
        specific_model = self.predict_specific_model_combo.currentText()
        input_type = self.predict_input_type_combo.currentText()

        if model_type == "Select Model...":
            QMessageBox.warning(self, "No Model Selected", "Please select a model type")
            return

        if specific_model == "Select Specific Model...":
            QMessageBox.warning(self, "No Model Selected", "Please select a specific model")
            return

        # Get input data based on type
        input_data = None
        parsed_csv_data = None

        if input_type == "Text":
            input_data = self.predict_input_text.toPlainText().strip()
            if not input_data:
                QMessageBox.warning(self, "No Input", "Please enter text input")
                return
        elif input_type in ["Image", "Audio"]:
            input_data = self.predict_file_path_edit.text().strip()
            if not input_data:
                QMessageBox.warning(self, "No Input", "Please select a file")
                return
        elif input_type == "File":
            file_path = self.predict_file_path_edit.text().strip()
            if not file_path:
                QMessageBox.warning(self, "No Input", "Please select a file")
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
                    QMessageBox.warning(self, "CSV Parse Error", f"Failed to parse CSV file: {parsed_csv_data['error']}")
                    return
                input_data = parsed_csv_data

                self.predict_progress.setVisible(False)
                self.status_bar.showMessage("CSV file parsed successfully")
            else:
                input_data = file_path
        elif input_type == "URL":
            input_data = self.predict_file_path_edit.text().strip()
            if not input_data:
                QMessageBox.warning(self, "No Input", "Please enter a URL")
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
            result = await self.client.web_search(input_data)
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
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        model_identifier = self.load_model_identifier_edit.text().strip()
        if not model_identifier:
            QMessageBox.warning(self, "No Identifier", "Please enter a model identifier (dataset ID or name)")
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

        # Show/hide automation config group based on test type
        if test_type == "System Automation":
            self.automation_config_group.setVisible(True)
            self.performance_config_group.setVisible(False)
        elif test_type == "Performance Test":
            self.automation_config_group.setVisible(False)
            self.performance_config_group.setVisible(True)
        else:
            self.automation_config_group.setVisible(False)
            self.performance_config_group.setVisible(False)

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
        try:
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
                reply = QMessageBox.question(
                    self, "File Exists",
                    f"FastText embeddings file already exists at {download_path}.\nDo you want to overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.status_bar.showMessage("Download cancelled - file already exists")
                    return

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
            QMessageBox.warning(self, "Download Error", f"Failed to start download: {str(e)}")

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
                            QTimer.singleShot(0, lambda p=progress: self._update_fasttext_progress(p))

            # Download complete
            QTimer.singleShot(0, lambda: self._fasttext_download_success(download_path))

        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            QTimer.singleShot(0, lambda msg=error_msg: self._fasttext_download_error(msg))

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
            QMessageBox.information(
                self, "Download Complete",
                f"FastText embeddings downloaded successfully!\n\n"
                f"File: {os.path.basename(download_path)}\n"
                f"Size: {file_size:.1f} MB\n"
                f"Location: {download_path}"
            )

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
            QMessageBox.warning(
                self, "Download Failed",
                f"Failed to download FastText embeddings:\n\n{error_msg}"
            )

            self.status_bar.showMessage("FastText download failed")

        except Exception as e:
            logger.error(f"Error in download error handler: {str(e)}")

    @qasync.asyncSlot()
    async def refresh_neural_data(self):
        """Refresh neural network data"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
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

            # Handle topology
            if topology_result.success:
                topology_data = topology_result.data
                if isinstance(topology_data, dict) and topology_data.get('status') == 'success':
                    # Format topology data nicely
                    formatted_topology = self._format_topology_data(topology_data)
                    self.topology_text.setPlainText(formatted_topology)
                else:
                    topology = json.dumps(topology_data, indent=2)
                    self.topology_text.setPlainText(topology)
            else:
                error_msg = topology_result.data.get('error', 'Unknown error') if isinstance(topology_result.data, dict) else 'Unknown error'
                self.topology_text.setPlainText(f"Error: {error_msg}")

            # Handle activity
            if activity_result.success:
                activity_data = activity_result.data
                if isinstance(activity_data, dict) and activity_data.get('status') == 'active':
                    # Format activity data nicely
                    formatted_activity = self._format_activity_data(activity_data)
                    if MATPLOTLIB_AVAILABLE:
                        self.update_neural_plot(activity_data)
                    else:
                        self.activity_text.setPlainText(formatted_activity)
                else:
                    if MATPLOTLIB_AVAILABLE:
                        self.update_neural_plot(activity_data)
                    else:
                        activity = json.dumps(activity_data, indent=2)
                        self.activity_text.setPlainText(activity)
            else:
                error_msg = activity_result.data.get('error', 'Unknown error') if isinstance(activity_result.data, dict) else 'Unknown error'
                if MATPLOTLIB_AVAILABLE:
                    self.activity_canvas.figure.clear()
                    self.activity_canvas.draw()
                else:
                    self.activity_text.setPlainText(f"Error: {error_msg}")

        except Exception as e:
            error_msg_full = f"Error refreshing neural data: {str(e)}"
            self.topology_text.setPlainText(error_msg_full)
            if MATPLOTLIB_AVAILABLE:
                self.activity_canvas.figure.clear()
                self.activity_canvas.draw()
            else:
                self.activity_text.setPlainText(error_msg_full)

    def _handle_neural_data_result_direct(self, topology_result, activity_result):
        """Handle neural data refresh result directly"""
        try:
            # Handle topology
            if topology_result.success:
                topology_data = topology_result.data
                if isinstance(topology_data, dict) and topology_data.get('status') == 'success':
                    # Format topology data nicely
                    formatted_topology = self._format_topology_data(topology_data)
                    self.topology_text.setPlainText(formatted_topology)
                else:
                    topology = json.dumps(topology_data, indent=2)
                    self.topology_text.setPlainText(topology)
            else:
                error_msg = topology_result.data.get('error', 'Unknown error') if isinstance(topology_result.data, dict) else 'Unknown error'
                self.topology_text.setPlainText(f"Error: {error_msg}")

            # Handle activity
            if activity_result.success:
                activity_data = activity_result.data
                if isinstance(activity_data, dict) and activity_data.get('status') == 'active':
                    # Format activity data nicely
                    formatted_activity = self._format_activity_data(activity_data)
                    if MATPLOTLIB_AVAILABLE:
                        self.update_neural_plot(activity_data)
                    else:
                        self.activity_text.setPlainText(formatted_activity)
                else:
                    if MATPLOTLIB_AVAILABLE:
                        self.update_neural_plot(activity_data)
                    else:
                        activity = json.dumps(activity_data, indent=2)
                        self.activity_text.setPlainText(activity)
            else:
                error_msg = activity_result.data.get('error', 'Unknown error') if isinstance(activity_result.data, dict) else 'Unknown error'
                if MATPLOTLIB_AVAILABLE:
                    self.activity_canvas.figure.clear()
                    self.activity_canvas.draw()
                else:
                    self.activity_text.setPlainText(f"Error: {error_msg}")

        except Exception as e:
            error_msg_full = f"Error refreshing neural data: {str(e)}"
            self.topology_text.setPlainText(error_msg_full)
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
            lines.append("=" * 30)

            if 'layers' in data:
                lines.append(f"Layers: {data['layers']}")
            if 'neurons' in data:
                lines.append(f"Neurons per Layer: {data['neurons']}")
            if 'input_size' in data:
                lines.append(f"Input Size: {data['input_size']}")
            if 'output_size' in data:
                lines.append(f"Output Size: {data['output_size']}")
            if 'learning_rate' in data:
                lines.append(f"Learning Rate: {data['learning_rate']}")
            if 'num_samples' in data:
                lines.append(f"Training Samples: {data['num_samples']}")

            if 'normalization_range' in data:
                norm_range = data['normalization_range']
                if isinstance(norm_range, dict):
                    min_val = norm_range.get('min', 'N/A')
                    max_val = norm_range.get('max', 'N/A')
                    lines.append(f"Normalization Range: [{min_val}, {max_val}]")

            if 'core_initialized' in data:
                status = "Yes" if data['core_initialized'] else "No"
                lines.append(f"Core Initialized: {status}")

            if 'network_structure' in data:
                struct = data['network_structure']
                if isinstance(struct, dict):
                    lines.append("")
                    lines.append("Network Structure:")
                    if 'input_layer' in struct:
                        lines.append(f"  Input Layer: {struct['input_layer']} neurons")
                    if 'hidden_layers' in struct:
                        lines.append(f"  Hidden Layers: {struct['hidden_layers']}")
                    if 'output_layer' in struct:
                        lines.append(f"  Output Layer: {struct['output_layer']} neurons")

            return "\n".join(lines)

        except Exception as e:
            return f"Error formatting topology data: {str(e)}\n\nRaw data: {json.dumps(data, indent=2)}"

    def _format_activity_data(self, data):
        """Format neural network activity data for display"""
        try:
            lines = []
            lines.append("Neural Network Activity")
            lines.append("=" * 30)

            if 'status' in data:
                lines.append(f"Status: {data['status']}")
            if 'samples_processed' in data:
                lines.append(f"Samples Processed: {data['samples_processed']}")
            if 'timestamp' in data:
                # Convert timestamp to readable format
                try:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(data['timestamp'] / 1000000000)  # nanoseconds to seconds
                    lines.append(f"Last Update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    lines.append(f"Timestamp: {data['timestamp']}")

            # Show current metrics if available
            if 'current_metrics' in data:
                metrics = data['current_metrics']
                if isinstance(metrics, dict):
                    lines.append("")
                    lines.append("Current Metrics:")
                    for key, value in metrics.items():
                        lines.append(f"  {key}: {value}")

            # Show recent activity if available
            if 'recent_activity' in data:
                activity = data['recent_activity']
                if isinstance(activity, dict):
                    lines.append("")
                    lines.append("Recent Activity:")
                    for key, value in activity.items():
                        lines.append(f"  {key}: {value}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error formatting activity data: {str(e)}\n\nRaw data: {json.dumps(data, indent=2)}"

    def update_neural_plot(self, activity_data):
        """Update neural network activity plot"""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.activity_canvas.figure.clear()
        ax = self.activity_canvas.figure.add_subplot(111)

        # Simple bar plot of activations (placeholder)
        if 'activations' in activity_data:
            activations = activity_data['activations']
            if isinstance(activations, list) and activations:
                ax.bar(range(len(activations)), activations)
                ax.set_xlabel('Neuron')
                ax.set_ylabel('Activation')
                ax.set_title('Neural Network Activity')

        self.activity_canvas.draw()

    # Training parameter change handlers
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

    def cancel_all_tasks(self):
        """Cancel all pending async tasks"""
        try:
            async def cancel_tasks():
                async with self.task_cleanup_lock:
                    tasks_to_cancel = list(self.pending_tasks)
                    self.pending_tasks.clear()

                    for task in tasks_to_cancel:
                        if task and not task.done():
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

        except Exception as e:
            logger.error(f"Error cancelling tasks: {str(e)}")

    def closeEvent(self, event):
        """Handle application close event with proper async cleanup"""
        try:
            logger.info("Starting application shutdown sequence")

            # Cancel all active tasks first
            self.cancel_all_tasks()

            # Stop health check timer to prevent new tasks during shutdown
            if hasattr(self, 'health_timer') and self.health_timer.isActive():
                self.health_timer.stop()
                logger.debug("Health check timer stopped during shutdown")

            # Stop training if running (using global async lock)
            async def stop_training_on_shutdown():
                async with self.global_async_lock:
                    if self.training_in_progress:
                        logger.info("Stopping training during shutdown")
                        self.training_in_progress = False

            # Schedule training stop in event loop
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(stop_training_on_shutdown())

            # Stop sandbox if running
            if hasattr(self, 'sandbox_manager') and self.sandbox_manager.is_running:
                logger.info("Stopping sandbox manager")
                self.sandbox_manager.stop_sandbox()

            # Close client connection using the existing qasync event loop
            if hasattr(self, 'client_initialized') and self.client_initialized and self.client:
                logger.info("Closing client connection")
                try:
                    # Use the existing qasync event loop instead of creating a new one
                    if hasattr(self.client, 'close'):
                        # Schedule the close operation in the current event loop
                        async def close_client():
                            await self.client.close()

                        # Use qasync-compatible approach for closing client
                        # Schedule the close operation using asyncio.create_task() for qasync compatibility
                        asyncio.create_task(close_client())
                except Exception as e:
                    logger.error(f"Error closing client: {str(e)}")

            # Save configuration
            self._save_config()

            logger.info("CoreAI3D Dashboard closed successfully")
            event.accept()

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            event.accept()

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

        dashboard = CoreAI3DDashboard()
        dashboard.show()

        # Create unified Qt/asyncio event loop
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        # Run the event loop and schedule client initialization
        with loop:
            # Schedule client initialization to run in the event loop using qasync
            def init_client():
                try:
                    logger.info("Running async client initialization...")
                    # Schedule the coroutine in the existing event loop
                    asyncio.create_task(dashboard._initialize_client_async())
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