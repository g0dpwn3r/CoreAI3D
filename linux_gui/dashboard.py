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
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
from coreai3d_client import CoreAI3DClient
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
    api_url: str = "http://0.0.0.0:8080/api/v1"
    ws_url: str = "ws://0.0.0.0:8081"
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
        self.sandbox_name = "CoreAI3D_Sandbox"
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
                'startup_command': config.get('startup_command', 'echo "Container started"'),
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
                'metadata': {}
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

    def add_file_to_dataset(self, dataset_name: str, file_path: str) -> bool:
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

            # Update dataset metadata
            self.datasets[dataset_name]['files'].append({
                'name': file_name,
                'path': dest_path,
                'added_at': datetime.now().isoformat(),
                'size': os.path.getsize(dest_path)
            })

            self._save_dataset_metadata(dataset_name)
            self.file_added.emit(dataset_name, file_name)

            logger.info(f"Added file {file_name} to dataset {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding file to dataset: {str(e)}")
            return False

    def _save_dataset_metadata(self, dataset_name: str):
        """Save dataset metadata to file"""
        try:
            dataset_path = os.path.join(self.config.data_dir, dataset_name)
            metadata_path = os.path.join(dataset_path, 'metadata.json')

            with open(metadata_path, 'w') as f:
                json.dump(self.datasets[dataset_name], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dataset metadata: {str(e)}")

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
                result = await self.client.solve_math(test_data.get('expression', ''))
                success = result.success
                message = "Math solution completed" if success else "Math solution failed"

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

class AsyncTaskThread(QThread):
    def __init__(self, coro, callback=None, error_callback=None):
        super().__init__()
        self.coro = coro
        self.callback = callback
        self.error_callback = error_callback
        self._finished = False

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.coro)
            loop.close()
            if self.callback:
                QTimer.singleShot(0, lambda: self.callback(result))
        except Exception as e:
            error_msg = f"Async task failed: {str(e)}"
            logger.error(error_msg)
            if self.error_callback:
                QTimer.singleShot(0, lambda: self.error_callback(str(e)))
        finally:
            self._finished = True

class AsyncTaskManager(QObject):
    """Manages async tasks with proper Qt integration"""

    task_completed = pyqtSignal(object)
    task_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.active_threads = []
        self._shutdown = False

    def run_async_task(self, coro, callback=None, error_callback=None):
        """Run an async coroutine safely with Qt integration"""
        if self._shutdown:
            logger.warning("AsyncTaskManager is shutting down, ignoring new task")
            return

        if not asyncio.iscoroutine(coro):
            coro = self._make_coroutine(coro)

        thread = AsyncTaskThread(coro, callback, error_callback)
        self.active_threads.append(thread)

        # Remove thread from active list when finished
        def cleanup_thread():
            if thread in self.active_threads:
                self.active_threads.remove(thread)

        thread.finished.connect(cleanup_thread)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _make_coroutine(self, func):
        async def wrapper():
            return func()
        return wrapper

    def shutdown(self):
        """Shutdown the task manager and wait for active threads"""
        logger.info(f"Shutting down AsyncTaskManager with {len(self.active_threads)} active threads")
        self._shutdown = True

        # Wait for all active threads to finish
        for thread in self.active_threads[:]:  # Copy list to avoid modification during iteration
            if thread.isRunning():
                logger.info("Waiting for thread to finish...")
                thread.wait(5000)  # Wait up to 5 seconds per thread
                if thread.isRunning():
                    logger.warning("Thread did not finish within timeout, terminating")
                    thread.terminate()
                    thread.wait(1000)  # Give it 1 more second
                # Ensure thread is properly cleaned up
                if thread in self.active_threads:
                    self.active_threads.remove(thread)

        self.active_threads.clear()
        logger.info("AsyncTaskManager shutdown complete")


class CoreAI3DDashboard(QMainWindow):
    """Main dashboard window"""

    def __init__(self):
        super().__init__()

        # Initialize configuration
        self.config = self._load_config()

        # Initialize async task manager
        self.async_manager = AsyncTaskManager()

        # Initialize components
        self.client: Optional[CoreAI3DClient] = None
        self.sandbox_manager = SandboxManager(self.config)
        self.training_manager = TrainingDataManager(self.config)
        self.debugger: Optional[CodeDebugger] = None

        # Chat response handler
        self.chat_response_handler = None

        # Initialize training argument variables - consolidated learning settings
        self.norm_min = 0.0
        self.norm_max = 1.0
        self.neurons = 10
        self.samples = 0
        self.epochs = 10
        self.layers = 3
        self.learning_rate = 0.01

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

        # Force reset API and WebSocket URLs to 0.0.0.0, overriding any saved QSettings
        config.api_url = "http://0.0.0.0:8080/api/v1"
        config.ws_url = "ws://0.0.0.0:8081"

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

        if len(api_key) < 10 and api_key:
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
        self.predict_model_combo.currentTextChanged.connect(self.on_predict_model_type_changed)
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
    
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_models)
        search_layout.addWidget(search_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        search_layout.addWidget(refresh_btn)

        layout.addWidget(search_group)

        # Model list
        list_group = QGroupBox("Available Models")
        list_layout = QVBoxLayout(list_group)

        self.model_table = QTableWidget(0, 5)
        self.model_table.setHorizontalHeaderLabels([
            "Model Name", "Type", "Size", "Version", "Status"
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

        # Initialize with sample models
        self.load_sample_models()

    def refresh_containers(self):
        """Refresh available containers"""
        try:
            # This would typically query Docker for running containers
            # For now, just update the status
            self.linux_sandbox_status_label.setText("Linux Sandbox: Containers refreshed")
            self.status_bar.showMessage("Container list refreshed")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh containers: {str(e)}")

    def execute_linux_command(self):
        """Execute Linux command in sandbox"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        command = self.linux_command_input.text().strip()
        if not command:
            QMessageBox.warning(self, "No Command", "Please enter a command to execute")
            return

        container = self.container_combo.currentText()

        try:
            async def execute():
                try:
                    # Execute command via LinuxModule
                    result = await self.client.execute_linux_command(container, command)
                    return result
                except Exception as e:
                    raise e

            def on_success(result):
                self.linux_training_progress.setVisible(False)
                if result.success:
                    self.linux_output.appendPlainText(f"$ {command}")
                    self.linux_output.appendPlainText(result.data.get('output', ''))
                    self.linux_sandbox_status_label.setText(f"Linux Sandbox: Command executed in {container}")
                    self.status_bar.showMessage("Command executed successfully")
                else:
                    self.linux_output.appendPlainText(f"$ {command}")
                    self.linux_output.appendPlainText(f"Error: {result.data.get('error', 'Unknown error')}")
                    self.status_bar.showMessage("Command execution failed")

            def on_error(error_msg):
                self.linux_training_progress.setVisible(False)
                self.linux_output.appendPlainText(f"Error: {error_msg}")
                self.status_bar.showMessage("Command execution error")

            # Show progress
            self.linux_training_progress.setVisible(True)
            self.linux_training_progress.setRange(0, 0)

            self.async_manager.run_async_task(execute(), on_success, on_error)

        except Exception as e:
            self.linux_training_progress.setVisible(False)
            QMessageBox.warning(self, "Error", f"Failed to execute command: {str(e)}")

    def start_linux_container(self):
        """Start Linux container"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        container = self.container_combo.currentText()

        try:
            async def start():
                result = await self.client.start_linux_container(container)
                return result

            def on_success(result):
                if result.success:
                    self.linux_sandbox_status_label.setText(f"Linux Sandbox: {container} started")
                    self.status_bar.showMessage(f"Container {container} started successfully")
                else:
                    QMessageBox.warning(self, "Start Error", result.data.get('error', 'Unknown error'))

            def on_error(error_msg):
                QMessageBox.warning(self, "Error", f"Failed to start container: {error_msg}")

            self.async_manager.run_async_task(start(), on_success, on_error)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start container: {str(e)}")

    def stop_linux_container(self):
        """Stop Linux container"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        container = self.container_combo.currentText()

        try:
            async def stop():
                result = await self.client.stop_linux_container(container)
                return result

            def on_success(result):
                if result.success:
                    self.linux_sandbox_status_label.setText("Linux Sandbox: Container stopped")
                    self.status_bar.showMessage(f"Container {container} stopped successfully")
                else:
                    QMessageBox.warning(self, "Stop Error", result.data.get('error', 'Unknown error'))

            def on_error(error_msg):
                QMessageBox.warning(self, "Error", f"Failed to stop container: {error_msg}")

            self.async_manager.run_async_task(stop(), on_success, on_error)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop container: {str(e)}")

    def clear_linux_output(self):
        """Clear Linux command output"""
        self.linux_output.clear()

    def run_linux_training(self):
        """Run Linux training scenario"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        scenario = self.scenario_combo.currentText()
        container = self.container_combo.currentText()

        try:
            async def train():
                # Run training scenario
                result = await self.client.run_linux_training(container, scenario)
                return result

            def on_success(result):
                self.linux_training_progress.setVisible(False)
                if result.success:
                    self.training_results.setPlainText(result.data.get('report', 'Training completed'))
                    self.status_bar.showMessage(f"Training scenario '{scenario}' completed")
                else:
                    self.training_results.setPlainText(f"Training failed: {result.data.get('error', 'Unknown error')}")
                    self.status_bar.showMessage("Training failed")

            def on_error(error_msg):
                self.linux_training_progress.setVisible(False)
                self.training_results.setPlainText(f"Training error: {error_msg}")
                self.status_bar.showMessage("Training error")

            # Show progress
            self.linux_training_progress.setVisible(True)
            self.linux_training_progress.setRange(0, 0)

            self.async_manager.run_async_task(train(), on_success, on_error)

        except Exception as e:
            self.linux_training_progress.setVisible(False)
            QMessageBox.warning(self, "Error", f"Failed to run training: {str(e)}")

    def generate_training_report(self):
        """Generate training report"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        container = self.container_combo.currentText()

        try:
            async def generate():
                result = await self.client.generate_training_report(container)
                return result

            def on_success(result):
                if result.success:
                    self.training_results.setPlainText(result.data.get('report', 'Report generated'))
                    self.status_bar.showMessage("Training report generated")
                else:
                    QMessageBox.warning(self, "Report Error", result.data.get('error', 'Unknown error'))

            def on_error(error_msg):
                QMessageBox.warning(self, "Error", f"Failed to generate report: {error_msg}")

            self.async_manager.run_async_task(generate(), on_success, on_error)

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

    def generate_code(self):
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
            async def generate():
                # Use AI to generate code
                result = await self._generate_code_ai(language, code_type, complexity, description)
                return result

            def on_success(result):
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

            def on_error(error_msg):
                # Hide progress
                self.generation_progress.setVisible(False)
                self.generate_btn.setEnabled(True)

                QMessageBox.warning(self, "Generation Error", error_msg)
                self.status_bar.showMessage("Code generation error")

            self.async_manager.run_async_task(generate(), on_success, on_error)

        except Exception as e:
            self.generation_progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            QMessageBox.warning(self, "Error", f"Failed to start code generation: {str(e)}")

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

    def _get_generic_template(self, language: str, code_type: str, complexity: str, description: str) -> str:
        """Generic code template"""
        return f"""# {language} {code_type} - {complexity} Level
# Generated for: {description}

# TODO: Implement {code_type.lower()} in {language}
# This is a placeholder template that needs to be customized
# based on the specific requirements.

def main():
    # Main function implementation
    pass

if __name__ == "__main__":
    main()
"""

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
        self.norm_min_spin.setRange(-10.0, 10.0)
        self.norm_min_spin.setValue(self.norm_min)
        self.norm_min_spin.setSingleStep(0.1)
        self.norm_min_spin.setDecimals(1)
        self.norm_min_spin.valueChanged.connect(self.on_norm_min_changed)
        norm_layout.addRow("Minimum (--min):", self.norm_min_spin)

        # Max value input
        self.norm_max_spin = QDoubleSpinBox()
        self.norm_max_spin.setRange(-10.0, 10.0)
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
        self.neurons_spin.setRange(1, 100000)
        self.neurons_spin.setValue(self.neurons)
        self.neurons_spin.valueChanged.connect(self.on_neurons_changed)
        network_layout.addRow("Neurons (-n):", self.neurons_spin)

        # Samples input
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(0, 1000000)
        self.samples_spin.setValue(self.samples)
        self.samples_spin.setSpecialValueText("All samples (0)")
        self.samples_spin.valueChanged.connect(self.on_samples_changed)
        network_layout.addRow("Samples (-s):", self.samples_spin)

        # Epochs input
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(self.epochs)
        self.epochs_spin.valueChanged.connect(self.on_epochs_changed)
        network_layout.addRow("Epochs (-e):", self.epochs_spin)

        # Layers input
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 1000)
        self.layers_spin.setValue(self.layers)
        self.layers_spin.valueChanged.connect(self.on_layers_changed)
        network_layout.addRow("Layers (-l):", self.layers_spin)

        scroll_layout.addWidget(network_group)

        # Training Parameters Section
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout(training_group)

        # Learning rate input
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(self.learning_rate)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(4)
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
        start_training_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(start_training_btn)

        stop_training_btn = QPushButton("Stop Training")
        stop_training_btn.clicked.connect(self.stop_training)
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
        clear_log_btn.clicked.connect(self.clear_training_log)
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
        """Initialize CoreAI3D client"""
        try:
            logger.info("Starting client initialization...")
            logger.info(f"API URL: {self.config.api_url}")
            logger.info(f"WS URL: {self.config.ws_url}")
            logger.info(f"API Key configured: {bool(self.config.api_key)}")

            # Allow initialization even without API key for offline mode
            if not self.config.api_url:
                logger.warning("API URL not configured - client will work in offline mode")
                self.config.api_url = "http://0.0.0.0:8080/api/v1"  # Set default
    
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

            return True

        except ImportError as e:
            logger.error(f"Import error during client initialization: {str(e)}")
            logger.error("Make sure coreai3d_client.py is in the same directory or PYTHONPATH")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    # Event handlers
    def on_dataset_selected(self, item: QListWidgetItem):
        """Handle dataset selection"""
        dataset_name = item.text()
        if dataset_name in self.training_manager.datasets:
            dataset = self.training_manager.datasets[dataset_name]
            self.dataset_info.setPlainText(json.dumps(dataset, indent=2))

            # Update file list
            self.file_list.clear()
            for file_info in dataset.get('files', []):
                self.file_list.addItem(file_info['name'])

    def on_sandbox_started(self, message: str):
        """Handle sandbox started event"""
        self.sandbox_status_label.setText("Sandboxie: Running")
        self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.status_bar.showMessage("Sandboxie started successfully")

    def on_sandbox_stopped(self, message: str):
        """Handle sandbox stopped event"""
        self.sandbox_status_label.setText("Sandboxie: Not Running")
        self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.status_bar.showMessage("Sandboxie stopped")

    def on_sandbox_error(self, title: str, message: str):
        """Handle sandbox error event"""
        self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {message}")
        QMessageBox.warning(self, title, message)
        self.status_bar.showMessage(f"Sandboxie error: {title}")

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
    def test_connection(self):
        """Test API connection"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        try:
            # Show progress indicator
            self.status_bar.showMessage("Testing connection...")

            # Run health check using async manager
            async def check_health():
                result = await self.client.health_check()
                return result

            def on_success(result):
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

            def on_error(error_msg):
                self.status_bar.showMessage("Connection test failed")
                QMessageBox.critical(
                    self, "Connection Error",
                    f"✗ Connection test failed!\n\n"
                    f"Error: {error_msg}\n\n"
                    "Please check:\n"
                    "• Server configuration\n"
                    "• Network settings\n"
                    "• Firewall rules"
                )

            self.async_manager.run_async_task(check_health(), on_success, on_error)

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
            async def health_check():
                result = await self.client.health_check()
                return result

            def on_success(result):
                if not result.success:
                    self.status_bar.showMessage("System health check failed")
                    logger.warning("System health check failed")

            def on_error(error_msg):
                logger.error(f"Health check error: {error_msg}")

            self.async_manager.run_async_task(health_check(), on_success, on_error)

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")

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
                success = self.training_manager.add_file_to_dataset(dataset_name, file_path)
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
                        success = self.training_manager.add_file_to_dataset(dataset_name, file_path)
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

    def run_test(self):
        """Run selected test"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        test_type = self.test_type_combo.currentText()

        try:
            if test_type == "API Connectivity":
                async def api_test():
                    result = await self.debugger.run_api_test("health_check")
                    return result

                def on_success(result):
                    self.add_test_result(result)

                def on_error(error_msg):
                    QMessageBox.warning(self, "Test Error", f"API test failed: {error_msg}")

                self.async_manager.run_async_task(api_test(), on_success, on_error)

            elif test_type == "Module Functionality":
                module_name = self.test_module_combo.currentText().split()[0].lower()
                test_data = {"test": "functionality"}

                async def module_test():
                    result = await self.debugger.run_module_test(module_name, test_data)
                    return result

                def on_success(result):
                    self.add_test_result(result)

                def on_error(error_msg):
                    QMessageBox.warning(self, "Test Error", f"Module test failed: {error_msg}")

                self.async_manager.run_async_task(module_test(), on_success, on_error)

            else:
                QMessageBox.information(self, "Test", f"Test type '{test_type}' not implemented yet")

        except Exception as e:
            QMessageBox.warning(self, "Test Error", str(e))

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
        """Start Sandboxie sandbox"""
        config = {
            'sandbox_name': self.sandbox_name_edit.text(),
            'startup_command': self.startup_command_edit.text(),
            'networking': self.networking_combo.currentText().lower(),
            'drop_rights': self.drop_rights_check.isChecked()
        }

        success = self.sandbox_manager.start_sandbox(config)
        if not success:
            QMessageBox.warning(self, "Sandboxie Error", "Failed to start Sandboxie sandbox")

    def stop_sandbox(self):
        """Stop Sandboxie sandbox"""
        success = self.sandbox_manager.stop_sandbox()
        if not success:
            QMessageBox.warning(self, "Sandboxie Error", "Failed to stop Sandboxie sandbox")

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

    def run_api_debug(self):
        """Run API debugging"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        try:
            async def debug_api():
                # Test various API endpoints
                endpoints = [
                    ("health_check", {}),
                    ("system_metrics", {}),
                    ("module_status", {})
                ]

                results = []
                for endpoint, params in endpoints:
                    try:
                        if endpoint == "health_check":
                            result = await self.client.health_check()
                        elif endpoint == "system_metrics":
                            result = await self.client.get_system_metrics()
                        elif endpoint == "module_status":
                            result = await self.client.get_module_status()

                        results.append((endpoint, result.success, None))

                    except Exception as e:
                        results.append((endpoint, False, str(e)))

                return results

            def on_success(results):
                for endpoint, success, error in results:
                    if error:
                        message = f"{endpoint}: ERROR - {error}"
                    else:
                        message = f"{endpoint}: {'OK' if success else 'FAILED'}"
                    self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

            def on_error(error_msg):
                self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Debug API Error: {error_msg}")

            self.async_manager.run_async_task(debug_api(), on_success, on_error)

        except Exception as e:
            QMessageBox.warning(self, "Debug Error", str(e))

    def run_system_debug(self):
        """Run system debugging"""
        try:
            # Collect system diagnostics
            diagnostics = []

            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            diagnostics.append(f"CPU Cores: {cpu_count}")
            if cpu_freq:
                diagnostics.append(f"CPU Frequency: {cpu_freq.current:.2f} MHz")

            # Memory info
            memory = psutil.virtual_memory()
            diagnostics.append(f"Total Memory: {memory.total / (1024**3):.2f} GB")
            diagnostics.append(f"Available Memory: {memory.available / (1024**3):.2f} GB")

            # Disk info
            disk = psutil.disk_usage('/')
            diagnostics.append(f"Total Disk: {disk.total / (1024**3):.2f} GB")
            diagnostics.append(f"Free Disk: {disk.free / (1024**3):.2f} GB")

            # Network info
            network = psutil.net_if_addrs()
            diagnostics.append(f"Network Interfaces: {len(network)}")

            # Display diagnostics
            self.sandbox_logs.appendPlainText("=== SYSTEM DIAGNOSTICS ===")
            for diag in diagnostics:
                self.sandbox_logs.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {diag}")

        except Exception as e:
            QMessageBox.warning(self, "Debug Error", str(e))

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
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        message = self.chat_input.text().strip()
        if not message:
            return

        # Add user message to history
        self.chat_history.append(f"You: {message}")
        self.chat_input.clear()

        # Add debug logging
        logger.info(f"Sending chat message: {message}")
        logger.info(f"Client connected: {self.client.is_connected}")

        try:
            async def send_message():
                logger.info("Making HTTP request to send chat message")
                result = await self.client.send_chat_message(message)
                logger.info(f"HTTP response received: success={result.success}, status={result.status_code}")
                if result.data:
                    logger.info(f"Response data: {result.data}")
                return result

            def on_success(result):
                logger.info("HTTP request successful, waiting for WebSocket response")
                if result.success:
                    # HTTP request succeeded, but actual response comes via WebSocket
                    # The response will be handled by the registered chat response handler
                    pass
                else:
                    error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                    logger.error(f"HTTP request failed: {error_msg}")
                    self.chat_history.append(f"Error: {error_msg}")

            def on_error(error_msg):
                logger.error(f"Async task error: {error_msg}")
                self.chat_history.append(f"Error: {error_msg}")

            self.async_manager.run_async_task(send_message(), on_success, on_error)

        except Exception as e:
            logger.error(f"Exception in send_chat_message: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to send message: {str(e)}")

    def _handle_chat_response(self, data: Dict[str, Any]):
        """Handle chat response from WebSocket"""
        logger.info(f"Handling chat response: {data}")
        content = data.get('content', '')
        if content:
            logger.info(f"Adding chat response to history: {content}")
            self.chat_history.append(f"AI: {content}")
        else:
            logger.warning("Received chat response with empty content")

    def _handle_chat_error(self, error_msg: str):
        """Handle chat error"""
        logger.error(f"Chat error: {error_msg}")
        self.chat_history.append(f"Error: {error_msg}")

    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history.clear()

    def load_sample_models(self):
        """Load all available models from the CoreAI3D system"""
        try:
            # Vision Models - for image processing, classification, and analysis
            vision_models = [
                {"name": "ResNet-50", "type": "Vision", "size": "98 MB", "version": "1.0.0", "status": "Available"},
                {"name": "YOLOv5-Small", "type": "Vision", "size": "14 MB", "version": "6.1.0", "status": "Available"},
                {"name": "EfficientNet-B0", "type": "Vision", "size": "20 MB", "version": "1.0.0", "status": "Available"},
                {"name": "MobileNetV3-Small", "type": "Vision", "size": "5.4 MB", "version": "1.0.0", "status": "Available"},
                {"name": "FaceNet", "type": "Vision", "size": "23 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Tesseract-OCR", "type": "Vision", "size": "8.2 MB", "version": "5.3.0", "status": "Available"},
                {"name": "Mask-RCNN", "type": "Vision", "size": "170 MB", "version": "2.1.0", "status": "Available"},
                {"name": "SSD-MobileNet", "type": "Vision", "size": "24 MB", "version": "1.0.0", "status": "Available"}
            ]

            # Audio Models - for speech processing and audio analysis
            audio_models = [
                {"name": "Whisper-Tiny", "type": "Audio", "size": "39 MB", "version": "1.2.0", "status": "Available"},
                {"name": "Whisper-Base", "type": "Audio", "size": "74 MB", "version": "1.2.0", "status": "Available"},
                {"name": "Whisper-Small", "type": "Audio", "size": "244 MB", "version": "1.2.0", "status": "Available"},
                {"name": "Tacotron2", "type": "Audio", "size": "113 MB", "version": "1.0.0", "status": "Available"},
                {"name": "WaveNet", "type": "Audio", "size": "4.7 GB", "version": "1.0.0", "status": "Available"},
                {"name": "DeepSpeech", "type": "Audio", "size": "190 MB", "version": "0.9.3", "status": "Available"},
                {"name": "SpeakerNet", "type": "Audio", "size": "16 MB", "version": "1.0.0", "status": "Available"},
                {"name": "AudioNet", "type": "Audio", "size": "8.5 MB", "version": "1.0.0", "status": "Available"}
            ]

            # Text/Language Models - for natural language processing
            text_models = [
                {"name": "BERT-Base", "type": "Text", "size": "420 MB", "version": "2.1.0", "status": "Available"},
                {"name": "GPT-2 Small", "type": "Text", "size": "548 MB", "version": "1.5.0", "status": "Available"},
                {"name": "DistilBERT", "type": "Text", "size": "66 MB", "version": "1.0.0", "status": "Available"},
                {"name": "RoBERTa-Base", "type": "Text", "size": "476 MB", "version": "1.0.0", "status": "Available"},
                {"name": "ALBERT-Base", "type": "Text", "size": "43 MB", "version": "1.0.0", "status": "Available"},
                {"name": "XLNet-Base", "type": "Text", "size": "446 MB", "version": "1.0.0", "status": "Available"},
                {"name": "ELECTRA-Small", "type": "Text", "size": "13 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Sentence-BERT", "type": "Text", "size": "218 MB", "version": "1.0.0", "status": "Available"}
            ]

            # Multimodal Models - combining vision and text
            multimodal_models = [
                {"name": "CLIP-ViT", "type": "Multimodal", "size": "1.2 GB", "version": "1.0.1", "status": "Available"},
                {"name": "CLIP-RN50", "type": "Multimodal", "size": "1.0 GB", "version": "1.0.1", "status": "Available"},
                {"name": "VisualBERT", "type": "Multimodal", "size": "1.1 GB", "version": "1.0.0", "status": "Available"},
                {"name": "ViLBERT", "type": "Multimodal", "size": "1.3 GB", "version": "1.0.0", "status": "Available"},
                {"name": "LXMERT", "type": "Multimodal", "size": "1.8 GB", "version": "1.0.0", "status": "Available"},
                {"name": "Oscar", "type": "Multimodal", "size": "1.4 GB", "version": "1.0.0", "status": "Available"}
            ]

            # Neural Network Models - custom trained models from the system
            neural_models = [
                {"name": "CoreAI3D-NN-1", "type": "Neural", "size": "2.3 MB", "version": "1.0.0", "status": "Available"},
                {"name": "CoreAI3D-NN-2", "type": "Neural", "size": "5.7 MB", "version": "1.1.0", "status": "Available"},
                {"name": "CoreAI3D-NN-3", "type": "Neural", "size": "12 MB", "version": "1.2.0", "status": "Available"},
                {"name": "CoreAI3D-NN-4", "type": "Neural", "size": "8.9 MB", "version": "1.0.0", "status": "Available"}
            ]

            # Math Models - for mathematical computations and optimization
            math_models = [
                {"name": "MathSolver-NN", "type": "Math", "size": "15 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Optimization-Net", "type": "Math", "size": "8.2 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Statistics-Engine", "type": "Math", "size": "6.1 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Symbolic-Math", "type": "Math", "size": "22 MB", "version": "1.0.0", "status": "Available"}
            ]

            # Web Models - for web content processing and analysis
            web_models = [
                {"name": "WebScraper-NN", "type": "Web", "size": "45 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Sentiment-Analyzer", "type": "Web", "size": "18 MB", "version": "1.0.0", "status": "Available"},
                {"name": "Content-Classifier", "type": "Web", "size": "27 MB", "version": "1.0.0", "status": "Available"},
                {"name": "News-Aggregator", "type": "Web", "size": "33 MB", "version": "1.0.0", "status": "Available"}
            ]

            # Combine all models
            all_models = (vision_models + audio_models + text_models +
                          multimodal_models + neural_models + math_models + web_models)

            self.model_table.setRowCount(len(all_models))
            for row, model in enumerate(all_models):
                self.model_table.setItem(row, 0, QTableWidgetItem(model["name"]))
                self.model_table.setItem(row, 1, QTableWidgetItem(model["type"]))
                self.model_table.setItem(row, 2, QTableWidgetItem(model["size"]))
                self.model_table.setItem(row, 3, QTableWidgetItem(model["version"]))
                status_item = QTableWidgetItem(model["status"])
                self.model_table.setItem(row, 4, status_item)

                # Color code status
                if model["status"] == "Available":
                    status_item.setBackground(QColor(144, 238, 144))  # Light green
                elif model["status"] == "Downloaded":
                    status_item.setBackground(QColor(173, 216, 230))  # Light blue
                elif model["status"] == "Downloading":
                    status_item.setBackground(QColor(255, 255, 224))  # Light yellow
                elif model["status"] == "Error":
                    status_item.setBackground(QColor(255, 182, 193))  # Light red
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
            self.download_timer = QTimer()
            self.download_timer.timeout.connect(lambda: self.update_download_progress(current_row, model_name))
            self.download_timer.start(500)
            logger.debug("Download timer started")

    def update_download_progress(self, row, model_name):
        """Update download progress"""
        try:
            logger.debug(f"Updating download progress for {model_name}, row {row}")
            current_value = self.download_progress.value()
            logger.debug(f"Current progress value: {current_value}")

            if current_value < 100:
                new_value = current_value + 10
                self.download_progress.setValue(new_value)
                self.download_status_label.setText(f"Downloading {model_name}... {new_value}%")
                logger.debug(f"Progress updated to {new_value}%")
            else:
                # Download complete
                logger.info(f"Download completed for {model_name}")
                if hasattr(self, 'download_timer') and self.download_timer.isActive():
                    self.download_timer.stop()
                    logger.debug("Download timer stopped")

                # Update status in main thread
                QTimer.singleShot(0, lambda: self._update_download_complete(row, model_name))
        except Exception as e:
            logger.error(f"Error updating download progress: {str(e)}")

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
        """Download model from Kaggle"""
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

                    # Download command
                    cmd = ['kaggle', 'datasets', 'download', dataset_path, '-p', models_dir]

                    # Run download
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    # Monitor progress (simplified)
                    while process.poll() is None:
                        time.sleep(1)

                    if process.returncode == 0:
                        QTimer.singleShot(0, lambda: self._download_success(row, model_name))
                    else:
                        stderr = process.stderr.read() if process.stderr else ""
                        error_msg = f"Kaggle download failed: {stderr}"
                        QTimer.singleShot(0, lambda: self._download_error(row, model_name, error_msg))

                except Exception as e:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name, str(e)))

            # Start download in background thread
            threading.Thread(target=download_worker, daemon=True).start()

        except Exception as e:
            self._download_error(row, model_name, f"Failed to start Kaggle download: {str(e)}")

    def _download_from_huggingface(self, row, model_name):
        """Download model from Hugging Face"""
        try:
            from huggingface_hub import snapshot_download
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

                    # Download with progress callback
                    def progress_callback(size, total):
                        if total > 0:
                            progress = int((size / total) * 100)
                            QTimer.singleShot(0, lambda: self._update_hf_progress(row, model_name, progress))

                    # Download model
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=models_dir,
                        local_dir_use_symlinks=False
                    )

                    QTimer.singleShot(0, lambda: self._download_success(row, model_name))

                except ImportError:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name,
                        "HuggingFace Hub not installed. Install with: pip install huggingface-hub"))
                except Exception as e:
                    QTimer.singleShot(0, lambda: self._download_error(row, model_name, str(e)))

            # Start download in background thread
            threading.Thread(target=download_worker, daemon=True).start()

        except Exception as e:
            self._download_error(row, model_name, f"Failed to start HuggingFace download: {str(e)}")

    def _update_kaggle_progress(self, row, model_name):
        """Update Kaggle download progress"""
        logger.debug(f"Updating Kaggle progress for {model_name}")
        # Simplified progress update - in real implementation would parse actual progress
        current_progress = self.download_progress.value()
        if current_progress < 90:  # Leave room for final processing
            new_progress = current_progress + 5
            self.download_progress.setValue(new_progress)
            self.download_status_label.setText(f"Downloading {model_name} from Kaggle... {new_progress}%")
            logger.debug(f"Kaggle progress updated to {new_progress}%")

    def _update_hf_progress(self, row, model_name, progress):
        """Update HuggingFace download progress"""
        logger.debug(f"Updating HuggingFace progress for {model_name} to {progress}%")
        self.download_progress.setValue(progress)
        self.download_status_label.setText(f"Downloading {model_name} from HuggingFace... {progress}%")

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

    def start_training(self):
        """Start training with current configuration"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        try:
            # Show progress with actual range
            self.training_progress.setVisible(True)
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(0)

            # Log training start
            self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training with configuration:")
            self.training_log.appendPlainText(json.dumps(self.learning_config, indent=2))

            async def train():
                # Call training with consolidated configuration
                result = await self.client.start_training(self.learning_config)
                return result

            def on_success(result):
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

            def on_error(error_msg):
                self.training_progress.setVisible(False)
                self.training_progress.setValue(0)  # Reset progress on error
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training error: {error_msg}")
                QMessageBox.warning(self, "Training Error", error_msg)

            self.async_manager.run_async_task(train(), on_success, on_error)

        except Exception as e:
            self.training_progress.setVisible(False)
            QMessageBox.warning(self, "Error", f"Failed to start training: {str(e)}")

    def stop_training(self):
        """Stop current training"""
        if not self.client:
            QMessageBox.warning(self, "Connection Error", "No active client connection")
            return

        try:
            async def stop():
                result = await self.client.stop_training()
                return result

            def on_success(result):
                if result.success:
                    self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Training stopped successfully")
                    self.training_progress.setVisible(False)
                    QMessageBox.information(self, "Training Stopped", "Training stopped successfully")
                else:
                    error_msg = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
                    self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to stop training: {error_msg}")
                    QMessageBox.warning(self, "Stop Failed", error_msg)

            def on_error(error_msg):
                self.training_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Stop training error: {error_msg}")
                QMessageBox.warning(self, "Stop Error", error_msg)

            self.async_manager.run_async_task(stop(), on_success, on_error)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop training: {str(e)}")

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

    def run_prediction(self):
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
            async def predict():
                # Call appropriate prediction method based on model type
                if model_type == "Vision Models":
                    result = await self.client.analyze_image(input_data)
                elif model_type == "Audio Models":
                    result = await self.client.analyze_audio(input_data)
                elif model_type == "Text Models":
                    result = await self.client.process_with_ai(input_data, 'text')
                elif model_type == "Math Models":
                    result = await self.client.solve_math(input_data)
                elif model_type == "Web Models":
                    result = await self.client.web_search(input_data)
                else:
                    # Generic prediction call with data format options
                    result = await self.client.run_prediction(
                        specific_model, input_data, input_type.lower(),
                        contains_header=contains_header, contains_text=contains_text
                    )
                return result

            def on_success(result):
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

            def on_error(error_msg):
                self.predict_progress.setVisible(False)
                self.predict_results_text.setPlainText(f"Error: {error_msg}")
                self.status_bar.showMessage("Prediction error")

            self.async_manager.run_async_task(predict(), on_success, on_error)

        except Exception as e:
            self.predict_progress.setVisible(False)
            QMessageBox.warning(self, "Error", f"Failed to start prediction: {str(e)}")

    def clear_prediction_results(self):
        """Clear prediction results"""
        self.predict_results_text.clear()
        self.predict_progress.setVisible(False)

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

                # Handle headers
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

    def refresh_neural_data(self):
        """Refresh neural network data"""
        if not self.client:
            if not self.initialize_client():
                QMessageBox.warning(self, "Connection Error", "Failed to initialize client")
                return

        try:
            async def refresh_data():
                # Get topology
                topology_result = await self.client.get_neural_topology()
                # Get activity
                activity_result = await self.client.get_neural_activity()
                return topology_result, activity_result

            def on_success(results):
                topology_result, activity_result = results

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

            def on_error(error_msg):
                error_msg_full = f"Error refreshing neural data: {error_msg}"
                self.topology_text.setPlainText(error_msg_full)
                if MATPLOTLIB_AVAILABLE:
                    self.activity_canvas.figure.clear()
                    self.activity_canvas.draw()
                else:
                    self.activity_text.setPlainText(error_msg_full)

            self.async_manager.run_async_task(refresh_data(), on_success, on_error)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh neural data: {str(e)}")

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
    def on_norm_min_changed(self, value):
        """Handle normalization min value change"""
        self.norm_min = value

    def on_norm_max_changed(self, value):
        """Handle normalization max value change"""
        self.norm_max = value

    def on_neurons_changed(self, value):
        """Handle neurons value change"""
        self.neurons = value

    def on_samples_changed(self, value):
        """Handle samples value change"""
        self.samples = value

    def on_epochs_changed(self, value):
        """Handle epochs value change"""
        self.epochs = value

    def on_layers_changed(self, value):
        """Handle layers value change"""
        self.layers = value

    def on_learning_rate_changed(self, value):
        """Handle learning rate value change"""
        self.learning_rate = value

    def closeEvent(self, event):
        """Handle application close event"""
        try:
            logger.info("Starting application shutdown sequence")

            # Stop sandbox if running
            if self.sandbox_manager.is_running:
                logger.info("Stopping sandbox manager")
                self.sandbox_manager.stop_sandbox()

            # Close client connection synchronously to avoid race conditions
            if self.client:
                logger.info("Closing client connection")
                try:
                    # Close synchronously to ensure proper cleanup
                    if hasattr(self.client, 'close'):
                        # Create a new event loop for synchronous close
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.client.close())
                        loop.close()
                except Exception as e:
                    logger.error(f"Error closing client: {str(e)}")

            # Shutdown async manager (this will wait for threads)
            logger.info("Shutting down async manager")
            self.async_manager.shutdown()

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
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='CoreAI3D Linux GUI Dashboard')
        parser.add_argument('-v', '--verbose', action='store_true',
                           help='Enable debug messages')
        args = parser.parse_args()

        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            # Also set specific loggers to DEBUG for more detailed output
            logging.getLogger('coreai3d_client').setLevel(logging.DEBUG)
            logging.getLogger('automation_helper').setLevel(logging.DEBUG)
            logger.info("Debug messages enabled via -v flag")
        else:
            # For release builds, ensure only WARNING and above are shown
            logging.getLogger().setLevel(logging.WARNING)
            logger.info("Running in release mode - debug messages disabled")

        # Create logs directory
        os.makedirs('logs', exist_ok=True)

        # Create application
        app = QApplication(sys.argv)

        # Set application properties
        app.setApplicationName("CoreAI3D Linux Dashboard")
        app.setApplicationVersion("1.3.0")
        app.setOrganizationName("CoreAI3D")

        # Create and show dashboard
        dashboard = CoreAI3DDashboard()
        dashboard.show()

        # Run application
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        QMessageBox.critical(None, "Application Error", f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()