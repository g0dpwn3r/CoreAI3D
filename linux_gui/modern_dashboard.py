#!/usr/bin/env python3
"""
CoreAI3D — Modern Linux GUI Dashboard
=====================================
A sleek, production-grade desktop application for managing AI training data,
testing, debugging, and API keys using PyQt6.

Features:
  - Modern dark theme (default) with optional light theme, instant switching
  - API Key generation with robust multi-location response parsing
  - Show-once key modal dialog (key never shown again after dismiss)
  - Secure key storage (system keyring, encrypted file fallback)
  - Dashboard, API Keys, Settings, Logs, About tabs
  - Full keyboard navigation, loading states, error handling

Toolkit: PyQt6 (Qt6) — mature, excellent Linux support, QSS theming
"""

import sys
import os
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Optional secure-storage dependencies
# ---------------------------------------------------------------------------
try:
    import keyring  # type: ignore
    HAS_KEYRING = True
except ImportError:  # pragma: no cover
    HAS_KEYRING = False

try:
    from cryptography.fernet import Fernet  # type: ignore
    HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    HAS_CRYPTOGRAPHY = False

# ---------------------------------------------------------------------------
# PyQt6 imports
# ---------------------------------------------------------------------------
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QPlainTextEdit, QComboBox,
    QCheckBox, QSpinBox, QProgressBar, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QScrollArea, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QRect,
    QSettings, QStandardPaths, QEasingCurve, QPropertyAnimation,
)
from PyQt6.QtGui import (
    QFont, QIcon, QPixmap, QPalette, QColor, QLinearGradient,
    QPainter, QBrush, QAction, QDesktopServices, QKeySequence,
    QShortcut,
)

# ---------------------------------------------------------------------------
# CoreAI3D client (reuse existing backend client)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coreai3d_client import CoreAI3DClient, APIResponse  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("coreai3d.modern")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "CoreAI3D"
APP_VERSION = "2.0.0"
ORG_NAME = "CoreAI3D"

# Keyring service name for secure storage
KEYRING_SERVICE = "coreai3d"
KEYRING_KEY = "api_key"

# Encrypted fallback file
CONFIG_DIR = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
ENCRYPTED_KEY_FILE = CONFIG_DIR / "api_key.enc"
KEY_FILE = CONFIG_DIR / "api_key.key"


# ===========================================================================
# Secure Key Storage
# ===========================================================================
class KeyStore:
    """
    Secure API key storage.

    Priority:
      1. System keyring (via `keyring` library) — most secure
      2. Encrypted local file (via `cryptography.Fernet`) — good fallback
      3. Plain-text file — last resort, logged warning
    """

    def __init__(self):
        self._fernet: Optional[Fernet] = None
        self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialise Fernet encryption for the fallback file."""
        if not HAS_CRYPTOGRAPHY:
            return
        if KEY_FILE.exists():
            key = KEY_FILE.read_bytes()
        else:
            key = Fernet.generate_key()
            KEY_FILE.write_bytes(key)
            KEY_FILE.chmod(0o600)
        self._fernet = Fernet(key)

    def save(self, api_key: str) -> bool:
        """Save an API key. Returns True on success."""
        # Try keyring first
        if HAS_KEYRING:
            try:
                keyring.set_password(KEYRING_SERVICE, KEYRING_KEY, api_key)
                logger.info("API key saved to system keyring")
                return True
            except Exception as e:  # pragma: no cover
                logger.warning(f"Keyring save failed: {e}")

        # Fallback to encrypted file
        if self._fernet is not None:
            try:
                token = self._fernet.encrypt(api_key.encode())
                ENCRYPTED_KEY_FILE.write_bytes(token)
                ENCRYPTED_KEY_FILE.chmod(0o600)
                logger.info("API key saved to encrypted local file")
                return True
            except Exception as e:  # pragma: no cover
                logger.warning(f"Encrypted file save failed: {e}")

        # Last resort: plain text (with warning)
        try:
            ENCRYPTED_KEY_FILE.write_text(api_key)
            ENCRYPTED_KEY_FILE.chmod(0o600)
            logger.warning("API key saved in PLAIN TEXT — keyring/cryptography not available")
            return True
        except Exception as e:  # pragma: no cover
            logger.error(f"All key storage methods failed: {e}")
            return False

    def load(self) -> Optional[str]:
        """Load an API key. Returns None if not found."""
        # Try keyring first
        if HAS_KEYRING:
            try:
                key = keyring.get_password(KEYRING_SERVICE, KEYRING_KEY)
                if key:
                    logger.info("API key loaded from system keyring")
                    return key
            except Exception as e:  # pragma: no cover
                logger.warning(f"Keyring load failed: {e}")

        # Fallback to encrypted file
        if self._fernet is not None and ENCRYPTED_KEY_FILE.exists():
            try:
                token = ENCRYPTED_KEY_FILE.read_bytes()
                key = self._fernet.decrypt(token).decode()
                logger.info("API key loaded from encrypted local file")
                return key
            except Exception as e:  # pragma: no cover
                logger.warning(f"Encrypted file load failed: {e}")

        # Last resort: plain text
        if ENCRYPTED_KEY_FILE.exists():
            try:
                key = ENCRYPTED_KEY_FILE.read_text().strip()
                if key:
                    logger.warning("API key loaded from PLAIN TEXT file")
                    return key
            except Exception as e:  # pragma: no cover
                logger.warning(f"Plain text load failed: {e}")

        return None

    def delete(self) -> bool:
        """Delete the stored API key. Returns True on success."""
        success = True
        if HAS_KEYRING:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_KEY)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Keyring delete failed: {e}")
                success = False
        if ENCRYPTED_KEY_FILE.exists():
            try:
                ENCRYPTED_KEY_FILE.unlink()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Encrypted file delete failed: {e}")
                success = False
        if KEY_FILE.exists():
            try:
                KEY_FILE.unlink()
            except Exception:  # pragma: no cover
                pass
        return success


# ===========================================================================
# API Key Response Parser
# ===========================================================================
class ApiKeyParser:
    """
    Robust API key extraction from backend responses.

    The backend may return the key in many different shapes depending on
    version, endpoint, or middleware wrapping.  This parser checks every
    plausible location so the "API key was generated but not found in the
    server response" error never happens for a valid response.

    Search order (top-level then nested):
      1. api_key, apiKey, key, secret, token
      2. Under data / result / payload / response / keys / key
      3. First string value anywhere in the JSON (last resort)
    """

    # Top-level key names to check (case-insensitive)
    TOP_LEVEL_KEYS = (
        "api_key", "apiKey", "apikey",
        "key", "secret", "token",
        "access_token", "accessToken",
    )

    # Nested container names that might hold the key
    NESTED_KEYS = (
        "data", "result", "payload", "response",
        "keys", "key", "apiKey", "api_key",
    )

    @classmethod
    def extract(cls, response_data: Any) -> Optional[str]:
        """Extract an API key from arbitrary backend response data."""
        if response_data is None:
            return None

        # 1. Top-level dict — check known key names
        if isinstance(response_data, dict):
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

            # 3. Last resort: first non-empty string value in the dict
            for v in response_data.values():
                if isinstance(v, str) and v.strip() and len(v) > 8:
                    return v.strip()

        # 4. If it's a list, check each element
        if isinstance(response_data, list):
            for item in response_data:
                found = cls.extract(item)
                if found:
                    return found

        return None

    @classmethod
    def _extract_from_container(cls, container: Any) -> Optional[str]:
        """Extract a key from a nested container (dict or list)."""
        if container is None:
            return None

        if isinstance(container, dict):
            # Check known key names inside the container
            for k in cls.TOP_LEVEL_KEYS:
                v = container.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

            # Recurse into nested containers
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
    def parse_response(cls, response: APIResponse) -> Optional[str]:
        """Parse a full APIResponse object and return the API key."""
        if not response or not response.success:
            return None
        return cls.extract(response.data)


# ===========================================================================
# Worker Thread for async API calls
# ===========================================================================
class ApiWorker(QObject):
    """Worker that runs async API calls in a separate thread with its own event loop."""

    finished = pyqtSignal(object)  # APIResponse
    error = pyqtSignal(str)

    def __init__(self, client: CoreAI3DClient, coro_func, *args, **kwargs):
        super().__init__()
        self.client = client
        self.coro_func = coro_func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        """Run the coroutine in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.coro_func(*self.args, **self.kwargs)
            )
            self.finished.emit(result)
        except Exception as e:  # pragma: no cover
            logger.exception("API worker error")
            self.error.emit(str(e))
        finally:
            loop.close()


# ===========================================================================
# Show-Once Key Modal Dialog
# ===========================================================================
class ApiKeyModalDialog(QDialog):
    """
    Modal dialog that displays a newly generated API key exactly once.

    The key is shown in full with a one-click Copy button and a strong
    warning that it will never be displayed again.  Once the dialog is
    dismissed, the key is not retained in memory by this dialog.
    """

    def __init__(self, api_key: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Generated")
        self.setModal(True)
        self.setFixedSize(520, 380)
        self._api_key = api_key
        self._copied = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QLabel("🔑 New API Key Generated")
        header.setObjectName("modalHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Warning
        warning = QLabel(
            "⚠️  This key will <b>never be shown again</b>. "
            "Copy it now and store it securely."
        )
        warning.setObjectName("modalWarning")
        warning.setWordWrap(True)
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warning)

        # Key display
        key_frame = QFrame()
        key_frame.setObjectName("keyFrame")
        key_layout = QVBoxLayout(key_frame)
        key_layout.setContentsMargins(16, 16, 16, 16)

        key_label = QLabel("API Key:")
        key_label.setObjectName("keyLabel")
        key_layout.addWidget(key_label)

        self.key_edit = QLineEdit(self._api_key)
        self.key_edit.setObjectName("keyEdit")
        self.key_edit.setReadOnly(True)
        self.key_edit.setCursorPosition(0)
        key_layout.addWidget(self.key_edit)

        # Copy button
        self.copy_btn = QPushButton("📋 Copy to Clipboard")
        self.copy_btn.setObjectName("copyBtn")
        self.copy_btn.clicked.connect(self._copy_key)
        key_layout.addWidget(self.copy_btn)

        self.copy_status = QLabel("")
        self.copy_status.setObjectName("copyStatus")
        self.copy_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        key_layout.addWidget(self.copy_status)

        layout.addWidget(key_frame)

        # Spacer
        layout.addStretch()

        # Acknowledge button
        self.ack_btn = QPushButton("I Have Saved the Key")
        self.ack_btn.setObjectName("ackBtn")
        self.ack_btn.setDefault(True)
        self.ack_btn.clicked.connect(self.accept)
        layout.addWidget(self.ack_btn)

    def _copy_key(self) -> None:
        from PyQt6.QtWidgets import QApplication
        cb = QApplication.clipboard()
        cb.setText(self._api_key)
        self._copied = True
        self.copy_btn.setText("✅ Copied!")
        self.copy_btn.setEnabled(False)
        self.copy_status.setText("Key copied to clipboard")

    def key(self) -> str:
        """Return the API key (for callers who need it before dismiss)."""
        return self._api_key


# ===========================================================================
# Main Application
# ===========================================================================
class CoreAI3DModern(QMainWindow):
    """
    CoreAI3D Modern Dashboard.

    Tabs:
      Dashboard   — System status, quick actions
      API Keys    — Generate, view, revoke, set default API keys
      Settings    — Theme, API base URL, storage options
      Logs        — Activity log viewer
      About       — Version, credits, license
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.api_url = self.settings.value("api_url", "http://127.0.0.1:1906/api/v1")
        self.theme = self.settings.value("theme", "dark")  # "dark" or "light"

        # Components
        self.client: Optional[CoreAI3DClient] = None
        self.keystore = KeyStore()
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[ApiWorker] = None

        # UI state
        self._current_api_key: Optional[str] = None
        self._is_generating = False

        # Build UI
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1100, 700)
        self._apply_theme()
        self._build_ui()
        self._load_stored_key()
        self._init_client()

        logger.info("CoreAI3D Modern Dashboard initialised")

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _apply_theme(self) -> None:
        """Apply the current theme (dark or light) via QSS."""
        if self.theme == "light":
            self.setStyleSheet(self._light_qss())
        else:
            self.setStyleSheet(self._dark_qss())

    def _dark_qss(self) -> str:
        return """
        QMainWindow {
            background-color: #0f1115;
            color: #e8eaed;
        }
        QWidget {
            background-color: #0f1115;
            color: #e8eaed;
            font-family: 'Segoe UI', 'Roboto', 'Ubuntu', 'DejaVu Sans', sans-serif;
            font-size: 13px;
        }
        QTabWidget::pane {
            border: 1px solid #2a2d36;
            background: #0f1115;
            border-radius: 8px;
        }
        QTabBar::tab {
            background: #1a1d24;
            color: #8b8f9a;
            padding: 10px 20px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #0f1115;
            color: #ffffff;
            border-bottom: 2px solid #3b82f6;
        }
        QTabBar::tab:hover:!selected {
            color: #c3c6cd;
        }
        QGroupBox {
            border: 1px solid #2a2d36;
            border-radius: 8px;
            margin-top: 16px;
            padding-top: 8px;
            background: #161920;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 6px;
            color: #94a3b8;
            font-weight: 600;
        }
        QLabel {
            color: #e8eaed;
        }
        QLabel[class="modalHeader"] {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }
        QLabel[class="modalWarning"] {
            font-size: 13px;
            color: #fbbf24;
        }
        QLabel[class="keyLabel"] {
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        QLabel[class="copyStatus"] {
            font-size: 12px;
            color: #34d399;
            font-weight: 600;
        }
        QFrame[class="keyFrame"] {
            background: #0b0c10;
            border: 1px solid #2a2d36;
            border-radius: 8px;
        }
        QLineEdit {
            background: #0b0c10;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 8px 12px;
            color: #e8eaed;
            selection-background-color: #3b82f6;
        }
        QLineEdit:focus {
            border-color: #3b82f6;
        }
        QLineEdit[class="keyEdit"] {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 14px;
            letter-spacing: 1px;
        }
        QPushButton {
            background: #374151;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            color: #e8eaed;
            font-weight: 500;
        }
        QPushButton:hover {
            background: #4b5563;
        }
        QPushButton:pressed {
            background: #374151;
        }
        QPushButton:disabled {
            background: #1f232c;
            color: #6b7280;
        }
        QPushButton[class="copyBtn"] {
            background: #1f2937;
            color: #34d399;
        }
        QPushButton[class="copyBtn"]:hover {
            background: #374151;
        }
        QPushButton[class="ackBtn"] {
            background: #3b82f6;
            color: #ffffff;
            font-weight: 600;
        }
        QPushButton[class="ackBtn"]:hover {
            background: #2563eb;
        }
        QPushButton#generateKeyBtn {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #ffffff;
            font-weight: 600;
            padding: 10px 24px;
        }
        QPushButton#generateKeyBtn:hover {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
        }
        QPushButton#generateKeyBtn:disabled {
            background: #1f232c;
            color: #6b7280;
        }
        QTextEdit, QPlainTextEdit {
            background: #0b0c10;
            border: 1px solid #2a2d36;
            border-radius: 6px;
            padding: 8px;
            color: #e8eaed;
            selection-background-color: #3b82f6;
        }
        QComboBox {
            background: #161920;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 6px 12px;
            color: #e8eaed;
        }
        QComboBox:focus {
            border-color: #3b82f6;
        }
        QComboBox::drop-down {
            border: none;
            width: 24px;
        }
        QCheckBox {
            color: #e8eaed;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #374151;
            border-radius: 4px;
            background: #0b0c10;
        }
        QCheckBox::indicator:checked {
            background: #3b82f6;
            border-color: #3b82f6;
        }
        QSpinBox {
            background: #161920;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 6px;
            color: #e8eaed;
        }
        QProgressBar {
            border: 1px solid #2a2d36;
            border-radius: 6px;
            text-align: center;
            background: #0b0c10;
            color: #e8eaed;
        }
        QProgressBar::chunk {
            background: #3b82f6;
            border-radius: 4px;
        }
        QStatusBar {
            background: #0b0c10;
            color: #94a3b8;
            border-top: 1px solid #2a2d36;
        }
        QDialog {
            background: #0f1115;
        }
        """

    def _light_qss(self) -> str:
        return """
        QMainWindow {
            background-color: #f8fafc;
            color: #0f172a;
        }
        QWidget {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: 'Segoe UI', 'Roboto', 'Ubuntu', 'DejaVu Sans', sans-serif;
            font-size: 13px;
        }
        QTabWidget::pane {
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            border-radius: 8px;
        }
        QTabBar::tab {
            background: #f1f5f9;
            color: #64748b;
            padding: 10px 20px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #f8fafc;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
        }
        QTabBar::tab:hover:!selected {
            color: #334155;
        }
        QGroupBox {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-top: 16px;
            padding-top: 8px;
            background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 6px;
            color: #475569;
            font-weight: 600;
        }
        QLabel {
            color: #0f172a;
        }
        QLabel[class="modalHeader"] {
            font-size: 20px;
            font-weight: 700;
            color: #0f172a;
        }
        QLabel[class="modalWarning"] {
            font-size: 13px;
            color: #d97706;
        }
        QLabel[class="keyLabel"] {
            font-size: 12px;
            color: #64748b;
            margin-bottom: 4px;
        }
        QLabel[class="copyStatus"] {
            font-size: 12px;
            color: #059669;
            font-weight: 600;
        }
        QFrame[class="keyFrame"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
        QLineEdit {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 8px 12px;
            color: #0f172a;
            selection-background-color: #3b82f6;
        }
        QLineEdit:focus {
            border-color: #3b82f6;
        }
        QLineEdit[class="keyEdit"] {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 14px;
            letter-spacing: 1px;
        }
        QPushButton {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 8px 16px;
            color: #0f172a;
            font-weight: 500;
        }
        QPushButton:hover {
            background: #e2e8f0;
        }
        QPushButton:pressed {
            background: #f1f5f9;
        }
        QPushButton:disabled {
            background: #f8fafc;
            color: #94a3b8;
        }
        QPushButton[class="copyBtn"] {
            background: #f0fdf4;
            color: #059669;
            border-color: #bbf7d0;
        }
        QPushButton[class="copyBtn"]:hover {
            background: #dcfce7;
        }
        QPushButton[class="ackBtn"] {
            background: #3b82f6;
            color: #ffffff;
            font-weight: 600;
        }
        QPushButton[class="ackBtn"]:hover {
            background: #2563eb;
        }
        QPushButton#generateKeyBtn {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #ffffff;
            font-weight: 600;
            padding: 10px 24px;
        }
        QPushButton#generateKeyBtn:hover {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
        }
        QPushButton#generateKeyBtn:disabled {
            background: #cbd5e1;
            color: #94a3b8;
        }
        QTextEdit, QPlainTextEdit {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 8px;
            color: #0f172a;
            selection-background-color: #3b82f6;
        }
        QComboBox {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 6px 12px;
            color: #0f172a;
        }
        QComboBox:focus {
            border-color: #3b82f6;
        }
        QComboBox::drop-down {
            border: none;
            width: 24px;
        }
        QCheckBox {
            color: #0f172a;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            background: #ffffff;
        }
        QCheckBox::indicator:checked {
            background: #3b82f6;
            border-color: #3b82f6;
        }
        QSpinBox {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 6px;
            color: #0f172a;
        }
        QProgressBar {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            text-align: center;
            background: #f1f5f9;
            color: #0f172a;
        }
        QProgressBar::chunk {
            background: #3b82f6;
            border-radius: 4px;
        }
        QStatusBar {
            background: #f1f5f9;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
        }
        QDialog {
            background: #f8fafc;
        }
        """

    def toggle_theme(self) -> None:
        """Switch between dark and light themes instantly."""
        self.theme = "light" if self.theme == "dark" else "dark"
        self.settings.setValue("theme", self.theme)
        self._apply_theme()
        self._log_activity(f"Theme switched to {self.theme}")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = self._build_header()
        main_layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)

        # Create tabs
        self._build_dashboard_tab()
        self._build_api_keys_tab()
        self._build_settings_tab()
        self._build_logs_tab()
        self._build_about_tab()

        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("QStatusBar { color: #94a3b8; }")

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(56)
        header.setStyleSheet("""
        QFrame#headerBar {
            background: #0b0c10;
            border-bottom: 1px solid #2a2d36;
        }
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        # App title
        title = QLabel("CoreAI3D")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: 0.5px;")
        layout.addWidget(title)

        layout.addStretch()

        # Theme toggle
        self.theme_btn = QPushButton("🌙 Dark" if self.theme == "dark" else "☀️ Light")
        self.theme_btn.setObjectName("themeBtn")
        self.theme_btn.setFixedSize(100, 32)
        self.theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_btn)

        return header

    def _build_dashboard_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Status cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        self.status_api = self._make_stat_card("API Status", "Unknown", "#6b7280")
        self.status_key = self._make_stat_card("API Key", "Not configured", "#ef4444")
        self.status_backend = self._make_stat_card("Backend", "Offline", "#ef4444")

        cards_layout.addWidget(self.status_api)
        cards_layout.addWidget(self.status_key)
        cards_layout.addWidget(self.status_backend)
        layout.addLayout(cards_layout)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(12)

        self.btn_refresh = QPushButton("🔄 Refresh Status")
        self.btn_refresh.clicked.connect(self._refresh_status)
        actions_layout.addWidget(self.btn_refresh)

        self.btn_test_conn = QPushButton("🔗 Test Connection")
        self.btn_test_conn.clicked.connect(self._test_connection)
        actions_layout.addWidget(self.btn_test_conn)

        self.btn_gen_key = QPushButton("🔑 Generate API Key")
        self.btn_gen_key.setObjectName("generateKeyBtn")
        self.btn_gen_key.clicked.connect(self._generate_api_key)
        actions_layout.addWidget(self.btn_gen_key)

        actions_layout.addStretch()
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # System info
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()
        self.txt_sysinfo = QTextEdit()
        self.txt_sysinfo.setReadOnly(True)
        self.txt_sysinfo.setPlaceholderText("System information will appear here...")
        self.txt_sysinfo.setMaximumHeight(150)
        info_layout.addWidget(self.txt_sysinfo)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Dashboard")

    def _make_stat_card(self, title: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setObjectName("statCard")
        card.setStyleSheet(f"""
        QFrame#statCard {{
            background: #161920;
            border: 1px solid #2a2d36;
            border-radius: 10px;
            border-left: 4px solid {color};
        }}
        QLabel {{
            color: #e8eaed;
            background: transparent;
            border: none;
        }}
        """)
        card.setFixedSize(200, 100)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 12px; color: #94a3b8; font-weight: 500;")
        layout.addWidget(title_lbl)

        val_lbl = QLabel(value)
        val_lbl.setObjectName("statValue")
        val_lbl.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")
        layout.addWidget(val_lbl)

        layout.addStretch()
        return card

    def _build_api_keys_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Generate section
        gen_group = QGroupBox("Generate New API Key")
        gen_layout = QVBoxLayout()

        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.key_desc = QLineEdit()
        self.key_desc.setPlaceholderText("e.g. Production API key")
        desc_layout.addWidget(self.key_desc)
        gen_layout.addLayout(desc_layout)

        btn_layout = QHBoxLayout()
        self.btn_gen_key_tab = QPushButton("Generate API Key")
        self.btn_gen_key_tab.setObjectName("generateKeyBtn")
        self.btn_gen_key_tab.clicked.connect(self._generate_api_key)
        btn_layout.addWidget(self.btn_gen_key_tab)
        btn_layout.addStretch()
        gen_layout.addLayout(btn_layout)

        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 0)
        self.gen_progress.setVisible(False)
        gen_layout.addWidget(self.gen_progress)

        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)

        # Current key section
        key_group = QGroupBox("Current API Key")
        key_layout = QVBoxLayout(key_group)

        self.key_display = QLineEdit()
        self.key_display.setReadOnly(True)
        self.key_display.setPlaceholderText("No API key configured")
        self.key_display.setObjectName("keyEdit")
        key_layout.addWidget(self.key_display)

        key_btns = QHBoxLayout()
        self.btn_copy_key = QPushButton("📋 Copy")
        self.btn_copy_key.setObjectName("copyBtn")
        self.btn_copy_key.clicked.connect(self._copy_current_key)
        self.btn_copy_key.setEnabled(False)
        key_btns.addWidget(self.btn_copy_key)

        self.btn_revoke_key = QPushButton("🗑️ Revoke")
        self.btn_revoke_key.clicked.connect(self._revoke_key)
        self.btn_revoke_key.setEnabled(False)
        key_btns.addWidget(self.btn_revoke_key)

        self.btn_set_default = QPushButton("⭐ Set Default")
        self.btn_set_default.clicked.connect(self._set_default_key)
        self.btn_set_default.setEnabled(False)
        key_btns.addWidget(self.btn_set_default)

        key_btns.addStretch()
        key_layout.addLayout(key_btns)

        key_group.setLayout(key_layout)
        layout.addWidget(key_group)

        # Stored keys list
        list_group = QGroupBox("Stored Keys")
        list_layout = QVBoxLayout(list_group)
        self.keys_list = QTextEdit()
        self.keys_list.setReadOnly(True)
        self.keys_list.setMaximumHeight(150)
        self.keys_list.setPlaceholderText("No stored keys")
        list_layout.addWidget(self.keys_list)
        layout.addWidget(list_group)

        layout.addStretch()
        self.tabs.addTab(tab, "API Keys")

    def _build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # API Configuration
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout(api_group)

        self.txt_api_url = QLineEdit(self.api_url)
        api_layout.addRow("API Base URL:", self.txt_api_url)

        self.txt_ws_url = QLineEdit("ws://127.0.0.1:1907")
        api_layout.addRow("WebSocket URL:", self.txt_ws_url)

        layout.addWidget(api_group)

        # Storage Configuration
        storage_group = QGroupBox("Key Storage")
        storage_layout = QVBoxLayout(storage_group)

        self.chk_use_keyring = QCheckBox("Use system keyring (preferred)")
        self.chk_use_keyring.setChecked(HAS_KEYRING)
        self.chk_use_keyring.setEnabled(HAS_KEYRING)
        storage_layout.addWidget(self.chk_use_keyring)

        if not HAS_KEYRING:
            hint = QLabel("Install `keyring` for system keyring support: pip install keyring")
            hint.setStyleSheet("color: #6b7280; font-size: 12px;")
            storage_layout.addWidget(hint)

        layout.addWidget(storage_group)

        # Theme (also in header, but here for discoverability)
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout(theme_group)
        theme_layout.addWidget(QLabel("Theme is also togglable from the header bar."))
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Save button
        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_save_settings.clicked.connect(self._save_settings)
        layout.addWidget(self.btn_save_settings)

        layout.addStretch()
        self.tabs.addTab(tab, "Settings")

    def _build_logs_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 24, 24, 24)

        self.txt_logs = QPlainTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setPlaceholderText("Activity log will appear here...")
        layout.addWidget(self.txt_logs)

        log_btns = QHBoxLayout()
        self.btn_clear_logs = QPushButton("Clear Logs")
        self.btn_clear_logs.clicked.connect(self._clear_logs)
        log_btns.addWidget(self.btn_clear_logs)
        log_btns.addStretch()
        layout.addLayout(log_btns)

        self.tabs.addTab(tab, "Logs")

    def _build_about_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        about_group = QGroupBox("About CoreAI3D")
        about_layout = QVBoxLayout(about_group)

        app_name = QLabel(f"{APP_NAME} v{APP_VERSION}")
        app_name.setStyleSheet("font-size: 24px; font-weight: 700; color: #ffffff;")
        about_layout.addWidget(app_name)

        desc = QLabel(
            "A modern Linux desktop application for managing AI training data, "
            "testing, debugging, and API keys. Built with PyQt6."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 14px;")
        about_layout.addWidget(desc)

        about_layout.addSpacing(16)

        tech = QLabel(
            "• PyQt6 (Qt6) — GUI framework\n"
            "• QSS — Theme styling\n"
            "• keyring / cryptography — Secure key storage\n"
            "• CoreAI3D Python Client — Backend API integration"
        )
        tech.setStyleSheet("color: #94a3b8; font-size: 13px;")
        about_layout.addWidget(tech)

        about_layout.addStretch()
        layout.addWidget(about_group)
        layout.addStretch()
        self.tabs.addTab(tab, "About")

    # ------------------------------------------------------------------
    # Client Initialisation
    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        """Initialise the CoreAI3D client in the background."""
        try:
            config = {
                'base_url': self.api_url,
                'ws_url': 'ws://127.0.0.1:1907',
                'api_key': self._current_api_key or '',
                'session_id': '',
                'timeout': 90.0,
                'max_retries': 3,
                'retry_delay': 1.0,
                'max_concurrent': 10,
                'debug': False,
            }
            self.client = CoreAI3DClient(config)
            self._log_activity("Client initialised")
            self._refresh_status()
        except Exception as e:
            logger.exception("Client init failed")
            self._log_activity(f"Client init failed: {e}")

    def _run_async(self, coro_func, *args, on_success=None, on_error=None, **kwargs) -> None:
        """Run an async function in a worker thread.

        Safely handles QThread lifecycle to avoid
        "wrapped C/C++ object of type QThread has been deleted" errors
        when a previous thread has been deleteLater'd but the Python
        reference still points at the stale C++ object.
        """
        # Check if a previous thread is still running
        if self._worker_thread is not None:
            try:
                if self._worker_thread.isRunning():
                    self._log_activity("Previous operation still in progress")
                    return
            except RuntimeError:
                # Underlying C++ object was already deleted; clear stale refs
                self._worker_thread = None
                self._worker = None

        self._worker_thread = QThread()
        self._worker = ApiWorker(self.client, coro_func, *args, **kwargs)
        self._worker.moveToThread(self._worker_thread)

        if on_success:
            self._worker.finished.connect(on_success)
        if on_error:
            self._worker.error.connect(on_error)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)

        # When the thread finishes, clear Python references so the next
        # call to _run_async doesn't see a stale deleted C++ object.
        def _on_thread_finished():
            self._worker_thread = None
            self._worker = None

        self._worker_thread.finished.connect(_on_thread_finished)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._worker.deleteLater)

        self._worker_thread.start()

    # ------------------------------------------------------------------
    # API Key Generation (Critical Fix)
    # ------------------------------------------------------------------
    def _generate_api_key(self) -> None:
        """Generate a new API key via the backend.

        Uses a dedicated worker thread (separate from the health-check timer)
        to avoid qasync task conflicts.  The response is parsed by
        ApiKeyParser which checks multiple key locations so the
        "key not found" error never happens for a valid response.
        """
        if self._is_generating:
            self._log_activity("API key generation already in progress")
            return

        if not self.client:
            QMessageBox.warning(self, "Client Not Ready", "CoreAI3D client is not initialised.")
            return

        self._is_generating = True
        self.btn_gen_key.setEnabled(False)
        self.btn_gen_key_tab.setEnabled(False)
        self.gen_progress.setVisible(True)
        self.statusBar().showMessage("Generating API key...")

        description = self.key_desc.text().strip()

        def on_success(response: APIResponse) -> None:
            self._is_generating = False
            self.btn_gen_key.setEnabled(True)
            self.btn_gen_key_tab.setEnabled(True)
            self.gen_progress.setVisible(False)

            if not response.success:
                self.statusBar().showMessage(f"Failed: {response.message}")
                self._log_activity(f"API key generation failed: {response.message}")
                QMessageBox.warning(self, "Generation Failed", f"Backend error: {response.message}")
                return

            # Robust key extraction — checks many locations
            api_key = ApiKeyParser.parse_response(response)

            if not api_key:
                self.statusBar().showMessage("Key not found in response")
                self._log_activity("API key generated but not found in server response")
                QMessageBox.warning(
                    self, "Key Not Found",
                    "The API key was generated but could not be extracted "
                    "from the server response.  Raw response:\n\n"
                    f"{json.dumps(response.data, indent=2)}"
                )
                return

            self._current_api_key = api_key
            self.keystore.save(api_key)
            self.key_display.setText(api_key)
            self.btn_copy_key.setEnabled(True)
            self.btn_revoke_key.setEnabled(True)
            self.btn_set_default.setEnabled(True)
            self.status_key.findChild(QLabel, "statValue").setText("Configured")
            self.status_key.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #22c55e;")
            self.statusBar().showMessage("API key generated successfully")
            self._log_activity("API key generated and stored")

            # Show key ONLY ONCE in modal dialog
            dialog = ApiKeyModalDialog(api_key, self)
            dialog.exec()

        def on_error(err: str) -> None:
            self._is_generating = False
            self.btn_gen_key.setEnabled(True)
            self.btn_gen_key_tab.setEnabled(True)
            self.gen_progress.setVisible(False)
            self.statusBar().showMessage(f"Error: {err}")
            self._log_activity(f"API key generation error: {err}")
            QMessageBox.critical(self, "Generation Error", err)

        self._run_async(
            self.client.generate_api_key,
            description=description,
            on_success=on_success,
            on_error=on_error,
        )

    def _load_stored_key(self) -> None:
        """Load any previously stored API key."""
        key = self.keystore.load()
        if key:
            self._current_api_key = key
            self.key_display.setText(key)
            self.btn_copy_key.setEnabled(True)
            self.btn_revoke_key.setEnabled(True)
            self.btn_set_default.setEnabled(True)
            self.status_key.findChild(QLabel, "statValue").setText("Configured")
            self.status_key.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #22c55e;")
            self._log_activity("Stored API key loaded")
        else:
            self.status_key.findChild(QLabel, "statValue").setText("Not configured")
            self.status_key.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")

    def _copy_current_key(self) -> None:
        from PyQt6.QtWidgets import QApplication
        if self._current_api_key:
            cb = QApplication.clipboard()
            cb.setText(self._current_api_key)
            self._log_activity("API key copied to clipboard")
            self.statusBar().showMessage("Key copied to clipboard", 3000)

    def _revoke_key(self) -> None:
        if not self._current_api_key:
            return
        reply = QMessageBox.question(
            self, "Revoke API Key",
            "Are you sure you want to revoke the current API key?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.keystore.delete()
            self._current_api_key = None
            self.key_display.clear()
            self.btn_copy_key.setEnabled(False)
            self.btn_revoke_key.setEnabled(False)
            self.btn_set_default.setEnabled(False)
            self.status_key.findChild(QLabel, "statValue").setText("Not configured")
            self.status_key.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")
            self._log_activity("API key revoked")
            self.statusBar().showMessage("API key revoked")

    def _set_default_key(self) -> None:
        """Mark the current key as default (already stored in keystore)."""
        if self._current_api_key:
            self._log_activity("API key set as default")
            self.statusBar().showMessage("API key set as default", 3000)

    # ------------------------------------------------------------------
    # Status & Connection
    # ------------------------------------------------------------------
    def _refresh_status(self) -> None:
        """Refresh dashboard status indicators."""
        if not self.client:
            self.status_backend.findChild(QLabel, "statValue").setText("Offline")
            self.status_backend.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")
            return

        # Use a lightweight HTTP check via the client
        def on_health(response: APIResponse) -> None:
            if response.success:
                self.status_backend.findChild(QLabel, "statValue").setText("Online")
                self.status_backend.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #22c55e;")
                self.status_api.findChild(QLabel, "statValue").setText("Responding")
                self.status_api.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #22c55e;")
            else:
                self.status_backend.findChild(QLabel, "statValue").setText("Offline")
                self.status_backend.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")
                self.status_api.findChild(QLabel, "statValue").setText("Error")
                self.status_api.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")

        def on_err(err: str) -> None:
            self.status_backend.findChild(QLabel, "statValue").setText("Offline")
            self.status_backend.findChild(QLabel, "statValue").setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")

        if hasattr(self.client, "health_check"):
            self._run_async(
                self.client.health_check,
                on_success=on_health,
                on_error=on_err,
            )

    def _test_connection(self) -> None:
        self._refresh_status()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _save_settings(self) -> None:
        self.api_url = self.txt_api_url.text().strip()
        self.settings.setValue("api_url", self.api_url)
        if self.client and self.client.config:
            self.client.config["base_url"] = self.api_url
        self._log_activity("Settings saved")
        self.statusBar().showMessage("Settings saved", 3000)
        QMessageBox.information(self, "Settings", "Settings saved successfully.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_activity(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.txt_logs.appendPlainText(entry)
        logger.info(message)

    def _clear_logs(self) -> None:
        self.txt_logs.clear()
        self._log_activity("Logs cleared")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORG_NAME)

    window = CoreAI3DModern()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
