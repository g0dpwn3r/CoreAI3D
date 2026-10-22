"""
conftest.py - Shared fixtures for CoreAI3D Linux GUI test suite.

Provides:
- app fixture: QApplication instance
- dashboard fixture: CoreAI3DDashboard main window
- screenshot_on_failure hook: captures screenshots when tests fail
- xvfb setup for headless testing
"""

import os
import sys
import logging
import time
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap

# Add linux_gui directory to path so we can import dashboard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy logging during tests
logging.basicConfig(level=logging.ERROR)
logging.getLogger('dashboard').setLevel(logging.ERROR)
logging.getLogger('coreai3d_client').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('aiohttp').setLevel(logging.ERROR)

from dashboard import CoreAI3DDashboard

# Directory for failure screenshots
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def app():
    """Create a QApplication instance for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("CoreAI3D Test")
    yield app


@pytest.fixture
def dashboard(app, request):
    """
    Create and show the main dashboard window.
    
    Yields the dashboard instance. On test failure, captures a screenshot.
    """
    dashboard = CoreAI3DDashboard()
    dashboard.show()
    
    # Process events to ensure UI is rendered
    app.processEvents()
    
    yield dashboard
    
    # Teardown: close dashboard and clean up
    try:
        dashboard.close()
    except Exception:
        pass
    app.processEvents()


@pytest.fixture
def wait_for_window():
    """Helper to wait for a window/dialog to appear."""
    def _wait(widget, timeout_ms=5000):
        """Wait for widget to become visible, processing events."""
        start = time.time()
        while not widget.isVisible():
            if (time.time() - start) * 1000 > timeout_ms:
                return False
            app = QApplication.instance()
            app.processEvents(50)
            time.sleep(0.05)
        return True
    return _wait


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture screenshot on test failure."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call" and report.failed:
        # Try to capture a screenshot from the dashboard
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for widget in app.topLevelWidgets():
                    if isinstance(widget, CoreAI3DDashboard):
                        screenshot_path = SCREENSHOT_DIR / f"{item.name}.png"
                        pixmap = widget.grab()
                        pixmap.save(str(screenshot_path))
                        report.extra = getattr(report, 'extra', [])
                        report.extra.append({
                            'screenshot': str(screenshot_path),
                            'test': item.name
                        })
                        break
        except Exception as e:
            # Screenshot capture failure should not mask the original failure
            pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "needs_backend: marks tests that require the CoreAI3D backend"
    )
    config.addinivalue_line(
        "markers", "ui: marks tests that exercise UI elements"
    )

