"""
test_dashboard_tab.py - Tests for the Dashboard tab (system overview).
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QProgressBar, QLabel
from PyQt6.QtCore import Qt


class TestDashboardTab:
    """Tests for the Dashboard tab."""

    def test_dashboard_tab_switches(self, dashboard):
        """Switching to Dashboard tab works."""
        dashboard.tab_widget.setCurrentIndex(0)
        assert dashboard.tab_widget.currentIndex() == 0
        assert dashboard.tab_widget.tabText(0) == "Dashboard"

    def test_system_status_labels_exist(self, dashboard):
        """System status labels are present and have expected text."""
        dashboard.tab_widget.setCurrentIndex(0)
        assert dashboard.cpu_label.text() == "CPU Usage: 0%"
        assert dashboard.memory_label.text() == "Memory Usage: 0%"
        assert dashboard.disk_label.text() == "Disk Usage: 0%"
        assert dashboard.network_label.text() == "Network: 0 KB/s"

    def test_progress_bars_exist(self, dashboard):
        """Progress bars are present with valid ranges."""
        dashboard.tab_widget.setCurrentIndex(0)
        assert dashboard.cpu_bar.minimum() == 0
        assert dashboard.cpu_bar.maximum() == 100
        assert dashboard.memory_bar.minimum() == 0
        assert dashboard.memory_bar.maximum() == 100
        assert dashboard.disk_bar.minimum() == 0
        assert dashboard.disk_bar.maximum() == 100

    def test_health_score_label(self, dashboard):
        """Health score label shows initial value."""
        dashboard.tab_widget.setCurrentIndex(0)
        assert "100%" in dashboard.health_score_label.text()

    def test_health_bar_range(self, dashboard):
        """Health bar has range 0-100."""
        dashboard.tab_widget.setCurrentIndex(0)
        assert dashboard.health_bar.minimum() == 0
        assert dashboard.health_bar.maximum() == 100
        assert dashboard.health_bar.value() == 100

    def test_test_connection_button_exists(self, dashboard):
        """Test Connection button exists in Quick Actions."""
        dashboard.tab_widget.setCurrentIndex(0)
        buttons = dashboard.findChildren(QPushButton)
        test_btn = None
        for btn in buttons:
            if btn.text() == "Test Connection":
                test_btn = btn
                break
        assert test_btn is not None, "Test Connection button not found"
        assert test_btn.isEnabled()

    def test_refresh_metrics_button_exists(self, dashboard):
        """Refresh Metrics button exists in Quick Actions."""
        dashboard.tab_widget.setCurrentIndex(0)
        buttons = dashboard.findChildren(QPushButton)
        refresh_btn = None
        for btn in buttons:
            if btn.text() == "Refresh Metrics":
                refresh_btn = btn
                break
        assert refresh_btn is not None, "Refresh Metrics button not found"

    def test_test_connection_click(self, dashboard):
        """Clicking Test Connection does not crash (may fail without backend)."""
        dashboard.tab_widget.setCurrentIndex(0)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Test Connection":
                btn.click()
                # Process events to allow any dialogs to appear
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_refresh_metrics_click(self, dashboard):
        """Clicking Refresh Metrics does not crash."""
        dashboard.tab_widget.setCurrentIndex(0)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Refresh Metrics":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_dashboard_tab_has_no_crash_on_render(self, dashboard):
        """Dashboard tab renders without exceptions."""
        dashboard.tab_widget.setCurrentIndex(0)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        # If we get here without exception, the tab rendered successfully
        assert True