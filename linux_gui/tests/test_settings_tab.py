"""
test_settings_tab.py - Tests for the Settings tab including API key generation.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox, QMessageBox
from PyQt6.QtCore import Qt


class TestSettingsTab:
    """Tests for the Settings tab."""

    def test_settings_tab_switches(self, dashboard):
        """Switching to Settings tab works."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert dashboard.tab_widget.currentIndex() == 13
        assert dashboard.tab_widget.tabText(13) == "Settings"

    def test_api_url_edit_has_value(self, dashboard):
        """API URL edit has a default value."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert dashboard.api_url_edit.text() != ""

    def test_ws_url_edit_has_value(self, dashboard):
        """WebSocket URL edit has a default value."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert dashboard.ws_url_edit.text() != ""

    def test_api_key_edit_is_password(self, dashboard):
        """API Key edit uses password echo mode."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert dashboard.api_key_edit.echoMode() == QLineEdit.EchoMode.Password

    def test_generate_api_key_button_exists(self, dashboard):
        """Generate API Key button exists."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'generate_api_key_btn')
        assert dashboard.generate_api_key_btn.text() == "Generate API Key"
        assert dashboard.generate_api_key_btn.isEnabled()
        tooltip = dashboard.generate_api_key_btn.toolTip()
        assert "Generate" in tooltip or "API key" in tooltip

    def test_test_connection_button_exists(self, dashboard):
        """Test Connection button exists in Settings."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'test_settings_conn_btn')
        assert dashboard.test_settings_conn_btn.text() == "Test Connection"

    def test_save_settings_button_exists(self, dashboard):
        """Save Settings button exists."""
        dashboard.tab_widget.setCurrentIndex(13)
        buttons = dashboard.findChildren(QPushButton)
        save_btn = None
        for btn in buttons:
            if btn.text() == "Save Settings":
                save_btn = btn
                break
        assert save_btn is not None, "Save Settings button not found"

    def test_data_dir_edit_exists(self, dashboard):
        """Data Directory edit exists."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'data_dir_edit')

    def test_sandbox_timeout_spin_exists(self, dashboard):
        """Sandbox timeout spin box exists."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'sandbox_timeout_edit')
        assert isinstance(dashboard.sandbox_timeout_edit, QSpinBox)

    def test_log_level_combo_exists(self, dashboard):
        """Log level combo exists with expected values."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'log_level_combo')
        assert isinstance(dashboard.log_level_combo, QComboBox)
        assert dashboard.log_level_combo.count() >= 4

    def test_diagnostics_check_exists(self, dashboard):
        """Enable Diagnostics checkbox exists."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'diagnostics_check')
        assert isinstance(dashboard.diagnostics_check, QCheckBox)

    def test_browse_button_exists(self, dashboard):
        """Browse button for data directory exists."""
        dashboard.tab_widget.setCurrentIndex(13)
        buttons = dashboard.findChildren(QPushButton)
        browse_btn = None
        for btn in buttons:
            if btn.text() == "Browse...":
                browse_btn = btn
                break
        assert browse_btn is not None

    def test_api_key_edit_accepts_text(self, dashboard):
        """API Key edit accepts and displays text."""
        dashboard.tab_widget.setCurrentIndex(13)
        test_key = "test_api_key_12345"
        dashboard.api_key_edit.setText(test_key)
        # Password mode hides text but text() returns the actual value
        assert dashboard.api_key_edit.text() == test_key

    def test_api_url_edit_accepts_text(self, dashboard):
        """API URL edit accepts and displays text."""
        dashboard.tab_widget.setCurrentIndex(13)
        test_url = "http://localhost:1906/api/v1"
        dashboard.api_url_edit.setText(test_url)
        assert dashboard.api_url_edit.text() == test_url

    def test_ws_url_edit_accepts_text(self, dashboard):
        """WebSocket URL edit accepts and displays text."""
        dashboard.tab_widget.setCurrentIndex(13)
        test_ws = "ws://localhost:1907"
        dashboard.ws_url_edit.setText(test_ws)
        assert dashboard.ws_url_edit.text() == test_ws

    def test_data_dir_edit_accepts_text(self, dashboard):
        """Data Directory edit accepts and displays text."""
        dashboard.tab_widget.setCurrentIndex(13)
        test_dir = "/tmp/test_data"
        dashboard.data_dir_edit.setText(test_dir)
        assert dashboard.data_dir_edit.text() == test_dir

    def test_sandbox_timeout_range(self, dashboard):
        """Sandbox timeout spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert dashboard.sandbox_timeout_edit.minimum() == 60
        assert dashboard.sandbox_timeout_edit.maximum() == 3600

    def test_log_level_combo_values(self, dashboard):
        """Log level combo has DEBUG, INFO, WARNING, ERROR."""
        dashboard.tab_widget.setCurrentIndex(13)
        items = [dashboard.log_level_combo.itemText(i) for i in range(dashboard.log_level_combo.count())]
        assert "DEBUG" in items
        assert "INFO" in items
        assert "WARNING" in items
        assert "ERROR" in items

    def test_generate_api_key_button_click(self, dashboard):
        """Clicking Generate API Key button does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        # The button requires a client to be initialized; without backend
        # it may show a warning dialog. We just verify it doesn't crash.
        dashboard.generate_api_key_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_test_settings_connection_click(self, dashboard):
        """Clicking Test Connection does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        dashboard.test_settings_conn_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_save_settings_click(self, dashboard):
        """Clicking Save Settings does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Save Settings":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break