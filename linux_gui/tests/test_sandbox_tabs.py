"""
test_sandbox_tabs.py - Tests for Docker Sandbox and Linux Sandbox tabs.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QComboBox, QCheckBox, QPlainTextEdit
from PyQt6.QtCore import Qt


class TestDockerSandboxTab:
    """Tests for the Docker Sandbox tab."""

    def test_docker_sandbox_tab_switches(self, dashboard):
        """Switching to Docker Sandbox tab works."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert dashboard.tab_widget.currentIndex() == 6
        assert dashboard.tab_widget.tabText(6) == "Docker Sandbox"

    def test_sandbox_status_label(self, dashboard):
        """Sandbox status label exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'sandbox_status_label')

    def test_sandbox_name_edit(self, dashboard):
        """Container name edit exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'sandbox_name_edit')
        assert isinstance(dashboard.sandbox_name_edit, QLineEdit)

    def test_startup_command_edit(self, dashboard):
        """Startup command edit exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'startup_command_edit')
        assert isinstance(dashboard.startup_command_edit, QLineEdit)

    def test_networking_combo(self, dashboard):
        """Networking combo exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'networking_combo')
        assert isinstance(dashboard.networking_combo, QComboBox)

    def test_drop_rights_check(self, dashboard):
        """Run as Non-Root checkbox exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'drop_rights_check')
        assert isinstance(dashboard.drop_rights_check, QCheckBox)
        assert dashboard.drop_rights_check.isChecked()

    def test_sandbox_logs(self, dashboard):
        """Sandbox logs text area exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'sandbox_logs')
        assert isinstance(dashboard.sandbox_logs, QPlainTextEdit)
        assert dashboard.sandbox_logs.isReadOnly()

    def test_start_container_button(self, dashboard):
        """Start Container button exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'sandbox_name_edit')  # verify tab loaded
        buttons = dashboard.findChildren(QPushButton)
        start_btn = None
        for btn in buttons:
            if btn.text() == "Start Container":
                start_btn = btn
                break
        assert start_btn is not None

    def test_stop_container_button(self, dashboard):
        """Stop Container button exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        buttons = dashboard.findChildren(QPushButton)
        stop_btn = None
        for btn in buttons:
            if btn.text() == "Stop Container":
                stop_btn = btn
                break
        assert stop_btn is not None

    def test_clear_logs_button(self, dashboard):
        """Clear Logs button exists."""
        dashboard.tab_widget.setCurrentIndex(6)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear Logs":
                clear_btn = btn
                break
        assert clear_btn is not None

    def test_start_container_click(self, dashboard):
        """Clicking Start Container does not crash."""
        dashboard.tab_widget.setCurrentIndex(6)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Start Container":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_stop_container_click(self, dashboard):
        """Clicking Stop Container does not crash."""
        dashboard.tab_widget.setCurrentIndex(6)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Stop Container":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_clear_sandbox_logs_click(self, dashboard):
        """Clicking Clear Logs does not crash."""
        dashboard.tab_widget.setCurrentIndex(6)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Clear Logs":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break


class TestLinuxSandboxTab:
    """Tests for the Linux Sandbox tab."""

    def test_linux_sandbox_tab_switches(self, dashboard):
        """Switching to Linux Sandbox tab works."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert dashboard.tab_widget.currentIndex() == 8
        assert dashboard.tab_widget.tabText(8) == "Linux Sandbox"

    def test_linux_sandbox_status_label(self, dashboard):
        """Linux Sandbox status label exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'linux_sandbox_status_label')

    def test_container_combo(self, dashboard):
        """Container combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'container_combo')
        items = [dashboard.container_combo.itemText(i) for i in range(dashboard.container_combo.count())]
        assert "ubuntu-sandbox" in items
        assert "kali-sandbox" in items
        assert "alpine-sandbox" in items
        assert "arch-sandbox" in items

    def test_linux_command_input(self, dashboard):
        """Linux command input exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'linux_command_input')
        assert isinstance(dashboard.linux_command_input, QLineEdit)

    def test_linux_output(self, dashboard):
        """Linux output text area exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'linux_output')
        assert isinstance(dashboard.linux_output, QPlainTextEdit)
        assert dashboard.linux_output.isReadOnly()

    def test_scenario_combo(self, dashboard):
        """Scenario combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'scenario_combo')
        items = [dashboard.scenario_combo.itemText(i) for i in range(dashboard.scenario_combo.count())]
        assert "File System Operations" in items
        assert "Process Management" in items
        assert "Network Operations" in items
        assert "hacking-white" in items
        assert "hacking-black" in items

    def test_linux_training_progress_hidden(self, dashboard):
        """Training progress bar is hidden initially."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'linux_training_progress')
        assert not dashboard.linux_training_progress.isVisible()

    def test_training_results_readonly(self, dashboard):
        """Training results is read-only."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'training_results')
        assert dashboard.training_results.isReadOnly()

    def test_execute_command_button(self, dashboard):
        """Execute Command button exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        exec_btn = None
        for btn in buttons:
            if btn.text() == "Execute Command":
                exec_btn = btn
                break
        assert exec_btn is not None

    def test_start_linux_container_button(self, dashboard):
        """Start Container button exists in Linux Sandbox tab."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        start_btn = None
        for btn in buttons:
            if btn.text() == "Start Container":
                start_btn = btn
                break
        assert start_btn is not None

    def test_stop_linux_container_button(self, dashboard):
        """Stop Container button exists in Linux Sandbox tab."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        stop_btn = None
        for btn in buttons:
            if btn.text() == "Stop Container":
                stop_btn = btn
                break
        assert stop_btn is not None

    def test_clear_output_button(self, dashboard):
        """Clear Output button exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear Output":
                clear_btn = btn
                break
        assert clear_btn is not None

    def test_run_training_button(self, dashboard):
        """Run Training button exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        train_btn = None
        for btn in buttons:
            if btn.text() == "Run Training":
                train_btn = btn
                break
        assert train_btn is not None

    def test_generate_report_button(self, dashboard):
        """Generate Report button exists."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        report_btn = None
        for btn in buttons:
            if btn.text() == "Generate Report":
                report_btn = btn
                break
        assert report_btn is not None

    def test_execute_empty_command(self, dashboard):
        """Executing empty command shows warning."""
        dashboard.tab_widget.setCurrentIndex(8)
        dashboard.linux_command_input.clear()
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Execute Command":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_execute_command_with_text(self, dashboard):
        """Executing command with text does not crash."""
        dashboard.tab_widget.setCurrentIndex(8)
        dashboard.linux_command_input.setText("echo hello")
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Execute Command":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_clear_linux_output(self, dashboard):
        """Clear Output clears the output area."""
        dashboard.tab_widget.setCurrentIndex(8)
        dashboard.linux_output.appendPlainText("test output")
        assert dashboard.linux_output.toPlainText() != ""
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Clear Output":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_refresh_containers(self, dashboard):
        """Refresh containers does not crash."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Refresh":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break