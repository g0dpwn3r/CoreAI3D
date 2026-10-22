"""
test_testing_tab.py - Tests for the Testing tab.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QComboBox, QTextEdit, QTableWidget, QSpinBox
from PyQt6.QtCore import Qt


class TestTestingTab:
    """Tests for the Testing tab."""

    def test_testing_tab_switches(self, dashboard):
        """Switching to Testing tab works."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.tab_widget.currentIndex() == 3
        assert dashboard.tab_widget.tabText(3) == "Testing"

    def test_test_type_combo(self, dashboard):
        """Test Type combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'test_type_combo')
        items = [dashboard.test_type_combo.itemText(i) for i in range(dashboard.test_type_combo.count())]
        assert "API Connectivity" in items
        assert "Module Functionality" in items
        assert "System Automation" in items
        assert "Performance Test" in items
        assert "Load Test" in items

    def test_test_module_combo(self, dashboard):
        """Module combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'test_module_combo')
        items = [dashboard.test_module_combo.itemText(i) for i in range(dashboard.test_module_combo.count())]
        assert "Vision Module" in items
        assert "Audio Module" in items
        assert "System Module" in items
        assert "Web Module" in items
        assert "Math Module" in items

    def test_automation_task_combo(self, dashboard):
        """Automation Task combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'automation_task_combo')
        items = [dashboard.automation_task_combo.itemText(i) for i in range(dashboard.automation_task_combo.count())]
        assert "file_backup" in items
        assert "system_cleanup" in items
        assert "log_rotation" in items
        assert "service_restart" in items
        assert "custom_script" in items

    def test_automation_params_text(self, dashboard):
        """Automation Parameters text edit exists."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'automation_params_text')
        assert isinstance(dashboard.automation_params_text, QTextEdit)

    def test_performance_config_group(self, dashboard):
        """Performance Test configuration group exists."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'performance_test_type_combo')
        assert hasattr(dashboard, 'performance_duration_spin')
        assert hasattr(dashboard, 'performance_iterations_spin')
        assert hasattr(dashboard, 'performance_concurrency_spin')

    def test_load_config_group(self, dashboard):
        """Load Test configuration group exists."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'load_test_type_combo')
        assert hasattr(dashboard, 'load_duration_spin')
        assert hasattr(dashboard, 'load_concurrent_users_spin')
        assert hasattr(dashboard, 'load_requests_per_user_spin')
        assert hasattr(dashboard, 'load_endpoint_combo')

    def test_test_results_table(self, dashboard):
        """Test results table exists with correct columns."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'test_results_table')
        assert isinstance(dashboard.test_results_table, QTableWidget)
        assert dashboard.test_results_table.columnCount() == 5
        headers = [
            dashboard.test_results_table.horizontalHeaderItem(i).text()
            for i in range(5)
        ]
        assert "Test ID" in headers
        assert "Type" in headers
        assert "Status" in headers
        assert "Duration" in headers
        assert "Message" in headers

    def test_run_test_button(self, dashboard):
        """Run Test button exists."""
        dashboard.tab_widget.setCurrentIndex(3)
        buttons = dashboard.findChildren(QPushButton)
        run_btn = None
        for btn in buttons:
            if btn.text() == "Run Test":
                run_btn = btn
                break
        assert run_btn is not None

    def test_clear_results_button(self, dashboard):
        """Clear Results button exists in Testing tab."""
        dashboard.tab_widget.setCurrentIndex(3)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear Results":
                clear_btn = btn
                break
        assert clear_btn is not None

    def test_performance_duration_spin_range(self, dashboard):
        """Performance duration spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.performance_duration_spin.minimum() == 1
        assert dashboard.performance_duration_spin.maximum() == 300

    def test_performance_iterations_spin_range(self, dashboard):
        """Performance iterations spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.performance_iterations_spin.minimum() == 100
        assert dashboard.performance_iterations_spin.maximum() == 1000000

    def test_performance_concurrency_spin_range(self, dashboard):
        """Performance concurrency spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.performance_concurrency_spin.minimum() == 1
        assert dashboard.performance_concurrency_spin.maximum() == 100

    def test_load_duration_spin_range(self, dashboard):
        """Load duration spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.load_duration_spin.minimum() == 10
        assert dashboard.load_duration_spin.maximum() == 600

    def test_load_concurrent_users_spin_range(self, dashboard):
        """Load concurrent users spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.load_concurrent_users_spin.minimum() == 1
        assert dashboard.load_concurrent_users_spin.maximum() == 1000

    def test_load_requests_per_user_spin_range(self, dashboard):
        """Load requests per user spin box has valid range."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert dashboard.load_requests_per_user_spin.minimum() == 10
        assert dashboard.load_requests_per_user_spin.maximum() == 10000

    def test_run_test_click(self, dashboard):
        """Clicking Run Test does not crash."""
        dashboard.tab_widget.setCurrentIndex(3)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Run Test":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_clear_test_results_click(self, dashboard):
        """Clicking Clear Results does not crash."""
        dashboard.tab_widget.setCurrentIndex(3)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Clear Results":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_test_type_change_triggers(self, dashboard):
        """Changing test type shows/hides config groups."""
        dashboard.tab_widget.setCurrentIndex(3)
        # Initially hidden
        assert not dashboard.automation_config_group.isVisible()
        assert not dashboard.performance_config_group.isVisible()
        assert not dashboard.load_config_group.isVisible()

        # Select System Automation
        dashboard.test_type_combo.setCurrentText("System Automation")
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        assert dashboard.automation_config_group.isVisible()

        # Select Performance Test
        dashboard.test_type_combo.setCurrentText("Performance Test")
        QApplication.instance().processEvents()
        assert dashboard.performance_config_group.isVisible()

        # Select Load Test
        dashboard.test_type_combo.setCurrentText("Load Test")
        QApplication.instance().processEvents()
        assert dashboard.load_config_group.isVisible()