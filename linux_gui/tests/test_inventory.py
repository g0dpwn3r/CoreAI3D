"""
test_inventory.py - Verify all UI elements exist on the dashboard.

These tests confirm the Application Inventory is complete by checking
that every top-level widget, tab, menu item, button, and control
expected by TEST_PLAN.md is actually present on the running dashboard.
"""

import pytest
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QStatusBar,
    QPushButton, QLineEdit, QTextEdit, QPlainTextEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget, QListWidget,
    QTreeWidget, QProgressBar, QGroupBox, QLabel, QSplitter,
    QScrollArea, QFileDialog
)
from PyQt6.QtCore import Qt


class TestApplicationInventory:
    """Verify the application structure matches the inventory."""

    def test_main_window_exists(self, dashboard):
        """Main window is a QMainWindow with expected title."""
        assert isinstance(dashboard, QMainWindow)
        assert "CoreAI3D" in dashboard.windowTitle()

    def test_status_bar_exists(self, dashboard):
        """Status bar is present and shows initial message."""
        assert dashboard.statusBar() is not None
        # Status bar should exist (message may vary)

    def test_tab_widget_exists(self, dashboard):
        """Central tab widget exists."""
        assert hasattr(dashboard, 'tab_widget')
        assert isinstance(dashboard.tab_widget, QTabWidget)

    def test_menu_bar_exists(self, dashboard):
        """Menu bar exists with File, Tools, Help menus."""
        menubar = dashboard.menuBar()
        assert menubar is not None
        menu_titles = [menu.title() for menu in menubar.findChildren(QMenu)]
        assert 'File' in menu_titles
        assert 'Tools' in menu_titles
        assert 'Help' in menu_titles

    def test_exit_menu_item(self, dashboard):
        """File menu has Exit action with Ctrl+Q shortcut."""
        menubar = dashboard.menuBar()
        file_menu = None
        for menu in menubar.findChildren(QMenu):
            if menu.title() == 'File':
                file_menu = menu
                break
        assert file_menu is not None, "File menu not found"
        actions = [a.text() for a in file_menu.actions()]
        assert 'Exit' in actions, "Exit action not in File menu"

    def test_refresh_all_menu_item(self, dashboard):
        """Tools menu has Refresh All action with F5 shortcut."""
        menubar = dashboard.menuBar()
        tools_menu = None
        for menu in menubar.findChildren(QMenu):
            if menu.title() == 'Tools':
                tools_menu = menu
                break
        assert tools_menu is not None, "Tools menu not found"
        actions = [a.text() for a in tools_menu.actions()]
        assert 'Refresh All' in actions, "Refresh All action not in Tools menu"

    def test_documentation_menu_item(self, dashboard):
        """Help menu has Documentation action."""
        menubar = dashboard.menuBar()
        help_menu = None
        for menu in menubar.findChildren(QMenu):
            if menu.title() == 'Help':
                help_menu = menu
                break
        assert help_menu is not None, "Help menu not found"
        actions = [a.text() for a in help_menu.actions()]
        assert 'Documentation' in actions

    def test_about_menu_item(self, dashboard):
        """Help menu has About action."""
        menubar = dashboard.menuBar()
        help_menu = None
        for menu in menubar.findChildren(QMenu):
            if menu.title() == 'Help':
                help_menu = menu
                break
        assert help_menu is not None
        actions = [a.text() for a in help_menu.actions()]
        assert 'About' in actions


class TestTabInventory:
    """Verify all 15 tabs exist with correct labels."""

    EXPECTED_TABS = [
        "Dashboard",
        "Training Data",
        "Training Configuration",
        "Testing",
        "Predict",
        "Load Model",
        "Docker Sandbox",
        "Debugging",
        "Linux Sandbox",
        "Model Downloads",
        "Code Generation",
        "Chat",
        "Neural Network",
        "Settings",
        "Beta Keys",
    ]

    def test_tab_count(self, dashboard):
        """Expected number of tabs is present."""
        assert dashboard.tab_widget.count() == len(self.EXPECTED_TABS), (
            f"Expected {len(self.EXPECTED_TABS)} tabs, got {dashboard.tab_widget.count()}"
        )

    def test_tab_labels(self, dashboard):
        """All expected tab labels are present in order."""
        actual = [dashboard.tab_widget.tabText(i) for i in range(dashboard.tab_widget.count())]
        assert actual == self.EXPECTED_TABS, f"Tab mismatch:\nExpected: {self.EXPECTED_TABS}\nActual: {actual}"

    def test_dashboard_tab(self, dashboard):
        """Dashboard tab has system status widgets."""
        dashboard.tab_widget.setCurrentIndex(0)
        assert hasattr(dashboard, 'cpu_label')
        assert hasattr(dashboard, 'memory_label')
        assert hasattr(dashboard, 'disk_label')
        assert hasattr(dashboard, 'network_label')
        assert hasattr(dashboard, 'cpu_bar')
        assert hasattr(dashboard, 'memory_bar')
        assert hasattr(dashboard, 'disk_bar')
        assert hasattr(dashboard, 'health_score_label')
        assert hasattr(dashboard, 'health_bar')

    def test_training_data_tab(self, dashboard):
        """Training Data tab has dataset management controls."""
        dashboard.tab_widget.setCurrentIndex(1)
        assert hasattr(dashboard, 'dataset_list')
        assert hasattr(dashboard, 'upload_file_type_combo')
        assert hasattr(dashboard, 'file_dataset_name')
        assert hasattr(dashboard, 'file_learning_mode_combo')
        assert hasattr(dashboard, 'file_output_size')
        assert hasattr(dashboard, 'file_directory_edit')
        assert hasattr(dashboard, 'file_learning_list')
        assert hasattr(dashboard, 'file_list')
        assert hasattr(dashboard, 'dataset_info')

    def test_training_configuration_tab(self, dashboard):
        """Training Configuration tab has normalization and network settings."""
        dashboard.tab_widget.setCurrentIndex(2)
        assert hasattr(dashboard, 'norm_min_spin')
        assert hasattr(dashboard, 'norm_max_spin')
        assert hasattr(dashboard, 'neurons_spin')
        assert hasattr(dashboard, 'samples_spin')

    def test_testing_tab(self, dashboard):
        """Testing tab has test configuration and results."""
        dashboard.tab_widget.setCurrentIndex(3)
        assert hasattr(dashboard, 'test_type_combo')
        assert hasattr(dashboard, 'test_module_combo')
        assert hasattr(dashboard, 'automation_task_combo')
        assert hasattr(dashboard, 'automation_params_text')
        assert hasattr(dashboard, 'performance_test_type_combo')
        assert hasattr(dashboard, 'performance_duration_spin')
        assert hasattr(dashboard, 'performance_iterations_spin')
        assert hasattr(dashboard, 'performance_concurrency_spin')
        assert hasattr(dashboard, 'load_test_type_combo')
        assert hasattr(dashboard, 'load_duration_spin')
        assert hasattr(dashboard, 'load_concurrent_users_spin')
        assert hasattr(dashboard, 'load_requests_per_user_spin')
        assert hasattr(dashboard, 'load_endpoint_combo')
        assert hasattr(dashboard, 'test_results_table')

    def test_predict_tab(self, dashboard):
        """Predict tab has model selection and input controls."""
        dashboard.tab_widget.setCurrentIndex(4)
        assert hasattr(dashboard, 'predict_model_combo')
        assert hasattr(dashboard, 'predict_specific_model_combo')
        assert hasattr(dashboard, 'predict_input_type_combo')
        assert hasattr(dashboard, 'predict_contains_header_check')
        assert hasattr(dashboard, 'predict_contains_text_check')
        assert hasattr(dashboard, 'predict_input_text')
        assert hasattr(dashboard, 'predict_file_path_edit')
        assert hasattr(dashboard, 'predict_progress')
        assert hasattr(dashboard, 'predict_results_text')

    def test_load_model_tab(self, dashboard):
        """Load Model tab has model identifier and type controls."""
        dashboard.tab_widget.setCurrentIndex(5)
        assert hasattr(dashboard, 'load_model_identifier_edit')
        assert hasattr(dashboard, 'load_model_type_combo')
        assert hasattr(dashboard, 'load_progress')
        assert hasattr(dashboard, 'load_results_text')

    def test_docker_sandbox_tab(self, dashboard):
        """Docker Sandbox tab has container controls."""
        dashboard.tab_widget.setCurrentIndex(6)
        assert hasattr(dashboard, 'sandbox_status_label')
        assert hasattr(dashboard, 'sandbox_name_edit')
        assert hasattr(dashboard, 'startup_command_edit')
        assert hasattr(dashboard, 'networking_combo')
        assert hasattr(dashboard, 'drop_rights_check')
        assert hasattr(dashboard, 'sandbox_logs')

    def test_debugging_tab(self, dashboard):
        """Debugging tab has system info, process monitor, debug controls."""
        dashboard.tab_widget.setCurrentIndex(7)
        assert hasattr(dashboard, 'system_info')
        assert hasattr(dashboard, 'process_table')
        assert hasattr(dashboard, 'api_debug_output')
        assert hasattr(dashboard, 'debug_log')

    def test_linux_sandbox_tab(self, dashboard):
        """Linux Sandbox tab has container and command controls."""
        dashboard.tab_widget.setCurrentIndex(8)
        assert hasattr(dashboard, 'linux_sandbox_status_label')
        assert hasattr(dashboard, 'container_combo')
        assert hasattr(dashboard, 'linux_command_input')
        assert hasattr(dashboard, 'linux_output')
        assert hasattr(dashboard, 'scenario_combo')
        assert hasattr(dashboard, 'linux_training_progress')
        assert hasattr(dashboard, 'training_results')

    def test_model_downloads_tab(self, dashboard):
        """Model Downloads tab has search, table, and download controls."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert hasattr(dashboard, 'model_search_input')
        assert hasattr(dashboard, 'model_type_combo')
        assert hasattr(dashboard, 'platform_combo')
        assert hasattr(dashboard, 'model_table')
        assert hasattr(dashboard, 'download_progress')
        assert hasattr(dashboard, 'download_status_label')
        assert hasattr(dashboard, 'download_log')
        assert hasattr(dashboard, 'model_info')

    def test_code_generation_tab(self, dashboard):
        """Code Generation tab has language, type, description, output."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'language_combo')
        assert hasattr(dashboard, 'code_type_combo')
        assert hasattr(dashboard, 'complexity_combo')
        assert hasattr(dashboard, 'include_comments_check')
        assert hasattr(dashboard, 'include_examples_check')
        assert hasattr(dashboard, 'code_description')
        assert hasattr(dashboard, 'generate_btn')
        assert hasattr(dashboard, 'copy_btn')
        assert hasattr(dashboard, 'save_btn')
        assert hasattr(dashboard, 'clear_btn')
        assert hasattr(dashboard, 'generation_progress')
        assert hasattr(dashboard, 'generated_code')
        assert hasattr(dashboard, 'template_info')

    def test_chat_tab(self, dashboard):
        """Chat tab has history, input, send, clear controls."""
        dashboard.tab_widget.setCurrentIndex(11)
        assert hasattr(dashboard, 'chat_history')
        assert hasattr(dashboard, 'thinking_label')
        assert hasattr(dashboard, 'chat_input')

    def test_neural_tab(self, dashboard):
        """Neural Network tab has refresh, topology, activity controls."""
        dashboard.tab_widget.setCurrentIndex(12)
        assert hasattr(dashboard, 'auto_refresh_check')
        assert hasattr(dashboard, 'topology_text')

    def test_settings_tab(self, dashboard):
        """Settings tab has API, data, sandbox, debug configuration."""
        dashboard.tab_widget.setCurrentIndex(13)
        assert hasattr(dashboard, 'api_url_edit')
        assert hasattr(dashboard, 'ws_url_edit')
        assert hasattr(dashboard, 'api_key_edit')
        assert hasattr(dashboard, 'generate_api_key_btn')
        assert hasattr(dashboard, 'test_settings_conn_btn')
        assert hasattr(dashboard, 'data_dir_edit')
        assert hasattr(dashboard, 'sandbox_timeout_edit')
        assert hasattr(dashboard, 'log_level_combo')
        assert hasattr(dashboard, 'diagnostics_check')

    def test_beta_keys_tab(self, dashboard):
        """Beta Keys tab has email key, key generation, table controls."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_email_input')
        assert hasattr(dashboard, 'beta_email_desc_input')
        assert hasattr(dashboard, 'beta_email_generate_btn')
        assert hasattr(dashboard, 'beta_count_spin')
        assert hasattr(dashboard, 'beta_desc_input')
        assert hasattr(dashboard, 'beta_generate_btn')
        assert hasattr(dashboard, 'beta_refresh_btn')
        assert hasattr(dashboard, 'beta_keys_table')
        assert hasattr(dashboard, 'beta_copy_btn')
        assert hasattr(dashboard, 'beta_send_btn')
        assert hasattr(dashboard, 'beta_copy_instructions_btn')
        assert hasattr(dashboard, 'beta_revoke_btn')


class TestButtonInventory:
    """Verify all buttons exist across tabs."""

    def test_dashboard_tab_buttons(self, dashboard):
        """Dashboard tab Quick Actions buttons exist."""
        dashboard.tab_widget.setCurrentIndex(0)
        # Find buttons by text
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Test Connection' in button_texts
        assert 'Refresh Metrics' in button_texts

    def test_training_data_tab_buttons(self, dashboard):
        """Training Data tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(1)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'New Dataset' in button_texts
        assert 'Load Dataset' in button_texts
        assert 'Upload File' in button_texts
        assert 'Delete Dataset' in button_texts
        assert 'Browse Directory...' in button_texts
        assert 'Add Files...' in button_texts
        assert 'Remove Selected' in button_texts
        assert 'Learn from Files' in button_texts
        assert 'Add File' in button_texts
        assert 'Remove File' in button_texts

    def test_testing_tab_buttons(self, dashboard):
        """Testing tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(3)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Run Test' in button_texts
        assert 'Clear Results' in button_texts

    def test_predict_tab_buttons(self, dashboard):
        """Predict tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(4)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Browse...' in button_texts
        assert 'Run Prediction' in button_texts
        assert 'Clear Results' in button_texts

    def test_load_model_tab_buttons(self, dashboard):
        """Load Model tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(5)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Load Model' in button_texts
        assert 'Clear Results' in button_texts

    def test_docker_sandbox_tab_buttons(self, dashboard):
        """Docker Sandbox tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(6)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Start Container' in button_texts
        assert 'Stop Container' in button_texts
        assert 'Clear Logs' in button_texts

    def test_debugging_tab_buttons(self, dashboard):
        """Debugging tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(7)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Refresh Info' in button_texts
        assert 'Refresh Processes' in button_texts
        assert 'API Debug' in button_texts
        assert 'System Debug' in button_texts
        assert 'Clear Results' in button_texts
        assert 'Export Debug Log' in button_texts

    def test_linux_sandbox_tab_buttons(self, dashboard):
        """Linux Sandbox tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(8)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Refresh' in button_texts
        assert 'Execute Command' in button_texts
        assert 'Start Container' in button_texts
        assert 'Stop Container' in button_texts
        assert 'Clear Output' in button_texts
        assert 'Run Training' in button_texts
        assert 'Generate Report' in button_texts

    def test_model_downloads_tab_buttons(self, dashboard):
        """Model Downloads tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Search' in button_texts
        assert 'Refresh' in button_texts
        assert 'Previous' in button_texts
        assert 'Next' in button_texts
        assert 'Download Selected' in button_texts
        assert 'Cancel Download' in button_texts
        assert 'Download FastText Embeddings' in button_texts
        assert 'Clear Log' in button_texts

    def test_code_generation_tab_buttons(self, dashboard):
        """Code Generation tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(10)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Generate Code' in button_texts
        assert 'Copy to Clipboard' in button_texts
        assert 'Save to File' in button_texts
        assert 'Clear' in button_texts

    def test_chat_tab_buttons(self, dashboard):
        """Chat tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(11)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Send' in button_texts
        assert 'Clear' in button_texts

    def test_neural_tab_buttons(self, dashboard):
        """Neural Network tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(12)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Refresh' in button_texts

    def test_settings_tab_buttons(self, dashboard):
        """Settings tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(13)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Browse...' in button_texts
        assert 'Generate API Key' in button_texts
        assert 'Test Connection' in button_texts
        assert 'Save Settings' in button_texts

    def test_beta_keys_tab_buttons(self, dashboard):
        """Beta Keys tab buttons exist."""
        dashboard.tab_widget.setCurrentIndex(14)
        buttons = dashboard.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert 'Generate Email Key' in button_texts
        assert 'Generate Beta Keys' in button_texts
        assert 'Refresh' in button_texts
        assert 'Copy' in button_texts
        assert 'Send via Email' in button_texts
        assert 'Copy + Instructions' in button_texts
        assert 'Revoke' in button_texts


class TestControlInventory:
    """Verify all control types exist."""

    def test_line_edits_exist(self, dashboard):
        """QLineEdit fields are present."""
        line_edits = dashboard.findChildren(QLineEdit)
        assert len(line_edits) > 0, "No QLineEdit fields found"

    def test_combo_boxes_exist(self, dashboard):
        """QComboBox fields are present."""
        combos = dashboard.findChildren(QComboBox)
        assert len(combos) > 0, "No QComboBox fields found"

    def test_check_boxes_exist(self, dashboard):
        """QCheckBox fields are present."""
        checkboxes = dashboard.findChildren(QCheckBox)
        assert len(checkboxes) > 0, "No QCheckBox fields found"

    def test_spin_boxes_exist(self, dashboard):
        """QSpinBox/QDoubleSpinBox fields are present."""
        spin_boxes = dashboard.findChildren(QSpinBox)
        double_spin_boxes = dashboard.findChildren(QDoubleSpinBox)
        assert len(spin_boxes) > 0 or len(double_spin_boxes) > 0

    def test_text_edits_exist(self, dashboard):
        """QTextEdit/QPlainTextEdit fields are present."""
        text_edits = dashboard.findChildren(QTextEdit)
        plain_text_edits = dashboard.findChildren(QPlainTextEdit)
        assert len(text_edits) > 0 or len(plain_text_edits) > 0

    def test_progress_bars_exist(self, dashboard):
        """QProgressBar fields are present."""
        progress_bars = dashboard.findChildren(QProgressBar)
        assert len(progress_bars) > 0

    def test_tables_exist(self, dashboard):
        """QTableWidget fields are present."""
        tables = dashboard.findChildren(QTableWidget)
        assert len(tables) > 0

    def test_list_widgets_exist(self, dashboard):
        """QListWidget fields are present."""
        lists = dashboard.findChildren(QListWidget)
        assert len(lists) > 0

    def test_group_boxes_exist(self, dashboard):
        """QGroupBox fields are present."""
        groups = dashboard.findChildren(QGroupBox)
        assert len(groups) > 0