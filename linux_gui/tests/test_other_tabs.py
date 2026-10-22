"""
test_other_tabs.py - Tests for remaining tabs: Predict, Load Model, Debugging,
Model Downloads, Neural Network, Training Data, Training Configuration.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QTextEdit, QComboBox, QCheckBox
from PyQt6.QtCore import Qt


class TestPredictTab:
    """Tests for the Predict tab."""

    def test_predict_tab_switches(self, dashboard):
        """Switching to Predict tab works."""
        dashboard.tab_widget.setCurrentIndex(4)
        assert dashboard.tab_widget.currentIndex() == 4
        assert dashboard.tab_widget.tabText(4) == "Predict"

    def test_predict_model_combo(self, dashboard):
        """Model type combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(4)
        items = [dashboard.predict_model_combo.itemText(i) for i in range(dashboard.predict_model_combo.count())]
        assert "Vision Models" in items
        assert "Text Models" in items

    def test_predict_input_type_combo(self, dashboard):
        """Input type combo has expected items."""
        dashboard.tab_widget.setCurrentIndex(4)
        items = [dashboard.predict_input_type_combo.itemText(i) for i in range(dashboard.predict_input_type_combo.count())]
        assert "Text" in items
        assert "Image" in items
        assert "File" in items

    def test_predict_input_text(self, dashboard):
        """Predict input text exists."""
        dashboard.tab_widget.setCurrentIndex(4)
        assert hasattr(dashboard, 'predict_input_text')
        assert isinstance(dashboard.predict_input_text, QTextEdit)

    def test_predict_checkboxes(self, dashboard):
        """Predict checkboxes exist."""
        dashboard.tab_widget.setCurrentIndex(4)
        assert hasattr(dashboard, 'predict_contains_header_check')
        assert hasattr(dashboard, 'predict_contains_text_check')

    def test_predict_file_path_edit(self, dashboard):
        """Predict file path edit exists."""
        dashboard.tab_widget.setCurrentIndex(4)
        assert hasattr(dashboard, 'predict_file_path_edit')

    def test_run_prediction_button(self, dashboard):
        """Run Prediction button exists."""
        dashboard.tab_widget.setCurrentIndex(4)
        buttons = dashboard.findChildren(QPushButton)
        run_btn = None
        for btn in buttons:
            if btn.text() == "Run Prediction":
                run_btn = btn
                break
        assert run_btn is not None

    def test_clear_prediction_button(self, dashboard):
        """Clear Results button exists in Predict tab."""
        dashboard.tab_widget.setCurrentIndex(4)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear Results":
                clear_btn = btn
                break
        assert clear_btn is not None

    def test_run_prediction_click(self, dashboard):
        """Clicking Run Prediction does not crash."""
        dashboard.tab_widget.setCurrentIndex(4)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Run Prediction":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break


class TestLoadModelTab:
    """Tests for the Load Model tab."""

    def test_load_model_tab_switches(self, dashboard):
        """Switching to Load Model tab works."""
        dashboard.tab_widget.setCurrentIndex(5)
        assert dashboard.tab_widget.currentIndex() == 5
        assert dashboard.tab_widget.tabText(5) == "Load Model"

    def test_load_model_identifier_edit(self, dashboard):
        """Model identifier edit exists."""
        dashboard.tab_widget.setCurrentIndex(5)
        assert hasattr(dashboard, 'load_model_identifier_edit')

    def test_load_model_type_combo(self, dashboard):
        """Model type combo exists."""
        dashboard.tab_widget.setCurrentIndex(5)
        assert hasattr(dashboard, 'load_model_type_combo')

    def test_load_model_button(self, dashboard):
        """Load Model button exists."""
        dashboard.tab_widget.setCurrentIndex(5)
        buttons = dashboard.findChildren(QPushButton)
        load_btn = None
        for btn in buttons:
            if btn.text() == "Load Model":
                load_btn = btn
                break
        assert load_btn is not None

    def test_clear_load_results_button(self, dashboard):
        """Clear Results button exists in Load Model tab."""
        dashboard.tab_widget.setCurrentIndex(5)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear Results":
                clear_btn = btn
                break
        assert clear_btn is not None


class TestDebuggingTab:
    """Tests for the Debugging tab."""

    def test_debugging_tab_switches(self, dashboard):
        """Switching to Debugging tab works."""
        dashboard.tab_widget.setCurrentIndex(7)
        assert dashboard.tab_widget.currentIndex() == 7
        assert dashboard.tab_widget.tabText(7) == "Debugging"

    def test_system_info(self, dashboard):
        """System info text edit exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        assert hasattr(dashboard, 'system_info')

    def test_process_table(self, dashboard):
        """Process table exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        assert hasattr(dashboard, 'process_table')

    def test_refresh_info_button(self, dashboard):
        """Refresh Info button exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        buttons = dashboard.findChildren(QPushButton)
        refresh_btn = None
        for btn in buttons:
            if btn.text() == "Refresh Info":
                refresh_btn = btn
                break
        assert refresh_btn is not None

    def test_refresh_processes_button(self, dashboard):
        """Refresh Processes button exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        buttons = dashboard.findChildren(QPushButton)
        refresh_btn = None
        for btn in buttons:
            if btn.text() == "Refresh Processes":
                refresh_btn = btn
                break
        assert refresh_btn is not None

    def test_api_debug_button(self, dashboard):
        """API Debug button exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        buttons = dashboard.findChildren(QPushButton)
        api_btn = None
        for btn in buttons:
            if btn.text() == "API Debug":
                api_btn = btn
                break
        assert api_btn is not None

    def test_system_debug_button(self, dashboard):
        """System Debug button exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        buttons = dashboard.findChildren(QPushButton)
        sys_btn = None
        for btn in buttons:
            if btn.text() == "System Debug":
                sys_btn = btn
                break
        assert sys_btn is not None

    def test_debug_log(self, dashboard):
        """Debug log text area exists."""
        dashboard.tab_widget.setCurrentIndex(7)
        assert hasattr(dashboard, 'debug_log')
        assert dashboard.debug_log.isReadOnly()


class TestModelDownloadsTab:
    """Tests for the Model Downloads tab."""

    def test_model_downloads_tab_switches(self, dashboard):
        """Switching to Model Downloads tab works."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert dashboard.tab_widget.currentIndex() == 9
        assert dashboard.tab_widget.tabText(9) == "Model Downloads"

    def test_model_search_input(self, dashboard):
        """Model search input exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert hasattr(dashboard, 'model_search_input')

    def test_model_type_combo(self, dashboard):
        """Model type filter combo exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert hasattr(dashboard, 'model_type_combo')

    def test_platform_combo(self, dashboard):
        """Platform combo exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert hasattr(dashboard, 'platform_combo')

    def test_model_table(self, dashboard):
        """Model table exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert hasattr(dashboard, 'model_table')

    def test_search_button(self, dashboard):
        """Search button exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        search_btn = None
        for btn in buttons:
            if btn.text() == "Search":
                search_btn = btn
                break
        assert search_btn is not None

    def test_refresh_models_button(self, dashboard):
        """Refresh button exists in Model Downloads tab."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        refresh_btn = None
        for btn in buttons:
            if btn.text() == "Refresh":
                refresh_btn = btn
                break
        assert refresh_btn is not None

    def test_download_selected_button(self, dashboard):
        """Download Selected button exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        download_btn = None
        for btn in buttons:
            if btn.text() == "Download Selected":
                download_btn = btn
                break
        assert download_btn is not None

    def test_cancel_download_button(self, dashboard):
        """Cancel Download button exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        cancel_btn = None
        for btn in buttons:
            if btn.text() == "Cancel Download":
                cancel_btn = btn
                break
        assert cancel_btn is not None

    def test_previous_page_button(self, dashboard):
        """Previous page button exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        prev_btn = None
        for btn in buttons:
            if btn.text() == "Previous":
                prev_btn = btn
                break
        assert prev_btn is not None

    def test_next_page_button(self, dashboard):
        """Next page button exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        next_btn = None
        for btn in buttons:
            if btn.text() == "Next":
                next_btn = btn
                break
        assert next_btn is not None

    def test_page_size_combo(self, dashboard):
        """Page size combo exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        assert hasattr(dashboard, 'page_size_combo')

    def test_download_fasttext_button(self, dashboard):
        """Download FastText Embeddings button exists."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        fasttext_btn = None
        for btn in buttons:
            if btn.text() == "Download FastText Embeddings":
                fasttext_btn = btn
                break
        assert fasttext_btn is not None

    def test_clear_download_log_button(self, dashboard):
        """Clear Log button exists in Model Downloads tab."""
        dashboard.tab_widget.setCurrentIndex(9)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear Log":
                clear_btn = btn
                break
        assert clear_btn is not None


class TestNeuralTab:
    """Tests for the Neural Network tab."""

    def test_neural_tab_switches(self, dashboard):
        """Switching to Neural Network tab works."""
        dashboard.tab_widget.setCurrentIndex(12)
        assert dashboard.tab_widget.currentIndex() == 12
        assert dashboard.tab_widget.tabText(12) == "Neural Network"

    def test_refresh_button_neural(self, dashboard):
        """Refresh button exists in Neural tab."""
        dashboard.tab_widget.setCurrentIndex(12)
        buttons = dashboard.findChildren(QPushButton)
        refresh_btn = None
        for btn in buttons:
            if btn.text() == "Refresh":
                refresh_btn = btn
                break
        assert refresh_btn is not None

    def test_auto_refresh_check(self, dashboard):
        """Auto-refresh checkbox exists."""
        dashboard.tab_widget.setCurrentIndex(12)
        assert hasattr(dashboard, 'auto_refresh_check')

    def test_topology_text(self, dashboard):
        """Topology text exists."""
        dashboard.tab_widget.setCurrentIndex(12)
        assert hasattr(dashboard, 'topology_text')


class TestTrainingDataTab:
    """Tests for the Training Data tab."""

    def test_training_data_tab_switches(self, dashboard):
        """Switching to Training Data tab works."""
        dashboard.tab_widget.setCurrentIndex(1)
        assert dashboard.tab_widget.currentIndex() == 1
        assert dashboard.tab_widget.tabText(1) == "Training Data"

    def test_dataset_list(self, dashboard):
        """Dataset list exists."""
        dashboard.tab_widget.setCurrentIndex(1)
        assert hasattr(dashboard, 'dataset_list')

    def test_new_dataset_button(self, dashboard):
        """New Dataset button exists."""
        dashboard.tab_widget.setCurrentIndex(1)
        buttons = dashboard.findChildren(QPushButton)
        new_btn = None
        for btn in buttons:
            if btn.text() == "New Dataset":
                new_btn = btn
                break
        assert new_btn is not None

    def test_load_dataset_button(self, dashboard):
        """Load Dataset button exists."""
        dashboard.tab_widget.setCurrentIndex(1)
        buttons = dashboard.findChildren(QPushButton)
        load_btn = None
        for btn in buttons:
            if btn.text() == "Load Dataset":
                load_btn = btn
                break
        assert load_btn is not None

    def test_upload_file_button(self, dashboard):
        """Upload File button exists."""
        dashboard.tab_widget.setCurrentIndex(1)
        buttons = dashboard.findChildren(QPushButton)
        upload_btn = None
        for btn in buttons:
            if btn.text() == "Upload File":
                upload_btn = btn
                break
        assert upload_btn is not None

    def test_delete_dataset_button(self, dashboard):
        """Delete Dataset button exists."""
        dashboard.tab_widget.setCurrentIndex(1)
        buttons = dashboard.findChildren(QPushButton)
        delete_btn = None
        for btn in buttons:
            if btn.text() == "Delete Dataset":
                delete_btn = btn
                break
        assert delete_btn is not None

    def test_learn_files_button(self, dashboard):
        """Learn from Files button exists."""
        dashboard.tab_widget.setCurrentIndex(1)
        buttons = dashboard.findChildren(QPushButton)
        learn_btn = None
        for btn in buttons:
            if btn.text() == "Learn from Files":
                learn_btn = btn
                break
        assert learn_btn is not None


class TestTrainingConfigurationTab:
    """Tests for the Training Configuration tab."""

    def test_training_config_tab_switches(self, dashboard):
        """Switching to Training Configuration tab works."""
        dashboard.tab_widget.setCurrentIndex(2)
        assert dashboard.tab_widget.currentIndex() == 2
        assert dashboard.tab_widget.tabText(2) == "Training Configuration"

    def test_norm_min_spin(self, dashboard):
        """Normalization min spin box exists."""
        dashboard.tab_widget.setCurrentIndex(2)
        assert hasattr(dashboard, 'norm_min_spin')

    def test_norm_max_spin(self, dashboard):
        """Normalization max spin box exists."""
        dashboard.tab_widget.setCurrentIndex(2)
        assert hasattr(dashboard, 'norm_max_spin')

    def test_neurons_spin(self, dashboard):
        """Neurons spin box exists."""
        dashboard.tab_widget.setCurrentIndex(2)
        assert hasattr(dashboard, 'neurons_spin')

    def test_samples_spin(self, dashboard):
        """Samples spin box exists."""
        dashboard.tab_widget.setCurrentIndex(2)
        assert hasattr(dashboard, 'samples_spin')