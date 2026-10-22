"""
test_edge_cases.py - Edge case and robustness tests.

Tests rapid clicking, tab switching, invalid input, window resizing,
and other edge conditions.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox
from PyQt6.QtCore import Qt

from dashboard import CoreAI3DDashboard


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_rapid_tab_switching(self, dashboard):
        """Rapidly switching between all tabs does not crash."""
        tab_count = dashboard.tab_widget.count()
        for i in range(tab_count):
            dashboard.tab_widget.setCurrentIndex(i)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        # Should not crash

    def test_rapid_button_clicking(self, dashboard):
        """Rapidly clicking buttons does not crash."""
        dashboard.tab_widget.setCurrentIndex(3)  # Testing tab
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.isEnabled():
                for _ in range(5):
                    btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_window_resize_small(self, dashboard):
        """Resizing window to very small size does not crash."""
        dashboard.resize(100, 100)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        dashboard.resize(1280, 720)  # Restore

    def test_window_resize_large(self, dashboard):
        """Resizing window to very large size does not crash."""
        dashboard.resize(3840, 2160)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        dashboard.resize(1280, 720)  # Restore

    def test_window_minimize_maximize(self, dashboard):
        """Minimizing and maximizing window does not crash."""
        dashboard.showMinimized()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        dashboard.showMaximized()
        QApplication.instance().processEvents()
        dashboard.showNormal()
        QApplication.instance().processEvents()

    def test_invalid_api_url(self, dashboard):
        """Entering invalid API URL does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)  # Settings
        dashboard.api_url_edit.setText("not-a-valid-url")
        dashboard.ws_url_edit.setText("also-not-valid")
        dashboard.test_settings_conn_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_empty_api_key_field(self, dashboard):
        """Leaving API key empty does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        dashboard.api_key_edit.clear()
        dashboard.generate_api_key_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_very_long_text_input(self, dashboard):
        """Very long text in input fields does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        long_text = "x" * 10000
        dashboard.api_url_edit.setText(long_text)
        dashboard.ws_url_edit.setText(long_text)
        dashboard.api_key_edit.setText(long_text)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_special_characters_in_input(self, dashboard):
        """Special characters in input fields do not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
        dashboard.api_url_edit.setText(special)
        dashboard.data_dir_edit.setText(special)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_unicode_text_input(self, dashboard):
        """Unicode text in input fields does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        unicode_text = "\u00e9\u00e8\u00ea\u00eb\u4e2d\u6587\u6f22\u5b57\u4e16\u754c\u2603\u2665"
        dashboard.api_url_edit.setText(unicode_text)
        dashboard.data_dir_edit.setText(unicode_text)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_spin_box_edge_values(self, dashboard):
        """Spin boxes at edge values do not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        dashboard.sandbox_timeout_edit.setValue(60)  # minimum
        dashboard.sandbox_timeout_edit.setValue(3600)  # maximum
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_combo_box_rapid_changes(self, dashboard):
        """Rapidly changing combo box selections does not crash."""
        dashboard.tab_widget.setCurrentIndex(3)  # Testing tab
        combo = dashboard.test_type_combo
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_text_edit_large_content(self, dashboard):
        """Large content in text edits does not crash."""
        dashboard.tab_widget.setCurrentIndex(4)  # Predict tab
        large_text = "x" * 50000
        dashboard.predict_input_text.setPlainText(large_text)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_table_selection_edge_cases(self, dashboard):
        """Table selection edge cases do not crash."""
        dashboard.tab_widget.setCurrentIndex(14)  # Beta Keys tab
        # Click copy with no selection
        dashboard.beta_copy_btn.click()
        # Click revoke with no selection
        dashboard.beta_revoke_btn.click()
        # Click send with no selection
        dashboard.beta_send_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_close_reopen_dashboard(self, dashboard, app):
        """Closing and reopening dashboard does not crash."""
        from PyQt6.QtWidgets import QApplication
        dashboard.close()
        QApplication.instance().processEvents()
        new_dashboard = CoreAI3DDashboard()
        new_dashboard.show()
        QApplication.instance().processEvents()
        new_dashboard.close()
        QApplication.instance().processEvents()
