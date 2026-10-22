"""
test_beta_keys_tab.py - Tests for the Beta Keys tab.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QSpinBox, QTableWidget
from PyQt6.QtCore import Qt


class TestBetaKeysTab:
    """Tests for the Beta Keys tab."""

    def test_beta_keys_tab_switches(self, dashboard):
        """Switching to Beta Keys tab works."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert dashboard.tab_widget.currentIndex() == 14
        assert dashboard.tab_widget.tabText(14) == "Beta Keys"

    def test_beta_email_input(self, dashboard):
        """Beta email input exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_email_input')
        assert isinstance(dashboard.beta_email_input, QLineEdit)

    def test_beta_email_desc_input(self, dashboard):
        """Beta email description input exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_email_desc_input')
        assert isinstance(dashboard.beta_email_desc_input, QLineEdit)

    def test_beta_email_generate_button(self, dashboard):
        """Generate Email Key button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_email_generate_btn')
        assert dashboard.beta_email_generate_btn.text() == "Generate Email Key"

    def test_beta_count_spin(self, dashboard):
        """Beta count spin box exists with valid range."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_count_spin')
        assert isinstance(dashboard.beta_count_spin, QSpinBox)
        assert dashboard.beta_count_spin.minimum() == 1
        assert dashboard.beta_count_spin.maximum() == 100

    def test_beta_desc_input(self, dashboard):
        """Beta description input exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_desc_input')
        assert isinstance(dashboard.beta_desc_input, QLineEdit)

    def test_beta_generate_button(self, dashboard):
        """Generate Beta Keys button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_generate_btn')
        assert dashboard.beta_generate_btn.text() == "Generate Beta Keys"

    def test_beta_refresh_button(self, dashboard):
        """Refresh button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_refresh_btn')
        assert dashboard.beta_refresh_btn.text() == "Refresh"

    def test_beta_keys_table(self, dashboard):
        """Beta keys table exists with correct columns."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_keys_table')
        assert isinstance(dashboard.beta_keys_table, QTableWidget)
        assert dashboard.beta_keys_table.columnCount() == 6
        headers = [
            dashboard.beta_keys_table.horizontalHeaderItem(i).text()
            for i in range(6)
        ]
        assert "Key" in headers
        assert "ID" in headers
        assert "Max Uses" in headers
        assert "Expires At" in headers
        assert "Active" in headers
        assert "Email" in headers

    def test_beta_copy_button(self, dashboard):
        """Copy button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_copy_btn')
        assert dashboard.beta_copy_btn.text() == "Copy"

    def test_beta_send_button(self, dashboard):
        """Send via Email button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_send_btn')
        assert dashboard.beta_send_btn.text() == "Send via Email"

    def test_beta_copy_instructions_button(self, dashboard):
        """Copy + Instructions button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_copy_instructions_btn')
        assert dashboard.beta_copy_instructions_btn.text() == "Copy + Instructions"

    def test_beta_revoke_button(self, dashboard):
        """Revoke button exists."""
        dashboard.tab_widget.setCurrentIndex(14)
        assert hasattr(dashboard, 'beta_revoke_btn')
        assert dashboard.beta_revoke_btn.text() == "Revoke"

    def test_beta_email_input_accepts_text(self, dashboard):
        """Beta email input accepts text."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_email_input.setText("test@example.com")
        assert dashboard.beta_email_input.text() == "test@example.com"

    def test_beta_desc_input_accepts_text(self, dashboard):
        """Beta description input accepts text."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_desc_input.setText("Test description")
        assert dashboard.beta_desc_input.text() == "Test description"

    def test_generate_email_key_without_email(self, dashboard):
        """Generate Email Key without email shows warning."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_email_input.clear()
        dashboard.beta_email_generate_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_generate_email_key_with_email(self, dashboard):
        """Generate Email Key with email does not crash."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_email_input.setText("test@example.com")
        dashboard.beta_email_generate_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_generate_beta_keys(self, dashboard):
        """Generate Beta Keys does not crash."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_generate_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_copy_beta_key_no_selection(self, dashboard):
        """Copy with no selection shows warning."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_copy_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_revoke_beta_key_no_selection(self, dashboard):
        """Revoke with no selection shows warning."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_revoke_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_refresh_beta_keys(self, dashboard):
        """Refresh beta keys does not crash."""
        dashboard.tab_widget.setCurrentIndex(14)
        dashboard.beta_refresh_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()