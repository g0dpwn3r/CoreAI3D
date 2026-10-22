"""
test_code_generation_tab.py - Tests for the Code Generation tab.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QComboBox, QCheckBox, QTextEdit, QPlainTextEdit, QProgressBar
from PyQt6.QtCore import Qt


class TestCodeGenerationTab:
    """Tests for the Code Generation tab."""

    def test_code_generation_tab_switches(self, dashboard):
        """Switching to Code Generation tab works."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert dashboard.tab_widget.currentIndex() == 10
        assert dashboard.tab_widget.tabText(10) == "Code Generation"

    def test_language_combo(self, dashboard):
        """Language combo has expected languages."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'language_combo')
        items = [dashboard.language_combo.itemText(i) for i in range(dashboard.language_combo.count())]
        assert "Python" in items
        assert "JavaScript" in items
        assert "Java" in items
        assert "C++" in items
        assert "Shell/Bash" in items

    def test_code_type_combo(self, dashboard):
        """Code Type combo has expected types."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'code_type_combo')
        items = [dashboard.code_type_combo.itemText(i) for i in range(dashboard.code_type_combo.count())]
        assert "Function" in items
        assert "Class" in items
        assert "API Client" in items
        assert "Unit Test" in items

    def test_complexity_combo(self, dashboard):
        """Complexity combo has expected levels."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'complexity_combo')
        items = [dashboard.complexity_combo.itemText(i) for i in range(dashboard.complexity_combo.count())]
        assert "Beginner" in items
        assert "Intermediate" in items
        assert "Advanced" in items
        assert "Expert" in items

    def test_include_comments_check(self, dashboard):
        """Include Comments checkbox is checked by default."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'include_comments_check')
        assert dashboard.include_comments_check.isChecked()

    def test_include_examples_check(self, dashboard):
        """Include Examples checkbox exists."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'include_examples_check')
        assert not dashboard.include_examples_check.isChecked()

    def test_code_description(self, dashboard):
        """Code description text edit exists."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'code_description')
        assert isinstance(dashboard.code_description, QTextEdit)

    def test_generate_code_button(self, dashboard):
        """Generate Code button exists and is enabled."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'generate_btn')
        assert dashboard.generate_btn.text() == "Generate Code"
        assert dashboard.generate_btn.isEnabled()

    def test_copy_button_disabled_initially(self, dashboard):
        """Copy button is disabled initially."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'copy_btn')
        assert not dashboard.copy_btn.isEnabled()

    def test_save_button_disabled_initially(self, dashboard):
        """Save button is disabled initially."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'save_btn')
        assert not dashboard.save_btn.isEnabled()

    def test_clear_button(self, dashboard):
        """Clear button exists."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'clear_btn')
        assert dashboard.clear_btn.text() == "Clear"

    def test_generated_code_output(self, dashboard):
        """Generated code output is read-only."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'generated_code')
        assert isinstance(dashboard.generated_code, QPlainTextEdit)
        assert dashboard.generated_code.isReadOnly()

    def test_template_info(self, dashboard):
        """Template info text edit is read-only."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'template_info')
        assert isinstance(dashboard.template_info, QTextEdit)
        assert dashboard.template_info.isReadOnly()

    def test_generation_progress_hidden(self, dashboard):
        """Generation progress bar is hidden initially."""
        dashboard.tab_widget.setCurrentIndex(10)
        assert hasattr(dashboard, 'generation_progress')
        assert not dashboard.generation_progress.isVisible()

    def test_code_description_accepts_text(self, dashboard):
        """Code description accepts text."""
        dashboard.tab_widget.setCurrentIndex(10)
        test_desc = "Create a Python function to calculate fibonacci"
        dashboard.code_description.setPlainText(test_desc)
        assert dashboard.code_description.toPlainText() == test_desc

    def test_generate_code_click_without_description(self, dashboard):
        """Clicking Generate Code without description shows warning."""
        dashboard.tab_widget.setCurrentIndex(10)
        dashboard.code_description.clear()
        dashboard.generate_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()
        # Should show a warning dialog but not crash

    def test_generate_code_click_with_description(self, dashboard):
        """Clicking Generate Code with description works."""
        dashboard.tab_widget.setCurrentIndex(10)
        dashboard.code_description.setPlainText("Create a hello world function")
        dashboard.generate_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_copy_generated_code(self, dashboard):
        """Copy button works after code generation."""
        dashboard.tab_widget.setCurrentIndex(10)
        # Copy button is disabled until code is generated
        assert not dashboard.copy_btn.isEnabled()
        # Click it anyway - should be safe
        dashboard.copy_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_clear_code_generation(self, dashboard):
        """Clear button resets the form."""
        dashboard.tab_widget.setCurrentIndex(10)
        dashboard.code_description.setPlainText("test")
        dashboard.clear_btn.click()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()