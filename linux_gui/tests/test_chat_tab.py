"""
test_chat_tab.py - Tests for the Chat tab.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QTextEdit
from PyQt6.QtCore import Qt


class TestChatTab:
    """Tests for the Chat tab."""

    def test_chat_tab_switches(self, dashboard):
        """Switching to Chat tab works."""
        dashboard.tab_widget.setCurrentIndex(11)
        assert dashboard.tab_widget.currentIndex() == 11
        assert dashboard.tab_widget.tabText(11) == "Chat"

    def test_chat_history_exists(self, dashboard):
        """Chat history text edit exists and is read-only."""
        dashboard.tab_widget.setCurrentIndex(11)
        assert hasattr(dashboard, 'chat_history')
        assert isinstance(dashboard.chat_history, QTextEdit)
        assert dashboard.chat_history.isReadOnly()

    def test_thinking_label_exists(self, dashboard):
        """Thinking indicator label exists."""
        dashboard.tab_widget.setCurrentIndex(11)
        assert hasattr(dashboard, 'thinking_label')
        assert not dashboard.thinking_label.isVisible()

    def test_chat_input_exists(self, dashboard):
        """Chat input line edit exists."""
        dashboard.tab_widget.setCurrentIndex(11)
        assert hasattr(dashboard, 'chat_input')
        assert isinstance(dashboard.chat_input, QLineEdit)
        assert "Type your message" in dashboard.chat_input.placeholderText()

    def test_send_button_exists(self, dashboard):
        """Send button exists."""
        dashboard.tab_widget.setCurrentIndex(11)
        buttons = dashboard.findChildren(QPushButton)
        send_btn = None
        for btn in buttons:
            if btn.text() == "Send":
                send_btn = btn
                break
        assert send_btn is not None

    def test_clear_chat_button_exists(self, dashboard):
        """Clear button exists in Chat tab."""
        dashboard.tab_widget.setCurrentIndex(11)
        buttons = dashboard.findChildren(QPushButton)
        clear_btn = None
        for btn in buttons:
            if btn.text() == "Clear":
                clear_btn = btn
                break
        assert clear_btn is not None

    def test_chat_input_accepts_text(self, dashboard):
        """Chat input accepts and displays text."""
        dashboard.tab_widget.setCurrentIndex(11)
        test_msg = "Hello, AI!"
        dashboard.chat_input.setText(test_msg)
        assert dashboard.chat_input.text() == test_msg

    def test_send_empty_message(self, dashboard):
        """Sending empty message does not crash."""
        dashboard.tab_widget.setCurrentIndex(11)
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Send":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_send_message(self, dashboard):
        """Sending a message does not crash."""
        dashboard.tab_widget.setCurrentIndex(11)
        dashboard.chat_input.setText("Test message")
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Send":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break

    def test_clear_chat_history(self, dashboard):
        """Clearing chat history works."""
        dashboard.tab_widget.setCurrentIndex(11)
        dashboard.chat_history.append("Test message")
        assert dashboard.chat_history.toPlainText() != ""
        buttons = dashboard.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() == "Clear":
                btn.click()
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
                break
        assert dashboard.chat_history.toPlainText() == ""

    def test_chat_input_return_pressed(self, dashboard):
        """Pressing Return in chat input sends message."""
        dashboard.tab_widget.setCurrentIndex(11)
        dashboard.chat_input.setText("Return key test")
        dashboard.chat_input.returnPressed.emit()
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()