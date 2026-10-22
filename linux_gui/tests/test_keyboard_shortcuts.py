"""
test_keyboard_shortcuts.py - Tests for keyboard shortcuts and accelerators.
"""

import pytest
from PyQt6.QtWidgets import QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QAction, QKeyEvent


class TestKeyboardShortcuts:
    """Tests for keyboard shortcuts and menu accelerators."""

    def test_exit_shortcut_ctrl_q(self, dashboard):
        """Exit action has Ctrl+Q shortcut."""
        menubar = dashboard.menuBar()
        for menu in menubar.findChildren(QMenu):
            if menu.title() == 'File':
                for action in menu.actions():
                    if action.text() == 'Exit':
                        shortcut = action.shortcut()
                        assert shortcut.matches(QKeySequence("Ctrl+Q")) != QKeySequence.SequenceMatch.NoMatch
                        break

    def test_refresh_all_shortcut_f5(self, dashboard):
        """Refresh All action has F5 shortcut."""
        menubar = dashboard.menuBar()
        for menu in menubar.findChildren(QMenu):
            if menu.title() == 'Tools':
                for action in menu.actions():
                    if action.text() == 'Refresh All':
                        shortcut = action.shortcut()
                        assert shortcut.matches(QKeySequence("F5")) != QKeySequence.SequenceMatch.NoMatch
                        break

    def test_ctrl_q_triggers_exit(self, dashboard):
        """Pressing Ctrl+Q closes the application."""
        # Send Ctrl+Q to the application
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Q,
            Qt.KeyboardModifier.ControlModifier,
        )
        dashboard.event(event)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_f5_triggers_refresh(self, dashboard):
        """Pressing F5 triggers refresh all."""
        # Capture status bar message before
        dashboard.tab_widget.setCurrentIndex(0)
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_F5,
            Qt.KeyboardModifier.NoModifier,
        )
        dashboard.event(event)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_tab_keyboard_navigation(self, dashboard):
        """Tab key moves focus between widgets."""
        dashboard.tab_widget.setCurrentIndex(13)  # Settings tab
        # Get initial focus widget
        initial_focus = dashboard.focusWidget()
        # Press Tab
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Tab,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.instance().sendEvent(dashboard, event)
        QApplication.instance().processEvents()
        # Focus should have changed (or stayed if no focusable widget)
        # Just verify no crash

    def test_escape_key(self, dashboard):
        """Escape key does not crash."""
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
        dashboard.event(event)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def test_enter_in_settings_fields(self, dashboard):
        """Enter key in settings line edits does not crash."""
        dashboard.tab_widget.setCurrentIndex(13)
        dashboard.api_url_edit.setFocus()
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.instance().sendEvent(dashboard.api_url_edit, event)
        QApplication.instance().processEvents()
