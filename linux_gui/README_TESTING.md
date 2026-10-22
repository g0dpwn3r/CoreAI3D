# CoreAI3D Linux GUI — Testing Guide

Complete guide to installing dependencies and running the automated test suite for the CoreAI3D Linux GUI Dashboard.

---

## 1. Prerequisites

### System Dependencies

The GUI requires PyQt6 and its system libraries. On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
  libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0
```

For headless testing (Xvfb):

```bash
sudo apt install xvfb
```

### Python Dependencies

From the `linux_gui` directory:

```bash
cd /mnt/data_ssd/CoreAI3D/linux_gui

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install GUI dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest pytest-html pytest-xvfb pytest-qt
```

> **Note:** `pytest-qt` is optional but recommended for Qt-specific testing utilities. The current suite uses standard `pytest` with `QApplication` fixtures directly.

---

## 2. Running the Tests

### From the linux_gui Directory

All test commands should be run from `/mnt/data_ssd/CoreAI3D/linux_gui/`.

### Full Test Suite (with HTML report)

```bash
# With display (desktop environment)
pytest tests/ -v --html=report.html --self-contained-html

# Headless (CI/server, no display)
xvfb-run -a pytest tests/ -v --html=report.html --self-contained-html
```

### Run Specific Test Files

```bash
# UI inventory tests (verify all elements exist)
pytest tests/test_inventory.py -v

# Settings tab (includes API key generation tests)
pytest tests/test_settings_tab.py -v

# Chat tab
pytest tests/test_chat_tab.py -v

# Edge cases
pytest tests/test_edge_cases.py -v

# Keyboard shortcuts
pytest tests/test_keyboard_shortcuts.py -v
```

### Run by Marker

```bash
# Run only UI tests (exclude slow/backend tests)
pytest tests/ -m "ui" -v

# Exclude tests that need backend
pytest tests/ -m "not needs_backend" -v
```

### Run a Single Test

```bash
pytest tests/test_settings_tab.py::TestSettingsTab::test_settings_tab_switches -v
```

---

## 3. Test Output

### HTML Report

When using `--html=report.html`, an HTML report is generated at:
```
/mnt/data_ssd/CoreAI3D/linux_gui/report.html
```

Open it in a browser to see pass/fail status for each test.

### Screenshots on Failure

When a test fails, a screenshot is captured and saved to:
```
/mnt/data_ssd/CoreAI3D/linux_gui/tests/screenshots/
```

Screenshots are named after the test function (e.g., `test_settings_tab_switches.png`).

### Logs

Application logs are written to:
```
/mnt/data_ssd/CoreAI3D/linux_gui/logs/dashboard.log
```

---

## 4. Test Suite Structure

```
tests/
├── __init__.py           # Package marker
├── conftest.py           # Shared fixtures (QApplication, dashboard, screenshots)
├── test_inventory.py     # All UI elements exist (windows, tabs, menus, buttons, controls)
├── test_dashboard_tab.py # Dashboard tab functionality
├── test_settings_tab.py  # Settings tab + API key generation
├── test_testing_tab.py   # Testing tab
├── test_chat_tab.py      # Chat tab
├── test_code_generation_tab.py  # Code generation tab
├── test_beta_keys_tab.py # Beta keys tab
├── test_beta_email.py    # Beta key email delivery (Resend transport, SMTP fallback)
├── test_sandbox_tabs.py  # Docker Sandbox + Linux Sandbox tabs
├── test_other_tabs.py    # Predict, Load Model, Debugging, Model Downloads, Neural, Training tabs
├── test_keyboard_shortcuts.py  # Ctrl+Q, F5, Enter, Tab, Escape
├── test_edge_cases.py    # Rapid clicking, resize, invalid input, etc.
└── screenshots/          # Auto-captured on test failure
```

---

## 5. Writing New Tests

### Adding a Test for a New UI Element

1. Add the element to `TEST_PLAN.md` inventory
2. Add a test method to the appropriate test class
3. Use `dashboard.findChildren(QPushButton)` or direct attribute access (e.g., `dashboard.api_url_edit`)

### Example: Testing a New Button

```python
def test_my_new_button(self, dashboard):
    """Test that My Button exists and clicking doesn't crash."""
    dashboard.tab_widget.setCurrentIndex(0)  # or appropriate tab
    buttons = dashboard.findChildren(QPushButton)
    my_btn = None
    for btn in buttons:
        if btn.text() == "My Button":
            my_btn = btn
            break
    assert my_btn is not None, "My Button not found"
    assert my_btn.isEnabled()
    
    my_btn.click()
    from PyQt6.QtWidgets import QApplication
    QApplication.instance().processEvents()
```

### Fixture Usage

- `dashboard` — Provides a fresh `CoreAI3DDashboard` instance per test, shown and ready
- `app` — Session-scoped `QApplication`
- `wait_for_window` — Helper to wait for a widget to become visible

---

## 6. Troubleshooting

### "No display environment variable"

Use `xvfb-run`:
```bash
xvfb-run -a pytest tests/ -v
```

### "ModuleNotFoundError: No module named 'PyQt6'"

Install requirements:
```bash
pip install -r requirements.txt
```

### "QXcbConnection: Could not connect to display"

Install XCB libraries:
```bash
sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
  libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0
```

### Tests hang or time out

Some tests interact with async operations (qasync). If a test hangs, it may be waiting for a backend response. Increase pytest timeout or run with `--timeout=30` (requires `pytest-timeout`).

---

## 7. CI Integration

Example GitLab CI / GitHub Actions step:

```yaml
test_gui:
  script:
    - apt-get update && apt-get install -y xvfb libxcb-cursor0
    - python3 -m venv .venv && source .venv/bin/activate
    - pip install -r requirements.txt pytest pytest-html
    - xvfb-run -a pytest tests/ -v --html=report.html --self-contained-html
  artifacts:
    paths:
      - report.html
      - tests/screenshots/
```
