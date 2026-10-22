# CoreAI3D Linux GUI — Test Plan & Master Checklist

**Application:** CoreAI3D Linux GUI Dashboard v1.3.0
**Toolkit:** PyQt6 (Qt6)
**Main Window:** `CoreAI3DDashboard` (QMainWindow) — "CoreAI3D Linux Dashboard"
**Entry Point:** `linux_gui/run_dashboard.sh` or `python3 run_dashboard.py`

---

## 1. Application Inventory

### 1.1 Top-Level Windows & Dialogs

| Window/Dialog | Class | How to Open |
|---|---|---|
| Main Dashboard Window | `CoreAI3DDashboard` | Application launch |
| About Dialog | `QMessageBox.about` | Help > About |
| Documentation Dialog | `QMessageBox.information` | Help > Documentation |
| QFileDialog | `QFileDialog.getExistingDirectory` / `getOpenFileName` | Settings > Browse... / Predict > Browse... |
| QMessageBox (warning/info) | `QMessageBox.warning` / `information` / `critical` | Various error/confirmation dialogs |

### 1.2 Tabs / Notebook Pages (15 tabs)

| Index | Tab Label | Key Controls |
|---|---|---|
| 0 | Dashboard | CPU/memory/disk labels, progress bars, Test Connection, Refresh Metrics, Health Score |
| 1 | Training Data | Dataset list, New Dataset, Load Dataset, Upload File, Delete Dataset, file learning controls, dataset details |
| 2 | Training Configuration | Normalization (min/max), Network architecture (neurons, samples, epochs, layers, learning rate) |
| 3 | Testing | Test Type, Module, Automation/Performance/Load config, Run Test, Clear Results, results table |
| 4 | Predict | Model Type, Specific Model, Input Type, checkboxes, input text, file path, Run Prediction, Clear Results |
| 5 | Load Model | Model Identifier, Model Type, Load Model, Clear Results, results |
| 6 | Docker Sandbox | Status, Container Name, Startup Command, Networking, Non-Root checkbox, Start/Stop Container, Clear Logs, logs |
| 7 | Debugging | System Info, Process Table, Refresh Info/Processes, API Debug, System Debug, Clear Results, Export Debug Log, debug messages |
| 8 | Linux Sandbox | Status, Container combo, Refresh, Command Input, Execute/Start/Stop Container, Clear Output, Output, Scenario combo, Run Training, Generate Report, progress, results |
| 9 | Model Downloads | Search input, Type/Platform combos, Search/Refresh, Pagination (Prev/Next/Page size), Model table, Download/Cancel, FastText, Clear Log, progress, model info |
| 10 | Code Generation | Language, Code Type, Complexity, Include Comments/Examples, Description, Generate/Copy/Save/Clear, progress, generated code, template info |
| 11 | Chat | Chat history, Thinking label, Chat input, Send, Clear |
| 12 | Neural Network | Refresh, Auto-refresh checkbox, Network Topology, Network Activity |
| 13 | Settings | API URL, WebSocket URL, API Key (password), Generate API Key, Test Connection, Save Settings, Data Dir, Browse, Sandbox Timeout, Log Level, Enable Diagnostics |
| 14 | Beta Keys | Email input, Description, Generate Email Key, Count, Description, Generate Beta Keys, Refresh, Keys table (6 cols), Copy, Send via Email, Copy + Instructions, Revoke |

### 1.3 Menu Bar

| Menu | Items | Shortcut |
|---|---|---|
| File | Exit | Ctrl+Q |
| Tools | Refresh All | F5 |
| Help | Documentation, About | — |

### 1.4 Keyboard Shortcuts & Accelerators

| Shortcut | Action |
|---|---|
| Ctrl+Q | Exit application (File > Exit) |
| F5 | Refresh All data (Tools > Refresh All) |
| Enter (in Chat input) | Send chat message |
| Enter (in Linux command input) | Execute Linux command |
| Tab | Keyboard navigation between widgets |

### 1.5 Control Inventory Summary

| Control Type | Count | Example |
|---|---|---|
| QPushButton | ~80+ | Test Connection, Generate API Key, Send, etc. |
| QLineEdit | ~20+ | API URL, Chat input, Command input, etc. |
| QTextEdit | ~15+ | Chat history, Dataset info, Predict input, etc. |
| QPlainTextEdit | ~10+ | Debug log, Sandbox logs, Generated code, etc. |
| QComboBox | ~25+ | Test Type, Language, Container, etc. |
| QCheckBox | ~10+ | Include Comments, Auto-refresh, Non-Root, etc. |
| QSpinBox/QDoubleSpinBox | ~15+ | Timeout, Neurons, Duration, etc. |
| QProgressBar | ~8+ | CPU, Memory, Disk, Health, Download, etc. |
| QTableWidget | ~5+ | Test results, Process table, Model table, Beta keys, etc. |
| QListWidget | ~5+ | Dataset list, File learning list, etc. |
| QGroupBox | ~40+ | System Status, API Configuration, etc. |

---

## 2. Master Manual Test Checklist

Copy this checklist into your test management tool or print it for manual execution.

### 2.1 Main Window

- [ ] **MW-01**: Application launches and shows main window titled "CoreAI3D Linux Dashboard"
  - Steps: Run `run_dashboard.sh`
  - Expected: Window appears, 15 tabs visible, menu bar present, status bar shows "Ready"
- [ ] **MW-02**: Window resizes without crashing
  - Steps: Drag window edges to small (200x200) and large (3840x2160) sizes
  - Expected: No crash, layout adapts
- [ ] **MW-03**: Window minimize/maximize/restore works
  - Steps: Click minimize, then maximize, then restore
  - Expected: Window state changes correctly, no crash
- [ ] **MW-04**: Status bar present with initial message
  - Steps: Observe status bar at bottom
  - Expected: Status bar visible, shows "Ready"

### 2.2 Menu Bar

- [ ] **MB-01**: File > Exit closes application (Ctrl+Q)
  - Steps: Click File > Exit, or press Ctrl+Q
  - Expected: Application closes
- [ ] **MB-02**: Tools > Refresh All refreshes all data (F5)
  - Steps: Click Tools > Refresh All, or press F5
  - Expected: Status bar shows "All data refreshed", data updates
- [ ] **MB-03**: Help > Documentation shows info dialog
  - Steps: Click Help > Documentation
  - Expected: Dialog with documentation text appears
- [ ] **MB-04**: Help > About shows about dialog
  - Steps: Click Help > About
  - Expected: Dialog with version 1.3.0 appears

### 2.3 Dashboard Tab

- [ ] **DB-01**: Tab switches to Dashboard
  - Steps: Click "Dashboard" tab
  - Expected: Dashboard content visible
- [ ] **DB-02**: CPU/memory/disk/network labels present
  - Steps: Observe labels
  - Expected: Labels show "CPU Usage: 0%", etc.
- [ ] **DB-03**: Progress bars present with 0-100 range
  - Steps: Observe progress bars
  - Expected: Bars visible, range 0-100
- [ ] **DB-04**: Health score shows 100%
  - Steps: Observe health score label and bar
  - Expected: Label shows "Overall Health Score: 100%", bar at 100
- [ ] **DB-05**: Test Connection button works
  - Steps: Click "Test Connection"
  - Expected: No crash (may show warning without backend)
- [ ] **DB-06**: Refresh Metrics button works
  - Steps: Click "Refresh Metrics"
  - Expected: No crash, metrics update

### 2.4 Training Data Tab

- [ ] **TD-01**: Tab switches to Training Data
  - Steps: Click "Training Data" tab
  - Expected: Training Data content visible
- [ ] **TD-02**: Dataset list present
  - Steps: Observe list widget
  - Expected: List widget visible
- [ ] **TD-03**: New Dataset button works
  - Steps: Click "New Dataset"
  - Expected: No crash
- [ ] **TD-04**: Load Dataset button works
  - Steps: Click "Load Dataset"
  - Expected: No crash
- [ ] **TD-05**: Upload File button works
  - Steps: Click "Upload File"
  - Expected: No crash
- [ ] **TD-06**: Delete Dataset button works
  - Steps: Click "Delete Dataset"
  - Expected: No crash
- [ ] **TD-07**: Browse Directory button opens file dialog
  - Steps: Click "Browse Directory..."
  - Expected: QFileDialog opens
- [ ] **TD-08**: Add Files button works
  - Steps: Click "Add Files..."
  - Expected: No crash
- [ ] **TD-09**: Remove Selected button works
  - Steps: Click "Remove Selected"
  - Expected: No crash
- [ ] **TD-10**: Learn from Files button works
  - Steps: Click "Learn from Files"
  - Expected: No crash

### 2.5 Settings Tab (Critical — API Key Generation)

- [ ] **ST-01**: Tab switches to Settings
  - Steps: Click "Settings" tab
  - Expected: Settings content visible
- [ ] **ST-02**: API URL field populated
  - Steps: Observe API URL field
  - Expected: Non-empty URL (default http://127.0.0.1:1906/api/v1)
- [ ] **ST-03**: WebSocket URL field populated
  - Steps: Observe WebSocket URL field
  - Expected: Non-empty URL
- [ ] **ST-04**: API Key field uses password mode
  - Steps: Observe API Key field echo mode
  - Expected: Characters masked (PasswordEchoOnEdit)
- [ ] **ST-05**: Generate API Key button present and enabled
  - Steps: Observe button
  - Expected: Button visible, enabled, tooltip mentions "Generate"
- [ ] **ST-06**: Test Connection button present
  - Steps: Observe button
  - Expected: Button visible
- [ ] **ST-07**: Save Settings button present
  - Steps: Observe button
  - Expected: Button visible
- [ ] **ST-08**: API key text entry works
  - Steps: Type test key into API Key field
  - Expected: Text accepted (masked)
- [ ] **ST-09**: API URL text entry works
  - Steps: Type URL into API URL field
  - Expected: Text accepted
- [ ] **ST-10**: Data Directory field accepts text
  - Steps: Type path into Data Directory field
  - Expected: Text accepted
- [ ] **ST-11**: Sandbox timeout spin box range 60-3600
  - Steps: Observe spin box
  - Expected: Min 60, Max 3600
- [ ] **ST-12**: Log level combo has DEBUG/INFO/WARNING/ERROR
  - Steps: Open combo dropdown
  - Expected: All four levels present
- [ ] **ST-13**: Enable Diagnostics checkbox toggles
  - Steps: Click checkbox
  - Expected: Checked state toggles

### 2.6 Beta Keys Tab

- [ ] **BK-01**: Tab switches to Beta Keys
  - Steps: Click "Beta Keys" tab
  - Expected: Beta Keys content visible
- [ ] **BK-02**: Email input present
  - Steps: Observe email field
  - Expected: Field visible with placeholder
- [ ] **BK-03**: Generate Email Key button works
  - Steps: Click "Generate Email Key"
  - Expected: No crash (shows warning without email/backend)
- [ ] **BK-04**: Count spin box range 1-100
  - Steps: Observe spin box
  - Expected: Min 1, Max 100
- [ ] **BK-05**: Generate Beta Keys button works
  - Steps: Click "Generate Beta Keys"
  - Expected: No crash
- [ ] **BK-06**: Keys table has 6 columns
  - Steps: Observe table headers
  - Expected: Key, ID, Max Uses, Expires At, Active, Email
- [ ] **BK-07**: Copy button warns without selection
  - Steps: Click "Copy" with no row selected
  - Expected: Warning dialog "No Selection"
- [ ] **BK-08**: Revoke button warns without selection
  - Steps: Click "Revoke" with no row selected
  - Expected: Warning dialog "No Selection"
- [ ] **BK-09**: Refresh button works
  - Steps: Click "Refresh"
  - Expected: No crash

### 2.7 Testing Tab

- [ ] **TT-01**: Tab switches to Testing
  - Steps: Click "Testing" tab
  - Expected: Testing content visible
- [ ] **TT-02**: Test Type combo has 5 items
  - Steps: Open combo
  - Expected: API Connectivity, Module Functionality, System Automation, Performance Test, Load Test
- [ ] **TT-03**: Run Test button works
  - Steps: Click "Run Test"
  - Expected: No crash
- [ ] **TT-04**: Clear Results button works
  - Steps: Click "Clear Results"
  - Expected: No crash
- [ ] **TT-05**: Test Type change shows/hides config groups
  - Steps: Change Test Type between options
  - Expected: Automation/Performance/Load groups show/hide accordingly

### 2.8 Predict Tab

- [ ] **PD-01**: Tab switches to Predict
  - Steps: Click "Predict" tab
  - Expected: Predict content visible
- [ ] **PD-02**: Run Prediction button works
  - Steps: Click "Run Prediction"
  - Expected: No crash
- [ ] **PD-03**: Browse button opens file dialog
  - Steps: Click "Browse..."
  - Expected: QFileDialog opens

### 2.9 Chat Tab

- [ ] **CH-01**: Tab switches to Chat
  - Steps: Click "Chat" tab
  - Expected: Chat content visible
- [ ] **CH-02**: Send message works
  - Steps: Type message, click Send
  - Expected: Message appears in history (or error without backend)
- [ ] **CH-03**: Enter key sends message
  - Steps: Type message, press Enter
  - Expected: Message sent
- [ ] **CH-04**: Clear clears history
  - Steps: Click Clear
  - Expected: History cleared

### 2.10 Code Generation Tab

- [ ] **CG-01**: Tab switches to Code Generation
  - Steps: Click "Code Generation" tab
  - Expected: Code Generation content visible
- [ ] **CG-02**: Generate Code button works
  - Steps: Click "Generate Code" (without description)
  - Expected: Warning dialog "Missing Description"
- [ ] **CG-03**: Copy button disabled initially
  - Steps: Observe Copy button
  - Expected: Disabled until code generated
- [ ] **CG-04**: Save button disabled initially
  - Steps: Observe Save button
  - Expected: Disabled until code generated
- [ ] **CG-05**: Clear button works
  - Steps: Click Clear
  - Expected: Form cleared

### 2.11 Docker Sandbox Tab

- [ ] **DS-01**: Tab switches to Docker Sandbox
  - Steps: Click "Docker Sandbox" tab
  - Expected: Docker Sandbox content visible
- [ ] **DS-02**: Start Container button works
  - Steps: Click "Start Container"
  - Expected: No crash (may fail without Docker)
- [ ] **DS-03**: Stop Container button works
  - Steps: Click "Stop Container"
  - Expected: No crash
- [ ] **DS-04**: Clear Logs button works
  - Steps: Click "Clear Logs"
  - Expected: Logs cleared

### 2.12 Linux Sandbox Tab

- [ ] **LS-01**: Tab switches to Linux Sandbox
  - Steps: Click "Linux Sandbox" tab
  - Expected: Linux Sandbox content visible
- [ ] **LS-02**: Container combo has distro options
  - Steps: Open combo
  - Expected: ubuntu-sandbox, kali-sandbox, alpine-sandbox, etc.
- [ ] **LS-03**: Execute Command button works
  - Steps: Click "Execute Command" (empty)
  - Expected: Warning dialog "No Command"
- [ ] **LS-04**: Run Training button works
  - Steps: Click "Run Training"
  - Expected: No crash
- [ ] **LS-05**: Generate Report button works
  - Steps: Click "Generate Report"
  - Expected: No crash
- [ ] **LS-06**: Clear Output button works
  - Steps: Click "Clear Output"
  - Expected: Output cleared

### 2.13 Debugging Tab

- [ ] **DG-01**: Tab switches to Debugging
  - Steps: Click "Debugging" tab
  - Expected: Debugging content visible
- [ ] **DG-02**: Refresh Info button works
  - Steps: Click "Refresh Info"
  - Expected: No crash
- [ ] **DG-03**: Refresh Processes button works
  - Steps: Click "Refresh Processes"
  - Expected: No crash
- [ ] **DG-04**: API Debug button works
  - Steps: Click "API Debug"
  - Expected: No crash
- [ ] **DG-05**: System Debug button works
  - Steps: Click "System Debug"
  - Expected: No crash
- [ ] **DG-06**: Export Debug Log button works
  - Steps: Click "Export Debug Log"
  - Expected: No crash

### 2.14 Model Downloads Tab

- [ ] **MD-01**: Tab switches to Model Downloads
  - Steps: Click "Model Downloads" tab
  - Expected: Model Downloads content visible
- [ ] **MD-02**: Search works
  - Steps: Type in search, click Search
  - Expected: No crash
- [ ] **MD-03**: Previous/Next buttons work
  - Steps: Click Previous, then Next
  - Expected: No crash
- [ ] **MD-04**: Download Selected button works
  - Steps: Click "Download Selected"
  - Expected: No crash
- [ ] **MD-05**: Clear Log button works
  - Steps: Click "Clear Log"
  - Expected: Log cleared

### 2.15 Neural Network Tab

- [ ] **NN-01**: Tab switches to Neural Network
  - Steps: Click "Neural Network" tab
  - Expected: Neural Network content visible
- [ ] **NN-02**: Refresh button works
  - Steps: Click "Refresh"
  - Expected: No crash
- [ ] **NN-03**: Auto-refresh checkbox toggles
  - Steps: Click checkbox
  - Expected: Checked state toggles

### 2.16 Edge Cases & Robustness

- [ ] **EC-01**: Rapid tab switching (all tabs in sequence)
  - Steps: Click through all 15 tabs rapidly
  - Expected: No crash, no freeze
- [ ] **EC-02**: Rapid button clicking (5x on each enabled button)
  - Steps: Click each enabled button 5 times rapidly
  - Expected: No crash
- [ ] **EC-03**: Window resize to minimum
  - Steps: Resize to 200x200
  - Expected: No crash
- [ ] **EC-04**: Window resize to 4K
  - Steps: Resize to 3840x2160
  - Expected: No crash
- [ ] **EC-05**: Invalid API URL
  - Steps: Enter garbage in API URL, click Test Connection
  - Expected: Error dialog, no crash
- [ ] **EC-06**: Very long text input (10k chars)
  - Steps: Paste 10000 chars into API URL field
  - Expected: No crash
- [ ] **EC-07**: Unicode/special characters
  - Steps: Enter unicode and special chars in text fields
  - Expected: No crash
- [ ] **EC-08**: Empty table selection actions
  - Steps: Click Copy/Revoke/Send with no table row selected
  - Expected: Warning dialogs, no crash
- [ ] **EC-09**: Spin box boundary values
  - Steps: Set spin boxes to min and max values
  - Expected: No crash
- [ ] **EC-10**: Close and reopen application
  - Steps: Close app, relaunch
  - Expected: App starts cleanly

### 2.17 Accessibility

- [ ] **AC-01**: All buttons have tooltips where applicable
  - Steps: Hover over buttons
  - Expected: Tooltips visible for complex actions
- [ ] **AC-02**: Tab order is logical
  - Steps: Press Tab repeatedly through Settings tab
  - Expected: Focus moves in logical order
- [ ] **AC-03**: Password fields mask input
  - Steps: Type in API Key field
  - Expected: Characters masked

### 2.18 API Key Generation (End-to-End)

- [ ] **AK-01**: Generate API Key button click produces key
  - Steps: With backend running, click "Generate API Key" in Settings
  - Expected: API Key field populated with new key, dialog shows key
- [ ] **AK-02**: Generated key can be copied
  - Steps: Select text in API Key field, copy
  - Expected: Key in clipboard
- [ ] **AK-03**: Save Settings persists key
  - Steps: Generate key, click Save Settings, restart app
  - Expected: Key restored from settings

---

## 3. Automated Test Suite

The automated test suite is located in the `tests/` directory.

| Test File | Coverage |
|---|---|
| `conftest.py` | Fixtures: QApplication, dashboard, screenshot on failure |
| `test_inventory.py` | All UI elements exist (windows, tabs, menus, buttons, controls) |
| `test_dashboard_tab.py` | Dashboard tab functionality |
| `test_settings_tab.py` | Settings tab, API key generation |
| `test_testing_tab.py` | Testing tab |
| `test_chat_tab.py` | Chat tab |
| `test_code_generation_tab.py` | Code generation tab |
| `test_beta_keys_tab.py` | Beta keys tab |
| `test_sandbox_tabs.py` | Docker Sandbox and Linux Sandbox tabs |
| `test_other_tabs.py` | Predict, Load Model, Debugging, Model Downloads, Neural, Training tabs |
| `test_keyboard_shortcuts.py` | Keyboard shortcuts (Ctrl+Q, F5, Enter, Tab, Escape) |
| `test_edge_cases.py` | Edge cases (rapid clicking, resize, invalid input, etc.) |

### Run Command

```bash
# From linux_gui directory
cd /mnt/data_ssd/CoreAI3D/linux_gui

# Install test dependencies
pip install pytest pytest-html pytest-xvfb

# Run all tests with HTML report
pytest tests/ -v --html=report.html --self-contained-html

# Run specific test file
pytest tests/test_settings_tab.py -v

# Run headless (requires Xvfb)
xvfb-run -a pytest tests/ -v --html=report.html
```

---

## 4. Coverage Summary

| Area | Automated | Manual | Total Items |
|---|---|---|---|
| Main Window | Yes | Yes | 4 |
| Menu Bar | Yes | Yes | 4 |
| Tabs (15) | Yes | Yes | 15 |
| Buttons (~80) | Yes | Yes | ~80 |
| Combo Boxes (~25) | Yes | Yes | ~25 |
| Check Boxes (~10) | Yes | Yes | ~10 |
| Spin Boxes (~15) | Yes | Yes | ~15 |
| Text Edits (~25) | Yes | Yes | ~25 |
| Tables (~5) | Yes | Yes | ~5 |
| Keyboard Shortcuts | Yes | Yes | 5 |
| Edge Cases | Yes | Yes | 10 |
| API Key Generation | Partial* | Yes | 3 |

*Automated tests verify button click doesn't crash, but end-to-end key generation requires backend running.

---

**File locations:**
- Test plan: `TEST_PLAN.md`
- Automated tests: `tests/`
- Test README: `README_TESTING.md`
