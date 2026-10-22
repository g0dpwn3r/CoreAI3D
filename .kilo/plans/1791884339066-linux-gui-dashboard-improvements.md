# Plan: CoreAI3D Linux GUI Dashboard Improvements

## Goal

Improve the PyQt6 Linux GUI (`linux_gui/dashboard.py`) with three requirements:
1. Beta-key email feature (generate + send via email)
2. Non-fullscreen/smaller window on startup (1280×800, centered, resizable)
3. Bug fixes (timer thread warnings, connection timeouts, API key generation, chat hang, etc.)

## Affected Files

- `linux_gui/dashboard.py` — main dashboard (window sizing, Beta Keys tab, timer fixes, bug fixes)
- `linux_gui/coreai3d_client.py` — API key generation, beta key endpoints, connection timeout
- `linux_gui/tests/test_inventory.py` — expects 15 tabs including "Beta Keys" at index 14
- `linux_gui/tests/test_beta_keys_tab.py` — expects specific Beta Keys tab widgets

## Key Findings from Code Inspection

### Window sizing
- `setup_ui()` calls `self.resize(1280, 720)` (line 2226) — needs to be 1280×800, centered
- No `showMaximized()` or `showFullScreen()` found in `main()` — `dashboard.show()` is used (line 10640)
- `QSettings` already used for config persistence — can extend for window geometry

### Beta Keys tab (from tests)
- Tests expect tab at index 14 with label "Beta Keys" (15 total tabs)
- Required widgets and exact button text:
  - `beta_email_input` (QLineEdit) — recipient email
  - `beta_email_desc_input` (QLineEdit) — description for email key
  - `beta_email_generate_btn` (QPushButton, text "Generate Email Key")
  - `beta_count_spin` (QSpinBox, range 1-100)
  - `beta_desc_input` (QLineEdit) — description for bulk keys
  - `beta_generate_btn` (QPushButton, text "Generate Beta Keys")
  - `beta_refresh_btn` (QPushButton, text "Refresh")
  - `beta_keys_table` (QTableWidget, 6 columns: "Key", "ID", "Max Uses", "Expires At", "Active", "Email")
  - `beta_copy_btn` (QPushButton, text "Copy")
  - `beta_send_btn` (QPushButton, text "Send via Email")
  - `beta_copy_instructions_btn` (QPushButton, text "Copy + Instructions")
  - `beta_revoke_btn` (QPushButton, text "Revoke")
- `coreai3d_client.py` already has `generate_beta_keys()`, `list_beta_keys()`, `toggle_beta_key()`
- No `create_beta_keys_tab()` method exists yet — needs to be added after `create_settings_tab()`

### Timer/thread issues
- `setup_timers()` (line 5225) creates `QTimer` on main thread — OK
- Neural refresh timer (line 4653) — needs thread-affinity check
- Download timers (lines 7909, 7987, 8241, 8411) — created in worker threads
- WebSocket thread (line 10500) — `start()` called on QThread
- `QThread` + Worker pattern already used for API key generation (lines 7274-7358)

### Connection/API key issues
- `coreai3d_client.py` has `login()`, `generate_api_key()`, `parse_api_key()`, `close()`
- API key generation uses QThread + `ApiKeyWorker` (already implemented)
- Need to show real backend error messages

### Chat hang
- Chat tab at line 4571 — needs timeout + cancel mechanism

## Implementation Plan

### Phase 1: Window behavior

1. In `setup_ui()`, change `self.resize(1280, 720)` → `self.resize(1280, 800)`
2. Add screen-centering logic using `QApplication.primaryScreen().geometry()`
3. Persist window geometry via `QSettings` — save in `closeEvent()`, restore in `__init__`
4. Set minimum window size to prevent unusable small windows (e.g., 800×600)
5. Verify no `showMaximized()`/`showFullScreen()` calls exist

### Phase 2: Beta Keys tab

1. Add `self.create_beta_keys_tab()` call after `self.create_settings_tab()` in `setup_ui()` (line 2251)
2. Create `create_beta_keys_tab()` method with all widgets matching test expectations:
   - Email section: `beta_email_input`, `beta_email_desc_input`, `beta_email_generate_btn`
   - Generation section: `beta_count_spin` (1-100), `beta_desc_input`, `beta_generate_btn`
   - Table: `beta_keys_table` (6 columns)
   - Action buttons: `beta_copy_btn`, `beta_send_btn`, `beta_copy_instructions_btn`, `beta_revoke_btn`, `beta_refresh_btn`
3. Implement handlers:
   - `on_generate_email_key_clicked()` — calls `generate_beta_keys(count=1, email=..., description=...)`
   - `on_generate_beta_keys_clicked()` — calls `generate_beta_keys(count=..., description=...)`
   - `on_refresh_beta_keys_clicked()` — calls `list_beta_keys()`, populates table
   - `on_copy_beta_key_clicked()` — copies selected key to clipboard
   - `on_send_beta_key_clicked()` — opens email dialog or sends via Resend/SMTP
   - `on_copy_instructions_clicked()` — copies key + instructions to clipboard
   - `on_revoke_beta_key_clicked()` — calls `toggle_beta_key(id, enable=False)`
4. Email sending: use `smtplib` + `email.mime` with SMTP settings from env vars or QSettings
   - Env vars: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
   - Fallback: Resend API (already integrated via `ResendEmailService`)
5. Use QThread + Worker pattern for all network calls (consistent with API key generation)

### Phase 3: Bug fixes

1. **Timer thread affinity:**
   - Audit all `QTimer` creation sites; ensure they're on main thread
   - Neural refresh timer (line 4653): guard creation with thread check
   - Download timers (lines 7909, 7987, 8241, 8411): move creation to main thread, start via signal
   - Add `QTimer.singleShot(0, ...)` pattern for cross-thread timer starts

2. **Connection timeout:**
   - Make timeout configurable via Settings (already `sandbox_timeout` exists, add `connection_timeout`)
   - Add user-facing feedback (QMessageBox or status bar) on timeout
   - Retry mechanism with exponential backoff (already in client `_monitor_connection`)

3. **API key generation error display:**
   - Show real backend error in `ApiKeyDialog` or QMessageBox
   - Parse error response from `APIResponse` properly (check `.error` field)

4. **Chat hang fix:**
   - Add timeout to chat requests (configurable, default 30s)
   - Add cancel button to abort in-flight chat requests
   - Use `asyncio.wait_for()` with timeout in chat handler

5. **Log noise reduction:**
   - Change `logger.info("No API key configured...")` to `logger.debug`
   - Suppress Kaggle/offline warnings unless relevant (only when feature is used)
   - Only show connection warnings when actually disconnected

6. **Crash-on-exit:**
   - Verify `closeEvent()` stops all timers, quits worker threads, closes aiohttp client
   - Add try/except around cleanup to prevent crash

### Phase 4: Validation

1. Run existing tests: `pytest linux_gui/tests/`
2. Verify window opens at 1280×800, centered, not maximized
3. Test beta key generation + email send flow
4. Verify no timer-thread warnings in logs
5. Verify connection and API key generation work cleanly

## Risks

- The dashboard.py is 10,668 lines — changes must be surgical
- QThread/Worker patterns must maintain thread affinity
- Beta Keys tab must match exact test expectations (widget names, button text, column headers)
- Email sending requires SMTP config — must handle missing config gracefully
- Chat timeout must not break existing WebSocket message handling
- Tests create dashboard without event loop — timer creation must not crash

## Open Questions

1. Should beta key email use SMTP (smtplib) or the Resend API that's already integrated?
   - Recommendation: Use Resend API (already integrated via `ResendEmailService`) for consistency; fall back to SMTP if Resend not configured
2. Should window geometry persistence use QSettings or a separate config file?
   - Recommendation: QSettings (already used throughout)
3. The tests expect 15 tabs but currently 13 exist — the Beta Keys tab must be added at index 14 (after Settings)
4. Should email sending be synchronous (blocking) or async (QThread)?
   - Recommendation: QThread + Worker (consistent with API key generation pattern)
