# CoreAI3D Modern GUI

A sleek, production-grade Linux desktop application for managing AI training data,
testing, debugging, and API keys. Built with **PyQt6 (Qt6)**.

## Features

- **Modern dark theme** (default) with optional light theme — instant switching, no restart
- **API Key generation** — robust multi-location response parsing eliminates the
  "API key was generated but not found in the server response" error
- **Show-once key modal** — key is displayed in full only once, with Copy button
- **Secure key storage** — system keyring (preferred), encrypted file (fallback)
- **Tabs**: Dashboard, API Keys, Settings, Logs, About
- Full keyboard navigation, loading states, error handling

## Installation

### 1. System dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
  libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0
```

### 2. Python virtual environment

```bash
cd /mnt/data_ssd/CoreAI3D/linux_gui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_modern.txt
```

### 3. Run

```bash
python3 modern_dashboard.py
```

## How API Key Parsing Works

The `ApiKeyParser` class in `modern_dashboard.py` extracts the API key from
backend responses by checking **multiple possible locations** in order:

1. **Top-level keys**: `api_key`, `apiKey`, `apikey`, `key`, `secret`, `token`,
   `access_token`, `accessToken`
2. **Nested containers**: `data`, `result`, `payload`, `response`, `keys`
3. **Last resort**: first non-empty string value longer than 8 characters

This prevents the common "key not found" error that occurs when the backend
wraps the key in an unexpected structure (e.g. `{"data": {"api_key": "..."}}`
or `{"result": {"key": "..."}}`).

## How the Show-Once Modal Works

When an API key is generated:
1. The key is saved to the keystore (keyring or encrypted file)
2. A modal dialog displays the key **in full**
3. A one-click **Copy to Clipboard** button is provided
4. A strong warning states: *"This key will never be shown again"*
5. After the dialog is dismissed, the key is **not** re-displayed by the dialog
   (though it remains in the key display field and keystore)

## Key Storage Priority

| Priority | Method | Library |
|----------|--------|---------|
| 1 | System keyring | `keyring` |
| 2 | Encrypted local file | `cryptography` (Fernet) |
| 3 | Plain-text file (warning) | built-in |

## Project Structure

```
linux_gui/
├── modern_dashboard.py    # Main application
├── requirements_modern.txt # Python dependencies
├── README_MODERN.md        # This file
├── coreai3d_client.py      # Backend API client (existing)
├── logs/                   # Application logs
└── tests/                  # Test suite
```

## Troubleshooting

**"No display" / "Could not connect to display"**
- Use `QT_QPA_PLATFORM=offscreen` for headless/CI
- Or install X11 libraries and run under an X server

**"ModuleNotFoundError: No module named 'PyQt6'"**
- `pip install -r requirements_modern.txt`

**Key not saved**
- Install `keyring` for system keyring: `pip install keyring`
- Install `cryptography` for encrypted file fallback: `pip install cryptography`
