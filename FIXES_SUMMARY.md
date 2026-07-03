# Linux GUI Connection Fixes - Summary

## Issues Identified and Fixed

### 1. **WebSocket URL Construction Error (CRITICAL)**
**File:** `linux_gui/dashboard.py` (lines 9671-9701)

**Problem:**
- The `_setup_websocket_connection()` method was incorrectly deriving the WebSocket URL from the API URL
- It forced `localhost` as the hostname instead of using the Docker service name `coreai3d`
- It extracted port 8080 from the API URL, but WebSocket runs on port 8081

**Fix:**
```python
# BEFORE (broken):
port = 8080  # Default port
if "localhost:" in api_url:
    port = int(api_url.split("localhost:")[1].split("/")[0])
ws_url = f"ws://localhost:{port}/ws"

# AFTER (fixed):
ws_url = self.config.ws_url  # Uses configured ws://coreai3d:8081
```

### 2. **Missing WebSocket Retry Mechanism**
**File:** `linux_gui/dashboard.py` (line 9706-9736)

**Problem:**
- WebSocket connections would fail silently with no retry logic
- Connection errors were logged but not handled gracefully

**Fix:**
- Added `_websocket_run_with_retry()` method with 5 retry attempts
- Added ping/pong keepalive (30s interval, 10s timeout)
- Added `_on_websocket_open()` handler for connection success logging

### 3. **CoreAI3DClient Incorrect Instantiation**
**File:** `linux_gui/dashboard.py` (lines 1919-1923)

**Problem:**
- CoreAI3DClient was being instantiated with keyword arguments
- The client expects a config dictionary as the first positional parameter

**Fix:**
```python
# BEFORE (broken):
self.client = CoreAI3DClient(
    api_url=self.config.api_url,
    ws_url=self.config.ws_url,
    api_key=self.config.api_key
)

# AFTER (fixed):
self.client = CoreAI3DClient({
    'base_url': self.config.api_url,
    'ws_url': self.config.ws_url,
    'api_key': self.config.api_key
})
```

### 4. **Windows Sandbox Dockerfile Issues**
**File:** `docker/sandbox/Dockerfile.windows`

**Problems Fixed:**
- Missing sandbox user creation → Added `useradd -m -s /bin/bash sandbox`
- Missing workspace directories → Added `mkdir -p /workspace /training_data`
- Ownership not set → Added `chown -R sandbox:sandbox /workspace /training_data`
- Python packages not installed → Added comprehensive pip install block
- Node.js packages installation → Added npm install with proper env vars

### 5. **Admin GUI XCB Dependencies**
**File:** `docker/admin-gui/Dockerfile.admin-gui`

**Problem:**
- PySide6/Qt requires XCB libraries for X11 rendering
- Missing dependencies caused GUI initialization failures

**Fix:**
Added comprehensive XCB and X11 library installation:
```dockerfile
apt-get install -y \
    libxcb-cursor0 libxcb-xkb1 libxcb-xinerama0 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
    libxkbcommon-x11-0 libx11-xcb1 libxext6 libxi6 \
    libxtst6 libxcomposite1 libxdamage1 libxcursor1 \
    libxrandr2 libxfixes3 libxrender1 libxcb1 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-sync1 \
    libxcb-util1 libxcb-xfixes0 libxcb-shm0 libxcb-glx0 \
    libxkbcommon0 libxkbcommon-x11-0 mesa-utils
```

## Configuration Summary

The dashboard now correctly uses these default configurations:
- **API URL:** `http://coreai3d:8080/api/v1`
- **WebSocket URL:** `ws://coreai3d:8081`
- **Timeout:** 30.0 seconds
- **Retry Attempts:** 3 (API), 5 (WebSocket)

## Testing Instructions

### 1. Rebuild and Restart Containers
```bash
docker compose down
docker compose up -d --build
```

### 2. Verify Service Connectivity
From inside the admin-gui container:
```bash
docker exec -it coreai3d_admin_gui bash

# Test API endpoint
curl -X GET http://coreai3d:8080/api/health

# Test authentication
curl -X POST http://coreai3d:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### 3. Run Dashboard Connection Test
```bash
cd /app/linux_gui
source .env/bin/activate
python3 run_dashboard.py
```

### 4. Check Logs for Successful Connection
Look for these log messages indicating success:
- `Setting up WebSocket connection to: ws://coreai3d:8081`
- `WebSocket connection established successfully`
- `Connection test passed`
- `Health check passed`

## Expected Behavior

After these fixes:
1. ✅ Dashboard will connect to `ws://coreai3d:8081` (not localhost)
2. ✅ WebSocket connection will retry up to 5 times on failure
3. ✅ Connection errors will be logged with clear messages
4. ✅ API calls will use `http://coreai3d:8080/api/v1`
5. ✅ Windows sandbox will start without user/workspace errors
6. ✅ Admin GUI will display properly with XCB support

## Files Modified

1. `/CoreAI3D/linux_gui/dashboard.py` - WebSocket connection logic
2. `/CoreAI3D/docker/sandbox/Dockerfile.windows` - User/directory setup
3. `/CoreAI3D/docker/admin-gui/Dockerfile.admin-gui` - XCB dependencies
4. `/CoreAI3D/FIXES_SUMMARY.md` - This documentation

## Troubleshooting

If you still see connection issues:

1. **Check container networking:**
   ```bash
   docker network inspect coreai3d_default
   ```

2. **Verify service is running:**
   ```bash
   docker compose ps
   ```

3. **Check WebSocket server logs:**
   ```bash
   docker logs coreai3d | grep -i websocket
   ```

4. **Test from host machine:**
   ```bash
   curl http://localhost:1906/api/health  # API through mapped port
   ```

5. **Verify DNS resolution from admin-gui:**
   ```bash
   docker exec -it coreai3d_admin_gui ping coreai3d
   ```
