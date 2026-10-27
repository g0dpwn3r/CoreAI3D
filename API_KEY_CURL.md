# CoreAI3D API Key Generation — curl Reference

This document covers curl commands for generating API keys across the CoreAI3D stack:
- **Web frontend** (Next.js): `/api/user/regenerate-api-key`
- **C++ backend** (PyQt6 GUI client): `/api/v1/auth/apikey`

---

## 1. Web Frontend — Regenerate API Key

**Endpoint:** `POST /api/user/regenerate-api-key`
**Auth:** Session cookie (`auth_token`) — the browser must be logged in.
**Base URL:** `http://localhost:3000` (dev) or your deployed web URL.

### curl command
```bash
curl -X POST http://localhost:3000/api/user/regenerate-api-key \
  -H "Content-Type: application/json" \
  -b "auth_token=<YOUR_SESSION_COOKIE>" \
  -c /tmp/cookies.txt \
  -v
```

### Example response
```json
{
  "api_key": "ca3d_<32-hex-chars>"
}
```

### Notes
- The `auth_token` cookie is set automatically after logging in via `/api/auth/login`.
- If using a browser, the Next.js `fetch` call in `ProfileCard.tsx` sends `credentials: 'include'`, so cookies are attached automatically.
- To test from curl outside a browser, first log in and capture the cookie:
  ```bash
  curl -X POST http://localhost:3000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"admin@coreai3d.local","password":"admin123"}' \
    -c /tmp/cookies.txt
  ```

---

## 2. C++ Backend — Generate API Key

**Endpoint:** `POST /api/v1/auth/apikey`
**Auth:** Requires a valid `session_id` (obtained from login).
**Base URL:** `http://localhost:1906` (local) or `http://coreai3d:8080` (Docker).

### Step A: Log in to get a session
```bash
curl -X POST http://localhost:1906/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  -c /tmp/cookies.txt
```

Response includes a `session_id` and/or `jwt_token`.

### Step B: Generate an API key
```bash
curl -X POST http://localhost:1906/api/v1/auth/apikey \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<YOUR_SESSION_ID>",
    "user_id": "1",
    "description": "My API key"
  }'
```

### Example response
```json
{
  "api_key": "<generated-key>",
  "description": "My API key",
  "created": true,
  "timestamp": "2026-10-18T21:32:06Z"
}
```

### Notes
- The `session_id` is required — without it the server returns `400 Session ID is required`.
- The `user_id` is optional; if provided and > 0, the key is persisted to the `api_keys` table.
- The `description` is optional metadata.

---

## 3. Linux GUI — Generate API Key (via PyQt6)

The linux_gui dashboard now has a **"Generate API Key"** button on the **Settings** tab under **API Configuration**. Clicking it calls `coreai3d_client.generate_api_key()` (POST `/auth/generate_api_key`) and populates the API Key field automatically.

There is no direct curl equivalent for the GUI flow — it goes through the PyQt6 client using the `COREAI3D_API_URL` base URL.

---

## 4. Using an API Key

Once you have an API key, pass it as a Bearer token:

```bash
curl -X GET http://localhost:1906/api/v1/some-endpoint \
  -H "Authorization: Bearer <YOUR_API_KEY>"
```

Or via the web frontend's API routes (session cookie based, no Bearer needed).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COREAI3D_API_URL` | `http://localhost:1906` | C++ backend base URL |
| `NEXT_PUBLIC_API_URL` | `` | Web frontend API base URL |
| `MYSQL_HOST` | `mysql` | Database host |
| `MYSQL_PORT` | `3306` | Database port |
