# CoreAI3D Web Frontend

A modern, black-and-blue themed web application for CoreAI3D with Grok-style UI, real-time chat streaming, beta-key gated registration, and admin features.

## Features

- **Grok-style UI**: Pure black background (#000), electric blue accents (#3b82f6), smooth animations
- **Real-time Chat**: Thinking → Generating → Streaming token flow
- **Beta-Key Gated Registration**: Users must have a valid beta key to register
- **Model Selector**: Multiple AI models with capabilities
- **User Dashboard**: Profile, API keys, usage stats, billing
- **Admin Panel**: User management, beta key generation/revocation
- **Linux GUI Integration**: Desktop app can generate and distribute beta keys

## Tech Stack

- **Next.js 14** (App Router) with TypeScript
- **Tailwind CSS** for styling
- **SQLite** (better-sqlite3) for local data
- **JWT** for authentication
- **WebSocket** for real-time chat streaming

## Prerequisites

- Node.js 18+
- CoreAI3D C++ backend running (for chat, beta key validation)

## Environment Variables

Create a `.env.local` file in the `web/` directory:

```env
# CoreAI3D Backend API URL (Docker container or bare metal)
COREAI3D_API_URL=http://localhost:1906

# JWT Secret (change in production!)
JWT_SECRET=your-super-secret-key-change-in-production

# Optional: Database path (defaults to ./data/coreai3d.db)
DATABASE_PATH=./data/coreai3d.db
```

## Quick Start

### 1. Install Dependencies

```bash
cd /mnt/data_ssd/CoreAI3D/web
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### 3. Build for Production

```bash
npm run build
npm start
```

## Beta Key System

### How It Works

1. **Admin generates keys** via:
   - Web Dashboard → Admin → Beta Keys tab
   - Linux GUI → Beta Keys tab

2. **Key format**: `CA3D-XXXXXXXXXXXXXXXXXXXX` (24 chars after prefix)

3. **Registration flow**:
   - User visits `/register`
   - Enters beta key → validates against backend
   - If valid, proceeds to create account
   - On success, key is marked as **used** in backend

4. **Key states**:
   - **Active**: Generated, not used, can be used for registration
   - **Used**: Someone registered with this key
   - **Revoked**: Admin disabled the key

### Generating Beta Keys (Linux GUI)

1. Open the Linux GUI dashboard:
   ```bash
   cd /mnt/data_ssd/CoreAI3D/linux_gui
   ./run_dashboard.sh
   ```

2. Go to **Beta Keys** tab (last tab)

3. **Generate keys**:
   - Enter count (1-100)
   - Optional description
   - Click "Generate Beta Keys"
   - Keys appear in table with plaintext (shown once!)

4. **Distribute keys**:
   - **Copy**: Copy plaintext to clipboard
   - **Send via Email**: Opens default mail client with pre-filled email
   - **Copy + Instructions**: Copies key with registration guide

### Generating Beta Keys (Web Dashboard)

1. Login as admin (`admin@coreai3d.local` / `admin123`)
2. Go to **Dashboard** → **Beta Keys** tab
3. Click "Generate Keys"
4. Enter count and notes
5. Keys appear in table with full details

## Admin Access

Default admin credentials:
- Email: `admin@coreai3d.local`
- Password: `admin123`

Access admin features:
- **Web**: Dashboard → Users / Beta Keys tabs
- **Linux GUI**: Beta Keys tab

## Project Structure

```
web/
├── app/
│   ├── api/                 # API routes
│   │   ├── auth/            # Login, register, validate
│   │   ├── beta-keys/       # Beta key CRUD (proxies to C++ backend)
│   │   ├── chat/            # Chat streaming
│   │   ├── conversations/   # Conversation CRUD
│   │   ├── models/          # Model listing
│   │   ├── user/            # Profile, API keys
│   │   └── admin/           # Admin user list
│   ├── chat/                # Main chat page
│   ├── dashboard/           # User dashboard
│   ├── models/              # Models page
│   ├── login/               # Login page
│   ├── register/            # Registration (beta key required)
│   ├── layout.tsx
│   ├── page.tsx             # Redirects to chat/login
│   └── globals.css          # Global styles
├── components/
│   ├── ChatWindow.tsx       # Main chat interface
│   ├── Sidebar.tsx          # Conversations + nav
│   ├── MessageBubble.tsx    # Message display
│   ├── ThinkingIndicator.tsx # Thinking/Generating animation
│   ├── ModelSelector.tsx    # Model dropdown
│   ├── ProfileCard.tsx      # User profile
│   ├── BillingSection.tsx   # Billing plans
│   ├── AdminBetaKeys.tsx    # Admin beta key management
│   └── BetaKeyForm.tsx      # Beta key input form
├── lib/
│   ├── api-client.ts        # API client
│   ├── auth.ts              # JWT, password hashing
│   ├── auth-context.tsx     # React auth context
│   ├── beta-keys.ts         # Beta key operations (legacy, now proxied)
│   ├── db.ts                # SQLite database
│   └── utils.ts             # Helpers
├── package.json
├── next.config.js
├── tailwind.config.js
└── postcss.config.js
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - Login
- `POST /api/auth/register` - Register (requires beta_key)
- `POST /api/auth/validate` - Validate session
- `POST /api/auth/validate-beta-key` - Validate beta key

### Beta Keys (Admin only)
- `GET /api/beta-keys` - List keys with stats
- `POST /api/beta-keys` - Generate keys `{count, notes, email?}`
- `DELETE /api/beta-keys/[id]` - Revoke key

### Chat
- `POST /api/chat` - Send message (streams response)
- `GET /api/conversations` - List conversations
- `POST /api/conversations` - Create conversation
- `GET /api/conversations/[id]` - Get conversation + messages
- `DELETE /api/conversations/[id]` - Delete conversation

### Models
- `GET /api/models` - List available models

### User
- `GET /api/user/profile` - Get profile + stats
- `PUT /api/user/profile` - Update profile
- `POST /api/user/regenerate-api-key` - Regenerate API key

### Admin
- `GET /api/admin/users` - List all users

## Chat Streaming Protocol

The chat endpoint returns Server-Sent Events (SSE):

```
data: {"type": "thinking", "conversation_id": "..."}

data: {"type": "generating"}

data: {"type": "token", "content": "Hello"}

data: {"type": "token", "content": " world"}

data: {"type": "done", "conversation_id": "..."}
```

## Linux GUI Integration

The Linux GUI (`/mnt/data_ssd/CoreAI3D/linux_gui/dashboard.py`) communicates with the same C++ backend endpoints:

- `POST /api/v1/beta/generate` - Generate keys
- `GET /api/v1/beta/list` - List keys
- `POST /api/v1/beta/toggle` - Revoke/enable keys
- `POST /api/v1/beta/validate` - Validate key (used by web registration)
- `POST /api/v1/beta/use` - Mark key as used (called after registration)

Both web and desktop share the same beta key database via the C++ backend.

## Troubleshooting

### "Cannot reach backend" errors
- Ensure `COREAI3D_API_URL` points to the running C++ backend
- Default: `http://localhost:1906` (bare metal) or `http://coreai3d:8080` (Docker)
- Check backend is running: `curl http://localhost:1906/health`

### Beta key validation fails
- Key must be generated via admin endpoints
- Key must be **Active** (not used, not revoked)
- Check C++ backend logs for validation errors

### Database issues
- SQLite database is at `./data/coreai3d.db`
- Delete to reset: `rm -rf data/`

### Port conflicts
- Web runs on port 3000
- Change in `package.json`: `"dev": "next dev -p 3001"`

## Development

### Linting
```bash
npm run lint
```

### Type Checking
```bash
npx tsc --noEmit
```

## Production Deployment

1. Set strong `JWT_SECRET`
2. Use HTTPS (set `secure: true` in cookies)
3. Configure reverse proxy (nginx) for port 80/443
4. Use process manager (PM2) for Next.js
5. Ensure C++ backend is accessible

## License

MIT - CoreAI3D Project