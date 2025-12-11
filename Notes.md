# Open-NueChat - Development Notes

## Project Overview

Full-stack LLM chat application with:
- FastAPI backend (Python 3.13)
- React 19 + TypeScript frontend
- SQLite database (aiosqlite)
- Multi-GPU support (ROCm, CUDA, CPU)
- FAISS GPU for vector search
- OpenAI-compatible LLM API integration

**Current Version:** NC-0.6.27

---

## Architecture

### Deployment (Single Container)
```
┌─────────────────────────────────────┐
│           open-nuechat              │
│  ┌─────────────────────────────┐   │
│  │         FastAPI             │   │
│  │  ┌─────────┬─────────────┐  │   │
│  │  │ /api/*  │ /ws/*       │  │   │
│  │  │ Backend │ WebSocket   │  │   │
│  │  └─────────┴─────────────┘  │   │
│  │  ┌─────────────────────┐    │   │
│  │  │ /* (static files)   │    │   │
│  │  │ Frontend SPA        │    │   │
│  │  └─────────────────────┘    │   │
│  └─────────────────────────────┘   │
│   Host: BACKEND_HOST:BACKEND_PORT  │
└─────────────────────────────────────┘
```

### Backend Structure
```
backend/app/
├── api/routes/       # FastAPI route handlers
├── core/config.py    # Pydantic settings
├── db/database.py    # SQLAlchemy async engine
├── models/           # SQLAlchemy ORM models
├── services/         # Business logic
│   ├── auth.py       # JWT + bcrypt
│   ├── billing.py    # Token tracking
│   ├── document_queue.py  # Persistent doc processing
│   ├── llm.py        # OpenAI-compatible client
│   ├── rag.py        # FAISS + embeddings
│   └── stt.py        # Speech-to-text
├── filters/          # Bidirectional stream filters
└── tools/            # Built-in LLM tools
```

### Frontend Structure
```
frontend/src/
├── components/       # React components
├── hooks/            # Custom hooks
├── pages/            # Route pages
├── lib/              # API client, utilities
├── stores/           # Zustand state
└── types/            # TypeScript types
```

---

## Configuration

### Required .env Settings
```bash
LLM_API_KEY=your-api-key
LLM_API_BASE=http://your-llm-endpoint/v1
LLM_MODEL=your-model-name
JWT_SECRET=generate-secure-random-string
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=secure-admin-password
```

### Optional .env Settings
```bash
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
DATABASE_URL=sqlite+aiosqlite:///./data/nuechat.db
FREEFORALL=false
FREE_TIER_TOKENS=100000
IMAGE_SERVICE_URL=http://image-service:5000
TTS_SERVICE_URL=http://tts-service:8100
```

### OAuth Configuration
```bash
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxx
GOOGLE_REDIRECT_URI=https://yourdomain.com/api/auth/google/callback
```

---

## Core Features

### Admin Limit Bypasses

Admins (`is_admin=True`) bypass all tier-based limits:

| Limit Type | Location | Normal Limits |
|------------|----------|---------------|
| Knowledge Stores | `knowledge_stores.py` | free: 3, pro: 20, enterprise: 100 |
| Custom Assistants | `assistants.py` | free: 3, pro: 20, enterprise: 100 |
| API Keys | `api_keys.py` | free: 3, pro: 10, enterprise: 100 |
| Token Usage | `billing.py` | Tier-based monthly limits |

All checks use `if not current_user.is_admin:` guard before enforcing limits.

### Token Refill System

User token counts (`tokens_used_this_month`) are automatically reset based on the configurable `token_refill_interval_hours` setting in Admin → System.

**How It Works:**
1. Background task runs every hour
2. Compares current time vs `last_token_reset_timestamp`
3. If elapsed >= `token_refill_interval_hours`, resets ALL users' tokens to 0

**Times Compared:**
```python
now = datetime.now(timezone.utc)
last_reset = datetime.fromisoformat(setting.value)
hours_since_reset = (now - last_reset).total_seconds() / 3600
if hours_since_reset >= refill_hours:
    # Reset all users
```

**Admin Settings:**
- Setting: `token_refill_interval_hours` (default: 720 = 30 days)
- Location: Admin → System tab

### Debug Token Resets

Enable in Admin → Site Dev → Debug Token Resets

**When enabled, logs on every token reset check:**
```
============================================================
[TOKEN_RESET_DEBUG] Token reset check triggered
[TOKEN_RESET_DEBUG] Current time (UTC): 2024-12-11T10:00:00+00:00
[TOKEN_RESET_DEBUG] Last reset: 2024-11-11T00:00:00+00:00
[TOKEN_RESET_DEBUG] Refill interval: 720 hours
[TOKEN_RESET_DEBUG] User token counts (3 users):
[TOKEN_RESET_DEBUG]   admin@example.com: 150,000 tokens (tier: enterprise)
[TOKEN_RESET_DEBUG]   user1@example.com: 45,000 tokens (tier: pro)
[TOKEN_RESET_DEBUG]   user2@example.com: 12,500 tokens (tier: free)
============================================================
[Token Reset not necessary]
[TOKEN_RESET_DEBUG] 671.5 hours until next reset.
```

**When disabled:** Silent operation (no logging).

---

## Document Processing Queue

Documents uploaded to RAG or Knowledge Stores are processed asynchronously via a persistent queue that survives container restarts.

### How It Works
```
User uploads document
  → Document saved to disk
  → Document record created (is_processed=False)
  → Task added to queue (persisted to JSON file)
  → Return immediately to user
  
Background worker (runs continuously)
  → Pick up pending task
  → Extract text from document
  → Create embeddings
  → Update document (is_processed=True, chunk_count=N)
  → Remove task from queue
```

### Queue Persistence

Queue state stored in `/app/data/document_queue.json`:
```json
{
  "task-uuid-1": {
    "task_id": "task-uuid-1",
    "document_id": "doc-uuid",
    "user_id": "user-uuid",
    "knowledge_store_id": "store-uuid",
    "file_path": "/app/uploads/...",
    "file_type": "application/pdf",
    "status": "pending",
    "created_at": "2024-12-11T10:00:00+00:00",
    "retry_count": 0
  }
}
```

### Task States

| Status | Description |
|--------|-------------|
| `pending` | Waiting to be processed |
| `processing` | Currently being processed |
| `completed` | Successfully processed (removed from queue) |
| `failed` | Failed after 3 retries |

### Restart Behavior

On startup:
1. Load queue from disk
2. Reset any `processing` tasks to `pending` (interrupted)
3. Start background worker
4. Worker processes pending tasks

### Logging
```
[DOC_QUEUE] Worker started
[DOC_QUEUE] Added task abc123 for document def456
[DOC_QUEUE] Processing task abc123 - document def456
[DOC_QUEUE] Extracting text from /app/uploads/...
[DOC_QUEUE] Creating embeddings for document def456
[DOC_QUEUE] Completed processing document def456 (42 chunks)
[DOC_QUEUE] Removed task abc123
[DOC_QUEUE] Empty Document Queue
```

**Enable in:** Admin → Site Dev → Debug Document Queue

**When disabled:** Silent operation (no logging).

---

## Message Chain / Branching Architecture

Messages form a tree structure via `parent_id`:
```
User1 (parent_id=null)        <- First message is root
  └── Asst1 (parent_id=User1)
        └── User2 (parent_id=Asst1)
              └── Asst2 (parent_id=User2)
                    ├── User3a (parent_id=Asst2)  <- Branch A
                    └── User3b (parent_id=Asst2)  <- Branch B (edit/retry)
```

### Frontend Message Flow

**Sending a New Message:**
```typescript
// ChatPage.tsx - handleSendMessage
1. Calculate parentId from store messages (self-contained)
2. Call sendChatMessage(chatId, content, undefined, parentId)

// WebSocketContext.tsx - sendChatMessage
3. Create temp user message with parent_id
4. Send to backend via WebSocket

// Backend saves, streams response
5. stream_end returns assistant message_id
6. Assistant becomes new leaf for next message
```

**Voice Mode Parent ID Fix (NC-0.6.27):**
Calculate parentId directly from messages at send time to avoid stale closure issues:
```typescript
const handleSendMessage = async (content: string) => {
  const { messages } = useChatStore.getState();
  const sortedMsgs = [...messages].sort((a, b) => 
    new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );
  const lastAssistant = sortedMsgs.reverse().find(m => m.role === 'assistant');
  let parentId = lastAssistant?.id || null;
  // ... rest of send logic
};
```

---

## Filter Chains (Agentic Flows)

Admin-configurable message processing flows (Admin → Filter Chains tab).

### Step Types

| Type | Description |
|------|-------------|
| `to_llm` | Ask LLM a question |
| `query` | Generate search query |
| `to_tool` | Execute a tool |
| `go_to_llm` | Send to main LLM |
| `filter_complete` | Exit chain early |
| `set_var` | Set a variable |
| `compare` | Compare values |
| `context_insert` | Add to context |
| `call_chain` | Execute another chain |
| `stop` | Stop execution |
| `block` | Block with error |

### Variables

- `$Query` - Original user input
- `$PreviousResult` - Last step output
- `$Var[name]` - Named outputs from earlier steps

---

## Image Generation

### Request Detection

LLM decides if user wants an image via `[IMAGE_REQUEST]` tag:
```
User: "draw me a cat"
LLM: [IMAGE_REQUEST: A cute orange tabby cat sitting...]
```

### Generation Flow
```
Message → Detect [IMAGE_REQUEST] → Queue job → Poll status → Display
```

### Persistence

Images saved to `/app/data/generated_images/` with metadata in SQLite.

---

## Voice Features (TTS/STT)

### Text-to-Speech (TTS)
- Microservice: `tts-service` with Kokoro model
- Streaming audio chunks via WebSocket
- ROCm GPU acceleration

### Speech-to-Text (STT)
- Whisper model (multilingual)
- Voice Activity Detection (VAD)
- Streaming transcription

---

## API Response Formats

### Chats List
```json
{ "chats": [...], "total": 0, "page": 1, "page_size": 20 }
```

### Messages List
```json
[{ "id": "...", "role": "user", "content": "..." }, ...]
```

### WebSocket Events
- `stream_start` - LLM response starting
- `stream_chunk` - Content chunk
- `stream_end` - Response complete
- `message_saved` - User message confirmed

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| NC-0.6.27 | 2024-12-11 | Voice mode parent_id fix, token reset automation, document queue, admin limit bypasses |
| NC-0.6.26 | 2024-12-10 | Talk to Me voice mode debug panel |
| NC-0.6.25 | 2024-12-10 | Swipe gestures for retry/branch navigation |
| NC-0.6.24 | 2024-12-09 | Filter chain debug mode |
| NC-0.6.23 | 2024-12-09 | Configurable filter chains |
| NC-0.6.22 | 2024-12-09 | Error handling & file request fixes |
| NC-0.6.21 | 2024-12-09 | Image generation queue |
| NC-0.6.20 | 2024-12-09 | Image persistence |
| NC-0.6.19 | 2024-12-09 | Image size/aspect selection |
| NC-0.6.18 | 2024-12-08 | Image generation service |
| NC-0.6.17 | 2024-12-08 | Image request detection |
| NC-0.6.16 | 2024-12-08 | ROCm GPU for TTS |
| NC-0.6.15 | 2024-12-08 | Artifact persistence |
| NC-0.6.14 | 2024-12-08 | Mobile responsive buttons |
| NC-0.6.13 | 2024-12-08 | Streaming TTS/STT |
| NC-0.6.12 | 2024-12-08 | Voice UI polish |
| NC-0.6.11 | 2024-12-08 | Voice TTS/STT frontend |
| NC-0.6.10 | 2024-12-08 | STT backend |
| NC-0.6.9 | 2024-12-08 | TTS microservice |
| NC-0.6.8 | 2024-12-08 | Health check token reset |
| NC-0.6.7 | 2024-12-08 | Artifacts panel |
| NC-0.6.6 | 2024-12-08 | Persist login state |
| NC-0.6.5 | 2024-12-08 | UI modernization |
| NC-0.6.4 | 2024-12-08 | Stop generation |
| NC-0.6.3 | 2024-12-08 | Message edit/delete branching |
| NC-0.6.2 | 2024-12-08 | Code summary feature |
| NC-0.6.1 | 2024-12-08 | Tool usage tracking |
| NC-0.6.0 | 2024-12-08 | Agent flows editor |

---

## Database Schema Version

Current: **NC-0.6.27**

Migrations run automatically on startup in `backend/app/main.py`.
