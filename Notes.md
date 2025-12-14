# Open-NueChat - Development Notes

## Project Overview

Full-stack LLM chat application with:
- FastAPI backend (Python 3.13)
- React 19 + TypeScript frontend
- SQLite database (aiosqlite)
- Multi-GPU support (ROCm, CUDA, CPU)
- FAISS GPU for vector search
- OpenAI-compatible LLM API integration

**Current Version:** NC-0.6.37

---

## Architecture

### Deployment (Single Container)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Open-NueChat                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────────────────┐    │
│  │   React     │    │              FastAPI Backend            │    │
│  │  Frontend   │◄──►│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │    │
│  │  (Vite)     │    │  │  Auth   │ │  Chat   │ │   RAG    │  │    │
│  └─────────────┘    │  │ Service │ │ Service │ │ Service  │  │    │
│                     │  └────┬────┘ └────┬────┘ └────┬─────┘  │    │
│                     │       │           │           │        │    │
│                     │  ┌────▼───────────▼───────────▼─────┐  │    │
│                     │  │           SQLite + FAISS         │  │    │
│                     │  └──────────────────────────────────┘  │    │
│                     └─────────────────────────────────────────┘    │
│                                       │                            │
│  ┌─────────────┐    ┌─────────────┐   │   ┌─────────────────────┐  │
│  │ TTS Service │    │ Image Gen   │   │   │    LLM Backend      │  │
│  │  (Kokoro)   │    │ (Diffusers) │   └──►│ (Ollama/vLLM/etc.)  │  │
│  └─────────────┘    └─────────────┘       └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Backend Structure (Refactored NC-0.6.28)

```
backend/app/
├── api/
│   ├── routes/              # FastAPI route handlers
│   ├── helpers.py           # Shared route utilities (NEW)
│   ├── exception_handlers.py # Centralized error handling (NEW)
│   ├── ws_types.py          # WebSocket event types (NEW)
│   └── schemas.py           # Pydantic request/response schemas
├── core/
│   ├── config.py            # Pydantic settings
│   └── logging.py           # Structured logging (NEW)
├── db/database.py           # SQLAlchemy async engine
├── models/                  # SQLAlchemy ORM models (REFACTORED)
│   ├── __init__.py          # Re-exports all models
│   ├── base.py              # Base, generate_uuid, enums
│   ├── user.py              # User, OAuthAccount, APIKey
│   ├── chat.py              # Chat, Message, ChatParticipant
│   ├── document.py          # Document, DocumentChunk, KnowledgeStore
│   ├── assistant.py         # CustomAssistant, AssistantConversation
│   ├── billing.py           # TokenUsage
│   ├── tool.py              # Tool, ToolUsage
│   ├── filter.py            # ChatFilter
│   ├── upload.py            # UploadedFile, UploadedArchive
│   └── settings.py          # SystemSetting, Theme
├── services/
│   ├── auth.py              # JWT + bcrypt
│   ├── billing.py           # Token tracking
│   ├── document_queue.py    # Persistent doc processing
│   ├── llm.py               # OpenAI-compatible client
│   ├── rag.py               # FAISS + embeddings
│   ├── stt.py               # Speech-to-text
│   ├── token_manager.py     # JWT blacklisting (NEW)
│   ├── rate_limiter.py      # Token bucket rate limiting (NEW)
│   ├── zip_processor.py     # Secure zip extraction (SECURITY UPDATE)
│   └── microservice_utils.py # Shared microservice utilities (NEW)
├── filters/                 # Bidirectional stream filters
└── tools/                   # Built-in LLM tools
```

### Frontend Structure (Refactored NC-0.6.28)

```
frontend/src/
├── components/              # React components
├── hooks/
│   ├── useMobile.ts
│   ├── useVoice.ts
│   └── useKeyboardShortcuts.ts  # Keyboard shortcuts (NEW)
├── pages/                   # Route pages
├── lib/
│   ├── api.ts               # API client
│   ├── artifacts.ts         # Artifact extraction
│   ├── formatters.ts        # Shared formatting utilities (NEW)
│   └── wsTypes.ts           # WebSocket type guards (NEW)
├── stores/
│   ├── chatStore.ts         # Legacy (re-exports from chat/)
│   ├── chat/                # Modular chat store (NEW)
│   │   ├── index.ts         # Composed store
│   │   ├── types.ts         # Type definitions
│   │   ├── chatSlice.ts     # Chat CRUD
│   │   ├── messageSlice.ts  # Message handling
│   │   ├── streamSlice.ts   # Streaming state
│   │   ├── artifactSlice.ts # Artifacts
│   │   └── codeSummarySlice.ts  # Code tracking
│   └── modelsStore.ts
└── types/                   # TypeScript types
```

---

## Configuration

### Required .env Settings

```bash
# Security (REQUIRED - generate unique values!)
SECRET_KEY=your-secure-random-string-here  # openssl rand -hex 32
ADMIN_EMAIL=admin@example.com
ADMIN_PASS=secure-admin-password

# LLM (REQUIRED)
LLM_API_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=not-needed
LLM_MODEL=default
```

### Optional .env Settings

```bash
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
DATABASE_URL=sqlite+aiosqlite:///./data/nuechat.db
FREEFORALL=false
FREE_TIER_TOKENS=100000
IMAGE_SERVICE_URL=http://image-service:8002
TTS_SERVICE_URL=http://tts-service:8001
DEBUG=false
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

### Rate Limiting (NC-0.6.28)

Token bucket rate limiting with configurable limits:

| Action | Limit | Window | Burst |
|--------|-------|--------|-------|
| chat_message | 60 | 1 min | 10 |
| image_generation | 10 | 1 min | 3 |
| file_upload | 20 | 1 min | 5 |
| api_key_creation | 5 | 1 hour | - |
| login_attempt | 10 | 5 min | - |
| document_upload | 30 | 1 hour | - |

Rate limit headers returned:
- `X-RateLimit-Limit`: Maximum requests
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp
- `Retry-After`: Seconds to wait (when exceeded)

### Token Blacklisting (NC-0.6.28)

JWT tokens can be blacklisted on logout:
- In-memory LRU cache (max 10000 tokens)
- Automatic cleanup of expired tokens
- Refresh token rotation when older than half lifetime

### Zip Security (NC-0.6.28)

Enhanced zip extraction security:
- Path traversal prevention (`..` detection)
- Null byte detection
- Symlink detection
- Size limits (500MB uncompressed, 10000 files)
- Path depth limits (50 levels max)

---

## Image Generation

Image generation is triggered when the user's message appears to request image creation.

### Detection Flow (NC-0.6.37)

1. **Regex Pre-filter**: Quick pattern matching for image-related keywords
2. **LLM Confirmation**: If regex matches, ask the LLM to confirm the intent
3. **Queue Task**: If confirmed, add to image generation queue

**Important**: The LLM confirmation:
- Does NOT save to chat history
- Does NOT count against user's token quota
- Uses a direct API call with short timeout
- **On error, defaults to NOT generating** (safe fallback)

### Supported Keywords

The regex pre-filter catches potential image requests. The **LLM confirmation** is what actually determines intent.

**Regex triggers on:**
- Image words: image, picture, photo, illustration, artwork, graphic, icon, logo, banner, poster, avatar, thumbnail, pic, img, gfx
- Creation verbs + image words: "create an image", "generate a picture"
- Art verbs with object: "paint me a...", "draw a picture of..."

**LLM handles false positives like:**
- "to illustrate this point" → LLM says NO
- "draw conclusions from the data" → LLM says NO
- "the picture of health" → LLM says NO
- Long documents containing image-related words in non-generative context → LLM says NO

### Configuration

```bash
# Enable/disable LLM confirmation (default: true)
IMAGE_CONFIRM_WITH_LLM=true

# Image generation service URL
IMAGE_GEN_SERVICE_URL=http://localhost:8034

# Timeout for image generation (default: 600 seconds)
IMAGE_GEN_TIMEOUT=600
```

### LLM Confirmation Prompt

The LLM is asked to classify whether the user wants to generate an image:
- Returns `YES: <prompt>` if image generation is requested
- Returns `NO` if not an image request

This prevents false positives like:
- "What's in this picture?" (analyzing, not generating)
- "How do I upload an image?" (asking about images, not creating)
- "Tell me about image processing" (discussing the concept)
- "This pic is great" (referring to existing image)

---

## Token Refill System

User token counts (`tokens_used_this_month`) are automatically reset based on the configurable `token_refill_interval_hours` setting.

**How It Works:**
1. Background task runs every hour
2. Compares current time vs `last_token_reset_timestamp`
3. If elapsed >= `token_refill_interval_hours`, resets ALL users' tokens to 0

**Admin Settings:**
- Setting: `token_refill_interval_hours` (default: 720 = 30 days)
- Location: Admin → System tab

### Debug Token Resets

Enable in Admin → Site Dev → Debug Token Resets

When enabled, logs:
```
============================================================
[TOKEN_RESET_DEBUG] Token reset check triggered
[TOKEN_RESET_DEBUG] Current time (UTC): 2024-12-11T10:00:00+00:00
[TOKEN_RESET_DEBUG] Last reset: 2024-11-11T00:00:00+00:00
[TOKEN_RESET_DEBUG] Refill interval: 720 hours
[TOKEN_RESET_DEBUG] User token counts (3 users):
[TOKEN_RESET_DEBUG]   admin@example.com: 150,000 tokens (tier: enterprise)
============================================================
```

---

## Document Processing Queue

Documents uploaded to RAG are processed asynchronously via a persistent queue.

### Queue Flow
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
  → Update document (is_processed=True)
  → Remove task from queue
```

### Queue Persistence

Queue state stored in `/app/data/document_queue.json`:
```json
{
  "task-uuid-1": {
    "task_id": "task-uuid-1",
    "document_id": "doc-uuid",
    "status": "pending",
    "retry_count": 0
  }
}
```

### Task States
- `pending`: Waiting to be processed
- `processing`: Currently being processed
- `completed`: Successfully processed (removed)
- `failed`: Failed after 3 retries

---

## Message Branching

Messages form a tree structure via `parent_id`:
```
User1 (parent_id=null)
  └── Asst1 (parent_id=User1)
        └── User2 (parent_id=Asst1)
              └── Asst2 (parent_id=User2)
                    ├── User3a (parent_id=Asst2)  <- Branch A
                    └── User3b (parent_id=Asst2)  <- Branch B
```

### Frontend Message Flow

```typescript
// handleSendMessage
1. Calculate parentId from store messages (fresh read to avoid stale closures)
2. Call sendChatMessage(chatId, content, undefined, parentId)

// WebSocket flow
3. Create temp user message with parent_id
4. Send to backend via WebSocket
5. Backend saves, streams response
6. stream_end returns assistant message_id
7. Assistant becomes new leaf for next message
```

---

## Keyboard Shortcuts (NC-0.6.28)

| Shortcut | Action |
|----------|--------|
| Ctrl/⌘ + N | New chat |
| Ctrl/⌘ + / | Focus input |
| Ctrl/⌘ + B | Toggle sidebar |
| Ctrl/⌘ + Shift + Backspace | Delete chat |
| Ctrl/⌘ + Shift + A | Toggle artifacts |
| Ctrl/⌘ + K | Search |
| Ctrl/⌘ + , | Settings |
| Escape | Close panel |

---

## Filter Chains (Agentic Flows)

Admin-configurable message processing flows (Admin → Filter Chains tab).

### Step Types
- `to_llm`: Ask LLM a question
- `query`: Generate search query
- `to_tool`: Execute a tool
- `go_to_llm`: Send to main LLM
- `filter_complete`: Exit chain early
- `set_var`: Set a variable
- `compare`: Compare values
- `context_insert`: Add to context
- `call_chain`: Execute another chain
- `stop`: Stop execution
- `block`: Block with error

### Variables
- `$Query` - Original user input
- `$PreviousResult` - Last step output
- `$Var[name]` - Named outputs from earlier steps

---

## WebSocket Events

### Client → Server
- `subscribe`: Join a chat room
- `unsubscribe`: Leave a chat room
- `chat_message`: Send a message
- `stop_generation`: Stop LLM generation
- `ping`: Heartbeat
- `regenerate`: Retry a message

### Server → Client
- `stream_start`: LLM response starting
- `stream_chunk`: Content chunk
- `stream_end`: Response complete
- `stream_error`: Error during streaming
- `tool_call`: Tool being invoked
- `tool_result`: Tool response
- `image_generation`: Image status update
- `message_saved`: User message confirmed
- `pong`: Heartbeat response
- `error`: General error
- `subscribed`: Room joined
- `unsubscribed`: Room left

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| NC-0.6.37 | 2025-12-13 | LLM confirmation for image generation (safe fallback on error) |
| NC-0.6.36 | 2025-12-13 | API keys table migration, parent_id branching fix, shared chat "Show All" toggle, TTS ROCm MIOPEN_FIND_MODE |
| NC-0.6.35 | 2025-12-13 | Custom assistant chat association |
| NC-0.6.34 | 2025-12-13 | Message artifacts JSON column |
| NC-0.6.33 | 2025-12-13 | Procedural memory system |
| NC-0.6.28 | 2025-12-11 | Model split, security hardening, rate limiting, keyboard shortcuts |
| NC-0.6.27 | 2025-12-11 | Voice mode parent_id fix, token reset automation, admin bypasses |
| NC-0.6.26 | 2025-12-10 | Talk to Me voice mode debug panel |
| NC-0.6.25 | 2025-12-10 | Swipe gestures for retry/branch navigation |
| NC-0.6.24 | 2025-12-09 | Filter chain debug mode |
| NC-0.6.23 | 2025-12-09 | Configurable filter chains |
| NC-0.6.22 | 2025-12-09 | Error handling & file request fixes |
| NC-0.6.21 | 2025-12-09 | Image generation queue |
| NC-0.6.20 | 2025-12-09 | Image persistence |

---

## Development Guidelines

### ⚠️ IMPORTANT: Software Switches

**ALL software switches/toggles NOT related to deployment configuration should be in the Admin Panel, not in environment variables.**

- ✅ Admin Panel: Feature toggles, behavior switches, runtime configuration
- ✅ .env: Deployment-specific (URLs, ports, API keys, secrets, database paths)
- ❌ .env: `ENABLE_SOME_FEATURE=true` (should be admin toggle)

Examples:
- `IMAGE_CONFIRM_WITH_LLM` → Should be admin toggle, not .env
- `LLM_API_BASE_URL` → Correct as .env (deployment config)
- `DEBUG` → Acceptable in .env (deployment-specific)

### Adding New Models

1. Create new file in `backend/app/models/` (e.g., `mymodel.py`)
2. Import Base and generate_uuid from `base.py`
3. Define SQLAlchemy model class
4. Add re-export in `__init__.py`
5. Run migrations on startup

### Adding New API Routes

1. Create route file in `backend/app/api/routes/`
2. Use helpers from `helpers.py` for common patterns
3. Register router in `main.py`
4. Add exception handlers if needed

### Frontend Store Slices

1. Add new slice file in `frontend/src/stores/chat/`
2. Define types in `types.ts`
3. Implement slice creator function
4. Compose in `index.ts`

### Structured Logging

```python
from app.core.logging import StructuredLogger, log_duration

logger = StructuredLogger(__name__)

# Basic logging
logger.info("User logged in", user_id="123", ip="1.2.3.4")

# Timing operations
with log_duration(logger, "database_query", table="users"):
    result = await db.execute(query)
```

---

## Database Schema Version

Current: **NC-0.6.36**

Migrations run automatically on startup in `backend/app/main.py`.

Key tables added/modified:
- `api_keys` - User-generated API keys for programmatic access (NC-0.6.36)

---

## Integration Status (NC-0.6.28)

### Backend Integration

| Component | Integration Point | Status |
|-----------|------------------|--------|
| Exception Handlers | `main.py` line 405: `setup_exception_handlers(app)` | ✅ Active |
| Rate Limiting | `auth.py` login, `knowledge_stores.py` create/upload | ✅ Active |
| Token Blacklisting | `dependencies.py` `get_current_user()`, `auth.py` logout | ✅ Active |
| Structured Logging | `llm.py` via `log_llm_request()`, `websocket.py` via `log_websocket_event()` | ✅ Active |
| Zip Security | `zip_processor.py` `validate_zip_archive()`, `chats.py` `ZipSecurityError` handling | ✅ Active |
| WebSocket Types | `websocket.py` imports typed helpers from `ws_types.py` | ✅ Active |
| Input Validators | `chats.py` zip upload, `knowledge_stores.py` document upload | ✅ Active |
| Path Traversal Fix | `knowledge_stores.py` and `documents.py` filename sanitization | ✅ Active |

### Frontend Integration

| Component | Integration Point | Status |
|-----------|------------------|--------|
| Chat Store Slices | `chatStore.ts` re-exports from `chat/` directory | ✅ Active |
| Keyboard Shortcuts | `Layout.tsx` and `ChatPage.tsx` via `useChatShortcuts` | ✅ Active |
| WebSocket Types | `WebSocketContext.tsx` uses `parseServerEvent()` | ✅ Active |
| Formatters | `ArtifactsPanel.tsx` uses `formatFileSize`, `formatMessageTime` | ✅ Active |
| ChatInput Ref | `ChatPage.tsx` passes `inputRef` for focus shortcut | ✅ Active |

### Security Hardening

- Path traversal prevention in zip files (`validate_zip_path()`)
- Path traversal prevention in file uploads (filename sanitization)
- Symlink detection in archives (`is_symlink()`)
- Dangerous file pattern detection (`is_dangerous_file()`)
- Size limits: 500MB uncompressed, 10000 files max
- Path depth: 50 levels max, 255 chars per component
- Login rate limiting: 10 attempts per 5 minutes per IP
- JWT invalidation on logout with LRU cache (10K tokens)

### Keyboard Shortcuts

| Shortcut | Action | Works In |
|----------|--------|----------|
| Ctrl/⌘+N | New chat | Anywhere |
| Ctrl/⌘+/ | Focus input | Chat page |
| Ctrl/⌘+B | Toggle sidebar | Anywhere |
| Ctrl/⌘+Shift+A | Toggle artifacts | Chat page |
| Escape | Close panel | Chat page |
