# Open-NueChat

**An open-source, full-featured LLM chat platform** with Custom GPTs, Knowledge Bases (RAG), OAuth2 authentication, billing/token tracking, tool calling, bidirectional WebSocket streaming, voice features (TTS/STT), image generation, and conversation branching.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Docker Profiles](#docker-profiles)
- [Control Script](#control-script)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

### 🤖 Custom GPTs (Assistants)
- Create custom AI assistants with specific personalities and instructions
- Configure model, temperature, max tokens, system prompts
- Attach Knowledge Bases for domain-specific context
- Publish GPTs to the marketplace for others to discover
- Subscribe to public GPTs and use them as models
- Rate and review assistants
- Welcome messages and suggested prompts

### 📚 Knowledge Bases (RAG)
- Create personal or shared knowledge stores
- Upload documents: PDF, TXT, MD, JSON, CSV, DOCX, and 40+ file types
- Local embeddings using sentence-transformers (no external API)
- FAISS vector search with GPU acceleration (ROCm, CUDA, or CPU)
- Configurable chunk size and overlap
- Persistent document processing queue (survives restarts)

### 💬 Real-time Chat
- Bidirectional WebSocket streaming
- Conversation branching (multiple response versions)
- Retry/regenerate messages with swipe gestures
- Export and share chats
- Upload zip files with full code context
- Signature extraction for 15+ programming languages

### 🎙️ Voice Features
- **Text-to-Speech (TTS)**: Kokoro model with GPU acceleration
- **Speech-to-Text (STT)**: Whisper model with VAD
- "Talk to Me" hands-free conversation mode
- Streaming audio for low latency

### 🖼️ Image Generation
- Integrated image generation service (Z-Image-Turbo)
- Support for multiple aspect ratios
- Persistent image storage
- Queue-based generation

### 🔐 Authentication & Authorization
- JWT-based authentication with access/refresh tokens
- OAuth2 integration (Google, GitHub)
- Password hashing with bcrypt
- Role-based access control
- API key management with scopes

### 🔧 Tool System
Built-in tools that the LLM can use:
- **calculator**: Safe math evaluation
- **get_current_time**: Current date/time with timezone
- **search_documents**: RAG search within documents
- **execute_python**: Sandboxed Python execution
- **format_json**: JSON validation and formatting
- **analyze_text**: Readability metrics
- **web_search**: Web search integration
- Custom MCP/OpenAPI tools support

### 💰 Billing & Token Tracking
- Three tiers: Free (100K), Pro (1M), Enterprise (10M)
- Per-message token tracking
- Automatic token reset with configurable intervals
- Admin-configurable pricing
- Admin bypass for all limits

### 🛠️ Admin Panel
- System prompt configuration
- Token pricing management
- User management
- Filter chains (no-code agentic flows)
- Debug settings for monitoring (Token Resets, Document Queue)
- Branding customization

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, SQLAlchemy (async), SQLite |
| **Frontend** | React 19, TypeScript, Tailwind CSS, Zustand |
| **LLM** | OpenAI-compatible API (Ollama, vLLM, LM Studio, etc.) |
| **Embeddings** | sentence-transformers (local) |
| **Vector Search** | FAISS (GPU: ROCm/CUDA, or CPU) |
| **TTS** | Kokoro |
| **STT** | OpenAI Whisper |
| **Image Gen** | Diffusers (Z-Image-Turbo) |

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- For GPU acceleration:
  - **AMD ROCm**: ROCm 6.0+ drivers
  - **NVIDIA CUDA**: CUDA 12.0+ and nvidia-container-toolkit
  - **CPU-only**: No additional requirements

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/open-nuechat.git
cd open-nuechat

# Copy and configure environment
cp .env.example .env
nano .env  # Set LLM_API_BASE_URL, SECRET_KEY, ADMIN_EMAIL, ADMIN_PASS
```

#### For AMD GPUs (ROCm)

```bash
# Build FAISS wheel (one-time, ~12 minutes)
./control.sh faiss-build --profile rocm

# Build and start
./control.sh build --profile rocm
./control.sh start -d --profile rocm
```

#### For NVIDIA GPUs (CUDA)

```bash
# Build and start (no FAISS wheel needed)
./control.sh build --profile cuda
./control.sh start -d --profile cuda
```

#### For CPU Only

```bash
# Build and start (no FAISS wheel needed)
./control.sh build --profile cpu
./control.sh start -d --profile cpu
```

#### Access the Application

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Docker Profiles

Open-NueChat supports multiple hardware configurations:

| Profile | Description | FAISS | Requirements |
|---------|-------------|-------|--------------|
| `rocm` | AMD GPU | ROCm GPU | ROCm 6.0+, requires `faiss-build` |
| `cuda` | NVIDIA GPU | CUDA GPU | CUDA 12.0+, nvidia-container-toolkit |
| `cpu` | CPU only | CPU | None (slower inference) |

### Services by Profile

| Profile | Main App | TTS | Image Gen |
|---------|----------|-----|-----------|
| `rocm` | ✅ ROCm | ✅ ROCm | ✅ ROCm |
| `cuda` | ✅ CUDA | ✅ CUDA | ✅ CUDA |
| `cpu` | ✅ CPU | ✅ CPU | ❌ Disabled |

---

## Control Script

The `control.sh` script manages the entire Docker deployment.

```bash
./control.sh [command] [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `faiss-build` | Build FAISS wheel for ROCm profile |
| `build` | Build Docker images |
| `start` | Start containers |
| `stop` | Stop containers |
| `down` | Stop and remove containers |
| `restart` | Restart containers |
| `logs` | View container logs |
| `status` | Show container status |
| `shell` | Open shell in container |
| `clean` | Remove containers, volumes, images |
| `faiss` | Check FAISS wheel status |
| `tts` | TTS service control |
| `image` | Image generation service control |
| `db` | Database operations |
| `migrate` | Migrate data from volumes |
| `help` | Show help message |

### FAISS Build

Build FAISS wheel for ROCm (required before building ROCm profile):

```bash
./control.sh faiss-build --profile rocm         # Build wheel (~12 min)
./control.sh faiss-build --profile rocm --force # Force rebuild
./control.sh faiss                              # Check wheel status
```

> **Note**: CUDA and CPU profiles use pre-built packages from PyPI. No `faiss-build` required.

### Build

```bash
./control.sh build --profile <rocm|cuda|cpu>    # Build for profile
./control.sh build --profile rocm --no-cache    # Build without cache
```

### Start / Stop

```bash
./control.sh start -d --profile <rocm|cuda|cpu> # Start in background
./control.sh start -d --build --profile cuda    # Rebuild and start
./control.sh stop                               # Stop containers
./control.sh down                               # Stop and remove
./control.sh down -v                            # Also remove volumes
./control.sh restart                            # Restart containers
```

### Logs

```bash
./control.sh logs                # Show last 100 lines
./control.sh logs -f             # Follow logs
./control.sh logs -n 500         # Show last 500 lines
```

### Services

```bash
# TTS Service
./control.sh tts start
./control.sh tts stop
./control.sh tts status

# Image Generation
./control.sh image start
./control.sh image stop
./control.sh image logs
```

### Database

```bash
./control.sh db migrate          # Run migrations
./control.sh db seed             # Seed sample data
./control.sh db reset            # Reset database (DESTRUCTIVE)
```

### Cleanup

```bash
./control.sh clean               # Basic cleanup
./control.sh clean --all         # Remove everything
./control.sh clean --volumes     # Remove only volumes
./control.sh clean --images      # Remove only images
```

### Complete Examples

```bash
# === AMD ROCm Setup ===
cp .env.example .env && nano .env
./control.sh faiss-build --profile rocm   # One-time, ~12 min
./control.sh build --profile rocm
./control.sh start -d --profile rocm

# === NVIDIA CUDA Setup ===
cp .env.example .env && nano .env
./control.sh build --profile cuda
./control.sh start -d --profile cuda

# === CPU Setup ===
cp .env.example .env && nano .env
./control.sh build --profile cpu
./control.sh start -d --profile cpu

# === Daily Operations ===
./control.sh logs -f                      # Monitor logs
./control.sh status                       # Check status
./control.sh shell                        # Debug in container
./control.sh restart                      # Restart after config change

# === Updates ===
git pull
./control.sh build --profile <profile>
./control.sh start -d --profile <profile>
```

---

## Configuration

### Required Environment Variables

```bash
# .env file - minimum required
SECRET_KEY=your-secure-random-string-here
ADMIN_EMAIL=admin@example.com
ADMIN_PASS=your-admin-password

# LLM Configuration
LLM_API_BASE_URL=http://localhost:11434/v1  # Ollama
LLM_MODEL=llama3.2                          # Model name
```

### Optional Environment Variables

```bash
# Branding
APP_NAME=Open-NueChat
APP_TAGLINE=AI-Powered Chat Platform
DEFAULT_THEME=dark

# OAuth (optional)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=

# Features
ENABLE_REGISTRATION=true
ENABLE_BILLING=true
FREEFORALL=false              # true = unlimited tokens for all users

# GPU
FAISS_USE_GPU=true            # false for CPU-only FAISS
```

See `.env.example` for all available options.

---

## Project Structure

```
open-nuechat/
├── backend/
│   ├── Dockerfile            # ROCm build
│   ├── Dockerfile.cuda       # NVIDIA CUDA build
│   ├── Dockerfile.cpu        # CPU-only build
│   ├── app/
│   │   ├── api/routes/       # FastAPI endpoints
│   │   ├── services/         # Business logic
│   │   │   ├── llm.py        # LLM client
│   │   │   ├── rag.py        # FAISS + embeddings
│   │   │   ├── document_queue.py
│   │   │   ├── stt.py        # Speech-to-text
│   │   │   └── billing.py    # Token tracking
│   │   ├── filters/          # Stream filters
│   │   ├── tools/            # Built-in tools
│   │   └── models/           # SQLAlchemy models
│   ├── wheels/               # Pre-built FAISS wheel (ROCm)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Route pages
│   │   └── stores/           # Zustand state
│   └── package.json
├── tts-service/
│   ├── Dockerfile            # CPU build
│   ├── Dockerfile.rocm       # ROCm build
│   ├── Dockerfile.cuda       # CUDA build
│   └── main.py
├── image-service/
│   ├── Dockerfile.rocm       # ROCm build
│   ├── Dockerfile.cuda       # CUDA build
│   └── main.py
├── docker-compose.yml        # All profiles defined here
├── control.sh                # Management script
├── .env.example              # Environment template
├── LICENSE                   # Apache 2.0 + attribution
├── Notes.md                  # Development notes
├── Signatures.md             # API signatures
└── README.md
```

---

## API Endpoints

### Authentication
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Sign in
- `POST /api/auth/refresh` - Refresh token
- `GET /api/auth/me` - Current user

### Chats
- `GET /api/chats` - List chats
- `POST /api/chats` - Create chat
- `GET /api/chats/{id}` - Get chat
- `DELETE /api/chats/{id}` - Delete chat

### Assistants (Custom GPTs)
- `GET /api/assistants` - My assistants
- `POST /api/assistants` - Create assistant
- `GET /api/assistants/explore` - Browse marketplace

### Knowledge Stores
- `GET /api/knowledge-stores` - My stores
- `POST /api/knowledge-stores` - Create store
- `POST /api/knowledge-stores/{id}/documents` - Upload document

### WebSocket
- `WS /ws/ws?token={jwt}` - Real-time chat

Full API documentation available at `/docs` when running.

---

## Schema Version

**Current: NC-0.6.27**

The database schema is automatically migrated on startup.

---

## License

**Apache License 2.0** with additional attribution requirements.

```
Copyright 2024 Open-NueChat Contributors

Licensed under the Apache License, Version 2.0
```

### Additional Requirements

Any use, modification, or distribution requires:

1. **Attribution**: Credit "Open-NueChat" in derivative works and documentation
2. **Citation**: For academic/research use:
   ```
   Open-NueChat: An Open-Source LLM Chat Platform
   https://github.com/yourname/open-nuechat
   ```
3. **Visible Credit**: If deployed as a service, include credit in footer or about page

See [LICENSE](LICENSE) for full terms.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [React](https://react.dev/) - Frontend library
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Kokoro](https://github.com/hexgrad/kokoro) - Text-to-speech
- [Whisper](https://github.com/openai/whisper) - Speech-to-text
- [sentence-transformers](https://www.sbert.net/) - Embeddings
