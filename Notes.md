# Open-NueChat - Development Notes

## Project Overview

Full-stack LLM chat application with:
- FastAPI backend (Python 3.13)
- React 19 + TypeScript frontend
- SQLite database (aiosqlite)
- Multi-GPU support (ROCm, CUDA, CPU)
- FAISS GPU for vector search
- OpenAI-compatible LLM API integration

**Current Version:** NC-0.8.0.12

---

## Recent Changes (NC-0.8.0.12)

### New Tools: search_replace, web_search, web_extract, grep_files, sed_files

Six new tools added to the tool registry:

| Tool | Category | Description |
|------|----------|-------------|
| `search_replace` | file_manager | Find and replace exact text in a session file. Search must be unique. Updates memory, DB, and disk. |
| `web_search` | web | Search the web via DuckDuckGo HTML. Returns titles, URLs, snippets. No API key needed. |
| `web_extract` | web | Extract structured content from a URL: title, meta, headings, links, images, clean text. |
| `grep_files` | file_manager | Search a pattern across ALL session files. Supports regex, file glob filter, context lines. |
| `sed_files` | file_manager | Batch regex find/replace across multiple files. Preview mode by default, force=true to apply. |
| `execute_python` | code_exec | (existing) - already present, listed for completeness |

**Tool Category Updates:**
- `web` category: added `web_search`, `web_extract`
- `file_manager` category: added `search_replace`, `grep_files`, `sed_files`
- Legacy `web_search` / `file_ops` categories updated accordingly

**New Hallucinated Tool Delegates:**
- `str_replace`, `find_replace`, `edit_file` → `search_replace`
- `grep`, `find_in_files`, `search_files` → `grep_files`
- `sed`, `replace_all`, `batch_replace` → `sed_files`
- `scrape`, `scrape_url` → `web_extract`

---

## Recent Changes (NC-0.8.0.11)

### Enhanced execute_python Tool

The `execute_python` tool now supports direct output to chat:

**New Parameters:**
- `output_image` (boolean): Captures matplotlib figures or PIL images and sends directly to chat as base64
- `output_text` (boolean): Sends result/output directly to chat without LLM reformatting

**Available Modules:**
- math, statistics, datetime, random, json, re, collections, itertools, functools
- matplotlib.pyplot (as plt) - when output_image=True
- PIL/Pillow - when output_image=True

**WebSocket Events:**
- `direct_image`: Sends base64 image directly to frontend
- `direct_text`: Sends text directly to frontend

**Example Usage (LLM):**
```python
# Generate a chart
execute_python(
    code='''
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Sample Chart")
plt.savefig("chart.png")
''',
    output_image=True
)

# Output formatted data
execute_python(
    code='''
result = "| Name | Value |\\n|------|-------|\\n| A | 100 |\\n| B | 200 |"
print(result)
''',
    output_text=True
)
```

---

## Recent Changes (NC-0.8.0.10)

### Fresh Install Settings Seeding

**Problem:** Admin Panel showed "Failed to load settings" on fresh install because `system_settings` table was empty.

**Solution:** 
1. Added `seed_default_settings()` in `main.py` lifespan that populates `system_settings` table from `SETTING_DEFAULTS` if empty.
2. Fixed migration version detection: fresh installs now start at `NC-0.0.0` so ALL migrations run, not just those after `NC-0.5.0`.

**Detection logic:**
- If `schema_version` row exists → use stored version
- If no row AND `system_settings` is empty → fresh install → `NC-0.0.0`
- If no row AND `system_settings` has data → legacy DB → `NC-0.5.0`

---

## Recent Changes (NC-0.8.0.9)

### Centralized Settings Keys

All system setting keys are now defined in `/backend/app/core/settings_keys.py`:

```python
from app.core.settings_keys import SK, SETTING_DEFAULTS

# Use constants instead of strings
width = await SettingsService.get_int(db, SK.IMAGE_GEN_DEFAULT_WIDTH)
```

**Benefits:**
- Single source of truth for all setting keys and defaults
- IDE autocomplete with `SK.`
- Typos caught at import time
- No more duplicate SETTING_DEFAULTS dicts

**Key Categories:**
- `SK.IMAGE_GEN_*` - Image generation settings
- `SK.LLM_*` - LLM settings
- `SK.GOOGLE_*` / `SK.GITHUB_*` - OAuth settings
- `SK.DEBUG_*` - Debug flags
- `SK.HISTORY_COMPRESSION_*` - Compression settings
- `SK.API_RATE_LIMIT_*` - Rate limits

### New Tool: create_file

Creates files with arbitrary content. Used for saving code, documents, configs, or any text-based files.

```python
create_file(path="src/main.cpp", content="...", overwrite=True)
```

Files saved to `/app/data/artifacts/` (or `ARTIFACTS_DIR` env var).

### Hallucinated Tool Delegates

LLMs often hallucinate tools from their training data (like `create_artifact` from Claude). These are now automatically delegated to real tools:

| Hallucinated Tool | Delegates To |
|-------------------|--------------|
| `create_artifact`, `write_file`, `save_file`, `artifact` | `create_file` |
| `run_code`, `run_python`, `code_interpreter`, `execute_code` | `execute_python` |
| `web_search`, `search_web`, `browse`, `browser` | `fetch_webpage` |
| `create_image`, `dalle`, `dall-e`, `text_to_image` | `generate_image` |
| `read_file`, `view_file`, `cat_file`, `open_file` | `view_file_lines` |
| `math`, `calculate`, `compute` | `calculator` |
| `get_time`, `current_time`, `now`, `date` | `get_current_time` |
| `agent_search`, `agent_read` | `memory_search`, `memory_read` |
| `str_replace`, `find_replace`, `replace_in_file`, `edit_file` | `search_replace` |
| `grep`, `find_in_files`, `search_files`, `search_all` | `grep_files` |
| `sed`, `replace_all`, `batch_replace` | `sed_files` |
| `extract_webpage`, `scrape`, `scrape_url` | `web_extract` |

**Argument Mapping:** Common argument name variations are mapped:
- `data`, `text`, `body`, `source` → `content` (for create_file)
- `filename`, `file_path`, `name` → `path` (for create_file)
- `uri`, `link` → `url` (for fetch_webpage)
- `description`, `text` → `prompt` (for generate_image)
- `old_str`, `find`, `pattern` → `search` (for search_replace)
- `new_str`, `replacement` → `replace` (for search_replace)

**Rejection:** Unknown tools still get `TOOL_NOT_FOUND` error with available tools list.

### Image Gen Settings Fix

Fixed admin panel image generation settings not being applied:
- Auto-detection flow now uses admin settings
- Tool handler now uses admin settings
- Removed `or 1024` fallback pattern that masked issues
- All code paths now use `SK.IMAGE_GEN_DEFAULT_WIDTH/HEIGHT`

---

## Recent Changes (NC-0.8.0.8)

### Admin Image Generation Settings

New **Admin Panel → Image Gen** tab for configuring default image generation settings:

**Settings:**
- `image_gen_default_width` (default: 1024, range: 256-2048)
- `image_gen_default_height` (default: 1024, range: 256-2048)
- `image_gen_default_aspect_ratio` (default: "1:1", options: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
- `image_gen_available_resolutions` (JSON array of resolution options)

**Endpoints:**
- `GET /api/admin/image-gen-settings` - Admin fetch settings
- `PUT /api/admin/image-gen-settings` - Admin save settings
- `GET /api/admin/public/image-settings` - Public endpoint for frontend

### Image Persistence Across Reloads

**Problem:** Generated images disappeared when reloading the page or opening chat from another location.

**Solution:**
1. **Backend** (`tools/registry.py`): `generate_image` tool's `notify_completion` callback now saves image metadata to `message_metadata` in the database
2. **Frontend** (`stores/chat/messageSlice.ts`): `fetchMessages` now loads `generated_image` from message metadata into `generatedImages` store

**Key Changes:**
- Tool handler fetches default dimensions from admin settings via `SettingsService.get_int()`
- Image metadata saved with full details: `url`, `width`, `height`, `seed`, `prompt`, `job_id`
- Proper TypeScript typing with `GeneratedImage` interface

### Image Preview in Artifacts Panel

**Problem:** Generated images showed as "Code" view displaying the URL text.

**Solution:** Added `'image'` to the `canPreview()` function in `ArtifactsPanel.tsx`:

```typescript
const canPreview = (art: Artifact): boolean => {
  return ['html', 'react', 'svg', 'markdown', 'mermaid', 'image'].includes(art.type);
};
```

### Download All Includes Images

**Problem:** "Download All" button created ZIP with image URLs as text files instead of actual images.

**Solution:** Updated `handleDownloadAll()` in `ChatPage.tsx` to detect `artifact.type === 'image'` and:
- Fetch images from URLs (relative or absolute)
- Handle base64 data URLs
- Handle raw base64 strings
- Add actual image blob to ZIP

### Image Context Hidden from Display

**Problem:** When users attach generated images, technical context blocks were visible:
```
[IMAGE CONTEXT - USER-GENERATED, NOT BY LLM]
...
[END IMAGE CONTEXT]
```

**Solution:** Added regex to `preprocessContent()` in `MessageBubble.tsx` to strip these blocks from display while preserving in message history.

### Temporary MCP Server Installation

New tool category **mcp_install** allows LLM to install MCP (Model Context Protocol) servers on-demand:

**Tools:**
- `install_mcp_server` - Install a temporary MCP server by name and URL
- `uninstall_mcp_server` - Manually remove an installed server
- `list_mcp_servers` - List all temporary servers with expiry status

**Auto-Cleanup:**
- Servers automatically removed after 4 hours of non-use
- Background worker checks every 15 minutes
- Usage tracked via `updated_at` timestamp

**Backend Implementation:**
- New service: `app/services/temp_mcp_manager.py`
- `TempMCPManager` class with install/uninstall/list methods
- Cleanup worker started in `main.py` lifespan
- Temporary flag stored in `Tool.config['is_temporary']`

**Frontend:**
- New toggle in ActiveToolsBar: "MCP Install" (package/box icon)
- When enabled, LLM can call install/uninstall/list tools

### Force Trigger Keywords for Global KB

New feature allowing Global KB content to be ALWAYS injected when specific keywords are detected, bypassing the semantic search score threshold.

**Use Case:** Ensure authoritative content (policies, procedures, guidelines) is loaded whenever relevant topics are mentioned, regardless of semantic similarity scores.

**New Database Columns (knowledge_stores):**
- `force_trigger_enabled` - Enable force trigger mode
- `force_trigger_keywords` - JSON list of trigger keywords
- `force_trigger_max_chunks` - Max chunks to load when triggered (default: 5)

**Backend Logic (rag.py):**
1. Check each global store for force trigger keywords in query
2. If matched, bypass `global_min_score` threshold (use 0.0)
3. Load up to `force_trigger_max_chunks` instead of `global_max_results`
4. Log: `[GLOBAL_RAG] FORCE TRIGGER: Store 'X' triggered by keyword 'Y'`

**Admin UI (Admin.tsx → Global KB tab):**

| Setting | Control | Description |
|---------|---------|-------------|
| Relevance Threshold | Slider + Input (0-1) | Minimum semantic similarity score. Shows label: strict/balanced/lenient/very lenient |
| Max Results | Input (1-20) | Maximum chunks to return from search |
| Require Keywords | Toggle + Tags | Only search when query contains keywords |
| Force Trigger | Toggle + Tags + Max | Bypass threshold when trigger keywords match |

**Relevance Threshold Labels:**
- ≥0.8 = strict (fewer but highly relevant)
- 0.6-0.8 = balanced
- 0.4-0.6 = lenient (more results)
- <0.4 = very lenient

**Example:**
- Store: "Company Policies"
- Force trigger keywords: `["policy", "procedure", "guideline", "rule"]`
- When user asks "What is the vacation policy?", the KB content is ALWAYS loaded regardless of semantic score.

### Tool Debugging Settings

New debug settings in **Admin Panel → Site Dev** to trace tool-related data:

| Setting | Description |
|---------|-------------|
| `DEBUG_Tool_Advertisements` | Log all tool definitions sent to the LLM (names, descriptions, parameters) |
| `DEBUG_Tool_Calls` | Log all data exchanged with LLM and tools (call requests, arguments, results) |

**Log Tags:**
- `[DEBUG_TOOL_ADVERTISEMENTS]` - Tool definitions being sent to LLM
- `[DEBUG_TOOL_CALLS]` - Tool call requests and results

**Example Output:**
```
[DEBUG_TOOL_ADVERTISEMENTS] ========== TOOLS SENT TO LLM ==========
[DEBUG_TOOL_ADVERTISEMENTS] Total tools: 5
[DEBUG_TOOL_ADVERTISEMENTS] [1] fetch_webpage
[DEBUG_TOOL_ADVERTISEMENTS]     Description: Fetch content from a webpage URL...
[DEBUG_TOOL_ADVERTISEMENTS]     Parameters: ['url']
[DEBUG_TOOL_ADVERTISEMENTS] ==========================================

[DEBUG_TOOL_CALLS] ========== TOOL CALL REQUEST ==========
[DEBUG_TOOL_CALLS] Tool: fetch_webpage
[DEBUG_TOOL_CALLS] Arguments: {"url": "https://example.com"}

[DEBUG_TOOL_CALLS] ========== TOOL CALL RESULT ==========
[DEBUG_TOOL_CALLS] Tool: fetch_webpage
[DEBUG_TOOL_CALLS] Result length: 4532
[DEBUG_TOOL_CALLS] Result preview: <html>...
[DEBUG_TOOL_CALLS] ======================================
```

### Task Queue Auto-Execute

**Problem:** When the LLM stops responding without processing queued tasks, the tasks just sit there.

**Solution:** Auto-execute pending tasks when LLM goes silent.

**How it works:**
1. After each LLM response, check if there are pending tasks but no current task
2. If tasks are waiting and queue is not paused, automatically send the next task to the LLM
3. The task is wrapped in a `[TASK QUEUE - AUTO EXECUTION]` prompt with instructions to call `complete_task` or `fail_task`
4. Recursively processes tasks until queue is empty (max depth: 10 to prevent infinite loops)
5. When queue is empty, sends `task_queue_empty` WebSocket event

**WebSocket Events:**
- `task_auto_execute` - Sent when auto-starting a task
- `task_queue_empty` - Sent when all tasks are complete

**Log Tags:**
- `[TASK_QUEUE_AUTO]` - Auto-execution messages

**Example Flow:**
```
User: "Plan a 3-day trip to Paris"
LLM: [adds 5 tasks: research, hotels, restaurants, attractions, itinerary]
LLM: [stops responding]
System: [TASK_QUEUE_AUTO] Detected 5 pending tasks, auto-executing...
System: [sends first task to LLM]
LLM: [researches, calls complete_task]
System: [sends second task to LLM]
... continues until queue empty ...
System: [task_queue_empty] 5 tasks completed
```

### Active Tools Filtering Fix

**Problem:** Tool buttons in ActiveToolsBar had no effect - all tools were always sent to LLM.

**Root Cause:** WebSocket handler used `enable_tools` (boolean) from message payload but never read `chat.active_tools` (list of categories).

**Solution:** Added tool category mapping and filtering in `websocket.py`:

```python
# Category → Backend Tools mapping
TOOL_CATEGORY_MAP = {
    "web_search": ["fetch_webpage", "fetch_urls"],
    "code_exec": ["execute_python"],
    "file_ops": ["view_file_lines", "search_in_file", "list_uploaded_files", "view_signature", "request_file"],
    "kb_search": ["search_documents"],
    # user_chats_kb is UI-only toggle; agent tools always available
}

# Always available when tools enabled (includes Agent Memory)
UTILITY_TOOLS = [
    "calculator", "get_current_time", "format_json", "analyze_text",
    "agent_search", "agent_read",  # Agent Memory - always available
]
```

**Backend reads `chat.active_tools` before expunging and filters tools accordingly.**

### API Model Names

`/v1/models` endpoint now exposes assistants by cleaned name instead of UUID:
- Spaces → underscores
- Non-alphanumeric (except `-` and `_`) removed  
- Lowercased

Example: "My Research Assistant!" → `my_research_assistant`

Legacy `gpt:uuid` format still supported for backward compatibility.

### Performance Indexes (NC-0.8.0.6)

Added database indexes for query optimization:
- `idx_document_owner` - documents by owner
- `idx_document_store` - documents by knowledge store
- `idx_document_processed` - processing queue
- `idx_usage_user_created` - billing history
- `idx_message_id_chat` - message lookups

### Pre-emptive RAG Toggle

New admin setting: **Admin Panel → Features → Enable Pre-emptive RAG**

When **enabled** (default):
- Global KB search runs (always)
- Chat History KB search runs
- Assistant KB search runs
- If any non-Global KB finds results, filter chains with `skip_if_rag_hit=True` are skipped

When **disabled**:
- Global KB search runs (always)
- Chat History KB and Assistant KB searches are skipped
- Filter chains (including web search) run unconditionally
- LLM can still use `search_documents` tool on-demand

**Setting Key:** `enable_preemptive_rag` (boolean, default: true)

---

## Recent Changes (NC-0.8.0.6)

### Frontend File Operation Tools

LLM can now view and modify files in the Artifacts panel using XML-style tool tags.

**Tools:**
- `<request_file path="..." offset="..."/>` - Get file content (chunked, 20KB/request)
- `<find_line path="..." contains="..."/>` - Find line number containing text
- `<find path="..." search="..."/>` - Search for text in specific file
- `<find search="..."/>` - Search all artifacts
- `<search_replace path="...">===== SEARCH\n...\n===== Replace\n...</search_replace>` - Replace text

**Processing Flow:**
1. LLM outputs tool tag during streaming
2. Frontend detects complete tag via regex
3. Frontend sends `stop_generation` to backend
4. Frontend executes tool against artifacts in Zustand store
5. Frontend sends result via WebSocket (`save_user_message: false`)
6. Backend continues LLM with tool result injected

**Key Files:**
- `WebSocketContext.tsx` - Tool detection patterns, execution, result delivery
- `MessageBubble.tsx` - `preprocessContent()` strips tool tags from display

**Artifact Matching:** `findArtifactByPath()` matches by exact path, suffix, basename, or partial match (case-insensitive).

### File Chunking (request_file)

Large files are served in 20KB chunks to avoid context overflow.

**Frontend (`WebSocketContext.tsx`):**
```typescript
const FILE_CHUNK_SIZE = 20000;

// extractFileRequests returns {path, offset} pairs
// Streaming handler chunks content and sends continuation hint
```

**Backend (`registry.py`):**
```python
# request_file tool handler
# Returns: content, offset, end_offset, total_size, more_available, next_request
```

### File Truncation (Attachments)

Files >20KB are truncated when building messages for LLM.

**Backend (`llm.py` - `_build_user_content`):**
- Files >20KB truncated with `[... TRUNCATED at X of Y chars ...]`
- Includes hint: `[Use <request_file path="..." offset="X"/> to retrieve more]`

**Frontend (`WebSocketContext.tsx`):**
- Attachment processor creates manifest for large files
- Stores full content as artifact, sends placeholder to LLM

### Tool Tag Display Hiding

Tool tags hidden from user but kept in message history.

**`MessageBubble.tsx` - `preprocessContent()`:**
```typescript
// Strips from display:
.replace(/<request_file\s+[^>]*\/?>/gi, '')
.replace(/<find_line\s+[^>]*\/?>/gi, '')
.replace(/<find\s+[^>]*\/?>/gi, '')
.replace(/<search_replace\s+[^>]*>[\s\S]*?<\/search_replace>/gi, '')
```

### Smart RAG Compression

RAG context is now optimized before injection into the LLM prompt.

**1. Subject Re-Ranking**
Results are re-ranked by keyword overlap with the query subject, not just vector similarity.

```python
# Keywords extracted from query (stopwords removed)
# Each result scored by overlap ratio
boost = overlap_ratio * 0.2  # Max 20% boost
result['_rerank_score'] = min(1.0, original_score + boost)
```

**2. Extractive Summarization**
Large chunks (>500 tokens / ~2000 chars) are auto-summarized using extractive summarization:
- Sentences scored by relevance to query
- Top 30% of sentences kept (min 2, max 5)
- Re-ordered by original position for coherence
- Only used if summary is 30%+ smaller than original

**3. Token Budget**
New `max_context_tokens` parameter stops adding context when budget reached.

**New Methods in RAGService:**
```python
async def _rerank_by_subject(query, results, debug_enabled) -> List[Dict]
async def _summarize_chunk(content, query, debug_enabled) -> Optional[str]
```

**Updated Method Signature:**
```python
async def get_knowledge_store_context(
    db, user, query, knowledge_store_ids,
    top_k=5,
    bypass_access_check=False,
    chat_id=None,
    max_context_tokens=None,      # NEW: Token budget
    enable_summarization=True,    # NEW: Auto-summarize large chunks
) -> str
```

### Category Filter Fix

**Problem:** Marketplace category filter didn't work.

**Root Cause:** 
1. `AssistantPublicInfo` schema was missing `category` field - endpoints weren't returning category data
2. Misunderstanding of NC-0.8.0.4 architecture - Modes ARE Categories (unified)

**Fix:**
1. `/assistants/categories` queries `AssistantMode` table (Modes = Categories)
2. Returns `value` as slugified mode name: `m.name.lower().replace(' ', '_')`
3. Added `category` field to `AssistantPublicInfo` schema
4. Updated `/explore`, `/discover`, `/subscribed` endpoints to include `category=assistant.category`

Categories are managed in Admin Panel → Assistant Modes tab.

---

## Recent Changes (NC-0.8.0.5)

### Per-Chat Token Limits
Users can now set max input and output tokens per chat via the model selector dropdown.

**New Columns:**
- `chats.max_input_tokens` (INTEGER, nullable) - Limit on input context
- `chats.max_output_tokens` (INTEGER, nullable) - Limit on completion length

**Frontend UI:**
- Token Limits section added to model selector dropdown in ChatPage.tsx
- Max Input and Max Output number inputs
- Values saved immediately on change via `handleTokenLimitUpdate()`

**Backend:**
- Added to `ChatCreate`, `ChatUpdate`, `ChatResponse` schemas
- `update_chat` endpoint handles token limit updates
- `LLMService.stream_message()` uses chat-level limits:
  - `effective_max_output_tokens` = `chat.max_output_tokens` or service default
  - `effective_max_input_tokens` = `chat.max_input_tokens` or None (no limit)

### Agent Memory Tools
Two new tools for accessing archived conversation history:

```python
agent_search(query, max_results=3)  # Search archived history
agent_read(filename)                 # Read specific agent file
```

**Agent Memory Files:**
- Stored as `{AgentNNNN}.md` files in UploadedFile table
- Hidden from UI artifacts panel
- Created when conversation exceeds `max_input_tokens`
- Contains summary header + full conversation archive

### Error Sanitization
Raw API errors no longer reach the browser.

**New method:** `LLMService._sanitize_error_for_user(error_msg)`

**Mappings:**
| Raw Error | User Message |
|-----------|--------------|
| `max_tokens...context length` | "The conversation is too long. Try starting a new chat or reduce the Max Output Tokens setting." |
| `rate limit` | "Rate limit reached. Please wait a moment and try again." |
| `authentication` | "API authentication error. Please check your settings or contact support." |
| `model not found` | "The selected model is not available. Please choose a different model." |
| `content filter` | "The request was blocked by content filters. Please rephrase your message." |
| `timeout` | "The request timed out. Please try again." |
| `500/502/503` | "The AI service is temporarily unavailable. Please try again in a moment." |

### Max Output Token Validation
Before calling API, `max_output_tokens` is capped to remaining context space:
```python
max_possible_output = model_context_limit - current_input_tokens - 1000
if effective_max_output_tokens > max_possible_output:
    effective_max_output_tokens = max(1000, max_possible_output)
```

### Input Token Overflow Handling
When `max_input_tokens` is exceeded, overflow messages are archived to Agent Memory:
```python
# Archive oldest messages (keeping system + recent 10)
await agent_memory.compress_messages(...)
messages = [messages[0]] + messages[-(keep_recent):]
```

---

## Recent Changes (NC-0.8.0.3)

### Sidebar Real-Time Updates
Sidebar now automatically reloads when chats are created, deleted, or updated.

**Implementation:**
- Added `sidebarReloadTrigger` counter to chat store
- `triggerSidebarReload()` called after `createChat`, `deleteChat`, `addMessage`
- Sidebar useEffect watches trigger and calls `fetchChats()` without clearing existing chats

**Key Fix:** Sidebar reload does NOT pass `sortBy` parameter, so `sortChanged` is false and existing chats array is preserved while counts refresh.

### Timestamp Preservation (User's Temporal Relation)
**Philosophy:** `updated_at` represents when the USER last interacted with the chat, not when the database row was modified.

**Changes:**
1. **Removed `onupdate=func.now()`** from `Chat.updated_at` in `app/models/chat.py`
2. **Explicit timestamp updates** only in websocket.py when user sends a message:
   ```python
   chat.updated_at = datetime.now(timezone.utc)
   ```
3. **Import preserves original timestamps:**
   - `created_at` = FIRST message timestamp (when user started the chat)
   - `updated_at` = LAST message timestamp (when user last interacted)

### Migration System Fix
**Problem:** All migrations were running on EVERY server startup, causing mass timestamp updates via backfill UPDATE statements.

**Root Cause:** Migration loop at line 405-413 in `main.py` iterated through all versions without checking if already applied.

**Fix:** Added version check before running migrations:
```python
if version_tuple <= current:
    logger.debug(f"Skipping migrations for {version} (already at {current_version})")
    continue
```

### Chat Click Fix
**Problem:** Clicking a chat removed it from sidebar until reload.

**Root Cause:** ChatPage useEffect fetched fresh chat data from backend and replaced it in the chats array, but the fresh data lacked `_assignedPeriod` tag.

**Fix:** Preserve `_assignedPeriod` when updating chat:
```typescript
updatedChats[chatIndex] = { ...chat, _assignedPeriod: existingChat._assignedPeriod };
```

---

## Recent Changes (NC-0.8.0.1.2)

### Chat Source Field
Added `source` column to `chats` table for tracking where chats were imported from.

**Values:** `native`, `chatgpt`, `grok`, `claude`

**Migration (NC-0.8.0.1.2):**
1. Add `source VARCHAR(50) DEFAULT 'native'` column
2. Backfill from title prefixes (`ChatGPT: ` → `chatgpt`, etc.)
3. Clean title prefixes after backfill

**Changes:**
- `app/models/chat.py`: Added `source` column
- `app/api/routes/chats.py`: Import sets `source` field, sidebar filtering uses `source` instead of title parsing
- `app/api/routes/user_settings.py`: Export includes `source` field
- `frontend/src/components/Sidebar.tsx`: Groups by `chat.source` instead of title prefix matching

### Generated Images in Artifacts Panel
Images generated via `nuechat_image_generation` now appear in the Artifacts panel.

**Naming:** `image001.png`, `image002.jpg`, etc. (numbered, extension from URL or default png)

**Changes:**
- `frontend/src/types/index.ts`: Added `'image'` to Artifact type, added `imageData` field
- `frontend/src/pages/ChatPage.tsx`: `artifacts` useMemo includes `generatedImages` as artifacts
- `frontend/src/components/ArtifactsPanel.tsx`: 
  - `getPreviewContent()`: Renders image with dimensions/seed info
  - `getLanguageLabel()`: Returns 'Image'
  - `TypeIcon`: Uses image icon
  - `getFileExtension()`: Detects extension from URL
  - `downloadArtifact()`: Handles image download (URL fetch or base64)

### Image Context for LLM
When building chat history, the backend injects generated image metadata so the LLM knows what was previously created.

**Location:** `app/services/llm.py` `_build_messages()`

**Context injected for assistant messages with completed image generation:**
```
[PREVIOUSLY GENERATED IMAGE - NOT BY THIS LLM RESPONSE]
The user requested an image and one was generated with these details:
- Prompt: "..."
- Dimensions: WxH
- Seed: N
The image exists and the user can see it. Do NOT attempt to generate a new image unless explicitly asked.
[END IMAGE CONTEXT]
```

### User Data Export Fixes (NC-0.8.0.1)
- Fixed `chat.model_id` → `chat.model`
- Fixed `user.display_name` → `user.username`

---

## Recent Changes (NC-0.8.0.0)

### Dynamic Tools & Assistant Modes

Major architecture change making RAG and tools **composable via filter chains** instead of hardcoded.

#### Key Concepts

**Three Trigger Types:**
| Node Type | Trigger | UI Location | Variable |
|-----------|---------|-------------|----------|
| `user_hint` | Text selection | Popup bubble over selection | `$Selected` |
| `user_action` | Button click | Below messages | `$MessageContent` |
| `export_tool` | LLM pattern | In LLM output | `$1`, `$2` (capture groups) |

**Tool Categories:**
| Category | Examples | User Control |
|----------|----------|--------------|
| Always Active | Global KB | None (invisible, always on) |
| Mode Controlled | Web, Artifacts, Images | Mode preset + toggle |
| User Toggleable | Citation style, verbosity | Full user control |

#### New Filter Chain Primitives

**RAG Evaluators:**
```python
_prim_local_rag      # User's uploaded docs for current chat
_prim_kb_rag         # Current assistant's knowledge base
_prim_global_kb      # Global knowledge bases (always active, no icon)
_prim_user_chats_kb  # User's chat history KB
```

**Dynamic Tool Nodes:**
```python
_prim_export_tool    # LLM-triggered ($WebSearch="...")
_prim_user_hint      # Text selection popup menu
_prim_user_action    # Buttons below messages
```

#### ExecutionContext Enhancement

```python
from_llm: bool = False   # True when processing LLM output
from_user: bool = False  # True when processing user input
```

#### Assistant Modes

Admin-defined presets controlling which tools are active:

| Mode | Active Tools | Advertised |
|------|--------------|------------|
| Creative Writing | None | None |
| Coding | artifacts, file_ops, code_exec | artifacts, file_ops |
| Deep Research | web_search, citations | web_search |
| General | web_search, artifacts | None |

- Dropdown in Custom GPT modal ("Assistant Style")
- Admin Panel tab to add/edit modes
- User can toggle individual tools (becomes "Custom" mode)

#### Active Tools Bar

Visual toolbar showing tool status:
- Line-based SVG icons (admin uploadable)
- **Active**: Glow (brighter, same hue as background)
- **Inactive**: No glow (muted but visible)
- **Hover**: Subtle brightness increase + tooltip
- Global KB has **no icon** (always active, invisible guardrails)

#### User Triggers

**Text Selection (user_hint):**
```
User selects "quantum entanglement"
         ↓
┌─────────────────────┐
│ 🔍 Search Web       │  → on_trigger: [to_tool(web_search)]
│ 💡 Explain          │  → on_trigger: [go_to_llm("Explain: {$Selected}")]
│ 📖 Tell me more     │  → on_trigger: [go_to_llm("Elaborate: {$Selected}")]
└─────────────────────┘
```

**Action Buttons (user_action):**
- Buttons below messages (next to Retry, Delete)
- Configurable per-mode

#### Database Changes

```sql
-- New table
CREATE TABLE assistant_modes (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(500),
    active_tools JSON,
    advertised_tools JSON,
    filter_chain_id UUID REFERENCES filter_chains(id),
    sort_order INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    is_global BOOLEAN DEFAULT TRUE,
    created_by UUID,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Modified tables
ALTER TABLE custom_assistants ADD COLUMN mode_id UUID;
ALTER TABLE chats ADD COLUMN mode_id UUID;
ALTER TABLE chats ADD COLUMN active_tools JSON;
```

#### Removed from websocket.py

| Removed | Replacement |
|---------|-------------|
| Hardcoded Global KB search | `global_kb` primitive (always active) |
| Hardcoded User Chats KB search | `user_chats_kb` primitive (mode-controlled) |
| Hardcoded Assistant KB search | `kb_rag` primitive (mode-controlled) |
| Hardcoded Local Docs search | `local_rag` primitive (mode-controlled) |

---

## Recent Changes (NC-0.7.17)

### Agentic Task Queue
FIFO task queue system for multi-step agentic workflows.

**Key Design Principles**:
- NO automatic LLM verification calls (wasteful and unreliable)
- LLM manages its own task flow via `complete_task` and `fail_task` tools
- Tasks auto-advance when `auto_continue=true` (default)
- Priority support (higher = more urgent)
- Pause/resume control for user override

**Task Structure**:
```json
{
  "id": "uuid",
  "description": "Short task description",
  "instructions": "Detailed instructions (up to 512 tokens)",
  "status": "queued|in_progress|completed|failed|paused",
  "auto_continue": true,
  "priority": 0,
  "result_summary": "Brief summary of what was done",
  "source": "user|llm"
}
```

**LLM Tools**:
| Tool | Description |
|------|-------------|
| `add_task` | Add task with description, instructions, priority, auto_continue |
| `add_tasks_batch` | Add multiple tasks at once (planning) |
| `complete_task` | Mark current task done (with optional summary) |
| `fail_task` | Mark current task failed (with reason) |
| `skip_task` | Skip current task, move to next |
| `get_task_queue` | Get queue status |
| `clear_task_queue` | Clear all pending tasks |
| `pause_task_queue` | Pause auto-execution |
| `resume_task_queue` | Resume auto-execution |

**System Prompt Integration**:
When tasks are pending, the system prompt includes:
- Current task with instructions
- Queued tasks (first 5)
- Recently completed tasks (last 3)
- Instructions to use `complete_task` or `fail_task`

**WebSocket Events**:
- `task_queue_status` - Sent after each LLM response
  - `queue_length`, `current_task`, `paused`, `completed_count`

**Database Changes**:
- `chats.chat_metadata` - JSON column for queue storage (NC-0.7.17 migration)

---

## Recent Changes (NC-0.7.16)

### Hybrid RAG Search
RAG now uses **hybrid search** combining semantic similarity with exact identifier matching.

**Problem Solved**: Queries like "Rule 1S-2.053" were failing because:
- Semantic search embeds the full phrase as one vector
- Documents containing "1S-2.053" without "Rule" nearby scored below threshold
- Specific identifiers have weak semantic meaning

**Solution**: Detect identifiers in queries and run parallel search:
1. **Semantic search** (existing FAISS cosine similarity)
2. **Identifier search** (exact substring matching)
3. **Merge results** with identifier matches boosted

**Identifier Patterns Detected**:
- Legal statutes: `1S-2.053`, `768.28`, `119.07(1)`
- Rule references: `FAC 61G15-30`, `Rule 12-345`
- Section citations: `§ 119.07`, `Section 768.28`
- Case numbers: `2024-CF-001234`, `23-cv-12345`
- Product codes: `ABC-123-XYZ`, `ICD-10-CM`

**Unicode Normalization**:
Documents often contain typographic characters that don't match user input:
- Em dash `—` (U+2014) vs hyphen `-` (U+002D)
- En dash `–` (U+2013) vs hyphen
- Curly quotes `""` vs straight quotes `""`
- Range words: "1 to 9" → "1-9", "5 through 10" → "5-10"

The `_normalize_text_for_matching()` helper converts all variants to ASCII equivalents before comparison:
- "1—9" in document matches "1-9" in query
- "1-9" in document matches "1 to 9" in query
- Range word conversion only applies when at least one side contains a digit (avoids "go to the" → "go-the")

**Key Methods Added to RAGService**:
- `_normalize_text_for_matching(text)` - Convert Unicode dashes/quotes to ASCII
- `_extract_identifiers(query)` - Regex extraction of alphanumeric codes
- `_identifier_search(db, identifiers, knowledge_store_ids, top_k)` - Substring search in chunks
- `_merge_search_results(semantic, identifier, threshold, boost, top_k)` - Combine with boost

**Behavior**:
- Identifier matches get +0.15 score boost
- Chunks found by both methods use higher score
- Identifier-only matches are included even if semantic score is below threshold
- Logs show `[RAG_HYBRID]` prefix for hybrid search activity

---

## Recent Changes (NC-0.7.15)

### Admin Panel Refactoring (Partial)
Started breaking up the 4500+ line Admin.tsx:
- Created `/frontend/src/components/admin/` folder
- Extracted type definitions to `types.ts`
- Created standalone tab components: `SystemTab.tsx`, `OAuthTab.tsx`, `FeaturesTab.tsx`, `CategoriesTab.tsx`
- Types are now imported from `@/components/admin`

**Note**: Full refactor ongoing. Additional tabs to be extracted: LLM, Tiers, Users, Chats, Tools, Filters, FilterChains, GlobalKB, Dev.

### Source-Specific RAG Prompts
Admin can now configure separate prompt templates for each RAG source:
- **Global Knowledge Base** - authoritative org-wide knowledge
- **Custom GPT Knowledge Base** - GPT-specific context
- **User Documents** - user's uploaded files
- **Chat History Knowledge** - user's conversation history

Each prompt supports `{context}` placeholder (and `{sources}` for Global KB).
Empty prompts fall back to legacy `rag_context_prompt`.

### Keyword Check Filter Node
New `keyword_check` node type for filter chains:
- Place at start of chain for quick exit if keywords don't match
- Supports: contains, whole word, exact, starts_with, ends_with, regex
- Match modes: "any" keyword or "all" keywords
- On no match: skip_chain (fastest), go_to_llm, or continue
- Case sensitive option
- Sets `$keyword_matched` variable (true/false)

### search_documents Tool Enhancement
The `search_documents` tool now searches all RAG sources:
1. Global Knowledge Stores
2. Custom GPT Knowledge Stores (when in GPT conversation)
3. User's Documents

Results include `source_type` field showing origin.

### Bug Fixes
- **Admin panel scroll**: Changed root container to `h-full overflow-y-auto` so content scrolls within Layout's main area
- **SQLAlchemy cartesian product warning**: Fixed admin user/chat list queries that used `select_from(query.subquery())` pattern causing cross-join warnings

---

## Recent Changes (NC-0.7.14)

### Migration System Fix
- **Bug**: Migrations were skipped if schema version was already updated (e.g., version bumped for code change, then migration added later)
- **Fix**: Migration system now always checks all migrations regardless of version
- Each migration individually checks if it needs to run:
  - ADD COLUMN: checks PRAGMA table_info
  - CREATE TABLE: checks sqlite_master
  - CREATE INDEX: checks sqlite_master
  - INSERT system_settings: checks if key exists
- Safely skips already-applied migrations
- Logs at debug level for skips, info level for actual changes

---

## Recent Changes (NC-0.7.13)

### RAG Context-Aware Query Enhancement
When users ask follow-up questions like "Where did it come from?" or "Tell me more about that", RAG searches now use conversation context to find relevant results.

**How it works:**
1. Detects short/vague queries (≤5 words or containing pronouns like "it", "that", "where")
2. Extracts keywords from recent messages (last 3)
3. Focuses on proper nouns (capitalized words) and longer meaningful words
4. Combines original query with context keywords

**Example:**
- User: "Tell me about Covid 19"
- AI: [discusses COVID-19]
- User: "Where did it come from?"
- Enhanced query: "Where did it come from? COVID-19 coronavirus Wuhan"
- RAG now finds relevant results instead of 0

**Updated functions:**
- `search_global_stores()` - accepts optional `chat_id`
- `get_knowledge_store_context()` - accepts optional `chat_id`
- `get_context_for_query()` - accepts optional `chat_id`
- New helper: `_enhance_query_with_context()` - extracts context keywords

---

## Recent Changes (NC-0.7.12)

### Bug Fix: Interrupted Stream Tree Structure
- **Bug**: When user interrupts a streaming response and sends a new message, the new message was becoming a new root instead of continuing the conversation tree
- **Root Cause**: The `stream_stopped` payload in the backup handler was missing `parent_id`, so the interrupted assistant message wasn't properly linked to the conversation tree
- **Fix**: Added `parent_id: assistant_parent_id` to the `stream_stopped` payload at websocket.py line ~1143
- Now interrupted messages maintain proper parent-child relationships in the conversation tree

---

## Recent Changes (NC-0.7.11)

### Admin-Managed GPT Categories
- New database model `AssistantCategory` for storing categories
- Admin panel "GPT Categories" tab to add, edit, disable, and delete categories
- Categories have: value (slug), label, icon (emoji), description, sort_order, is_active
- Default categories seeded on first request if none exist
- Deleting a category moves GPTs to "general" category
- Cannot delete the "general" category
- CustomGPTs page now fetches categories from API instead of hardcoded list

### New API Endpoints
- `GET /assistants/categories` - List all categories (public)
- `POST /assistants/categories` - Create category (admin)
- `PATCH /assistants/categories/{id}` - Update category (admin)
- `DELETE /assistants/categories/{id}` - Delete category (admin)

---

## Recent Changes (NC-0.7.10)

### Custom GPT Enhancements
- **Avatar Upload**: Users can upload images for Custom GPTs, resized to 64x64 pixels
  - New endpoint: `POST /assistants/{id}/avatar` (multipart form data)
  - Serves avatars via `GET /assistants/avatars/{filename}`
  - Images stored in `uploads/avatars/` directory
- **Categories**: GPTs now have categories (general, writing, coding, research, education, business, creative, productivity, lifestyle, other)
- **Explore Page Improvements**:
  - Search by name, tagline, and description
  - Category filter dropdown
  - Sort by: Popularity (default), Recent, A-Z
  - Pagination (12 items per page)
  - Category badges on cards
- **My GPTs**: Shows avatar images and category badges

### Backend Changes
- Added `category` column to `custom_assistants` table
- New avatar upload/delete endpoints with PIL image processing
- Avatar images auto-resized to 64x64 JPEG

---

## Recent Changes (NC-0.7.09)

### Global Knowledge Store Visibility Fix
- **Bug**: Knowledge stores marked as "Global" in Admin still showed as "Private" in Documents panel, and other users couldn't see them
- **Root Cause**: `is_global` (auto-search) and `is_public` (visibility) were separate flags - marking global didn't make it visible
- **Fix**: 
  1. When a store is set as Global, it's now also automatically set as Public and Discoverable
  2. Added "Global" badge (purple) in Documents page to show which stores are auto-searched
  3. Added `is_global` field to frontend `KnowledgeStore` type

---

## Recent Changes (NC-0.7.08)

### Concurrent Streaming Fix (CRITICAL)
- **Bug**: Sending a new message while LLM was still responding caused content from both responses to mix together into gibberish
- **Root Cause**: `StreamingHandler.set_streaming_task()` simply overwrote the previous task without cancelling it, so both tasks would run concurrently sending chunks to the same connection
- **Fix**:
  1. `websocket.py`: `set_streaming_task()` now cancels any existing task before setting new one
  2. `websocket.py` routes: Added explicit `request_stop()` call before starting new `chat_message`
  3. Added `is_streaming()` method to check if currently streaming

### Streaming Markdown Rendering Fix
- **Bug**: Streaming messages showed raw markdown (`**text**`, `###`) instead of rendered formatting
- **Root Cause**: Imperative DOM streaming component was appending raw text without markdown rendering
- **Fix**: Reverted to React state-based streaming with `<MessageMarkdown>` component for proper real-time rendering

### Logger Fix in LLM Service
- **Bug**: `name 'logger' is not defined` error in v1 completions
- **Root Cause**: `llm.py` used `logger` throughout but only defined `llm_logger`
- **Fix**: Added `import logging` and `logger = logging.getLogger(__name__)` to `llm.py`

### UI Close Button Margins
- Added proper margins (20px+) to close buttons near scrollable areas
- Fixed in: `ArtifactsPanel.tsx`, `SummaryPanel.tsx`, `GeneratedImageCard.tsx`

### Global Knowledge Base Debugging
- Added improved logging to diagnose why global KBs may not work across users
- Logs now show: setting existence/value, number of global stores found
- Look for `[GLOBAL_RAG]` logs to diagnose issues

---

## Recent Changes (NC-0.6.50)

### RAG Embedding Model Loading Fix
- Fixed "meta tensor" errors from `accelerate` library
- Root cause: `low_cpu_mem_usage=True` creates placeholder tensors that can't be copied
- Solution: Set `low_cpu_mem_usage=False` in all model loading:
  - `rag.py`: Multiple loading strategies with fallbacks
  - `procedural_memory.py`: Reuses RAGService model when available
  - `stt.py`: Whisper model loading
- Added environment variable workarounds (`ACCELERATE_USE_CPU=1`, etc.)

### RAG Admin Endpoints
- `GET /api/admin/rag/status` - Check model loading status
- `POST /api/admin/rag/reset` - Reset failed model state and retry loading
- Useful for debugging model loading issues without container restart

### Artifact Streaming Detection
- Added patterns to detect completed artifacts during streaming:
  - XML artifacts: `<artifact title="...">...</artifact>`
  - Equals format: `<artifact=filename>...</artifact>`
  - Filename tags: `<Menu.cpp>...</Menu.cpp>`
  - Code fences: `filename.ts\n```typescript...```
- Artifacts are **tracked** but don't interrupt the stream (avoids race conditions)

### Tool Result Notifications
- When LLM uses tools (search_replace, find, request_file), the result includes:
  ```
  [FILES SAVED] The following files were saved during your response: file1.ts, file2.tsx
  You can reference these files in your continued response.
  ```
- Gives LLM context about what files exist when editing them

### Code Summary API Path Fix
- Fixed frontend calling `/code-summary` when backend uses `/summary`
- Corrected in `codeSummarySlice.ts`

---

## Recent Changes (NC-0.6.49)

### Streaming Timeout Fix
- Changed `AsyncOpenAI` timeout from simple int to `httpx.Timeout`
- `read` timeout is per-chunk, so streams don't timeout while tokens flow
- Affects: `llm.py`, `vision_router.py`

### Chat Model Persistence on Reload
- Now fetches individual chat via `GET /chats/{id}` on page load
- Ensures model/assistant_id are correct even if not in paginated list
- Updates chat in list with fresh data if present

### New Chat Behavior
- Now creates chat immediately with last used model
- Falls back to default model if no current chat
- Supports Custom GPTs (`gpt:` prefix) as last used model

### Model Switching Fix
- Can now switch from Custom GPT to regular model (clears `assistant_id`)
- Backend clears `assistant_id`, `assistant_name`, and `AssistantConversation` link
- UI correctly highlights only the active model/GPT (not both)

### All Models Prompt
- New admin setting: "All Models Prompt"
- Appended to ALL system prompts including Custom GPT system prompts
- Allows universal instructions (e.g., artifact format, feature awareness) to apply everywhere
- Located in Admin → System Settings → Prompts

### Stop Button Fix (Major)
- **Root cause**: Streaming blocked the WebSocket receive loop, so `stop_generation` couldn't be processed
- **Fix**: Run `handle_chat_message` and `handle_regenerate_message` in background tasks via `asyncio.create_task()`
- Main loop now stays free to receive `stop_generation` immediately
- `request_stop()` now cancels task FIRST, then closes stream with timeouts
- Added error handling for cancelled tasks

### Chat Deletion Removes Knowledge Index
- Deleting a chat now removes its entry from the user's chat knowledge store
- Uses `RAGService.delete_document()` to remove document and rebuild FAISS index
- Deleting all chats removes the entire chat knowledge store
- Prevents stale/orphaned data in search indexes

### Artifact Extraction Improvements
- Pattern 4b now handles text before `<artifact=file>` tag (e.g., `File main.cpp // comment <artifact=src/main.cpp>`)
- End-of-content extraction for complete code fences without closing tags
- Prefix text preserved in output placeholder

### Model Selection Fixes
- `newChatModel` initializes empty, syncs with `defaultModel` when loaded
- New Chat navigates to `/chat` (empty state) instead of creating chat immediately
- Chat switching clears all state via `setCurrentChat(null)`

---

## Recent Changes (NC-0.6.48)

### Multi-Model / Hybrid Routing
- New `llm_providers` table for configuring multiple LLM backends
- Set one model as **Default** (text/reasoning) and another as **Vision Default** (image understanding)
- **Hybrid routing**: When images are sent and primary model isn't multimodal:
  1. Vision model describes images
  2. Descriptions are injected into context
  3. Primary model responds with full context
- Admin UI: LLM tab now shows provider list with add/edit/delete/test
- Legacy LLM settings preserved as fallback in collapsible section

### Vision Router Service (`vision_router.py`)
- Detects images in current turn attachments
- Routes to vision model for descriptions
- Caches descriptions in message metadata
- Falls back to stripping images if no MM capability (Option A)

---

## Recent Changes (NC-0.6.47)

### Agent Memory System
- Automatically archives older conversation history to `{AgentNNNN}.md` files when context exceeds 50% of model limit
- Agent files contain 50-line summary headers with searchable keywords
- Files are automatically searched when LLM needs historical context
- Hidden from Artifacts panel and excluded from zip downloads
- Deleted automatically when chat is deleted (cascade)

### Large Attachment Processing
- Files > 100KB or total attachments > 200KB are stored as searchable artifacts
- LLM receives compact manifest with:
  - File list with sizes and languages
  - Code signatures (functions, classes with line numbers)
  - Instructions to use `<request_file path="..."/>` for full content
- Prevents context window overflow while maintaining full file access

### Export Data Feature
- Settings → Preferences → Export All Data button
- Downloads zip with all chats, messages, and knowledge store metadata

### Bug Fixes
- Fixed `MultipleResultsFound` errors in file queries
- Fixed missing `update` import in LLM service
- Increased default LLM timeout to 300 seconds

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
│   ├── fileProcessor.ts     # File upload processing (NEW NC-0.6.38)
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

### Global Knowledge Stores (NC-0.6.40)

Admins can mark knowledge stores as "global" - these are automatically searched on every chat message.

**Authoritative Injection**: When matches are found in global stores, they are injected into the system prompt as **definitive and trusted** information:

```
## AUTHORITATIVE KNOWLEDGE BASE

<trusted_knowledge source="{store names}">
IMPORTANT: The following information comes from the organization's verified 
global knowledge base. This content is DEFINITIVE and TRUSTED - treat it as 
the authoritative source of truth for the topics it covers. When this 
knowledge conflicts with your general training, defer to this information.

{matched content with source and confidence}
</trusted_knowledge>

When answering questions related to the above topics, you MUST use this 
authoritative information as your primary source.
```

**Configuration** (Admin Panel → Knowledge Stores → Set Global):
- `is_global`: Enable/disable global search
- `global_min_score`: Minimum relevance threshold (default 0.7)
- `global_max_results`: Max results per store (default 3)

### Knowledge Store Search Architecture (NC-0.6.40)

Knowledge stores are searched based on context:

| Search Type | When Triggered | Access Check |
|-------------|----------------|--------------|
| Global KB | Always (every message) | None - public |
| Custom GPT KB | When using that Custom GPT | Bypassed via assistant |
| User Documents | When `enable_rag=true` and no Custom GPT | User ownership |

**Important Rules:**
- Global KBs are searched on EVERY message, regardless of Custom GPT or RAG settings
- Custom GPT KBs are ONLY searched when using that specific Custom GPT
- User's unitemized documents are searched ONLY when `enable_rag=true` AND no Custom GPT is active
- Both REST API (`chats.py`) and WebSocket (`websocket.py`) follow the same search logic

### Chat Title Generation (NC-0.6.40)

**Single Path**: Title generation happens ONLY in `websocket.py` via `generate_chat_title()`.

**Location**: `websocket.py` → `generate_chat_title(first_message, db)`

**Flow**:
1. New chat created with title "New Chat"
2. After first message streamed via WebSocket, title is generated via fresh LLM instance
3. Frontend notified via `chat_updated` WebSocket event

**Key Design Decisions:**
- Uses a **fresh LLMService instance** (not the chat's LLM) to prevent Custom GPT configuration pollution
- REST API (`chats.py`) does NOT generate titles - relies on WebSocket path
- Single code path ensures consistent, high-quality titles

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

## File Upload to Artifacts (NC-0.6.38)

Users can upload any supported file type directly to the chat. Files are:
1. Added to the artifacts panel for viewing
2. Sent to the LLM as context
3. Available for partial viewing via tools

### Supported File Types

**Code Files**: `.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.java`, `.go`, `.rs`, `.rb`, `.php`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.swift`, `.kt`, `.scala`, `.lua`, `.r`, `.sh`, `.bash`, `.asm`, `.s`

**Config/Data Files**: `.yaml`, `.yml`, `.toml`, `.xml`, `.json`, `.csv`, `.sql`, `.graphql`, `.proto`

**Web Files**: `.html`, `.css`, `.scss`, `.sass`

**Documents**: `.md`, `.txt`, `.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.rtf`

**Images**: All common image formats (sent as base64)

**Archives**: `.zip` (special processing with signature extraction)

### Binary Document Handling

Binary documents (PDF, DOCX, XLSX, RTF) are processed via:
1. Frontend sends file to `/documents/extract-text` endpoint
2. Backend uses `DocumentProcessor.extract_text()` (Tika/PyMuPDF for PDF, python-docx, openpyxl)
3. Extracted text becomes artifact content and LLM context

### Partial File Viewing Tools

For large files, the LLM can use these tools instead of reading entire content:

| Tool | Description |
|------|-------------|
| `list_uploaded_files` | List all files in the session with line count and preview |
| `view_file_lines` | View specific line range (e.g., lines 100-150) |
| `search_in_file` | Search for pattern and show matches with context |
| `view_signature` | View code around a function/class definition |

### Signature Extraction

Code files automatically have signatures extracted:
- **Python**: `def`, `class`, `CONSTANTS`
- **JavaScript/TypeScript**: `function`, `class`, `const`, `interface`, `type`
- **Java/C#**: `class`, `interface`, public methods
- **Go**: `func`, `type`
- **Rust**: `fn`, `struct`, `enum`, `trait`, `impl`
- **C/C++**: functions, `struct`, `class`, `namespace`, `#define`
- **Assembly**: labels, `.global` directives

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

### Visual Editor (NC-0.6.41)

Filter chains now use a **visual node-based editor** similar to n8n:

- **Drag-and-drop nodes** from the palette (left side)
- **Click nodes** to expand and configure
- **Connect nodes** by dragging from output handles
- **Jump connections** shown as animated edges
- **JSON mode** toggle for advanced editing

The editor uses React Flow and supports all step types with intuitive configuration panels.

### Step Types

**Core:**
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
- `keyword_check`: Quick keyword match (NC-0.7.15)

**RAG Evaluators (NC-0.8.0.0):**
- `local_rag`: Search user's uploaded docs for current chat
- `kb_rag`: Search current assistant's knowledge base
- `global_kb`: Search global knowledge bases (always active)
- `user_chats_kb`: Search user's chat history KB

**Dynamic Tools (NC-0.8.0.0):**
- `export_tool`: Register LLM-triggered tool (`$WebSearch="..."`)
- `user_hint`: Text selection popup menu item
- `user_action`: Button below chat messages

### Variables
- `$Query` - Original user input
- `$PreviousResult` - Last step output
- `$Var[name]` - Named outputs from earlier steps
- `$Selected` - User's selected text (user_hint only)
- `$MessageContent` - Full message content (user_action only)
- `$1`, `$2`, etc. - Regex capture groups (export_tool only)

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
| NC-0.8.0.12 | 2026-02-11 | **New Tools** - `search_replace` (exact find/replace in files), `web_search` (DuckDuckGo search), `web_extract` (structured page extraction), `grep_files` (cross-file search), `sed_files` (batch regex replace). Updated tool categories and hallucinated tool delegates. |
| NC-0.8.0.8 | 2026-01-10 | **Force Trigger Keywords for Global KB** - Bypass score threshold when keywords match. New DB columns: `force_trigger_enabled`, `force_trigger_keywords`, `force_trigger_max_chunks`. Enhanced Admin UI with relevance threshold slider. **Tool Debugging** - New DEBUG_Tool_Advertisements and DEBUG_Tool_Calls settings in Admin Site Dev. **Task Queue Auto-Execute** - Automatically feeds pending tasks to LLM when it stops responding without processing them. |
| NC-0.8.0.7 | 2026-01-09 | Active Tools Filtering fix, Streaming Tool Calls fix, Pre-emptive RAG Toggle, API model names use cleaned assistant names, Performance indexes, Admin image gen settings, image persistence, image preview in artifacts, Download All includes images, image context hidden, generate_image tool, temporary MCP server installation (4h auto-cleanup) |
| NC-0.8.0.6 | 2026-01-08 | Smart RAG Compression - subject re-ranking (keyword overlap boost), extractive summarization for large chunks, token budget support |
| NC-0.8.0.5 | 2026-01-08 | Per-chat token limits (max_input_tokens, max_output_tokens), Agent Memory tools (agent_search, agent_read), error sanitization, max output token validation |
| NC-0.8.0.4 | 2026-01-08 | Mode/Category unification - Categories pull from AssistantModes, emojis derived from name, slug mapping for backward compatibility |
| NC-0.8.0.3 | 2026-01-08 | Sidebar real-time updates (sidebarReloadTrigger), timestamp preservation (removed onupdate trigger), migration system fix, chat click fix (_assignedPeriod preservation), import preserves original timestamps |
| NC-0.8.0.1.2 | 2026-01-07 | Chat source field for import tracking, sidebar filtering by source instead of title prefix |
| NC-0.8.0.1.1 | 2026-01-07 | Generated images in Artifacts panel (imageNNN.ext), LLM image context injection |
| NC-0.8.0.1 | 2026-01-07 | User data export fixes (model_id, display_name attributes) |
| NC-0.8.0.0 | 2026-01-05 | **Dynamic Tools & Assistant Modes** - Composable RAG via filter chains, user_hint/user_action nodes, export_tool for LLM triggers, Active Tools Bar, Assistant Modes presets |
| NC-0.7.17 | 2026-01-05 | Agentic Task Queue - FIFO queue for multi-step workflows, LLM-managed task flow |
| NC-0.7.16 | 2026-01-05 | Hybrid RAG search - combines semantic + identifier matching for legal/code references |
| NC-0.7.15 | 2026-01-05 | Source-specific RAG prompts, keyword_check filter node, Admin panel refactor (partial), scroll fix, SQLAlchemy warning fix |
| NC-0.7.14 | 2026-01-04 | Migration system fix - checks all migrations regardless of version |
| NC-0.7.13 | 2026-01-04 | RAG context-aware query enhancement for follow-up questions |
| NC-0.7.12 | 2026-01-04 | Interrupted stream tree structure fix |
| NC-0.7.11 | 2026-01-04 | Admin-managed GPT categories |
| NC-0.7.10 | 2026-01-03 | Custom GPT avatar upload, categories, explore page improvements |
| NC-0.7.09 | 2026-01-02 | Global Knowledge Store visibility fix |
| NC-0.7.08 | 2026-01-01 | Concurrent streaming fix, markdown rendering fix, logger fix |
| NC-0.6.50 | 2025-12-19 | RAG model loading fix (meta tensor errors), RAG admin endpoints, artifact streaming detection, tool result notifications |
| NC-0.6.49 | 2025-12-18 | Stop button fix, chat deletion removes knowledge index, streaming timeout, All Models Prompt |
| NC-0.6.46 | 2025-12-17 | Filter chain skip_if_rag_hit option, RAG search before filter chains (Global → Chat History → Assistant KB) |
| NC-0.6.45 | 2025-12-17 | Chat Knowledge Base feature - index all chats into personal knowledge store, green dot indicators |
| NC-0.6.44 | 2025-12-17 | Chat search searches message content, infinite scroll pagination, import chats sorted newest-to-oldest |
| NC-0.6.43 | 2025-12-17 | Grok chat import parser fix for nested response structure |
| NC-0.6.42 | 2025-12-17 | Anonymous sharing option for shared chats (hide owner name) |
| NC-0.6.41 | 2025-12-17 | Visual node-based filter chain editor (n8n-style), raw settings API endpoints |
| NC-0.6.40 | 2025-12-16 | Global KB authoritative injection, unified KB search architecture (Global always, Custom GPT only when active), single-path title generation |
| NC-0.6.39 | 2025-12-16 | Aggressive stop generation (closes LLM connection), file persistence to DB, DB fallback for LLM tools, remove KB upload rate limit |
| NC-0.6.38 | 2025-12-15 | File upload to artifacts, partial file viewing tools, remove 100K char filter limit |
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

Current: **NC-0.8.0.8**

Migrations run automatically on startup in `backend/app/main.py`.

Key tables added/modified:
- NC-0.8.0.8: New columns on `knowledge_stores`: `force_trigger_enabled`, `force_trigger_keywords`, `force_trigger_max_chunks`.
- NC-0.8.0.7: No schema changes. Admin settings stored in `system_settings` table for image generation. Image metadata stored in `messages.message_metadata`.
- NC-0.8.0.6: No schema changes (Smart RAG Compression is code-only)
- NC-0.8.0.5: `chats.max_input_tokens` and `chats.max_output_tokens` INTEGER columns for per-chat token limits
- NC-0.8.0.4: No schema changes (mode/category unification is code-only, emoji derived from name)
- NC-0.8.0.3: No schema changes (behavior changes only: removed `onupdate` from Chat.updated_at, explicit timestamp updates on user message, import uses first/last message timestamps)
- NC-0.8.0.1.2: `chats.source` VARCHAR(50) column for import tracking (native, chatgpt, grok, claude)
- NC-0.8.0.1.1: No schema changes (frontend-only: images in artifacts panel)
- NC-0.8.0.1: No schema changes (export endpoint bug fixes)
- NC-0.8.0.0: `assistant_modes` table, `mode_id` and `active_tools` columns on `chats`, `mode_id` on `custom_assistants`
- NC-0.7.17: `chats.chat_metadata` JSON column (task queue storage)
- NC-0.7.16: No schema changes (hybrid RAG search is code-only)
- NC-0.7.15: No schema changes (Admin refactor, RAG prompts stored in system_settings)
- NC-0.7.14: No schema changes (migration system logic fix)
- NC-0.7.13: No schema changes (RAG query enhancement)
- NC-0.7.12: No schema changes (websocket parent_id fix)
- NC-0.7.11: `assistant_categories` table, `category` column on `custom_assistants`
- NC-0.7.10: `category` column on `custom_assistants`
- NC-0.6.50: No schema changes (RAG model loading fix, frontend streaming improvements)
- NC-0.6.49: No schema changes (timeout/frontend fixes only)
- `filter_chains` - Added skip_if_rag_hit flag (NC-0.6.46)
- `users` - Added chat knowledge fields (NC-0.6.45)
- `chats` - Added is_knowledge_indexed flag (NC-0.6.45)
- `uploaded_files` - Persisted file uploads (NC-0.6.38/39)
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
