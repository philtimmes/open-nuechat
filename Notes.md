# Open-NueChat - Development Notes

## Project Overview

Full-stack LLM chat application with:
- FastAPI backend (Python 3.13)
- React 19 + TypeScript frontend
- SQLite database (aiosqlite)
- Multi-GPU support (ROCm, CUDA, CPU)
- FAISS GPU for vector search
- OpenAI-compatible LLM API integration

**Current Version:** NC-0.6.77

---

## Recent Changes (NC-0.6.77)

### Bugfix: Breadcrumb Navigation Closes Panel
- **Root cause**: `setSelectedArtifact(null)` automatically set `showArtifacts: false`
- **Fix**: Modified `artifactSlice.ts` to only auto-OPEN panel on selection, never auto-close
- Breadcrumb navigation now works correctly within the artifacts panel

### Feature: File Tree View in Artifacts Panel
- New tree/flat toggle buttons in artifacts panel header
- Tree view shows folders expandable/collapsible 
- "Expand all" / "Collapse all" controls
- Flat view enhanced to show full filepath

### Bugfix: Agent File Naming Convention
- Changed frontend chunking from `agent001.md` to `{Agent0001}.md`
- Now matches backend `{AgentNNNN}.md` convention from `agent_memory.py`
- Consistent naming prevents duplicate filtering issues

**Files Changed:**
- `frontend/src/stores/chat/artifactSlice.ts`: Fix setSelectedArtifact behavior
- `frontend/src/lib/artifacts.ts`: Added `buildFileTree()` and `FileTreeNode` type
- `frontend/src/components/ArtifactsPanel.tsx`: Added tree view with toggle
- `frontend/src/contexts/WebSocketContext.tsx`: Fixed agent file naming

---

## Recent Changes (NC-0.6.76)

### Feature: Billing APIs Admin Tab
- New "Billing APIs" tab in Admin panel for payment provider configuration
- Configure Stripe, PayPal, and Google Pay settings from the UI
- Test connection buttons for Stripe and PayPal

**Backend:**
- Added `BillingApiSettingsSchema` with all payment provider fields
- Added `GET/PUT /admin/billing-api-settings` endpoints
- Added `POST /admin/billing-api-settings/test-stripe` endpoint
- Added `POST /admin/billing-api-settings/test-paypal` endpoint
- Added defaults for all billing settings in `SETTING_DEFAULTS`

**Frontend (Admin.tsx):**
- Added `BillingApiSettings` interface
- Added `billing_apis` to `TabId` type
- Added state and fetch for billing API settings
- Added save and test connection functions
- Full UI for Stripe (API key, publishable key, webhook secret, price IDs)
- Full UI for PayPal (client ID/secret, webhook ID, mode, plan IDs)
- Full UI for Google Pay (merchant ID, merchant name)

---

## Recent Changes (NC-0.6.75)

### Feature: Context Window Overflow Protection
- Large tool results (>32k chars, ~1/4 of context) are automatically chunked into hidden agent files
- Prevents LLM context overflow while keeping data searchable

**How it works:**
1. When tool result exceeds `CONTEXT_CHUNK_THRESHOLD` (32000 chars)
2. Data is split into chunks of ~24000 chars each
3. Each chunk becomes a hidden artifact: `{Agent0001}.md`, `{Agent0002}.md`, etc.
4. LLM receives a summary with file list and preview
5. LLM can use `<find search="..."/>` or `<request_file path="{Agent0001}.md"/>` to access data

**Changes:**
- `types/index.ts`: Added `hidden?: boolean` to Artifact interface
- `WebSocketContext.tsx`: Added `chunkLargeToolResult()` function and constants
- `WebSocketContext.tsx`: Modified `sendToolResult()` to chunk large results
- `ChatPage.tsx`: Filter hidden artifacts from UI display (but still searchable)

---

## Recent Changes (NC-0.6.74)

### Fix: Auto-Close Incomplete Tool Tags
- If LLM stops mid-output without closing a tool tag, we now salvage the operation
- Applies to `<search_replace>` and `<replace_block>` tags
- Uses `INCOMPLETE_SEARCH_REPLACE_PATTERN` and `INCOMPLETE_REPLACE_BLOCK_PATTERN`
- Fallback only triggers when opening tag exists but closing tag is missing
- Example: `<search_replace path="f">===== SEARCH\nold\n===== Replace\nnew` → processed as complete

---

## Recent Changes (NC-0.6.73)

### Feature: KaTeX Math Rendering
- Added KaTeX support for rendering LaTeX math notation in messages
- Inline math: `$x^2$` renders as x²
- Block math: `$$\sum_{i=1}^n i$$` renders as centered equation

**Dependencies Added (package.json):**
- `katex`: ^0.16.11
- `remark-math`: ^6.0.0
- `rehype-katex`: ^7.0.1
- `@types/katex`: ^0.16.7 (dev)

**Files Updated:**
- `MessageBubble.tsx` - Main chat messages
- `ChatPage.tsx` - Streaming message markdown
- `SharedChat.tsx` - Shared chat view
- `VibeChat.tsx` - Vibe code assistant chat

Each ReactMarkdown component now uses:
```tsx
<ReactMarkdown
  remarkPlugins={[remarkGfm, remarkMath]}
  rehypePlugins={[rehypeKatex]}
  ...
/>
```

---

## Recent Changes (NC-0.6.72)

### Fix: Tool Call Closures Detection
- **Problem**: Tool call closing tags like `</search_replace>` weren't always detected when not alone on a line
- **Example**: `</body> </html> </search_replace>` - the closing tag would be missed

**Changes (WebSocketContext.tsx):**
- Added `STREAM_REPLACE_BLOCK_PATTERN` for streaming detection of `</replace_block>` tags
- Added streaming interrupt handler for `replace_block` tool (was only processed at stream end)
- Updated `hasToolTag` checks to include `<replace_block>`
- Both patterns (`search_replace` and `replace_block`) now properly handle closing tags that appear inline with other content

**Note:** The regex patterns were already correct - `([\s\S]*?)</search_replace>` matches everything up to the closing tag regardless of what else is on the line. The main addition was streaming detection for `replace_block`.

---

## Recent Changes (NC-0.6.71)

### UI: Sidebar Right Margin for Delete Icon
- Added right padding (`pr-[30px]`) to chat list container
- Increased right margin on chat items (`ml-2 mr-3` instead of `mx-2`)
- Delete icon now easily reachable without being cut off by scrollbar

---

## Recent Changes (NC-0.6.70)

### Fix: Filter Chain Executor Infinite Loop
- **Problem**: Chat got stuck when filter chains were enabled ("1 chains to execute..." and then hang)
- **Root cause**: Duplicate `_execute_steps()` call in `executor.py` - the chain would run twice and could cause infinite loops

**Fix:**
- Removed duplicate code block in `ChainExecutor.execute()` (lines 548-572 were duplicating 518-546)
- Now executes steps exactly once as intended

---

## Recent Changes (NC-0.6.69)

### Enhancement: Improved Artifact Tool Guidance
- Enhanced system prompt with CRITICAL section for artifact editing
- Clearer instructions about reading tool responses on failure
- Explicit warning against retrying identical failed operations
- Better explanation of how file state changes after edits

---

## Recent Changes (NC-0.6.68)

### Feature: Artifact Tools with State Tracking
- **Problem**: LLM repeatedly failed search_replace operations because it searched for original content after the file had been modified
- **Root cause**: No state tracking between tool calls; LLM didn't know actual file content

**Solution - Artifact State Manager:**
- New `artifact_tools.py` with comprehensive file editing tools
- Per-chat session state tracking for all created artifacts
- Smart error responses that include actual file content when search fails
- Duplicate operation detection to prevent repeated identical failures
- Similar match finding when exact search fails

**New Tools Added:**
- `create_artifact` - Create tracked files
- `read_artifact` - Read current file state (with line range support)
- `search_replace` - Edit with state awareness and smart error recovery
- `list_artifacts` - List all session artifacts
- `append_to_artifact` - Append content to files
- `insert_at_line` - Insert at specific line number
- `delete_lines` - Delete line ranges

**Key Features:**
- When search_replace fails, response includes:
  - `actual_content` - Current file content
  - `similar_matches` - Similar text that might be what was intended
  - `hint` - Guidance on what to do next
- Duplicate detection: If same search attempted twice, returns previous error with file content
- Version tracking: Each artifact tracks edit history
- System prompt guidance: LLM instructed to read actual_content before retrying

**Files Changed:**
- `backend/app/tools/artifact_tools.py` (NEW) - Complete artifact management system
- `backend/app/tools/registry.py` - Register artifact tools
- `backend/app/services/llm.py` - Add artifact guidance to system prompt

---

## Recent Changes (NC-0.6.67)

### Fix: Persist LLM responses when client disconnects
- **Problem**: Leaving a chat during generation would lose the LLM response and any artifacts
- **Root cause**: WebSocket disconnection cancelled the streaming task before content could be saved

**Solution - Detached Task Execution:**
- Streaming tasks now continue running even after client disconnects
- Tasks are registered in a detached task registry for tracking
- Connection marked as `is_disconnected` instead of cancelling tasks
- `send_to_connection()` silently returns False for disconnected clients

**Changes to websocket.py (service):**
- Added `is_disconnected` field to `Connection` dataclass
- Added `register_detached_task()` for tracking tasks that outlive connections
- `disconnect()` marks connection as disconnected but doesn't cancel tasks
- `send_to_connection()` checks `is_disconnected` before attempting send
- `StreamingHandler.send_chunk()` skips sending to disconnected clients

**Changes to websocket.py (routes):**
- `chat_message` and `regenerate_message` register as detached tasks
- `WebSocketDisconnect` handler no longer cancels tasks
- `CancelledError` handler now commits content before trying to send
- Exception handlers commit partial content before rollback
- Title notification wrapped in try/except for disconnected clients

**Changes to llm.py:**
- Cancel handling now commits the partial content immediately
- Added logging for cancelled message saves

**Result:**
- LLM continues generating even if user navigates away
- All content is saved to database when complete
- Artifacts are preserved (extracted from saved message content)
- When user returns, they see the full response

---

## Recent Changes (NC-0.6.66)

### Feature: Complete Payment System
Added full payment processing with Stripe, PayPal, and Google Pay support.

**Backend Changes:**
- **New Payment Models** (`billing.py`):
  - `Subscription`: User subscription tracking with provider references
  - `PaymentMethod`: Stored payment methods (cards, PayPal)
  - `Transaction`: Payment transaction history
  - `PaymentProvider`, `PaymentStatus`, `SubscriptionStatus` enums
  
- **New Payment Service** (`payments.py`):
  - `StripeProvider`: Full Stripe Checkout and Subscription API integration
  - `PayPalProvider`: PayPal Orders and Subscriptions API integration
  - `GooglePayProvider`: Google Pay via Stripe payment gateway
  - `PaymentService`: Unified service managing all providers
  - Webhook handlers for payment confirmations
  
- **New Billing Endpoints** (`billing.py` routes):
  - `GET /billing/subscription`: Get current subscription status
  - `POST /billing/subscribe/{tier}`: Create checkout session (with provider param)
  - `POST /billing/cancel-subscription`: Cancel subscription
  - `GET /billing/payment-methods`: List stored payment methods
  - `GET /billing/transactions`: Transaction history
  - `GET /billing/providers`: Get available providers and config
  - `POST /billing/webhooks/stripe`: Stripe webhook handler
  - `POST /billing/webhooks/paypal`: PayPal webhook handler

- **Config Updates** (`config.py`):
  - `STRIPE_PUBLISHABLE_KEY`, `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`
  - `PAYPAL_MODE` (sandbox/live), `PAYPAL_WEBHOOK_ID`
  - `GOOGLE_PAY_MERCHANT_ID`, `GOOGLE_PAY_MERCHANT_NAME`
  - `PAYMENT_CURRENCY`, `PAYMENT_SUCCESS_URL`, `PAYMENT_CANCEL_URL`
  - `PRO_TIER_PRICE`, `ENTERPRISE_TIER_PRICE`

**Frontend Changes** (`Billing.tsx`):
- Payment provider selection (Stripe, PayPal, Google Pay icons)
- Real subscription management (subscribe, cancel)
- Stored payment methods display
- Transaction history with status badges
- Provider availability check and warning
- Improved tier display with formatted token limits

**Dependencies Added:**
- `stripe>=7.0.0`: Stripe Python SDK
- `aiohttp>=3.9.0`: Async HTTP for PayPal API calls

---

## Recent Changes (NC-0.6.65)

### Fix: Artifact closing tag detection
- **Problem**: `</artifact>` tags not detected if not alone on the line
  - Tags with content before them (e.g., `code</artifact>`) were missed
  - Tags with content after them (e.g., `</artifact> more text`) were missed
- **Solution**: Updated Pattern 3 and Pattern 4 in `artifacts.ts`
  - Patterns now match closing tag anywhere in line
  - Content before closing tag is added to artifact buffer
  - Content after closing tag is preserved in output
- **Files changed**: `frontend/src/lib/artifacts.ts`

---

## Recent Changes (NC-0.6.64)

### Feature: Gzip file support & log error extraction
- **Gzip decompression**: Uploaded .gz files are automatically decompressed client-side
  - Uses browser's native DecompressionStream API
  - Safety limits: 10MB compressed max, 50MB decompressed max, 100x ratio limit (zip bomb protection)
  - Inner filename extracted (e.g., syslog.gz → syslog)
- **Log error extraction**: Files with "log" in name/extension get error analysis
  - Detects: error, failed, exception, traceback, critical, fatal, panic, warning, HTTP 4xx/5xx, errno
  - Extracts error lines with preceding and following non-error context lines
  - Shows up to 50 errors in summary
- **Error summary in LLM context**: Log errors automatically fed to LLM
  - Displayed in Summary Panel with expandable details
  - Included in system prompt for AI analysis
- **New warning type**: `log_error` added to SignatureWarning
- **Files changed**:
  - `fileProcessor.ts`: Gzip decompression, log error extraction, updated return types
  - `ChatInput.tsx`: Added onLogErrors callback prop
  - `ChatPage.tsx`: handleLogErrors to add warnings to summary panel
  - `SummaryPanel.tsx`: Expandable log error details display
  - `types/index.ts`: log_error type, errorSummary/errorCount fields
  - `websocket.py`: Include errorSummary in LLM context for log_error warnings

---

## Recent Changes (NC-0.6.63)

### Feature: Shared chat images & formatting
- **Images in shared chats**: Attachments now included in shared chat API response
  - Backend: `path_messages` and `all_messages` now include `attachments` field
  - Frontend: SharedChat renders image attachments with proper styling
- **Shared chat formatting matches original**: 
  - Updated SharedChat ReactMarkdown components to match MessageBubble
  - Added all heading handlers (h1-h6), strong, em, hr, li, etc.
- **User newlines preserved**:
  - User messages now render with `whitespace-pre-wrap` instead of markdown
  - Preserves line breaks exactly as typed
  - Applied to both MessageBubble and SharedChat
- **Files changed**:
  - `main.py`: Added attachments to shared chat message data
  - `SharedChat.tsx`: Added attachments interface, image rendering, formatting components, user message handling
  - `MessageBubble.tsx`: User messages use whitespace-pre-wrap

---

## Recent Changes (NC-0.6.62)

### Fix: RAG embedding model auto-retry
- **Problem**: Once embedding model failed to load, it never retried
- **Root cause**: `_model_load_failed` flag was permanent, preventing all retries
- **Solution**: 
  - Added time-based retry: after 60 seconds, model loading is retried automatically
  - Added `_model_load_last_attempt` timestamp tracking
  - Background task on startup attempts model load after 5 second delay
  - `get_model_status()` now shows `retry_in_seconds` when failed
- **Files changed**:
  - `rag.py`: Time-based retry mechanism for model loading
  - `main.py`: Background delayed model load task on startup

---

## Recent Changes (NC-0.6.61)

### Fix: Images display immediately & markdown formatting works
- **Image display fix**: 
  - Root cause: `setCurrentChat` was clearing messages when URL changed after `createChat`
  - Fix: Skip clearing if switching to the same chat (prevents race condition)
- **Markdown formatting fix**:
  - Root cause: Tailwind v4 doesn't include typography plugin by default
  - Fix: Removed prose classes, using explicit component styling instead
  - Added `li` component for list items
- **Files changed**:
  - `chatSlice.ts`: setCurrentChat now skips clear when same chat
  - `MessageBubble.tsx`: Removed prose classes, using direct styling

---

## Recent Changes (NC-0.6.60)

### Feature: Immediate image display & improved markdown formatting
- **Images display immediately**: Uploaded images now appear in chat instantly without refresh
  - User messages now include attachments data for display
  - Maps ProcessedAttachment (filename) to Attachment (name) format
- **Markdown formatting improvements**: 
  - Added explicit heading components (h1-h6) with proper sizing
  - Added strong/em/hr handlers for consistent text formatting
  - Headings scale: h1 (2xl) → h6 (sm)
- **Files changed**:
  - `WebSocketContext.tsx`: Convert attachments for display in user messages
  - `MessageBubble.tsx`: Added heading, strong, em, hr components to ReactMarkdown

---

## Recent Changes (NC-0.6.59)

### Fix: Image base64 data was being truncated
- **Problem**: Images sent to LLM were only sending first 50 characters of base64 data + "..."
- **Root cause**: Debug-style truncation in `_build_multimodal_content()` was used in actual payload
- **Solution**: Pass full base64 data to LLM API
- **Files changed**:
  - `llm.py`: Fixed `_build_multimodal_content()` to use complete base64 data

---

## Recent Changes (NC-0.6.58)

### Fix: Stream Cross-Chat Contamination
- **Problem**: Streams from 2 different chat IDs were getting placed in the last created chat
- **Root cause**: When switching chats during streaming, the global streaming state wasn't properly isolated
- **Solution**:
  - Added `currentStreamingChatIdRef` to track which chat owns the current stream
  - Buffer flush callback now checks if streaming chat matches current chat before appending
  - Clear streaming refs when chat changes via useEffect
  - Enhanced validation in stream_end and stream_stopped handlers
  - Clear refs in all stream termination paths (end, error, stopped)
- **Files changed**:
  - `WebSocketContext.tsx`: Added streaming chat ID tracking and validation

---

## Recent Changes (NC-0.6.57)

### Feature: Thinking Tokens Support
- **Admin Panel → LLM tab**: Added "Think Begin Token" and "Think End Token" fields
- **Usage**: Configure tokens like `<think>` / `</think>` to hide model reasoning
- **Rendering**: Content between thinking tokens is hidden behind a collapsible "🧠 Thinking..." panel
- **User Experience**: Click "Thinking..." to expand and see the model's reasoning
- **Streaming**: Shows animated "Thinking..." indicator during streaming
- **Admin Scrolling**: Fixed Admin panel scrolling by adding `overflow-y-auto` to container
- **Files changed**:
  - `admin.py`: Added `think_begin_token` and `think_end_token` to LLMSettingsSchema
  - `utils.py`: Added `/thinking-tokens` endpoint for frontend
  - `Admin.tsx`: Added Thinking Tokens UI section, fixed scrolling
  - `MessageBubble.tsx`: Added `ThinkingBlockPanel` component and extraction logic

---

## Recent Changes (NC-0.6.56)

### Fix: Pre-generate assistant message IDs BEFORE streaming
- **Problem**: Tool continuations created branches because message IDs were generated too late
- **Root cause**: Message ID was generated inside `llm.stream_message()` AFTER database operations started
- **Solution**: 
  - Generate `assistant_message_id` in `websocket.py` BEFORE calling `stream_message()`
  - Send `stream_start` to frontend IMMEDIATELY with that ID
  - Pass pre-generated ID to `llm.stream_message()` as `message_id` parameter
  - Frontend receives ID BEFORE any content, tracks it for tool continuations
  - Tool results use the tracked message ID as `parent_id`
- **Flow now**:
  1. Backend generates UUID for assistant message
  2. Backend sends `stream_start` with that ID to frontend
  3. Frontend stores ID in `currentStreamingMessageIdRef`
  4. Backend creates Message record with that ID
  5. LLM streams content
  6. If tool tag detected, frontend uses stored ID as `parent_id`
  7. Next message chains correctly
- **Files changed**:
  - `websocket.py`: Pre-generate ID, call `start_stream()` before `stream_message()`
  - `llm.py`: Accept `message_id` parameter, use it when creating Message

---

## Recent Changes (NC-0.6.55)

### Fix: Improved parent_id tracking for tool continuations
- **Problem**: Tool loops were still creating branches (21/21 version selector) - messages had no parent_id
- **Root cause**: Multiple issues:
  1. Tool continuation logic needed to validate frontend parent_id before using
  2. Needed fallback to latest message when frontend parent_id invalid
  3. Insufficient logging made debugging difficult
- **Solution**: 
  - For tool continuations: First validate frontend's parent_id, then fall back to latest message
  - Added comprehensive logging throughout parent_id flow
  - Explicitly convert user_message.id to string for safety
  - Log assistant_parent_id before LLM call and after message creation
- **Files changed**:
  - `websocket.py`: Improved tool continuation parent_id logic with validation
  - `llm.py`: Added logging for parent_id in stream_message

---

## Recent Changes (NC-0.6.54)

### Feature: kb_search tool for LLM knowledge base access
- **Usage**: `<kb_search query="SEARCH TERM">`
- **Description**: Allows LLM to search user's accessible knowledge bases during conversation
- **Searches**: All owned, shared, and public knowledge stores
- **Returns**: Top 5 results with document name, knowledge store, score, and content
- **Files added/changed**:
  - `knowledge_stores.py`: Added `/search` POST endpoint with `KBSearchRequest`/`KBSearchResponse`
  - `WebSocketContext.tsx`: Added `STREAM_KB_SEARCH_PATTERN` and handler

---

## Recent Changes (NC-0.6.53)

### Fix: Tool loops creating branches instead of linear conversation
- **Problem**: When tool calls looped (e.g., request_file, find_line), each continuation was creating a new branch instead of extending linearly, showing "21/21" in version selector
- **Root cause**: Tool continuations (`save_user_message=false`) used frontend-provided `parent_id`, which could be stale due to timing issues - all continuations ended up with the same parent
- **Solution**: 
  - Backend now automatically finds the LATEST message in the chat for tool continuations
  - Uses that message as parent regardless of what frontend sends
  - Guarantees linear conversation flow for all tool loops
- **Files changed**:
  - `websocket.py`: Query for latest message when `save_user_message=false`
  - `WebSocketContext.tsx`: Added streaming message ID tracking (secondary fix)

---

## Recent Changes (NC-0.6.52)

### Fix: Artifacts carrying over to New Chat
- **Problem**: When clicking "New Chat" from a chat with artifacts, the artifacts would carry over to the new chat
- **Root cause**: `createChat()` was always preserving `uploadedArtifacts`, `zipUploadResult`, and `zipContext` from the previous state
- **Solution**: 
  - Added `preserveArtifacts` parameter to `createChat()` (default: `false`)
  - "New Chat" button and keyboard shortcuts now clear artifacts (default behavior)
  - First message in empty state still preserves artifacts (passes `true`) for users who upload files before chatting
- **Files changed**:
  - `chatSlice.ts`: Added `preserveArtifacts` parameter with conditional preservation
  - `types.ts`: Updated `createChat` type signature
  - `ChatPage.tsx`: Pass `preserveArtifacts=true` when creating chat on first message

---

## Recent Changes (NC-0.6.51)

### Chat Knowledge Base Assistant Context Filtering
- **Fixed**: Chat Knowledge was pulling results from ALL chats regardless of which assistant was used
- **Problem**: When using Assistant B, knowledge from chats conducted with Assistant A was being returned
- **Solution**: 
  - Store `assistant_id` and `assistant_name` in chunk metadata when indexing chats
  - Filter chat knowledge search results by current assistant context
  - Chats without an assistant only match queries without an assistant
  - Chats with Assistant X only match queries with Assistant X
- **Files changed**:
  - `user_settings.py`: Added assistant_id to chunk metadata in both bulk and incremental indexing
  - `rag.py`: New `get_chat_knowledge_context()` method with assistant filtering
  - `websocket.py`: Updated chat KB search to use filtered method
- **Note**: Users should re-index their chat knowledge to populate assistant metadata for existing chats

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
| NC-0.6.77 | 2025-12-25 | File tree view in artifacts, breadcrumb nav fix, agent file naming fix |
| NC-0.6.76 | 2025-12-25 | Billing APIs admin tab for Stripe/PayPal/Google Pay configuration |
| NC-0.6.75 | 2025-12-24 | Context window overflow protection - chunk large tool results into hidden agent files |
| NC-0.6.74 | 2025-12-24 | Auto-close incomplete tool tags when LLM stops mid-output |
| NC-0.6.73 | 2025-12-24 | KaTeX math rendering for LaTeX notation in messages |
| NC-0.6.72 | 2025-12-24 | Tool call closures detection - streaming support for replace_block |
| NC-0.6.71 | 2025-12-24 | Sidebar right margin for delete icon accessibility |
| NC-0.6.70 | 2025-12-24 | Fix filter chain executor infinite loop (duplicate _execute_steps call) |
| NC-0.6.69 | 2025-12-24 | Enhanced artifact tool guidance in system prompt |
| NC-0.6.68 | 2025-12-24 | Artifact tools with state tracking - prevents search_replace confusion |
| NC-0.6.67 | 2025-12-24 | Persist LLM responses when client disconnects (detached task execution) |
| NC-0.6.66 | 2025-12-24 | Complete payment system: Stripe, PayPal, Google Pay integration |
| NC-0.6.65 | 2025-12-24 | Fix artifact closing tag detection when not alone on line |
| NC-0.6.64 | 2025-12-24 | Gzip file support, log error extraction with context, error summary in LLM |
| NC-0.6.63 | 2025-12-23 | Shared chat images & formatting, preserve user newlines (whitespace-pre-wrap) |
| NC-0.6.62 | 2025-12-23 | RAG embedding model auto-retry after 60s, background startup load with delay |
| NC-0.6.61 | 2025-12-23 | Fix image display after upload (setCurrentChat race), fix markdown (remove prose classes) |
| NC-0.6.60 | 2025-12-22 | Immediate image display after upload, improved markdown heading/formatting |
| NC-0.6.59 | 2025-12-22 | Fix image base64 truncation - images were only sending first 50 chars to LLM |
| NC-0.6.58 | 2025-12-22 | Fix stream cross-chat contamination - track streaming chat ID, validate before appending |
| NC-0.6.57 | 2025-12-21 | Thinking tokens support - hide model reasoning in collapsible panel, Admin panel scrolling fix |
| NC-0.6.56 | 2025-12-21 | Pre-generate assistant message IDs BEFORE streaming - fixes tool continuation branching |
| NC-0.6.55 | 2025-12-21 | Improved tool continuation parent_id tracking with validation and comprehensive logging |
| NC-0.6.54 | 2025-12-21 | Add kb_search tool - LLM can search knowledge bases with `<kb_search query="...">` |
| NC-0.6.53 | 2025-12-21 | Fix tool loops creating branches - backend now uses latest message as parent for continuations |
| NC-0.6.52 | 2025-12-20 | Fix artifacts carrying over to New Chat - added preserveArtifacts parameter |
| NC-0.6.51 | 2025-12-20 | Chat Knowledge assistant context filtering - prevents knowledge leakage between different GPTs |
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

Current: **NC-0.6.77**

Migrations run automatically on startup in `backend/app/main.py`.

Key tables added/modified:
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
