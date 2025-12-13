# Chat Message Architecture v2

## Core Principles

1. **IDs are immutable** - Generated once using `crypto.randomUUID()`, never changed
2. **Parent IDs are immutable** - Set at message creation, never modified
3. **State machine** - Messages follow defined state transitions
4. **Late commit** - Database save only happens after stream completes
5. **Optimistic UI** - Frontend shows messages immediately, confirms with server

## Message States

```
┌─────────┐     ┌─────────┐     ┌───────────┐     ┌───────────┐
│ pending │ ──► │ sending │ ──► │ streaming │ ──► │ committed │
└─────────┘     └─────────┘     └───────────┘     └───────────┘
     │               │               │
     │               │               │
     ▼               ▼               ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│  error  │     │  error  │     │  error  │
└─────────┘     └─────────┘     └─────────┘
```

### State Definitions

| State | Description |
|-------|-------------|
| `pending` | Created locally, not yet sent to server |
| `sending` | Sent to server, awaiting confirmation |
| `streaming` | Assistant response being streamed |
| `committed` | Saved to database, immutable |
| `error` | Failed to send or save |

### Valid Transitions

- `pending → sending` - User sends message
- `pending → error` - Local validation failed
- `sending → committed` - Server confirmed save
- `sending → error` - Server rejected message
- `streaming → committed` - Stream completed successfully
- `streaming → error` - Stream failed
- `error → pending` - User retries

## Message Flow

### User Sends Message

```
1. User clicks Send
   │
   ▼
2. Frontend generates IDs:
   - user_message_id = crypto.randomUUID()
   - request_id = "req_" + crypto.randomUUID()
   │
   ▼
3. Create user message (state: sending):
   {
     id: user_message_id,      // NEVER changes
     chat_id: current_chat,
     parent_id: last_assistant_id,  // NEVER changes
     role: "user",
     content: "...",
     state: "sending",
     created_at: now
   }
   │
   ▼
4. Add to store immediately (optimistic)
   │
   ▼
5. Send via WebSocket:
   {
     type: "chat_message",
     payload: {
       chat_id: current_chat,
       message_id: user_message_id,
       parent_id: last_assistant_id,
       content: "..."
     }
   }
   │
   ▼
6. Server responds: "message_saved"
   │
   ▼
7. Transition: sending → committed
```

### Assistant Response Streaming

```
1. Server sends: stream_start
   {
     chat_id: "...",
     message_id: "assistant_uuid",  // Server generates this
     parent_id: user_message_id     // Links to user message
   }
   │
   ▼
2. Create assistant placeholder (state: streaming):
   {
     id: assistant_uuid,       // From server, NEVER changes
     chat_id: current_chat,
     parent_id: user_message_id,  // NEVER changes
     role: "assistant",
     content: "",
     state: "streaming",
     created_at: now
   }
   │
   ▼
3. Server sends: stream_chunk (many times)
   {
     content: "..."
   }
   │
   ▼
4. Append to streaming content
   (only modifies content, nothing else)
   │
   ▼
5. Server sends: stream_end
   {
     message_id: assistant_uuid,
     usage: { input_tokens: N, output_tokens: M }
   }
   │
   ▼
6. Finalize message:
   - Extract artifacts from content
   - Set token counts
   - Transition: streaming → committed
   │
   ▼
7. Save artifacts to backend (fire-and-forget)
```

## Data Structures

### ChatMessage (Frontend)

```typescript
interface ChatMessage {
  // Immutable (set once)
  readonly id: string;
  readonly chat_id: string;
  readonly parent_id: string | null;
  readonly role: 'user' | 'assistant' | 'system' | 'tool';
  readonly created_at: string;
  
  // Mutable (during streaming only)
  content: string;
  
  // State
  state: MessageState;
  
  // Optional
  artifacts?: Artifact[];
  tool_calls?: ToolCall[];
  input_tokens?: number;
  output_tokens?: number;
  metadata?: {...};
  error?: string;
}
```

### WebSocket Messages

#### Client → Server

```typescript
// Send user message
{
  type: "chat_message",
  payload: {
    chat_id: string,
    message_id: string,      // Client-generated UUID
    parent_id: string | null,
    content: string,
    attachments?: any[]
  }
}

// Stop generation
{
  type: "stop_generation",
  payload: { chat_id: string }
}
```

#### Server → Client

```typescript
// User message saved
{
  type: "message_saved",
  payload: {
    message_id: string,
    parent_id: string | null  // Confirmed parent (should match what client sent)
  }
}

// Stream starting
{
  type: "stream_start",
  payload: {
    chat_id: string,
    message_id: string,       // Server-generated UUID for assistant
    parent_id: string         // The user message this responds to
  }
}

// Content chunk
{
  type: "stream_chunk",
  payload: {
    chat_id: string,
    content: string
  }
}

// Stream complete
{
  type: "stream_end",
  payload: {
    chat_id: string,
    message_id: string,
    parent_id: string,
    usage: {
      input_tokens: number,
      output_tokens: number
    }
  }
}

// Error
{
  type: "stream_error",
  payload: {
    chat_id: string,
    message_id?: string,
    error: string
  }
}
```

## Branching / Edit History

Messages form a tree, not a list:

```
           [user1]
              │
           [asst1]
              │
        ┌─────┴─────┐
     [user2]    [user2']  (edit)
        │          │
     [asst2]   [asst2']
        │
     [user3]
```

### Branch Selection

Store tracks which branch is "active" for each fork point:

```typescript
selectedVersions: {
  "asst1_id": "user2'_id",  // User edited, selected new branch
}
```

### Display Logic

1. Build tree from parent_id relationships
2. Walk tree from root
3. At each fork, use selectedVersions to pick branch
4. If no selection, use first (oldest) child

## Store API

```typescript
// Message lifecycle
loadMessages(chatId: string): Promise<void>
clearMessages(): void

// User messages
addUserMessage(params): { messageId, requestId }
confirmUserMessage(messageId): void
failUserMessage(messageId, error): void

// Assistant messages  
startAssistantStream(params): void
appendStreamContent(content): void
commitAssistantMessage(params): void
failStream(messageId, error): void
cancelStream(): void

// Branching
selectBranch(parentId, childId): void
getSiblings(messageId): ChatMessage[]

// Queries
getConversation(): ChatMessage[]
getMessages(): ChatMessage[]
getMessage(id): ChatMessage | undefined
isStreaming(): boolean
```

## Migration from Old System

### What to Remove

1. `replaceMessageId()` - IDs don't change
2. `updateMessage()` with parent_id changes - parent_id is immutable
3. Temp ID logic - always use real UUIDs
4. Multiple artifact extraction passes - extract once at commit

### What to Keep

1. WebSocket connection management
2. Streaming buffer (for performance)
3. Artifact extraction logic
4. Branch navigation UI

### Changes to WebSocketContext

```typescript
// OLD
const userMessage = {
  id: crypto.randomUUID(),
  ...
};
addMessage(userMessage);
wsRef.current.send(JSON.stringify({
  type: 'chat_message',
  payload: { ...data, message_id: userMessage.id }
}));

// NEW
const { messageId, requestId } = useMessageStore.getState().addUserMessage({
  chatId,
  content,
  parentId,
  attachments
});
wsRef.current.send(JSON.stringify({
  type: 'chat_message', 
  payload: {
    chat_id: chatId,
    message_id: messageId,
    parent_id: parentId,
    content
  }
}));
```

### Changes to stream_start Handler

```typescript
// OLD - created message with potentially wrong parent_id
const assistantMessage = {
  id: chunk.message_id,
  parent_id: chunk.parent_id,  // Server tells us
  ...
};
addMessage(assistantMessage);

// NEW - uses pre-validated parent from request
useMessageStore.getState().startAssistantStream({
  messageId: chunk.message_id,
  chatId: chunk.chat_id,
  parentId: chunk.parent_id  // Server confirms, should match our expectation
});
```

### Changes to stream_end Handler

```typescript
// OLD - complex extraction and updating
const { cleanContent, artifacts } = extractArtifacts(streamingContent);
const assistantMessage = { ... };
addMessage(assistantMessage);
api.put(...); // Save artifacts

// NEW - single commit call
const { streamingContent } = useMessageStore.getState();
const { artifacts } = extractArtifacts(streamingContent);

useMessageStore.getState().commitAssistantMessage({
  messageId: chunk.message_id,
  content: streamingContent,
  artifacts,
  inputTokens: chunk.usage?.input_tokens,
  outputTokens: chunk.usage?.output_tokens
});

// Background save
api.put(`/chats/${chatId}/messages/${messageId}/artifacts`, { artifacts });
```

## File Structure

```
frontend/src/stores/chat/
├── messageSystem.ts     # Types, state machine, utilities
├── messageStore.ts      # Zustand store implementation
├── types.ts             # Legacy types (keep for compatibility)
├── index.ts             # Re-exports
└── ...other slices      # Artifact, streaming UI state, etc.
```

## Testing Checklist

- [ ] User sends message → appears immediately
- [ ] Server confirms → state changes to committed
- [ ] Stream starts → placeholder appears with correct parent
- [ ] Chunks arrive → content updates
- [ ] Stream ends → artifacts extracted, message committed
- [ ] Error during stream → message shows error state
- [ ] Cancel stream → partial content preserved
- [ ] Reload page → all committed messages load
- [ ] Edit creates branch → old response preserved
- [ ] Branch navigation → correct messages shown
- [ ] Fast chat switching → no cross-contamination
