/**
 * WebSocket Integration Guide for Message System v2
 * 
 * This file shows how to update WebSocketContext.tsx to use the new message system.
 * Copy the relevant sections into WebSocketContext.tsx.
 */

import { useMessageStore } from '../stores/chat';
import { extractArtifacts } from '../lib/artifacts';
import api from '../lib/api';

// ============================================
// SENDING USER MESSAGES
// ============================================

/**
 * Updated sendChatMessage function
 */
const sendChatMessage_NEW = (chatId: string, content: string, attachments?: unknown[], parentId?: string | null) => {
  if (wsRef.current?.readyState !== WebSocket.OPEN) {
    console.error('WebSocket not connected');
    return;
  }
  
  // Get the message store
  const messageStore = useMessageStore.getState();
  
  // Create user message (generates ID, adds to store in 'sending' state)
  const { messageId, requestId } = messageStore.addUserMessage({
    chatId,
    content,
    parentId: parentId ?? null,
    attachments,
  });
  
  // Get zip context if available
  const { zipContext } = useChatStore.getState();
  
  console.log('[sendChatMessage] Sending:', { chatId, messageId, parentId, content: content.substring(0, 50) });
  
  // Send via WebSocket
  wsRef.current.send(JSON.stringify({
    type: 'chat_message',
    payload: {
      chat_id: chatId,
      message_id: messageId,      // Pre-generated ID
      parent_id: parentId ?? null, // Fixed parent
      content,
      attachments,
      zip_context: zipContext,
    },
  }));
};

// ============================================
// HANDLING SERVER MESSAGES
// ============================================

/**
 * Updated message_saved handler
 */
const handleMessageSaved_NEW = (payload: { message_id: string; parent_id?: string }) => {
  console.log('[message_saved] Confirmed:', payload.message_id);
  
  const messageStore = useMessageStore.getState();
  
  // Confirm the user message (state: sending → committed)
  messageStore.confirmUserMessage(payload.message_id);
  
  // Note: We do NOT update parent_id here. If the server returns a different
  // parent_id than we sent, that's a bug that should be fixed server-side.
  // The client's parent_id is authoritative for what it intended.
};

/**
 * Updated stream_start handler
 */
const handleStreamStart_NEW = (payload: {
  chat_id: string;
  message_id: string;
  parent_id: string;
}) => {
  console.log('[stream_start] Starting:', payload);
  
  const messageStore = useMessageStore.getState();
  const currentChatId = messageStore.currentChatId;
  
  // Validate this stream is for the current chat
  if (payload.chat_id && currentChatId && payload.chat_id !== currentChatId) {
    console.warn('[stream_start] Ignoring stream for different chat:', payload.chat_id);
    return;
  }
  
  // Clear streaming buffer
  streamingBufferRef.current?.clear();
  
  // Start assistant stream (creates placeholder in 'streaming' state)
  messageStore.startAssistantStream({
    messageId: payload.message_id,  // Server-generated ID for assistant
    chatId: payload.chat_id,
    parentId: payload.parent_id,    // The user message this responds to
  });
};

/**
 * Updated stream_chunk handler
 */
const handleStreamChunk_NEW = (payload: { chat_id: string; content: string }) => {
  const messageStore = useMessageStore.getState();
  const currentChatId = messageStore.currentChatId;
  
  // Validate this chunk is for the current chat
  if (payload.chat_id && currentChatId && payload.chat_id !== currentChatId) {
    return; // Silently ignore
  }
  
  if (payload.content) {
    // Use buffered append for performance
    streamingBufferRef.current?.append(payload.content);
  }
};

// Create streaming buffer that batches updates to the new store
const createStreamingBuffer_NEW = () => {
  return new StreamingBuffer((content) => {
    useMessageStore.getState().appendStreamContent(content);
  });
};

/**
 * Updated stream_end handler
 */
const handleStreamEnd_NEW = (payload: {
  chat_id: string;
  message_id: string;
  parent_id?: string;
  usage?: { input_tokens: number; output_tokens: number };
}) => {
  console.log('[stream_end] Ending:', payload);
  
  // Flush any remaining buffered content
  streamingBufferRef.current?.flushNow();
  
  const messageStore = useMessageStore.getState();
  const currentChatId = messageStore.currentChatId;
  
  // Validate this is for the current chat
  if (payload.chat_id && currentChatId && payload.chat_id !== currentChatId) {
    console.warn('[stream_end] Ignoring for different chat:', payload.chat_id);
    return;
  }
  
  // Get the accumulated streaming content
  const content = messageStore.streamingContent;
  
  // Extract artifacts from content
  const { artifacts } = extractArtifacts(content);
  
  console.log('[stream_end] Content length:', content.length, 'Artifacts:', artifacts.length);
  
  // Commit the assistant message (state: streaming → committed)
  messageStore.commitAssistantMessage({
    messageId: payload.message_id,
    content,
    artifacts: artifacts.length > 0 ? artifacts : undefined,
    inputTokens: payload.usage?.input_tokens,
    outputTokens: payload.usage?.output_tokens,
  });
  
  // Save artifacts to backend (fire-and-forget)
  if (artifacts.length > 0 && payload.chat_id && payload.message_id) {
    api.put(`/chats/${payload.chat_id}/messages/${payload.message_id}/artifacts`, {
      artifacts,
    }).then(() => {
      console.log('[stream_end] Saved', artifacts.length, 'artifacts to backend');
    }).catch(err => {
      console.error('[stream_end] Failed to save artifacts:', err);
    });
  }
};

/**
 * Updated stream_error handler
 */
const handleStreamError_NEW = (payload: {
  chat_id: string;
  message_id?: string;
  error: string;
}) => {
  console.error('[stream_error]', payload);
  
  const messageStore = useMessageStore.getState();
  
  if (payload.message_id) {
    messageStore.failStream(payload.message_id, payload.error);
  }
};

/**
 * Updated stopGeneration function
 */
const stopGeneration_NEW = (chatId: string) => {
  if (wsRef.current?.readyState !== WebSocket.OPEN) return;
  
  console.log('[stopGeneration] Stopping:', chatId);
  
  // Flush buffer and cancel stream
  streamingBufferRef.current?.flushNow();
  useMessageStore.getState().cancelStream();
  
  // Notify server
  wsRef.current.send(JSON.stringify({
    type: 'stop_generation',
    payload: { chat_id: chatId },
  }));
};

// ============================================
// CHAT SWITCHING
// ============================================

/**
 * When switching chats, load messages from new store
 */
const loadChatMessages_NEW = async (chatId: string) => {
  await useMessageStore.getState().loadMessages(chatId);
};

/**
 * When leaving chat, clear messages
 */
const clearChatMessages_NEW = () => {
  useMessageStore.getState().clearMessages();
};

// ============================================
// COMPLETE SWITCH CASE (for handleMessage)
// ============================================

const handleMessage_UPDATED = (message: WSMessage) => {
  console.log('[WS] Message:', message.type, message.payload);
  
  switch (message.type) {
    case 'pong':
      break;
      
    case 'subscribed':
      console.log('[WS] Subscribed to chat:', message.payload);
      break;
      
    case 'message_saved': {
      const payload = message.payload as { message_id: string; parent_id?: string };
      useMessageStore.getState().confirmUserMessage(payload.message_id);
      break;
    }
    
    case 'chat_updated': {
      // Keep existing logic for title updates
      const payload = message.payload as { chat_id: string; title?: string };
      if (payload.chat_id && payload.title) {
        updateChatLocally(payload.chat_id, { title: payload.title });
      }
      break;
    }
    
    case 'stream_start': {
      const payload = message.payload as {
        chat_id: string;
        message_id: string;
        parent_id: string;
      };
      
      const store = useMessageStore.getState();
      if (payload.chat_id !== store.currentChatId) {
        console.warn('[stream_start] Wrong chat');
        break;
      }
      
      streamingBufferRef.current?.clear();
      store.startAssistantStream({
        messageId: payload.message_id,
        chatId: payload.chat_id,
        parentId: payload.parent_id,
      });
      break;
    }
    
    case 'stream_chunk': {
      const payload = message.payload as { chat_id: string; content: string };
      const store = useMessageStore.getState();
      
      if (payload.chat_id !== store.currentChatId) break;
      if (payload.content) {
        streamingBufferRef.current?.append(payload.content);
      }
      break;
    }
    
    case 'tool_call_start': {
      // Keep existing tool call logic
      break;
    }
    
    case 'tool_call_end': {
      break;
    }
    
    case 'stream_end': {
      streamingBufferRef.current?.flushNow();
      
      const payload = message.payload as {
        chat_id: string;
        message_id: string;
        parent_id?: string;
        usage?: { input_tokens: number; output_tokens: number };
      };
      
      const store = useMessageStore.getState();
      if (payload.chat_id !== store.currentChatId) {
        console.warn('[stream_end] Wrong chat');
        break;
      }
      
      const content = store.streamingContent;
      const { artifacts } = extractArtifacts(content);
      
      store.commitAssistantMessage({
        messageId: payload.message_id,
        content,
        artifacts: artifacts.length > 0 ? artifacts : undefined,
        inputTokens: payload.usage?.input_tokens,
        outputTokens: payload.usage?.output_tokens,
      });
      
      // Save artifacts
      if (artifacts.length > 0) {
        api.put(`/chats/${payload.chat_id}/messages/${payload.message_id}/artifacts`, {
          artifacts,
        }).catch(console.error);
      }
      break;
    }
    
    case 'stream_error': {
      const payload = message.payload as {
        chat_id: string;
        message_id?: string;
        error: string;
      };
      
      if (payload.message_id) {
        useMessageStore.getState().failStream(payload.message_id, payload.error);
      }
      break;
    }
    
    // ... other cases remain similar
  }
};

// ============================================
// MIGRATION CHECKLIST
// ============================================

/*
To migrate WebSocketContext.tsx:

1. Add import:
   import { useMessageStore } from '../stores/chat';

2. Update sendChatMessage to use addUserMessage
   - Remove local message creation
   - Use returned messageId in WebSocket payload

3. Update message_saved handler
   - Call confirmUserMessage instead of replaceMessageId

4. Update stream_start handler
   - Call startAssistantStream instead of creating message locally

5. Update stream_chunk handler  
   - Use new streamingBuffer that calls appendStreamContent

6. Update stream_end handler
   - Call commitAssistantMessage instead of addMessage
   - Extract artifacts once, pass to commit

7. Update stream_error handler
   - Call failStream

8. Update stopGeneration
   - Call cancelStream

9. Update chat switching logic
   - Use loadMessages and clearMessages from new store

10. Remove deprecated functions:
    - replaceMessageId
    - updateMessage (for parent_id changes)
    - Temp ID generation logic

11. Test scenarios:
    - Send message → appears immediately
    - Server confirms → no visible change (already showing)
    - Stream arrives → content updates
    - Stream ends → message finalized
    - Error → message shows error state
    - Cancel → partial content preserved
    - Switch chat → messages clear, new ones load
    - Reload → all committed messages appear
*/
