import { createContext, useContext, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useAuthStore } from '../stores/authStore';
import { useChatStore } from '../stores/chatStore';
import { useBrandingStore } from '../stores/brandingStore';
import { chatApi } from '../lib/api';
import { extractArtifacts, cleanFilePath } from '../lib/artifacts';
import { createFileChangeFromCode } from '../lib/signatures';
import { 
  parseServerEvent, 
  isStreamStart, isStreamChunk, isStreamEnd, isStreamError,
  isToolCall, isToolResult, isImageGeneration, isMessageSaved,
  extractErrorMessage
} from '../lib/wsTypes';
import type { Message, StreamChunk, WSMessage, ZipFileResponse } from '../types';

// Regex to detect file request tags in LLM responses
const FILE_REQUEST_REGEX = /<request_file\s+path=["']([^"']+)["']\s*\/?>/gi;

// Extract all file request paths from content
function extractFileRequests(content: string): string[] {
  const paths: string[] = [];
  let match;
  while ((match = FILE_REQUEST_REGEX.exec(content)) !== null) {
    paths.push(match[1]);
  }
  FILE_REQUEST_REGEX.lastIndex = 0; // Reset regex state
  return paths;
}

// Construct WebSocket URL from current location
function getWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}`;
}

// Streaming buffer for batching updates - aggressive throttling
class StreamingBuffer {
  private buffer: string = '';
  private timeoutId: ReturnType<typeof setTimeout> | null = null;
  private lastFlush: number = 0;
  private flushInterval: number = 100; // 100ms - match backend
  private maxBufferSize: number = 500; // characters - larger buffer
  private onFlush: (content: string) => void;
  
  constructor(onFlush: (content: string) => void) {
    this.onFlush = onFlush;
  }
  
  append(chunk: string) {
    this.buffer += chunk;
    
    // Flush immediately if buffer is very large
    if (this.buffer.length >= this.maxBufferSize) {
      this.flushNow();
      return;
    }
    
    // Schedule flush if not already scheduled
    if (!this.timeoutId) {
      this.timeoutId = setTimeout(() => {
        this.timeoutId = null;
        this.flushNow();
      }, this.flushInterval);
    }
  }
  
  flushNow() {
    if (this.buffer.length > 0) {
      this.onFlush(this.buffer);
      this.buffer = '';
      this.lastFlush = performance.now();
    }
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
  
  clear() {
    this.buffer = '';
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
}

interface WebSocketContextValue {
  isConnected: boolean;
  connectionError: string | null;
  subscribe: (chatId: string) => void;
  unsubscribe: (chatId: string) => void;
  sendChatMessage: (chatId: string, content: string, attachments?: unknown[], parentId?: string | null) => void;
  sendClientMessage: (chatId: string, content: string) => void;
  regenerateMessage: (chatId: string, content: string, parentId: string) => void;
  stopGeneration: (chatId: string) => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  const { accessToken, isAuthenticated, logout } = useAuthStore();
  const {
    currentChat,
    addMessage,
    appendStreamingContent,
    setStreamingContent,
    setStreamingToolCall,
    clearStreaming,
    setIsSending,
    updateChatLocally,
    updateStreamingArtifacts,
  } = useChatStore();
  
  // Create streaming buffer that batches updates
  const streamingBufferRef = useRef<StreamingBuffer | null>(null);
  const lastArtifactCheckRef = useRef<number>(0);
  
  if (!streamingBufferRef.current) {
    streamingBufferRef.current = new StreamingBuffer((content) => {
      const store = useChatStore.getState();
      store.appendStreamingContent(content);
      
      // Check for artifacts periodically (every 500ms)
      const now = performance.now();
      if (now - lastArtifactCheckRef.current > 500) {
        lastArtifactCheckRef.current = now;
        const { streamingContent } = useChatStore.getState();
        store.updateStreamingArtifacts(streamingContent + content);
      }
    });
  }
  
  // Handle file requests detected in LLM responses - fetch files and auto-continue
  const handleFileRequests = useCallback(async (chatId: string, paths: string[], parentMessageId: string) => {
    console.log(`[handleFileRequests] Fetching ${paths.length} files for chat ${chatId}, parent=${parentMessageId}`);
    
    // Log current state before making changes
    const { artifacts: currentArtifacts, messages: currentMessages } = useChatStore.getState();
    console.log(`[handleFileRequests] Current state: ${currentMessages.length} messages, ${currentArtifacts.length} artifacts`);
    
    // Clear previous streaming content since we're about to start a new response
    const { clearStreaming } = useChatStore.getState();
    clearStreaming();
    
    const fileContents: string[] = [];
    const fetchedPaths: string[] = [];
    
    for (const path of paths) {
      try {
        const response = await chatApi.getZipFile(chatId, path);
        const fileData = response.data as ZipFileResponse;
        
        console.log(`Fetched file: ${path} (${fileData.content.length} chars)`);
        fileContents.push(fileData.formatted);
        fetchedPaths.push(path);
        
      } catch (error) {
        console.error(`Failed to fetch file ${path}:`, error);
        fileContents.push(`[FILE_ERROR: Could not retrieve ${path}]`);
      }
    }
    
    // Combine all file contents and send as continuation
    if (fetchedPaths.length > 0 || fileContents.length > 0) {
      const combinedContent = fileContents.join('\n\n');
      
      // Add as file content message (for display)
      const fileMessage: Message = {
        id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        chat_id: chatId,
        role: 'user',
        content: combinedContent,
        parent_id: parentMessageId,
        created_at: new Date().toISOString(),
        metadata: { 
          type: 'file_content', 
          path: fetchedPaths.join(', ') || paths.join(', '),
        },
      };
      addMessage(fileMessage);
      
      // Auto-send to continue the conversation with the file content
      // This triggers the LLM to analyze the file
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        console.log('[handleFileRequests] Auto-continuing conversation with file content');
        setIsSending(true);
        
        // Get zip context from store
        const { zipContext } = useChatStore.getState();
        
        // Create natural continuation prompt
        const continuationPrompt = fetchedPaths.length === 1
          ? `Here is the file you requested:\n\n${combinedContent}`
          : `Here are the ${fetchedPaths.length} files you requested:\n\n${combinedContent}`;
        
        console.log(`[handleFileRequests] Sending continuation with ${continuationPrompt.length} chars to LLM`);
        
        wsRef.current.send(JSON.stringify({
          type: 'chat_message',
          payload: {
            chat_id: chatId,
            content: continuationPrompt,
            parent_id: parentMessageId,
            zip_context: zipContext,
            save_user_message: false,  // Don't save again, we already added the message
          },
        }));
        
        console.log('[handleFileRequests] Continuation sent, waiting for LLM response...');
      } else {
        console.error('WebSocket not open, cannot send file content continuation');
        setIsSending(false);
      }
    } else {
      console.error('No files could be fetched');
      setIsSending(false);
    }
  }, [addMessage, setIsSending]);
  
  const handleMessage = useCallback((message: WSMessage) => {
    console.log('WebSocket message received:', message.type, message.payload);
    
    switch (message.type) {
      case 'pong':
        // Heartbeat response
        break;
        
      case 'subscribed':
        console.log('Subscribed to chat:', message.payload);
        break;
        
      case 'message_saved': {
        // Server confirms user message was saved
        // Since we use client-generated UUIDs, the ID should already match
        const payload = message.payload as { message_id: string; parent_id?: string };
        console.log('Message saved confirmed:', payload.message_id);
        // If parent_id was modified by server (e.g., validation), update it
        if (payload.parent_id !== undefined) {
          const { messages } = useChatStore.getState();
          const msg = messages.find(m => m.id === payload.message_id);
          if (msg && msg.parent_id !== payload.parent_id) {
            useChatStore.getState().updateMessage(payload.message_id, { parent_id: payload.parent_id });
          }
        }
        break;
      }
        
      case 'chat_updated': {
        // Update chat in list (e.g., when title is generated)
        const payload = message.payload as { chat_id: string; title?: string };
        if (payload.chat_id && payload.title) {
          console.log('Chat title updated:', payload.chat_id, payload.title);
          updateChatLocally(payload.chat_id, { title: payload.title });
          
          // Also update document title immediately if this is the current chat
          const { currentChat } = useChatStore.getState();
          if (currentChat?.id === payload.chat_id && payload.title !== 'New Chat') {
            const appName = useBrandingStore.getState().config?.app_name || 'Open-NueChat';
            document.title = `${appName} - ${payload.title}`;
          }
        }
        break;
      }
        
      case 'stream_start': {
        const payload = message.payload as StreamChunk;
        const { currentChat } = useChatStore.getState();
        
        // Validate this stream is for the current chat
        if (payload.chat_id && currentChat?.id && payload.chat_id !== currentChat.id) {
          console.warn('[stream_start] Ignoring stream for different chat:', payload.chat_id, 'current:', currentChat.id);
          break;
        }
        
        console.log('[stream_start] Stream starting for chat:', payload.chat_id);
        // Log current state
        const { messages, artifacts } = useChatStore.getState();
        console.log(`[stream_start] State before clear: ${messages.length} messages, ${artifacts.length} artifacts`);
        
        streamingBufferRef.current?.clear();
        setStreamingContent('');
        setIsSending(true);
        
        // Verify state wasn't accidentally cleared
        const { messages: afterMsgs, artifacts: afterArts } = useChatStore.getState();
        console.log(`[stream_start] State after clear: ${afterMsgs.length} messages, ${afterArts.length} artifacts`);
        break;
      }
        
      case 'stream_chunk': {
        const chunk = message.payload as StreamChunk;
        const { currentChat } = useChatStore.getState();
        
        // Validate this chunk is for the current chat
        if (chunk.chat_id && currentChat?.id && chunk.chat_id !== currentChat.id) {
          // Silently ignore chunks for other chats (don't spam logs)
          break;
        }
        
        if (chunk.content) {
          // Use buffered append for performance
          streamingBufferRef.current?.append(chunk.content);
        }
        break;
      }
        
      case 'tool_call_start': {
        const chunk = message.payload as StreamChunk;
        if (chunk.tool_call) {
          setStreamingToolCall({
            name: chunk.tool_call.name,
            input: JSON.stringify(chunk.tool_call.input, null, 2),
          });
        }
        break;
      }
        
      case 'tool_call_end':
        setStreamingToolCall(null);
        break;
        
      case 'stream_end': {
        // Flush any remaining buffered content
        streamingBufferRef.current?.flushNow();
        
        const chunk = message.payload as StreamChunk & { parent_id?: string };
        console.log('Stream ended:', chunk);
        // Add final message
        if (chunk.message_id) {
          const { streamingContent, currentChat, generatedImages } = useChatStore.getState();
          console.log('Final streaming content length:', streamingContent.length);
          
          // Only add message if it belongs to the current chat
          // This prevents cross-chat contamination when switching chats during streaming
          if (chunk.chat_id && currentChat?.id && chunk.chat_id !== currentChat.id) {
            console.warn('Ignoring stream_end for different chat:', chunk.chat_id, 'current:', currentChat.id);
            clearStreaming();
            setIsSending(false);
            break;
          }
          
          // Extract artifacts from the content and attach to message
          // cleanContent has [📦 Artifact: ...] placeholders, artifacts has the extracted data
          const { cleanContent, artifacts: extractedArtifacts } = extractArtifacts(streamingContent);
          
          // Check if there's a generated image for this message
          let generatedImage = generatedImages[chunk.message_id];
          if (generatedImage) {
            console.log('%c[STREAM_END]', 'background: purple; color: white', 
              'Found generated image for message:', chunk.message_id, 'base64 len:', generatedImage.base64?.length);
          } else {
            console.log('%c[STREAM_END]', 'background: orange; color: black', 
              'No image found yet for message:', chunk.message_id);
          }
          
          const assistantMessage: Message = {
            id: chunk.message_id,
            chat_id: chunk.chat_id || '',
            role: 'assistant',
            // Use cleanContent which has artifact placeholders, preserving code blocks for markdown rendering
            // But actually we want the ORIGINAL content so code blocks render via markdown
            // The artifacts are stored separately for the panel
            content: streamingContent,
            parent_id: chunk.parent_id,
            input_tokens: chunk.usage?.input_tokens,
            output_tokens: chunk.usage?.output_tokens,
            created_at: new Date().toISOString(),
            // Attach extracted artifacts to the message
            artifacts: extractedArtifacts.length > 0 ? extractedArtifacts : undefined,
            // Attach generated image directly to message metadata
            metadata: generatedImage ? { generated_image: generatedImage } : undefined,
          };
          addMessage(assistantMessage);
          
          console.log('Added assistant message with', extractedArtifacts.length, 'artifacts');
          
          // Update code summary with extracted artifacts
          if (extractedArtifacts.length > 0) {
            const { addFileToSummary, saveCodeSummary, codeSummary } = useChatStore.getState();
            
            console.log('[CODE_SUMMARY] Processing', extractedArtifacts.length, 'artifacts');
            console.log('[CODE_SUMMARY] Current summary files:', codeSummary?.files?.length || 0);
            
            // Track which files we're adding
            const existingPaths = new Set((codeSummary?.files || []).map(f => f.path));
            let filesAdded = 0;
            
            for (const artifact of extractedArtifacts) {
              // Use filename or title as the path, cleaned of any angle brackets/punctuation
              const rawPath = artifact.filename || artifact.title;
              const filepath = rawPath ? cleanFilePath(rawPath) : null;
              
              console.log(`[CODE_SUMMARY] Artifact: type=${artifact.type}, filename=${artifact.filename}, title=${artifact.title}, cleanPath=${filepath}, contentLen=${artifact.content?.length || 0}`);
              
              // Track all code-like artifacts
              const isCodeLike = ['code', 'react', 'html', 'json', 'csv'].includes(artifact.type);
              
              if (filepath && artifact.content && isCodeLike) {
                const action = existingPaths.has(filepath) ? 'modified' : 'created';
                const fileChange = createFileChangeFromCode(
                  filepath,
                  artifact.content,
                  action
                );
                addFileToSummary(fileChange);
                existingPaths.add(filepath); // Track for subsequent artifacts in same batch
                filesAdded++;
                console.log(`[CODE_SUMMARY] ✓ Added ${action} file: ${filepath} (${fileChange.signatures.length} signatures)`);
              } else {
                console.log(`[CODE_SUMMARY] ✗ Skipped: filepath=${!!filepath}, content=${!!artifact.content}, isCodeLike=${isCodeLike}`);
              }
            }
            
            console.log(`[CODE_SUMMARY] Total files added: ${filesAdded}`);
            
            // Save to backend if we added any files
            if (filesAdded > 0) {
              saveCodeSummary().then(() => {
                console.log('[CODE_SUMMARY] Saved to backend');
              }).catch(err => {
                console.error('[CODE_SUMMARY] Failed to save:', err);
              });
            }
          } else {
            console.log('[CODE_SUMMARY] No artifacts to process');
          }
          
          // If no image was found, schedule a check in case it arrives right after
          // (handles race condition where image_generated event is still processing)
          if (!generatedImage && chunk.message_id) {
            const messageId = chunk.message_id; // Capture for closure
            setTimeout(() => {
              const { generatedImages: imgs, messages: msgs, updateMessage } = useChatStore.getState();
              const img = imgs[messageId];
              if (img) {
                console.log('%c[DELAYED IMAGE CHECK]', 'background: green; color: white',
                  'Found image after delay, updating message');
                const msg = msgs.find(m => m.id === messageId);
                if (msg) {
                  updateMessage(messageId, {
                    metadata: { ...msg.metadata, generated_image: img }
                  });
                }
              }
            }, 200);
          }
          
          // Check for file request tags in the response
          const fileRequests = extractFileRequests(streamingContent);
          if (fileRequests.length > 0 && currentChat?.id) {
            console.log('Detected file requests:', fileRequests);
            // Fetch files and inject into conversation
            // Don't clear streaming state yet - handleFileRequests will manage it
            handleFileRequests(currentChat.id, fileRequests, chunk.message_id);
            // Don't call clearStreaming() or setIsSending(false) here
            // handleFileRequests will set isSending appropriately
            break;
          }
        }
        clearStreaming();
        setIsSending(false);
        break;
      }
        
      case 'stream_error': {
        const chunk = message.payload as StreamChunk;
        console.error('Stream error:', chunk.error);
        streamingBufferRef.current?.clear();
        clearStreaming();
        setIsSending(false);
        break;
      }
      
      case 'stream_stopped': {
        // Flush any remaining buffered content
        streamingBufferRef.current?.flushNow();
        
        // Generation was stopped by user
        const chunk = message.payload as StreamChunk & { parent_id?: string };
        console.log('Stream stopped by user');
        // Add partial message if content exists
        if (chunk.message_id) {
          const { streamingContent } = useChatStore.getState();
          if (streamingContent) {
            const assistantMessage: Message = {
              id: chunk.message_id,
              chat_id: chunk.chat_id || '',
              role: 'assistant',
              content: streamingContent + '\n\n*[Generation stopped]*',
              parent_id: chunk.parent_id,
              input_tokens: chunk.usage?.input_tokens,
              output_tokens: chunk.usage?.output_tokens,
              created_at: new Date().toISOString(),
            };
            addMessage(assistantMessage);
          }
        }
        clearStreaming();
        setIsSending(false);
        break;
      }
        
      case 'client_message': {
        // Message from another client in shared chat
        const msg = message.payload as Message;
        addMessage(msg);
        break;
      }
      
      case 'image_generation_started': {
        // Image generation has started - queue info available in payload
        break;
      }
      
      case 'image_generated': {
        const payload = message.payload as {
          chat_id: string;
          message_id: string;
          image: {
            base64?: string;
            url?: string;
            width: number;
            height: number;
            seed: number;
            prompt: string;
            generation_time?: number;
            job_id?: string;
          };
        };
        
        if (!payload.message_id || !payload.image) {
          break;
        }
        
        if (!payload.image.base64 && !payload.image.url) {
          break;
        }
        
        const { setGeneratedImage, messages, updateMessage } = useChatStore.getState();
        
        setGeneratedImage(payload.message_id, payload.image);
        
        const existingMessage = messages.find(m => m.id === payload.message_id);
        if (existingMessage) {
          updateMessage(payload.message_id, {
            content: "Here's the image I generated based on your request:",
            metadata: { ...existingMessage.metadata, generated_image: payload.image }
          });
        }
        break;
      }
      
      case 'image_generation_failed': {
        const failPayload = message.payload as {
          chat_id: string;
          message_id: string;
          error: string;
        };
        
        const { messages, updateMessage } = useChatStore.getState();
        const existingMsg = messages.find(m => m.id === failPayload.message_id);
        
        if (existingMsg) {
          updateMessage(failPayload.message_id, {
            content: `I wasn't able to generate the image: ${failPayload.error || 'Unknown error'}`,
            metadata: { 
              ...existingMsg.metadata, 
              image_generation: { 
                status: 'failed', 
                error: failPayload.error 
              }
            }
          });
        }
        break;
      }
        
      case 'error':
        console.error('WebSocket error:', message.payload);
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  }, [addMessage, appendStreamingContent, setStreamingContent, setStreamingToolCall, clearStreaming, setIsSending, updateChatLocally, updateStreamingArtifacts]);
  
  const connect = useCallback(() => {
    if (!accessToken || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    try {
      const wsUrl = `${getWebSocketUrl()}/ws/ws?token=${accessToken}`;
      console.log('Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      let wasConnected = false;
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        wasConnected = true;
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttempts.current = 0;
        
        // Subscribe to current chat if any
        const { currentChat } = useChatStore.getState();
        if (currentChat) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            payload: { chat_id: currentChat.id },
          }));
        }
      };
      
      ws.onmessage = (event) => {
        // Use type-safe parser from wsTypes
        const data = parseServerEvent(event.data);
        if (!data) {
          console.warn('Received invalid WebSocket message');
          return;
        }
        
        // Debug: log ALL incoming messages except pong
        if (data.type !== 'pong') {
          console.log('%c[WS]', 'color: green; font-weight: bold', data.type, data.payload ? Object.keys(data.payload) : 'no payload');
        }
        if (data.type === 'image_generation') {
          console.log('%c[IMAGE]', 'color: blue; font-weight: bold; font-size: 16px', 'Got image!', data.payload?.message_id);
        }
        handleMessage(data as WSMessage);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Don't set error message yet - wait for onclose to determine if it's auth failure
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason, 'wasConnected:', wasConnected);
        setIsConnected(false);
        wsRef.current = null;
        
        // Handle auth errors - code 4001 or immediate close without ever connecting
        const isAuthError = event.code === 4001 || event.code === 1008 || 
          (event.code === 1006 && !wasConnected) ||
          event.reason?.toLowerCase().includes('token') ||
          event.reason?.toLowerCase().includes('auth');
        
        if (isAuthError) {
          console.log('WebSocket auth failed, logging out');
          logout();
          window.location.href = '/login';
          return;
        }
        
        // Set connection error for non-auth failures
        if (!wasConnected) {
          setConnectionError('Connection failed');
        }
        
        // Reconnect after delay if authenticated (with backoff)
        const { isAuthenticated } = useAuthStore.getState();
        if (isAuthenticated && event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          reconnectAttempts.current++;
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setConnectionError('Connection lost. Please refresh the page.');
        }
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setConnectionError('Failed to connect');
    }
  }, [accessToken, handleMessage, logout]);
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);
  
  const subscribe = useCallback((chatId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        payload: { chat_id: chatId },
      }));
    }
  }, []);
  
  const unsubscribe = useCallback((chatId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        payload: { chat_id: chatId },
      }));
    }
  }, []);
  
  const sendChatMessage = useCallback((chatId: string, content: string, attachments?: unknown[], parentId?: string | null) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Generate a proper UUID for the user message (used by both frontend and backend)
      const messageId = crypto.randomUUID();
      const userMessage: Message = {
        id: messageId,
        chat_id: chatId,
        role: 'user',
        content,
        parent_id: parentId,
        created_at: new Date().toISOString(),
      };
      addMessage(userMessage);
      
      setIsSending(true);
      
      // Get zip context from store if available
      const { zipContext } = useChatStore.getState();
      
      console.log('Sending chat message:', { chatId, content: content.substring(0, 50), parentId, messageId, hasZipContext: !!zipContext });
      wsRef.current.send(JSON.stringify({
        type: 'chat_message',
        payload: {
          chat_id: chatId,
          content,
          attachments,
          parent_id: parentId,
          message_id: messageId,  // Send the ID so backend uses it
          zip_context: zipContext,  // Include zip manifest if available
        },
      }));
    } else {
      console.error('WebSocket not connected');
    }
  }, [addMessage, setIsSending]);
  
  // Regenerate without adding a new user message (for retry)
  const regenerateMessage = useCallback((chatId: string, content: string, parentId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setIsSending(true);
      
      console.log('Regenerating message:', { chatId, content: content.substring(0, 50), parentId });
      wsRef.current.send(JSON.stringify({
        type: 'regenerate_message',
        payload: {
          chat_id: chatId,
          content,
          parent_id: parentId,  // The user message whose response we're regenerating
        },
      }));
    } else {
      console.error('WebSocket not connected');
    }
  }, [setIsSending]);
  
  const sendClientMessage = useCallback((chatId: string, content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'client_message',
        payload: {
          chat_id: chatId,
          content,
        },
      }));
    }
  }, []);
  
  const stopGeneration = useCallback((chatId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('Stopping generation for chat:', chatId);
      
      // Immediately clear streaming state for responsive UI
      streamingBufferRef.current?.flushNow();
      clearStreaming();
      setIsSending(false);
      
      // Send stop request to server
      wsRef.current.send(JSON.stringify({
        type: 'stop_generation',
        payload: { chat_id: chatId },
      }));
    }
  }, [setIsSending, clearStreaming]);
  
  const ping = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);
  
  // Connect on mount if authenticated
  useEffect(() => {
    if (isAuthenticated && accessToken) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [isAuthenticated, accessToken, connect, disconnect]);
  
  // Track previous chat for proper unsubscribe
  const previousChatIdRef = useRef<string | null>(null);
  
  // Subscribe to chat when it changes - with proper cleanup
  useEffect(() => {
    if (!isConnected) return;
    
    const newChatId = currentChat?.id || null;
    const oldChatId = previousChatIdRef.current;
    
    // Unsubscribe from old chat first
    if (oldChatId && oldChatId !== newChatId) {
      console.log('[WS] Unsubscribing from old chat:', oldChatId);
      unsubscribe(oldChatId);
      // Clear any lingering streaming state when switching chats
      clearStreaming();
    }
    
    // Subscribe to new chat
    if (newChatId) {
      console.log('[WS] Subscribing to chat:', newChatId);
      subscribe(newChatId);
    }
    
    previousChatIdRef.current = newChatId;
    
    // Cleanup on unmount
    return () => {
      if (newChatId) {
        unsubscribe(newChatId);
      }
    };
  }, [isConnected, currentChat?.id, subscribe, unsubscribe, clearStreaming]);
  
  // Heartbeat
  useEffect(() => {
    if (!isConnected) return;
    
    const interval = setInterval(() => {
      ping();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [isConnected, ping]);
  
  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        connectionError,
        subscribe,
        unsubscribe,
        sendChatMessage,
        sendClientMessage,
        regenerateMessage,
        stopGeneration,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}
