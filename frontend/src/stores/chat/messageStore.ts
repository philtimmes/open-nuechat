/**
 * Message Store v2 - Clean Chat Hierarchy
 * 
 * This store manages messages with proper state transitions and immutability.
 * 
 * Key Guarantees:
 * - Message IDs never change after creation
 * - Parent IDs never change after creation
 * - State transitions follow defined rules
 * - Streaming content only updates pending messages
 */

import { create } from 'zustand';
import api from '../../lib/api';
import { extractArtifacts } from '../../lib/artifacts';
import type { Artifact } from '../../types';
import {
  type ChatMessage,
  type MessageState,
  type PendingRequest,
  generateMessageId,
  generateRequestId,
  createUserMessage,
  createAssistantPlaceholder,
  fromServerMessage,
  canTransition,
  getConversationPath,
  findSiblings,
} from './messageSystem';

// ============================================
// STORE STATE
// ============================================

interface MessageStoreState {
  // All messages for current chat (keyed by ID for O(1) lookup)
  messagesById: Record<string, ChatMessage>;
  
  // Ordered message IDs (maintains insertion order)
  messageOrder: string[];
  
  // Current chat ID
  currentChatId: string | null;
  
  // Pending requests (for tracking user → assistant pairs)
  pendingRequests: Record<string, PendingRequest>;
  
  // Streaming state
  streamingMessageId: string | null;
  streamingContent: string;
  
  // Artifacts (extracted from messages)
  artifacts: Artifact[];
  
  // Branch selection: { parent_id: selected_child_id }
  selectedVersions: Record<string, string>;
  
  // Loading state
  isLoading: boolean;
  error: string | null;
}

interface MessageStoreActions {
  // ===== LIFECYCLE =====
  loadMessages: (chatId: string) => Promise<void>;
  clearMessages: () => void;
  
  // ===== USER MESSAGES =====
  addUserMessage: (params: {
    chatId: string;
    content: string;
    parentId: string | null;
    attachments?: unknown[];
  }) => { messageId: string; requestId: string };
  confirmUserMessage: (messageId: string) => void;
  failUserMessage: (messageId: string, error: string) => void;
  
  // ===== ASSISTANT MESSAGES =====
  startAssistantStream: (params: {
    messageId: string;
    chatId: string;
    parentId: string;
  }) => void;
  appendStreamContent: (content: string) => void;
  commitAssistantMessage: (params: {
    messageId: string;
    content: string;
    artifacts?: Artifact[];
    inputTokens?: number;
    outputTokens?: number;
    metadata?: ChatMessage['metadata'];
  }) => void;
  failStream: (messageId: string, error: string) => void;
  cancelStream: () => void;
  
  // ===== BRANCHING =====
  selectBranch: (parentId: string, childId: string) => void;
  getSiblings: (messageId: string) => ChatMessage[];
  
  // ===== COMPUTED =====
  getConversation: () => ChatMessage[];
  getMessages: () => ChatMessage[];
  getMessage: (id: string) => ChatMessage | undefined;
  isStreaming: () => boolean;
}

type MessageStore = MessageStoreState & MessageStoreActions;

// ============================================
// STORE IMPLEMENTATION
// ============================================

export const useMessageStore = create<MessageStore>()((set, get) => ({
  // Initial state
  messagesById: {},
  messageOrder: [],
  currentChatId: null,
  pendingRequests: {},
  streamingMessageId: null,
  streamingContent: '',
  artifacts: [],
  selectedVersions: {},
  isLoading: false,
  error: null,
  
  // ===== LIFECYCLE =====
  
  loadMessages: async (chatId: string) => {
    // Clear if switching chats
    if (get().currentChatId !== chatId) {
      get().clearMessages();
    }
    
    set({ isLoading: true, error: null, currentChatId: chatId });
    
    try {
      const response = await api.get(`/chats/${chatId}/messages`);
      const serverMessages = Array.isArray(response.data) ? response.data : [];
      
      const messagesById: Record<string, ChatMessage> = {};
      const messageOrder: string[] = [];
      const allArtifacts: Artifact[] = [];
      
      for (const serverMsg of serverMessages) {
        const msg = fromServerMessage(serverMsg);
        messagesById[msg.id] = msg;
        messageOrder.push(msg.id);
        
        // Extract artifacts
        if (msg.artifacts && msg.artifacts.length > 0) {
          allArtifacts.push(...msg.artifacts);
        } else if (msg.content && msg.role === 'assistant') {
          const { artifacts } = extractArtifacts(msg.content);
          if (artifacts.length > 0) {
            // Note: We don't mutate msg here, artifacts stay on the extracted result
            allArtifacts.push(...artifacts);
          }
        }
      }
      
      set({
        messagesById,
        messageOrder,
        artifacts: allArtifacts,
        isLoading: false,
      });
      
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load messages';
      set({ error: message, isLoading: false });
      console.error('Failed to load messages:', error);
    }
  },
  
  clearMessages: () => {
    set({
      messagesById: {},
      messageOrder: [],
      pendingRequests: {},
      streamingMessageId: null,
      streamingContent: '',
      artifacts: [],
      selectedVersions: {},
      error: null,
    });
  },
  
  // ===== USER MESSAGES =====
  
  addUserMessage: ({ chatId, content, parentId, attachments }) => {
    const messageId = generateMessageId();
    const requestId = generateRequestId();
    
    const message = createUserMessage({
      id: messageId,
      chat_id: chatId,
      parent_id: parentId,
      content,
      attachments: attachments as ChatMessage['attachments'],
    });
    
    // Set to sending state
    message.state = 'sending';
    
    const request: PendingRequest = {
      request_id: requestId,
      user_message_id: messageId,
      chat_id: chatId,
      parent_id: parentId,
      content,
      attachments,
      created_at: new Date().toISOString(),
      sent_at: new Date().toISOString(),
      state: 'sent',
    };
    
    set((state) => ({
      messagesById: { ...state.messagesById, [messageId]: message },
      messageOrder: [...state.messageOrder, messageId],
      pendingRequests: { ...state.pendingRequests, [requestId]: request },
    }));
    
    return { messageId, requestId };
  },
  
  confirmUserMessage: (messageId: string) => {
    set((state) => {
      const msg = state.messagesById[messageId];
      if (msg && canTransition(msg.state, 'committed')) {
        return {
          messagesById: {
            ...state.messagesById,
            [messageId]: { ...msg, state: 'committed' as MessageState },
          },
        };
      }
      return state;
    });
  },
  
  failUserMessage: (messageId: string, error: string) => {
    set((state) => {
      const msg = state.messagesById[messageId];
      if (msg && canTransition(msg.state, 'error')) {
        return {
          messagesById: {
            ...state.messagesById,
            [messageId]: { ...msg, state: 'error' as MessageState, error },
          },
        };
      }
      return state;
    });
  },
  
  // ===== ASSISTANT MESSAGES =====
  
  startAssistantStream: ({ messageId, chatId, parentId }) => {
    const parent = get().messagesById[parentId];
    if (!parent) {
      console.warn(`[startAssistantStream] Parent ${parentId} not found, creating anyway`);
    }
    
    const placeholder = createAssistantPlaceholder({
      id: messageId,
      chat_id: chatId,
      parent_id: parentId,
    });
    
    set((state) => ({
      messagesById: { ...state.messagesById, [messageId]: placeholder },
      messageOrder: [...state.messageOrder, messageId],
      streamingMessageId: messageId,
      streamingContent: '',
    }));
    
    console.log(`[startAssistantStream] Created placeholder: id=${messageId}, parent_id=${parentId}`);
  },
  
  appendStreamContent: (content: string) => {
    const streamingId = get().streamingMessageId;
    if (!streamingId) return;
    
    set((state) => {
      const newContent = state.streamingContent + content;
      const msg = state.messagesById[streamingId];
      
      if (msg && msg.state === 'streaming') {
        return {
          streamingContent: newContent,
          messagesById: {
            ...state.messagesById,
            [streamingId]: { ...msg, content: newContent },
          },
        };
      }
      
      return { streamingContent: newContent };
    });
  },
  
  commitAssistantMessage: ({ messageId, content, artifacts, inputTokens, outputTokens, metadata }) => {
    set((state) => {
      const msg = state.messagesById[messageId];
      if (!msg) {
        console.error(`[commitAssistantMessage] Message ${messageId} not found`);
        return state;
      }
      
      if (!canTransition(msg.state, 'committed')) {
        console.warn(`[commitAssistantMessage] Invalid transition: ${msg.state} → committed`);
        return state;
      }
      
      const committedMsg: ChatMessage = {
        ...msg,
        content,
        state: 'committed',
        artifacts,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        metadata,
      };
      
      const newArtifacts = artifacts && artifacts.length > 0
        ? [...state.artifacts, ...artifacts]
        : state.artifacts;
      
      console.log(`[commitAssistantMessage] Committed: id=${messageId}, content=${content.length} chars, artifacts=${artifacts?.length || 0}`);
      
      return {
        messagesById: { ...state.messagesById, [messageId]: committedMsg },
        artifacts: newArtifacts,
        streamingMessageId: null,
        streamingContent: '',
      };
    });
  },
  
  failStream: (messageId: string, error: string) => {
    set((state) => {
      const msg = state.messagesById[messageId];
      if (msg && canTransition(msg.state, 'error')) {
        return {
          messagesById: {
            ...state.messagesById,
            [messageId]: { ...msg, state: 'error' as MessageState, error },
          },
          streamingMessageId: null,
          streamingContent: '',
        };
      }
      return {
        streamingMessageId: null,
        streamingContent: '',
      };
    });
  },
  
  cancelStream: () => {
    const streamingId = get().streamingMessageId;
    if (!streamingId) return;
    
    set((state) => {
      const msg = state.messagesById[streamingId];
      if (msg) {
        return {
          messagesById: {
            ...state.messagesById,
            [streamingId]: { ...msg, content: state.streamingContent, state: 'committed' as MessageState },
          },
          streamingMessageId: null,
          streamingContent: '',
        };
      }
      return {
        streamingMessageId: null,
        streamingContent: '',
      };
    });
  },
  
  // ===== BRANCHING =====
  
  selectBranch: (parentId: string, childId: string) => {
    set((state) => ({
      selectedVersions: { ...state.selectedVersions, [parentId]: childId },
    }));
  },
  
  getSiblings: (messageId: string) => {
    const messages = Object.values(get().messagesById);
    return findSiblings(messages, messageId);
  },
  
  // ===== COMPUTED =====
  
  getConversation: () => {
    const { messagesById, selectedVersions } = get();
    const messages = Object.values(messagesById);
    return getConversationPath(messages, selectedVersions);
  },
  
  getMessages: () => {
    const { messagesById, messageOrder } = get();
    return messageOrder.map(id => messagesById[id]).filter(Boolean);
  },
  
  getMessage: (id: string) => {
    return get().messagesById[id];
  },
  
  isStreaming: () => {
    return get().streamingMessageId !== null;
  },
}));

// ============================================
// CONVENIENCE HOOKS
// ============================================

export function useConversation() {
  return useMessageStore((state) => state.getConversation());
}

export function useIsStreaming() {
  return useMessageStore((state) => state.streamingMessageId !== null);
}

export function useStreamingContent() {
  return useMessageStore((state) => state.streamingContent);
}

export function useArtifacts() {
  return useMessageStore((state) => state.artifacts);
}
