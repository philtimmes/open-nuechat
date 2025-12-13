/**
 * Message slice - Message CRUD and branching operations
 */
import type { Message, Artifact, MessageBranch } from '../../types';
import type { MessageSlice, SliceCreator } from './types';
import api from '../../lib/api';
import { extractArtifacts, collectChatArtifacts } from '../../lib/artifacts';

export const createMessageSlice: SliceCreator<MessageSlice> = (set, get) => ({
  messages: [],
  isLoadingMessages: false,

  fetchMessages: async (chatId: string) => {
    set({ isLoadingMessages: true, error: null });
    try {
      const response = await api.get(`/chats/${chatId}/messages`);
      const messages: Message[] = Array.isArray(response.data) ? response.data : [];
      
      // Collect artifacts from all messages
      // Always extract from content to ensure we don't miss any
      // Use stored artifacts' timestamps when available
      const allArtifacts: Artifact[] = [];
      
      for (const msg of messages) {
        // Always extract from content to catch everything
        const { artifacts: extractedArtifacts } = extractArtifacts(msg.content || '');
        
        if (msg.artifacts && msg.artifacts.length > 0) {
          // Use stored artifacts (they have correct timestamps)
          // They should match extracted artifacts by content
          allArtifacts.push(...msg.artifacts);
        } else if (extractedArtifacts.length > 0) {
          // No stored artifacts - use extracted with message timestamp
          for (const art of extractedArtifacts) {
            art.created_at = msg.created_at || art.created_at;
          }
          allArtifacts.push(...extractedArtifacts);
        }
      }
      
      set({ 
        messages, 
        isLoadingMessages: false,
        artifacts: allArtifacts,
      });
      
      // Also fetch uploaded artifacts and code summary
      const { fetchUploadedData, fetchCodeSummary } = get();
      await Promise.all([
        fetchUploadedData(chatId),
        fetchCodeSummary(chatId),
      ]);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch messages';
      set({ error: errorMessage, isLoadingMessages: false, messages: [] });
      console.error('Failed to fetch messages:', error);
    }
  },

  addMessage: (message: Message) => {
    const { currentChat, messages: existingMessages } = get();
    
    // Validate: Only add messages for the current chat
    if (message.chat_id && currentChat?.id && message.chat_id !== currentChat.id) {
      console.warn('[addMessage] Ignoring message for different chat:', message.chat_id, 'current:', currentChat.id);
      return;
    }
    
    // Validate: Don't add duplicate messages
    if (existingMessages.some(m => m.id === message.id)) {
      console.log('[addMessage] Skipping duplicate message:', message.id);
      return;
    }
    
    // Use artifacts passed with message (from stream_end extraction)
    // Only extract from content if message.artifacts is not already set
    let newArtifacts: Artifact[] = [];
    if (message.artifacts && message.artifacts.length > 0) {
      // Use the already-extracted artifacts (with correct timestamps)
      newArtifacts = message.artifacts;
    } else if (message.content) {
      // Fallback: extract from content if no artifacts provided
      const { artifacts: extracted } = extractArtifacts(message.content);
      newArtifacts = extracted;
    }
    
    set((state) => ({
      messages: [...state.messages, message],
      // Append new artifacts to existing - don't replace!
      artifacts: [...state.artifacts, ...newArtifacts],
    }));
  },

  clearMessages: () => {
    set({ 
      messages: [], 
      artifacts: [], 
      selectedArtifact: null,
      streamingArtifacts: [],
    });
  },

  updateMessage: (messageId: string, updates: Partial<Message>) => {
    set((state) => {
      const updatedMessages = state.messages.map((m) =>
        m.id === messageId ? { ...m, ...updates } : m
      );
      
      // If content was updated, re-extract all artifacts from all messages
      if (updates.content !== undefined) {
        const messageArtifacts: Artifact[] = [];
        for (const msg of updatedMessages) {
          if (msg.artifacts) {
            messageArtifacts.push(...msg.artifacts);
          }
          if (msg.content) {
            const { artifacts } = extractArtifacts(msg.content);
            messageArtifacts.push(...artifacts);
          }
        }
        // Keep uploaded artifacts separate and merge
        const uploadedArtifacts = (state.uploadedArtifacts || []).map(a => ({ ...a, source: 'upload' as const }));
        return {
          messages: updatedMessages,
          artifacts: [...uploadedArtifacts, ...messageArtifacts],
        };
      }
      
      return { messages: updatedMessages };
    });
  },

  replaceMessageId: (tempId: string, realId: string, parentId?: string | null) => {
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === tempId 
          ? { ...m, id: realId, parent_id: parentId } 
          : m
      ),
    }));
  },

  retryMessage: async (messageId: string) => {
    const { messages, currentChat } = get();
    if (!currentChat) return;
    
    const messageIndex = messages.findIndex((m) => m.id === messageId);
    if (messageIndex === -1) return;
    
    const message = messages[messageIndex];
    if (message.role !== 'assistant') return;
    
    // Find the user message before this
    let userMessageIndex = messageIndex - 1;
    while (userMessageIndex >= 0 && messages[userMessageIndex].role !== 'user') {
      userMessageIndex--;
    }
    if (userMessageIndex < 0) return;
    
    const userMessage = messages[userMessageIndex];
    
    // Check if there are messages after this one
    const hasMessagesAfter = messageIndex < messages.length - 1;
    
    if (hasMessagesAfter) {
      // Create a branch - store current content as a branch
      const currentBranch: MessageBranch = {
        id: `branch-${Date.now()}`,
        content: message.content,
        artifacts: message.artifacts,
        tool_calls: message.tool_calls,
        input_tokens: message.input_tokens,
        output_tokens: message.output_tokens,
        created_at: message.created_at,
      };
      
      // Initialize branches array if needed
      const existingBranches = message.branches || [];
      const branches = [...existingBranches, currentBranch];
      
      // Update the message to have branches
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === messageId
            ? {
                ...m,
                branches,
                current_branch: branches.length,
                content: '',
              }
            : m
        ),
      }));
    } else {
      // No messages after - just clear and regenerate
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === messageId
            ? { ...m, content: '' }
            : m
        ),
      }));
    }
    
    // Store for WebSocket context to pick up
    (window as unknown as Record<string, unknown>).__retryMessageId = messageId;
    (window as unknown as Record<string, unknown>).__retryUserContent = userMessage.content;
  },

  switchBranch: (messageId: string, branchIndex: number) => {
    const { messages } = get();
    const message = messages.find((m) => m.id === messageId);
    if (!message || !message.branches) return;
    
    const totalBranches = message.branches.length + 1;
    if (branchIndex < 0 || branchIndex >= totalBranches) return;
    
    // If switching to the "main" branch (last one, current content)
    if (branchIndex === message.branches.length) {
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === messageId
            ? { ...m, current_branch: undefined }
            : m
        ),
      }));
      return;
    }
    
    // Switch to a stored branch
    const branch = message.branches[branchIndex];
    const currentContent = message.content;
    const currentArtifacts = message.artifacts;
    
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId
          ? {
              ...m,
              content: branch.content,
              artifacts: branch.artifacts,
              tool_calls: branch.tool_calls,
              input_tokens: branch.input_tokens,
              output_tokens: branch.output_tokens,
              current_branch: branchIndex,
              branches: m.branches?.map((b, i) =>
                i === (m.current_branch ?? m.branches!.length)
                  ? { ...b, content: currentContent, artifacts: currentArtifacts }
                  : b
              ),
            }
          : m
      ),
    }));
  },
});
