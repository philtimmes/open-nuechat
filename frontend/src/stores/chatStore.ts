import { create } from 'zustand';
import type { Chat, Message, Artifact, MessageBranch, CodeSummary, FileChange, SignatureWarning, ZipUploadResult, GeneratedImage } from '../types';
import api from '../lib/api';
import { useModelsStore } from './modelsStore';
import { extractArtifacts, collectChatArtifacts } from '../lib/artifacts';

interface ChatStore {
  chats: Chat[];
  currentChat: Chat | null;
  messages: Message[];
  isLoadingChats: boolean;
  isLoadingMessages: boolean;
  isSending: boolean;
  streamingContent: string;
  streamingToolCall: { name: string; input: string } | null;
  streamingArtifacts: Artifact[];
  error: string | null;
  
  // Generated images (message_id -> image data)
  generatedImages: Record<string, GeneratedImage>;
  setGeneratedImage: (messageId: string, image: GeneratedImage) => void;
  
  // Zip context for LLM (manifest from uploaded zip files)
  zipContext: string | null;
  setZipContext: (context: string | null) => void;
  
  // Artifacts
  artifacts: Artifact[];
  selectedArtifact: Artifact | null;
  showArtifacts: boolean;
  
  // Uploaded files (persisted to backend)
  uploadedArtifacts: Artifact[];
  zipUploadResult: Partial<ZipUploadResult> | null;
  setZipUploadResult: (result: Partial<ZipUploadResult> | null) => void;
  fetchUploadedData: (chatId: string) => Promise<void>;
  
  // Code Summary
  codeSummary: CodeSummary | null;
  showSummary: boolean;
  setShowSummary: (show: boolean) => void;
  updateCodeSummary: (files: FileChange[], warnings?: SignatureWarning[]) => void;
  addFileToSummary: (file: FileChange) => void;
  addWarning: (warning: SignatureWarning) => void;
  clearSummary: () => void;
  fetchCodeSummary: (chatId: string) => Promise<void>;
  saveCodeSummary: () => Promise<void>;
  
  // Chat actions
  fetchChats: () => Promise<void>;
  createChat: (model?: string, systemPrompt?: string) => Promise<Chat>;
  setCurrentChat: (chat: Chat | null) => void;
  deleteChat: (chatId: string) => Promise<void>;
  deleteAllChats: () => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  updateChatLocally: (chatId: string, updates: Partial<Chat>) => void;
  
  // Message actions
  fetchMessages: (chatId: string) => Promise<void>;
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  replaceMessageId: (tempId: string, realId: string, parentId?: string | null) => void;
  
  // Branching
  retryMessage: (messageId: string) => Promise<void>;
  switchBranch: (messageId: string, branchIndex: number) => void;
  
  // Artifact actions
  setSelectedArtifact: (artifact: Artifact | null) => void;
  setShowArtifacts: (show: boolean) => void;
  collectAllArtifacts: () => Artifact[];
  updateStreamingArtifacts: (content: string) => void;
  
  // Streaming actions
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (chunk: string) => void;
  setStreamingToolCall: (toolCall: { name: string; input: string } | null) => void;
  clearStreaming: () => void;
  setIsSending: (isSending: boolean) => void;
  
  // Error handling
  setError: (error: string | null) => void;
}

export const useChatStore = create<ChatStore>((set, get) => ({
  chats: [],
  currentChat: null,
  messages: [],
  isLoadingChats: false,
  isLoadingMessages: false,
  isSending: false,
  streamingContent: '',
  streamingToolCall: null,
  error: null,
  
  // Zip context for LLM
  zipContext: null,
  setZipContext: (context) => set({ zipContext: context }),
  
  // Generated images
  generatedImages: {} as Record<string, GeneratedImage>,
  setGeneratedImage: (messageId, image) => set((state) => ({
    generatedImages: { ...state.generatedImages, [messageId]: image }
  })),
  
  // Artifacts
  artifacts: [],
  streamingArtifacts: [] as Artifact[],
  selectedArtifact: null,
  showArtifacts: false,
  
  // Uploaded files (persisted)
  uploadedArtifacts: [] as Artifact[],
  zipUploadResult: null,
  
  // Code Summary
  codeSummary: null,
  showSummary: false,
  
  setShowSummary: (show) => set({ showSummary: show }),
  
  updateCodeSummary: (files, warnings = []) => {
    const { currentChat, codeSummary } = get();
    if (!currentChat) return;
    
    const newSummary: CodeSummary = {
      id: codeSummary?.id || `summary_${Date.now()}`,
      chat_id: currentChat.id,
      files,
      warnings,
      last_updated: new Date().toISOString(),
      auto_generated: true,
    };
    set({ codeSummary: newSummary });
  },
  
  addFileToSummary: (file) => {
    const { currentChat, codeSummary } = get();
    console.log('[addFileToSummary] Called with file:', file.path, 'currentChat:', currentChat?.id);
    if (!currentChat) {
      console.log('[addFileToSummary] No currentChat, skipping');
      return;
    }
    
    const existingFiles = codeSummary?.files || [];
    // Update or add file
    const fileIndex = existingFiles.findIndex(f => f.path === file.path);
    const updatedFiles = fileIndex >= 0
      ? existingFiles.map((f, i) => i === fileIndex ? file : f)
      : [...existingFiles, file];
    
    const newSummary: CodeSummary = {
      id: codeSummary?.id || `summary_${Date.now()}`,
      chat_id: currentChat.id,
      files: updatedFiles,
      warnings: codeSummary?.warnings || [],
      last_updated: new Date().toISOString(),
      auto_generated: true,
    };
    console.log('[addFileToSummary] Setting new summary with', updatedFiles.length, 'files');
    set({ codeSummary: newSummary });
  },
  
  addWarning: (warning) => {
    const { currentChat, codeSummary } = get();
    if (!currentChat) return;
    
    const newSummary: CodeSummary = {
      id: codeSummary?.id || `summary_${Date.now()}`,
      chat_id: currentChat.id,
      files: codeSummary?.files || [],
      warnings: [...(codeSummary?.warnings || []), warning],
      last_updated: new Date().toISOString(),
      auto_generated: true,
    };
    set({ codeSummary: newSummary });
  },
  
  clearSummary: () => set({ codeSummary: null }),
  
  fetchCodeSummary: async (chatId) => {
    try {
      const response = await api.get(`/chats/${chatId}/summary`);
      if (response.data) {
        set({ codeSummary: response.data });
      }
    } catch (err) {
      // Summary might not exist yet, that's OK
      console.log('No existing summary for chat');
    }
  },
  
  saveCodeSummary: async () => {
    const { currentChat, codeSummary } = get();
    if (!currentChat || !codeSummary) return;
    
    try {
      await api.put(`/chats/${currentChat.id}/summary`, codeSummary);
    } catch (err) {
      console.error('Failed to save code summary:', err);
    }
  },
  
  setZipUploadResult: (result) => {
    set({ zipUploadResult: result });
    if (result?.artifacts) {
      set({ uploadedArtifacts: result.artifacts });
    }
  },
  
  fetchUploadedData: async (chatId) => {
    try {
      const response = await api.get(`/chats/${chatId}/uploaded-files`);
      const { artifacts, archive } = response.data;
      
      if (artifacts && artifacts.length > 0) {
        set({ uploadedArtifacts: artifacts });
        
        // Reconstruct zipUploadResult from archive info
        if (archive) {
          const zipResult: Partial<import('../types').ZipUploadResult> = {
            filename: archive.filename,
            total_files: archive.total_files,
            total_size: archive.total_size,
            languages: archive.languages || {},
            artifacts: artifacts,
            llm_manifest: archive.llm_manifest,
            summary: archive.summary || '',  // Human-readable summary with signatures
            // Reconstruct signature_index from artifacts
            signature_index: artifacts.reduce((acc: Record<string, import('../types').CodeSignature[]>, art: Artifact) => {
              if (art.signatures && art.signatures.length > 0 && art.filename) {
                acc[art.filename] = art.signatures;
              }
              return acc;
            }, {}),
          };
          set({ 
            zipUploadResult: zipResult,
            zipContext: archive.llm_manifest || null,
          });
        }
        
        // Merge uploaded artifacts with message-extracted artifacts
        set((state) => ({
          artifacts: [...state.artifacts.filter(a => a.source !== 'upload'), ...artifacts],
        }));
      }
    } catch (err) {
      // No uploaded files yet, that's OK
      console.log('No uploaded files for chat');
    }
  },
  
  fetchChats: async () => {
    set({ isLoadingChats: true, error: null });
    try {
      const response = await api.get('/chats');
      // Backend returns { chats: [...], total, page, page_size }
      const chats = Array.isArray(response.data?.chats) ? response.data.chats : [];
      set({ chats, isLoadingChats: false });
    } catch (err) {
      set({ error: 'Failed to fetch chats', isLoadingChats: false, chats: [] });
      console.error('Failed to fetch chats:', err);
    }
  },
  
  createChat: async (model?: string, systemPrompt?: string) => {
    try {
      // Use provided model, or selected model from store, or let backend use default
      const modelToUse = model || useModelsStore.getState().selectedModel || undefined;
      
      // Check if this is an assistant model (gpt: prefix)
      if (modelToUse && modelToUse.startsWith('gpt:')) {
        const assistantId = modelToUse.substring(4);
        
        // Start a conversation with the assistant
        const response = await api.post(`/assistants/${assistantId}/start`);
        const { chat_id } = response.data;
        
        // Fetch the created chat
        const chatResponse = await api.get(`/chats/${chat_id}`);
        const newChat = chatResponse.data;
        
        set((state) => ({ chats: [newChat, ...state.chats], currentChat: newChat }));
        return newChat;
      }
      
      // Regular chat creation
      const response = await api.post('/chats', {
        model: modelToUse,
        system_prompt: systemPrompt,
      });
      const newChat = response.data;
      set((state) => ({ chats: [newChat, ...state.chats], currentChat: newChat }));
      return newChat;
    } catch (err) {
      console.error('Failed to create chat:', err);
      throw err;
    }
  },
  
  setCurrentChat: (chat) => {
    set({ currentChat: chat, messages: [], streamingContent: '', artifacts: [], selectedArtifact: null, codeSummary: null, showSummary: false });
    if (chat) {
      get().fetchMessages(chat.id);
      get().fetchCodeSummary(chat.id);
    }
  },
  
  deleteChat: async (chatId) => {
    try {
      await api.delete(`/chats/${chatId}`);
      set((state) => ({
        chats: state.chats.filter((c) => c.id !== chatId),
        currentChat: state.currentChat?.id === chatId ? null : state.currentChat,
        messages: state.currentChat?.id === chatId ? [] : state.messages,
      }));
    } catch (err) {
      console.error('Failed to delete chat:', err);
      throw err;
    }
  },
  
  deleteAllChats: async () => {
    try {
      await api.delete('/chats');
      set({ chats: [], currentChat: null, messages: [], artifacts: [] });
    } catch (err) {
      console.error('Failed to delete all chats:', err);
      throw err;
    }
  },
  
  updateChatTitle: async (chatId, title) => {
    try {
      await api.patch(`/chats/${chatId}`, { title });
      set((state) => ({
        chats: state.chats.map((c) =>
          c.id === chatId ? { ...c, title } : c
        ),
        currentChat:
          state.currentChat?.id === chatId
            ? { ...state.currentChat, title }
            : state.currentChat,
      }));
    } catch (err) {
      console.error('Failed to update chat title:', err);
      throw err;
    }
  },
  
  updateChatLocally: (chatId, updates) => {
    set((state) => ({
      chats: state.chats.map((c) =>
        c.id === chatId ? { ...c, ...updates } : c
      ),
      currentChat:
        state.currentChat?.id === chatId
          ? { ...state.currentChat, ...updates }
          : state.currentChat,
    }));
  },
  
  fetchMessages: async (chatId) => {
    set({ isLoadingMessages: true, error: null });
    try {
      const response = await api.get(`/chats/${chatId}/messages`);
      // Backend returns array directly
      const messages = Array.isArray(response.data) ? response.data : [];
      
      // Extract artifacts from all messages
      const allArtifacts: Artifact[] = [];
      for (const msg of messages) {
        if (msg.artifacts) {
          allArtifacts.push(...msg.artifacts);
        }
        const { artifacts } = extractArtifacts(msg.content);
        allArtifacts.push(...artifacts);
      }
      
      set({ messages, isLoadingMessages: false, artifacts: allArtifacts });
      
      // Also fetch uploaded artifacts and code summary
      const { fetchUploadedData, fetchCodeSummary } = get();
      await Promise.all([
        fetchUploadedData(chatId),
        fetchCodeSummary(chatId),
      ]);
    } catch (err) {
      set({ error: 'Failed to fetch messages', isLoadingMessages: false, messages: [] });
      console.error('Failed to fetch messages:', err);
    }
  },
  
  addMessage: (message) => {
    // Extract artifacts from new message
    const { artifacts: newArtifacts } = extractArtifacts(message.content);
    const allNewArtifacts = [...(message.artifacts || []), ...newArtifacts];
    
    console.log(`[addMessage] Adding message ${message.id}, role=${message.role}`);
    console.log(`[addMessage] Message has ${message.artifacts?.length || 0} direct artifacts, extracted ${newArtifacts.length} from content`);
    
    set((state) => {
      console.log(`[addMessage] Current artifacts count: ${state.artifacts.length}, adding ${allNewArtifacts.length}`);
      return {
        messages: [...state.messages, message],
        artifacts: [...state.artifacts, ...allNewArtifacts],
      };
    });
  },
  
  clearMessages: () => {
    set({ messages: [], artifacts: [] });
  },
  
  updateMessage: (messageId, updates) => {
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
  
  // Replace a temp message ID with the real server ID (used when message_saved comes back)
  replaceMessageId: (tempId, realId, parentId) => {
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === tempId ? { ...m, id: realId, parent_id: parentId } : m
      ),
    }));
  },
  
  // Retry creates a branch if there are messages after this one
  retryMessage: async (messageId) => {
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
                current_branch: branches.length, // New branch will be index branches.length
                content: '', // Will be filled by streaming
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
    
    // The actual regeneration will be triggered by WebSocket context
    // Store the message ID being retried so WebSocket knows
    (window as any).__retryMessageId = messageId;
    (window as any).__retryUserContent = userMessage.content;
  },
  
  switchBranch: (messageId, branchIndex) => {
    const { messages } = get();
    const message = messages.find((m) => m.id === messageId);
    if (!message || !message.branches) return;
    
    const totalBranches = message.branches.length + 1; // +1 for current
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
    
    // Store current as the "live" branch if not already in branches
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
              // Update the branch we're leaving with current content
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
  
  setSelectedArtifact: (artifact) => {
    set({ selectedArtifact: artifact, showArtifacts: artifact !== null });
  },
  
  setShowArtifacts: (show) => {
    set({ showArtifacts: show, selectedArtifact: show ? get().selectedArtifact : null });
  },
  
  collectAllArtifacts: () => {
    const { messages } = get();
    const artifactMap = collectChatArtifacts(messages);
    return Array.from(artifactMap.values());
  },
  
  updateStreamingArtifacts: (content) => {
    // Extract artifacts from streaming content in real-time
    const { artifacts: streamingArtifacts } = extractArtifacts(content);
    set({ streamingArtifacts });
  },
  
  setStreamingContent: (content) => {
    set({ streamingContent: content });
  },
  
  appendStreamingContent: (chunk) => {
    set((state) => ({ streamingContent: state.streamingContent + chunk }));
  },
  
  setStreamingToolCall: (toolCall) => {
    set({ streamingToolCall: toolCall });
  },
  
  clearStreaming: () => {
    set({ streamingContent: '', streamingToolCall: null, streamingArtifacts: [] });
  },
  
  setIsSending: (isSending) => {
    set({ isSending });
  },
  
  setError: (error) => {
    set({ error });
  },
}));
