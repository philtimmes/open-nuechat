/**
 * Chat slice - CRUD operations for chats
 */
import type { Chat } from '../../types';
import type { ChatSlice, SliceCreator } from './types';
import api from '../../lib/api';
import { useModelsStore } from '../modelsStore';

export const createChatSlice: SliceCreator<ChatSlice> = (set, get) => ({
  chats: [],
  currentChat: null,
  isLoadingChats: false,
  hasMoreChats: true,
  chatPage: 1,
  chatSearchQuery: '',

  // Add imported chats to the list without replacing existing ones
  addImportedChats: (importedChats: Chat[]) => {
    set((state) => {
      // Merge imported chats with existing ones, avoiding duplicates
      const existingIds = new Set(state.chats.map(c => c.id));
      const newChats = importedChats.filter(c => !existingIds.has(c.id));
      
      // Combine and sort by updated_at desc
      const allChats = [...newChats, ...state.chats].sort(
        (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
      );
      
      return { chats: allChats };
    });
  },

  fetchChats: async (loadMore = false, search?: string) => {
    const currentState = get();
    
    // If search changed, reset pagination
    const searchChanged = search !== undefined && search !== currentState.chatSearchQuery;
    const page = (loadMore && !searchChanged) ? currentState.chatPage : 1;
    
    console.log(`[fetchChats] loadMore=${loadMore}, page=${page}, hasMoreChats=${currentState.hasMoreChats}, isLoading=${currentState.isLoadingChats}`);
    
    // Don't fetch if already loading or no more chats (when loading more without search change)
    if (currentState.isLoadingChats || (loadMore && !searchChanged && !currentState.hasMoreChats)) {
      console.log(`[fetchChats] Skipping - isLoading=${currentState.isLoadingChats}, hasMore=${currentState.hasMoreChats}`);
      return;
    }
    
    // Update search query if provided
    if (search !== undefined) {
      set({ chatSearchQuery: search });
    }
    
    const searchQuery = search !== undefined ? search : currentState.chatSearchQuery;
    
    set({ isLoadingChats: true, error: null });
    try {
      const params: Record<string, unknown> = { page, page_size: 50 };
      if (searchQuery) {
        params.search = searchQuery;
      }
      
      const response = await api.get('/chats', { params });
      // Backend returns { chats: [...], total, page, page_size }
      const newChats = Array.isArray(response.data?.chats) ? response.data.chats : [];
      const total = response.data?.total || 0;
      const pageSize = response.data?.page_size || 50;
      
      // Calculate if there are more chats to load
      const hasMore = page * pageSize < total;
      
      console.log(`[fetchChats] Got ${newChats.length} chats, total=${total}, page=${page}, pageSize=${pageSize}, hasMore=${hasMore}`);
      
      set((state) => ({ 
        chats: (loadMore && !searchChanged) ? [...state.chats, ...newChats] : newChats,
        isLoadingChats: false,
        hasMoreChats: hasMore,
        chatPage: page + 1,
      }));
    } catch (err) {
      set({ error: 'Failed to fetch chats', isLoadingChats: false, chats: loadMore ? get().chats : [] });
      console.error('Failed to fetch chats:', err);
    }
  },

  createChat: async (model?: string, systemPrompt?: string, preserveArtifacts = false) => {
    try {
      // Use provided model, or selected model from store, or let backend use default
      const modelToUse = model || useModelsStore.getState().selectedModel || undefined;
      
      console.log('[createChat] Called with:', { 
        providedModel: model, 
        selectedModel: useModelsStore.getState().selectedModel,
        finalModelToUse: modelToUse,
        isAssistant: modelToUse?.startsWith('gpt:'),
        preserveArtifacts,
      });
      
      // Only preserve uploaded artifacts if explicitly requested (e.g., user uploaded files before chat existed)
      // Do NOT preserve when clicking "New Chat" from an existing chat
      const preservedArtifacts = preserveArtifacts ? get().uploadedArtifacts : [];
      const preservedZipResult = preserveArtifacts ? get().zipUploadResult : null;
      const preservedZipContext = preserveArtifacts ? get().zipContext : null;
      
      // Check if this is an assistant model (gpt: prefix)
      if (modelToUse && modelToUse.startsWith('gpt:')) {
        const assistantId = modelToUse.substring(4);
        console.log('[createChat] Starting assistant conversation:', assistantId);
        
        // Start a conversation with the assistant
        const response = await api.post(`/assistants/${assistantId}/start`);
        const { chat_id } = response.data;
        console.log('[createChat] Assistant chat created:', chat_id);
        
        // Fetch the created chat
        const chatResponse = await api.get(`/chats/${chat_id}`);
        const newChat = chatResponse.data;
        
        set((state) => ({ 
          chats: [newChat, ...state.chats], 
          currentChat: newChat,
          messages: [],
          artifacts: [],
          selectedArtifact: null,
          showArtifacts: false,
          generatedImages: {},
          uploadedArtifacts: preservedArtifacts,
          zipUploadResult: preservedZipResult,
          zipContext: preservedZipContext,
          codeSummary: null,
          showSummary: false,
        }));
        return newChat;
      }
      
      // Regular chat creation
      const response = await api.post('/chats', {
        model: modelToUse,
        system_prompt: systemPrompt,
      });
      const newChat = response.data;
      
      set((state) => ({
        chats: [newChat, ...state.chats],
        currentChat: newChat,
        messages: [],
        artifacts: [],
        selectedArtifact: null,
        showArtifacts: false,
        generatedImages: {},
        uploadedArtifacts: preservedArtifacts,
        zipUploadResult: preservedZipResult,
        zipContext: preservedZipContext,
        codeSummary: null,
        showSummary: false,
      }));
      
      return newChat;
    } catch (err) {
      console.error('Failed to create chat:', err);
      throw err;
    }
  },

  setCurrentChat: (chat) => {
    const { currentChat: existingChat, isSending } = get();
    
    // If switching to the same chat, don't clear state (prevents losing messages during streaming)
    if (chat && existingChat && chat.id === existingChat.id) {
      // Just update the chat metadata if needed, but don't clear messages
      set({ currentChat: chat });
      return;
    }
    
    // If currently sending, don't clear state (prevents losing user message with attachments)
    if (isSending && chat && existingChat && chat.id === existingChat.id) {
      return;
    }
    
    // Clear ALL state when switching chats to prevent cross-chat contamination
    set({ 
      currentChat: chat, 
      messages: [], 
      streamingContent: '', 
      streamingToolCall: null,
      streamingArtifacts: [],
      isSending: false,
      artifacts: [], 
      selectedArtifact: null, 
      showArtifacts: false,
      codeSummary: null, 
      showSummary: false,
      uploadedArtifacts: [],
      zipUploadResult: null,
      zipContext: null,
      generatedImages: {},
    });
    if (chat) {
      get().fetchMessages(chat.id);
      get().fetchCodeSummary(chat.id);
    }
  },

  deleteChat: async (chatId: string) => {
    try {
      await api.delete(`/chats/${chatId}`);
      set((state) => ({
        chats: state.chats.filter((c) => c.id !== chatId),
        currentChat: state.currentChat?.id === chatId ? null : state.currentChat,
        messages: state.currentChat?.id === chatId ? [] : state.messages,
        artifacts: state.currentChat?.id === chatId ? [] : state.artifacts,
        showArtifacts: state.currentChat?.id === chatId ? false : state.showArtifacts,
        generatedImages: state.currentChat?.id === chatId ? {} : state.generatedImages,
      }));
    } catch (err) {
      console.error('Failed to delete chat:', err);
      throw err;
    }
  },

  deleteAllChats: async () => {
    try {
      await api.delete('/chats');
      set({ 
        chats: [], 
        currentChat: null, 
        messages: [], 
        artifacts: [],
        selectedArtifact: null,
        showArtifacts: false,
        uploadedArtifacts: [],
        zipUploadResult: null,
        zipContext: null,
        codeSummary: null,
        showSummary: false,
        generatedImages: {},
      });
    } catch (err) {
      console.error('Failed to delete all chats:', err);
      throw err;
    }
  },

  updateChatTitle: async (chatId: string, title: string) => {
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

  updateChatLocally: (chatId: string, updates: Partial<Chat>) => {
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
});
