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
  chatSortBy: 'modified',
  chatGroupCounts: null,
  sidebarReloadTrigger: 0,
  
  // Per-group pagination state
  groupPages: {},
  groupHasMore: {},
  groupLoading: {},
  
  triggerSidebarReload: () => {
    set((state) => ({ sidebarReloadTrigger: state.sidebarReloadTrigger + 1 }));
  },

  fetchChats: async (loadMore = false, search?: string, sortBy?: string) => {
    const currentState = get();
    const currentSortBy = sortBy ?? currentState.chatSortBy;
    const searchQuery = search ?? currentState.chatSearchQuery;
    
    // Track if sort actually changed
    const sortChanged = sortBy !== undefined && sortBy !== currentState.chatSortBy;
    
    // Update sort/search in state if provided
    if (sortChanged) {
      set({ chatSortBy: sortBy as 'modified' | 'created' | 'alphabetical' | 'source' });
    }
    if (search !== undefined && search !== currentState.chatSearchQuery) {
      set({ chatSearchQuery: search });
    }
    
    // Don't fetch if already loading
    if (currentState.isLoadingChats) return;
    
    // For loadMore, check if there's more to load
    if (loadMore && !currentState.hasMoreChats) return;
    
    const page = loadMore ? currentState.chatPage : 1;
    
    set({ isLoadingChats: true, error: null });
    try {
      const params: Record<string, unknown> = { page, page_size: 50, sort_by: currentSortBy };
      if (searchQuery) params.search = searchQuery;
      
      const response = await api.get('/chats', { params });
      const newChats = Array.isArray(response.data?.chats) ? response.data.chats : [];
      const total = response.data?.total || 0;
      const pageSize = response.data?.page_size || 50;
      const groupCounts = response.data?.group_counts || null;
      const hasMore = page * pageSize < total;
      
      set((state) => {
        const isGroupedView = ['source', 'modified', 'created'].includes(currentSortBy);
        
        let chatsValue;
        if (isGroupedView) {
          // For grouped views: only clear chats when sort explicitly changes
          // Otherwise keep existing chats (preserves newly created chats)
          chatsValue = sortChanged ? [] : state.chats;
        } else {
          // For alphabetical: replace/append as normal
          chatsValue = loadMore ? [...state.chats, ...newChats] : newChats;
        }
        
        return { 
          chats: chatsValue,
          isLoadingChats: false,
          hasMoreChats: hasMore,
          chatPage: page + 1,
          chatGroupCounts: groupCounts,
          // Reset group pagination only on sort change
          groupPages: sortChanged ? {} : state.groupPages,
          groupHasMore: sortChanged ? {} : state.groupHasMore,
        };
      });
    } catch (err) {
      set({ error: 'Failed to fetch chats', isLoadingChats: false });
      console.error('Failed to fetch chats:', err);
    }
  },

  fetchGroupChats: async (groupType: 'source' | 'period', groupName: string, loadMore = false, dateField?: 'updated_at' | 'created_at') => {
    const currentState = get();
    const groupKey = `${groupType}:${groupName}`;
    
    // Check if already loading this group
    if (currentState.groupLoading[groupKey]) {
      return;
    }
    
    // Check if no more to load
    const currentPage = currentState.groupPages[groupKey] || 1;
    if (loadMore && currentState.groupHasMore[groupKey] === false) {
      return;
    }
    
    const page = loadMore ? currentPage : 1;
    
    set((state) => ({
      groupLoading: { ...state.groupLoading, [groupKey]: true },
    }));
    
    try {
      const params: Record<string, unknown> = { page, page_size: 50 };
      if (groupType === 'period' && dateField) {
        params.date_field = dateField;
      }
      
      const response = await api.get(`/chats/group/${groupType}/${encodeURIComponent(groupName)}`, { params });
      
      const rawChats = Array.isArray(response.data?.chats) ? response.data.chats : [];
      const total = response.data?.total || 0;
      const pageSize = response.data?.page_size || 50;
      const hasMore = page * pageSize < total;
      const returnedGroupName = response.data?.group_name || groupName;
      
      // Tag each chat with its assigned period so frontend knows where to display it
      const newChats = rawChats.map((c: Chat) => ({
        ...c,
        _assignedPeriod: groupType === 'period' ? returnedGroupName : undefined,
      }));
      
      set((state) => {
        // Merge new chats, avoiding duplicates
        const existingIds = new Set(state.chats.map(c => c.id));
        const uniqueNewChats = newChats.filter((c: Chat) => !existingIds.has(c.id));
        
        return {
          chats: [...state.chats, ...uniqueNewChats],
          groupPages: { ...state.groupPages, [groupKey]: page + 1 },
          groupHasMore: { ...state.groupHasMore, [groupKey]: hasMore },
          groupLoading: { ...state.groupLoading, [groupKey]: false },
        };
      });
    } catch (err) {
      console.error(`Failed to fetch group chats (${groupKey}):`, err);
      set((state) => ({
        groupLoading: { ...state.groupLoading, [groupKey]: false },
      }));
    }
  },

  setChatSortBy: (sortBy) => {
    const currentState = get();
    if (sortBy !== currentState.chatSortBy) {
      // Reset and fetch with new sort
      set({ chatSortBy: sortBy, chats: [], chatPage: 1, hasMoreChats: true, groupPages: {}, groupHasMore: {} });
      get().fetchChats(false, currentState.chatSearchQuery, sortBy);
    }
  },

  createChat: async (model?: string, systemPrompt?: string) => {
    try {
      // Use provided model, or selected model from store, or let backend use default
      const modelToUse = model || useModelsStore.getState().selectedModel || undefined;
      
      console.log('[createChat] Called with:', { 
        providedModel: model, 
        selectedModel: useModelsStore.getState().selectedModel,
        finalModelToUse: modelToUse,
        isAssistant: modelToUse?.startsWith('gpt:'),
      });
      
      // Preserve uploaded artifacts - they were uploaded before the chat was created
      const { uploadedArtifacts: preservedArtifacts, zipUploadResult: preservedZipResult, zipContext: preservedZipContext } = get();
      
      // Check if this is an assistant model (gpt: prefix)
      if (modelToUse && modelToUse.startsWith('gpt:')) {
        const assistantId = modelToUse.substring(4);
        console.log('[createChat] Starting assistant conversation:', assistantId);
        
        // 1. Create chat on backend FIRST
        const response = await api.post(`/assistants/${assistantId}/start`);
        const { chat_id } = response.data;
        console.log('[createChat] Assistant chat created:', chat_id);
        
        // 2. Fetch the created chat to confirm it exists
        const chatResponse = await api.get(`/chats/${chat_id}`);
        const newChat = chatResponse.data;
        const now = new Date().toISOString();
        
        // 3. Update local state
        set((state) => ({ 
          chats: [{ ...newChat, _assignedPeriod: 'Today', updated_at: now }, ...state.chats], 
          currentChat: { ...newChat, updated_at: now },
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
          chatGroupCounts: state.chatGroupCounts ? {
            ...state.chatGroupCounts,
            'Today': (state.chatGroupCounts['Today'] || 0) + 1
          } : null,
        }));
        
        // 4. Reload sidebar - chat is confirmed on backend
        get().triggerSidebarReload();
        return newChat;
      }
      
      // Regular chat creation
      // 1. Create chat on backend FIRST - wait for response
      const response = await api.post('/chats', {
        model: modelToUse,
        system_prompt: systemPrompt,
      });
      const newChat = response.data;
      const now = new Date().toISOString();
      
      // 2. Backend confirmed - update local state
      set((state) => ({
        chats: [{ ...newChat, _assignedPeriod: 'Today', updated_at: now }, ...state.chats],
        currentChat: { ...newChat, updated_at: now },
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
        chatGroupCounts: state.chatGroupCounts ? {
          ...state.chatGroupCounts,
          'Today': (state.chatGroupCounts['Today'] || 0) + 1
        } : null,
      }));
      
      // 3. Reload sidebar - chat is confirmed on backend
      get().triggerSidebarReload();
      return newChat;
    } catch (err) {
      console.error('Failed to create chat:', err);
      throw err;
    }
  },

  setCurrentChat: (chat) => {
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
      // Get the chat's period before deleting
      const chatToDelete = get().chats.find(c => c.id === chatId);
      const period = (chatToDelete as Chat & { _assignedPeriod?: string })?._assignedPeriod;
      
      await api.delete(`/chats/${chatId}`);
      set((state) => ({
        chats: state.chats.filter((c) => c.id !== chatId),
        currentChat: state.currentChat?.id === chatId ? null : state.currentChat,
        messages: state.currentChat?.id === chatId ? [] : state.messages,
        artifacts: state.currentChat?.id === chatId ? [] : state.artifacts,
        showArtifacts: state.currentChat?.id === chatId ? false : state.showArtifacts,
        generatedImages: state.currentChat?.id === chatId ? {} : state.generatedImages,
        // Update group counts - decrement the period
        chatGroupCounts: (state.chatGroupCounts && period) ? {
          ...state.chatGroupCounts,
          [period]: Math.max(0, (state.chatGroupCounts[period] || 0) - 1)
        } : state.chatGroupCounts,
      }));
      get().triggerSidebarReload();
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
        chatGroupCounts: null,
        groupPages: {},
        groupHasMore: {},
      });
    } catch (err) {
      console.error('Failed to delete all chats:', err);
      throw err;
    }
  },

  deleteGroupChats: async (groupType: 'source' | 'period', groupName: string) => {
    try {
      const response = await api.delete(`/chats/group/${groupType}/${encodeURIComponent(groupName)}`);
      const deletedCount = response.data?.count || 0;
      
      // Remove deleted chats from state
      set((state) => {
        const isInGroup = (chat: Chat) => {
          if (groupType === 'source') {
            const chatSource = chat.source || 'native';
            const targetSource = groupName === 'Native' ? 'native' : groupName.toLowerCase();
            return chatSource === targetSource;
          } else {
            // Period-based - we need to check dates
            // For simplicity, just refetch after deletion
            return false;
          }
        };
        
        const remainingChats = state.chats.filter(c => !isInGroup(c));
        const currentChatDeleted = state.currentChat && isInGroup(state.currentChat);
        
        // Update group counts
        const newGroupCounts = state.chatGroupCounts ? { ...state.chatGroupCounts } : null;
        if (newGroupCounts && newGroupCounts[groupName]) {
          newGroupCounts[groupName] = 0;
        }
        
        return {
          chats: remainingChats,
          currentChat: currentChatDeleted ? null : state.currentChat,
          messages: currentChatDeleted ? [] : state.messages,
          artifacts: currentChatDeleted ? [] : state.artifacts,
          showArtifacts: currentChatDeleted ? false : state.showArtifacts,
          generatedImages: currentChatDeleted ? {} : state.generatedImages,
          chatGroupCounts: newGroupCounts,
        };
      });
      
      // Refetch to get accurate counts
      get().fetchChats(false, get().chatSearchQuery, get().chatSortBy);
      
      return deletedCount;
    } catch (err) {
      console.error(`Failed to delete group chats (${groupType}/${groupName}):`, err);
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
