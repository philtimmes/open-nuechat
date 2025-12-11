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
        
        set((state) => ({ 
          chats: [newChat, ...state.chats], 
          currentChat: newChat,
          messages: [],
          artifacts: [],
          selectedArtifact: null,
          uploadedArtifacts: [],
          zipUploadResult: null,
          zipContext: null,
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
        uploadedArtifacts: [],
        zipUploadResult: null,
        zipContext: null,
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
    set({ 
      currentChat: chat, 
      messages: [], 
      streamingContent: '', 
      artifacts: [], 
      selectedArtifact: null, 
      codeSummary: null, 
      showSummary: false,
      uploadedArtifacts: [],
      zipUploadResult: null,
      zipContext: null,
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
        uploadedArtifacts: [],
        zipUploadResult: null,
        zipContext: null,
        codeSummary: null,
        showSummary: false,
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
