/**
 * Models Store
 * 
 * Fetches and manages available LLM models from the backend.
 * The backend proxies to the configured LLM server (Ollama, vLLM, etc.)
 * Also manages subscribed Custom GPT assistants.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Model {
  id: string;
  name?: string;
  owned_by?: string;
  created?: number;
}

export interface SubscribedAssistant {
  id: string;  // Format: "gpt:{assistant_id}"
  name: string;
  type: 'assistant';
  assistant_id: string;
  icon: string;
  color: string;
}

interface ModelsState {
  models: Model[];
  subscribedAssistants: SubscribedAssistant[];
  defaultModel: string;
  selectedModel: string;
  isLoading: boolean;
  error: string | null;
  lastFetched: number | null;
  
  // Actions
  fetchModels: (force?: boolean) => Promise<void>;
  setSelectedModel: (modelId: string) => void;
  getDisplayName: (modelId: string) => string;
  isAssistantModel: (modelId: string) => boolean;
  getAssistantId: (modelId: string) => string | null;
  addSubscribedAssistant: (assistant: SubscribedAssistant) => void;
  removeSubscribedAssistant: (assistantId: string) => void;
}

export const useModelsStore = create<ModelsState>()(
  persist(
    (set, get) => ({
      models: [],
      subscribedAssistants: [],
      defaultModel: 'default',
      selectedModel: '',
      isLoading: false,
      error: null,
      lastFetched: null,
      
      fetchModels: async (force = false) => {
        // Don't refetch if we fetched recently (within 5 minutes) unless forced
        const now = Date.now();
        const lastFetched = get().lastFetched;
        if (!force && lastFetched && now - lastFetched < 5 * 60 * 1000 && get().models.length > 0) {
          return;
        }
        
        set({ isLoading: true, error: null });
        
        try {
          // Include auth header for subscribed assistants
          const authData = localStorage.getItem('nexus-auth');
          const headers: HeadersInit = {};
          if (authData) {
            try {
              const { state } = JSON.parse(authData);
              if (state?.accessToken) {
                headers['Authorization'] = `Bearer ${state.accessToken}`;
              }
            } catch {
              // Ignore parse errors
            }
          }
          
          const response = await fetch('/api/models', { headers });
          
          if (!response.ok) {
            throw new Error('Failed to fetch models');
          }
          
          const data = await response.json();
          
          // Defensive: ensure models is an array
          const models: Model[] = Array.isArray(data.models) ? data.models : [];
          const defaultModel = data.default_model || 'default';
          const subscribedAssistants: SubscribedAssistant[] = Array.isArray(data.subscribed_assistants) 
            ? data.subscribed_assistants 
            : [];
          
          set({
            models,
            subscribedAssistants,
            defaultModel,
            selectedModel: get().selectedModel || defaultModel,
            isLoading: false,
            lastFetched: now,
          });
        } catch (error) {
          console.error('Failed to fetch models:', error);
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Failed to fetch models',
          });
        }
      },
      
      setSelectedModel: (modelId: string) => {
        set({ selectedModel: modelId });
      },
      
      getDisplayName: (modelId: string) => {
        // Handle undefined/null modelId
        if (!modelId) {
          return 'Unknown Model';
        }
        
        const { models, subscribedAssistants } = get();
        
        // Check if it's a subscribed assistant
        const assistant = subscribedAssistants.find(a => a.id === modelId);
        if (assistant) {
          return assistant.name;
        }
        
        const model = models.find(m => m.id === modelId);
        
        if (model?.name) {
          return model.name;
        }
        
        // Clean up common model ID patterns for display
        let displayName = modelId;
        
        // Remove version suffixes like :latest, :7b, etc.
        displayName = displayName.replace(/:latest$/, '');
        
        // Capitalize first letter of each word
        displayName = displayName
          .split(/[-_]/)
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
        
        return displayName;
      },
      
      isAssistantModel: (modelId: string) => {
        return modelId.startsWith('gpt:');
      },
      
      getAssistantId: (modelId: string) => {
        if (modelId.startsWith('gpt:')) {
          return modelId.substring(4);
        }
        return null;
      },
      
      addSubscribedAssistant: (assistant: SubscribedAssistant) => {
        const { subscribedAssistants } = get();
        if (!subscribedAssistants.find(a => a.id === assistant.id)) {
          set({ subscribedAssistants: [...subscribedAssistants, assistant] });
        }
      },
      
      removeSubscribedAssistant: (assistantId: string) => {
        const { subscribedAssistants } = get();
        set({ 
          subscribedAssistants: subscribedAssistants.filter(a => a.assistant_id !== assistantId) 
        });
      },
    }),
    {
      name: 'nexus-models',
      partialize: (state) => ({
        selectedModel: state.selectedModel,
        lastFetched: state.lastFetched,
      }),
    }
  )
);

// Hook for easy access to current model
export function useCurrentModel() {
  return useModelsStore((state) => state.selectedModel || state.defaultModel);
}

// Hook for model display name
export function useModelDisplayName(modelId: string) {
  return useModelsStore((state) => state.getDisplayName(modelId));
}
