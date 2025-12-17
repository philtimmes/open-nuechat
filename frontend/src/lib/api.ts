import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios';

// Use relative path - everything served from same origin
const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - add auth token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Get token from localStorage (set by authStore)
    const authData = localStorage.getItem('nexus-auth');
    if (authData) {
      try {
        const { state } = JSON.parse(authData);
        if (state?.accessToken) {
          config.headers.Authorization = `Bearer ${state.accessToken}`;
        }
      } catch {
        // Ignore parse errors
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config;
    
    // If 401 and we haven't already tried refreshing
    if (error.response?.status === 401 && originalRequest && !('_retry' in originalRequest)) {
      (originalRequest as InternalAxiosRequestConfig & { _retry: boolean })._retry = true;
      
      const authData = localStorage.getItem('nexus-auth');
      if (authData) {
        try {
          const { state } = JSON.parse(authData);
          if (state?.refreshToken) {
            // Try to refresh
            const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
              refresh_token: state.refreshToken,
            });
            
            const { access_token, refresh_token } = response.data;
            
            // Update stored tokens
            const newState = {
              ...state,
              accessToken: access_token,
              refreshToken: refresh_token || state.refreshToken,
            };
            localStorage.setItem('nexus-auth', JSON.stringify({ state: newState }));
            
            // Retry original request
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
            return api(originalRequest);
          }
        } catch {
          // Refresh failed - clear auth
          localStorage.removeItem('nexus-auth');
          window.location.href = '/login';
        }
      }
    }
    
    return Promise.reject(error);
  }
);

export default api;

// Helper functions for common operations
export const authApi = {
  login: (email: string, password: string) =>
    api.post('/auth/login', { email, password }),
  
  register: (email: string, username: string, password: string) =>
    api.post('/auth/register', { email, username, password }),
  
  me: () => api.get('/auth/me'),
  
  refresh: (refreshToken: string) =>
    api.post('/auth/refresh', { refresh_token: refreshToken }),
  
  oauthGoogle: () => {
    window.location.href = `${API_BASE_URL}/auth/oauth/google`;
  },
  
  oauthGitHub: () => {
    window.location.href = `${API_BASE_URL}/auth/oauth/github`;
  },
};

export const chatApi = {
  list: () => api.get('/chats'),
  
  create: (model?: string, systemPrompt?: string) =>
    api.post('/chats', { model, system_prompt: systemPrompt }),
  
  get: (chatId: string) => api.get(`/chats/${chatId}`),
  
  update: (chatId: string, data: { title?: string; system_prompt?: string; model?: string }) =>
    api.patch(`/chats/${chatId}`, data),
  
  // parent_id: message with multiple children (branch point)
  // child_id: the selected child message
  updateSelectedVersion: (chatId: string, parentId: string, childId: string) =>
    api.patch(`/chats/${chatId}/selected-version`, null, {
      params: { parent_id: parentId, child_id: childId }
    }),
  
  share: (chatId: string, anonymous: boolean = false) => api.post(`/chats/${chatId}/share`, { anonymous }),
  
  delete: (chatId: string) => api.delete(`/chats/${chatId}`),
  
  deleteAll: () => api.delete('/chats'),
  
  messages: (chatId: string) => api.get(`/chats/${chatId}/messages`),
  
  sendMessage: (chatId: string, content: string, attachments?: unknown[]) =>
    api.post(`/chats/${chatId}/messages`, { content, attachments }),
  
  // Edit a message - creates a new branch
  editMessage: (chatId: string, messageId: string, content: string, regenerateResponse: boolean = true) =>
    api.patch(`/chats/${chatId}/messages/${messageId}`, { content, regenerate_response: regenerateResponse }),
  
  // Delete a message and all its descendants
  deleteMessage: (chatId: string, messageId: string) =>
    api.delete(`/chats/${chatId}/messages/${messageId}`),
  
  uploadZip: (chatId: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/chats/${chatId}/upload-zip`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  getZipFile: (chatId: string, path: string) =>
    api.get(`/chats/${chatId}/zip-file`, { params: { path } }),
  
  getUploadedFiles: (chatId: string) =>
    api.get(`/chats/${chatId}/uploaded-files`),
};

export const billingApi = {
  usage: () => api.get('/billing/usage'),
  
  history: (days?: number) =>
    api.get('/billing/usage/history', { params: { days } }),
  
  byChat: () => api.get('/billing/usage/by-chat'),
  
  checkLimit: () => api.get('/billing/check-limit'),
  
  invoice: (year: number, month: number) =>
    api.get(`/billing/invoice/${year}/${month}`),
};

export const documentApi = {
  list: () => api.get('/documents'),
  
  upload: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/documents', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  get: (docId: string) => api.get(`/documents/${docId}`),
  
  delete: (docId: string) => api.delete(`/documents/${docId}`),
  
  search: (query: string, topK?: number) =>
    api.post('/documents/search', { query, top_k: topK }),
};

export const themeApi = {
  list: () => api.get('/themes'),
  
  system: () => api.get('/themes/system'),
  
  get: (themeId: string) => api.get(`/themes/${themeId}`),
  
  create: (data: { name: string; colors: unknown; fonts?: unknown; is_public?: boolean }) =>
    api.post('/themes', data),
  
  delete: (themeId: string) => api.delete(`/themes/${themeId}`),
  
  apply: (themeId: string) => api.post(`/themes/apply/${themeId}`),
};
