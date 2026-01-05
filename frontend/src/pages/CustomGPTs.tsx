import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useModelsStore } from '../stores/modelsStore';
import api from '../lib/api';
import type { CustomAssistant, KnowledgeStore } from '../types';

type Tab = 'my-gpts' | 'explore';

export default function CustomGPTs() {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const { models, defaultModel, fetchModels, addSubscribedAssistant, removeSubscribedAssistant } = useModelsStore();
  const [activeTab, setActiveTab] = useState<Tab>('my-gpts');
  const [myGpts, setMyGpts] = useState<CustomAssistant[]>([]);
  const [publicGpts, setPublicGpts] = useState<CustomAssistant[]>([]);
  const [knowledgeStores, setKnowledgeStores] = useState<KnowledgeStore[]>([]);
  const [subscribedIds, setSubscribedIds] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Form state
  const [showForm, setShowForm] = useState(false);
  const [editingGpt, setEditingGpt] = useState<CustomAssistant | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    slug: '',
    tagline: '',
    description: '',
    icon: 'AI',
    color: '#6366f1',
    system_prompt: '',
    welcome_message: '',
    suggested_prompts: [''],
    model: '',
    is_public: false,
    is_discoverable: false,
    knowledge_store_ids: [] as string[],
    category: 'general',
  });
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  
  // Filter/sort state for explore tab
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'popularity' | 'updated' | 'alphabetical'>('popularity');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 12;
  
  useEffect(() => {
    fetchModels();
    fetchData();
  }, []);
  
  // Set default model when models load
  useEffect(() => {
    if (defaultModel && !formData.model) {
      setFormData(prev => ({ ...prev, model: defaultModel }));
    }
  }, [defaultModel]);
  
  const fetchData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Load all data with graceful error handling
      let myData: CustomAssistant[] = [];
      let publicData: CustomAssistant[] = [];
      let ksData: KnowledgeStore[] = [];
      
      // Load my assistants
      try {
        const myRes = await api.get('/assistants');
        myData = myRes.data || [];
      } catch (err: unknown) {
        const error = err as { response?: { data?: unknown; status?: number }; message?: string };
        console.error('Failed to load my assistants:', {
          status: error?.response?.status,
          data: error?.response?.data,
          message: error?.message,
        });
      }
      
      // Load public assistants
      try {
        const publicRes = await api.get('/assistants/explore');
        publicData = publicRes.data || [];
      } catch (err: unknown) {
        const error = err as { response?: { data?: unknown }; message?: string };
        console.warn('Failed to load public GPTs:', error?.response?.data || error?.message);
      }
      
      // Load subscribed assistants
      try {
        const subscribedRes = await api.get('/assistants/subscribed');
        const subscribed = subscribedRes.data || [];
        console.log('Loaded subscribed assistants:', subscribed);
        const ids = subscribed.map((a: CustomAssistant) => a.id);
        console.log('Subscribed IDs:', ids);
        setSubscribedIds(new Set(ids));
      } catch (err: unknown) {
        const error = err as { response?: { data?: unknown; status?: number }; message?: string };
        console.error('Failed to load subscribed assistants:', {
          status: error?.response?.status,
          data: error?.response?.data,
          message: error?.message,
        });
      }
      
      // Load knowledge stores
      try {
        const ksRes = await api.get('/knowledge-stores');
        ksData = ksRes.data || [];
      } catch (err: unknown) {
        const error = err as { response?: { data?: unknown; status?: number }; message?: string };
        console.error('Failed to load knowledge stores:', {
          status: error?.response?.status,
          data: error?.response?.data,
          message: error?.message,
        });
      }
      
      setMyGpts(myData);
      setPublicGpts(publicData);
      setKnowledgeStores(ksData);
    } catch (err) {
      console.error('Unexpected error in fetchData:', err);
      setError('Failed to load data. Please refresh the page.');
    } finally {
      setIsLoading(false);
    }
  };
  
  const generateSlug = (name: string) => {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  };
  
  const handleNameChange = (name: string) => {
    setFormData({
      ...formData,
      name,
      slug: editingGpt ? formData.slug : generateSlug(name),
    });
  };
  
  const handleSubmit = async () => {
    if (!formData.name.trim()) {
      setError('Name is required');
      return;
    }
    
    if (!formData.system_prompt.trim()) {
      setError('System prompt is required');
      return;
    }
    
    if (formData.system_prompt.trim().length < 10) {
      setError('System prompt must be at least 10 characters');
      return;
    }
    
    try {
      // Build data object with only the fields the API expects
      const data = {
        name: formData.name,
        tagline: formData.tagline || null,
        description: formData.description || null,
        icon: formData.icon,
        color: formData.color,
        system_prompt: formData.system_prompt,
        welcome_message: formData.welcome_message || null,
        suggested_prompts: formData.suggested_prompts.filter(p => p.trim()),
        model: formData.model,
        enabled_tools: [],
        knowledge_store_ids: formData.knowledge_store_ids,
        is_public: formData.is_public,
        is_discoverable: formData.is_discoverable,
        category: formData.category,
      };
      
      console.log('Submitting assistant data:', data);
      
      let assistantId: string;
      if (editingGpt) {
        await api.patch(`/assistants/${editingGpt.id}`, data);
        assistantId = editingGpt.id;
      } else {
        const res = await api.post('/assistants', data);
        assistantId = res.data.id;
      }
      
      // Upload avatar if selected
      if (avatarFile) {
        setUploadingAvatar(true);
        const avatarFormData = new FormData();
        avatarFormData.append('file', avatarFile);
        try {
          await api.post(`/assistants/${assistantId}/avatar`, avatarFormData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
        } catch (err) {
          console.error('Failed to upload avatar:', err);
          // Don't fail the whole operation for avatar upload failure
        }
        setUploadingAvatar(false);
      }
      
      await fetchData();
      resetForm();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string | Array<{ msg?: string; loc?: string[] }> } } };
      const detail = error.response?.data?.detail;
      
      // Handle Pydantic validation errors (array of error objects)
      if (Array.isArray(detail)) {
        const messages = detail.map(e => {
          const field = e.loc?.slice(-1)[0] || 'field';
          return `${field}: ${e.msg}`;
        });
        setError(messages.join(', ') || 'Validation failed');
      } else if (typeof detail === 'string') {
        setError(detail);
      } else {
        setError('Failed to save');
      }
    }
  };
  
  const handleDelete = async (id: string) => {
    if (!confirm('Delete this Custom GPT?')) return;
    
    try {
      await api.delete(`/assistants/${id}`);
      await fetchData();
    } catch (err) {
      setError('Failed to delete');
    }
  };
  
  const handleStartChat = async (gpt: CustomAssistant) => {
    try {
      // Start a conversation with the assistant
      const res = await api.post(`/assistants/${gpt.id}/start`);
      const { chat_id } = res.data;
      
      // Navigate to the new chat
      navigate(`/chat/${chat_id}`);
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to start chat');
    }
  };
  
  const handleSubscribe = async (gpt: CustomAssistant) => {
    try {
      await api.post(`/assistants/${gpt.id}/subscribe`);
      
      // Update local state
      setSubscribedIds(prev => new Set([...prev, gpt.id]));
      
      // Add to models store
      addSubscribedAssistant({
        id: `gpt:${gpt.id}`,
        name: gpt.name,
        type: 'assistant',
        assistant_id: gpt.id,
        icon: gpt.icon,
        color: gpt.color,
      });
      
      // Force refresh models
      fetchModels(true);
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to subscribe');
    }
  };
  
  const handleUnsubscribe = async (gpt: CustomAssistant) => {
    try {
      await api.delete(`/assistants/${gpt.id}/subscribe`);
      
      // Update local state
      setSubscribedIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(gpt.id);
        return newSet;
      });
      
      // Remove from models store
      removeSubscribedAssistant(gpt.id);
      
      // Force refresh models
      fetchModels(true);
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to unsubscribe');
    }
  };
  
  const startEditing = (gpt: CustomAssistant) => {
    setEditingGpt(gpt);
    setFormData({
      name: gpt.name,
      slug: gpt.slug,
      tagline: gpt.tagline || '',
      description: gpt.description || '',
      icon: gpt.icon,
      color: gpt.color,
      system_prompt: gpt.system_prompt,
      welcome_message: gpt.welcome_message || '',
      suggested_prompts: gpt.suggested_prompts.length ? gpt.suggested_prompts : [''],
      model: gpt.model,
      is_public: gpt.is_public,
      is_discoverable: gpt.is_discoverable,
      knowledge_store_ids: gpt.knowledge_store_ids || [],
      category: gpt.category || 'general',
    });
    setAvatarPreview(gpt.avatar_url || null);
    setAvatarFile(null);
    setShowForm(true);
  };
  
  const resetForm = () => {
    setShowForm(false);
    setEditingGpt(null);
    setFormData({
      name: '',
      slug: '',
      tagline: '',
      description: '',
      icon: 'AI',
      color: '#6366f1',
      system_prompt: '',
      welcome_message: '',
      suggested_prompts: [''],
      model: defaultModel || '',
      is_public: false,
      is_discoverable: false,
      knowledge_store_ids: [],
      category: 'general',
    });
    setAvatarFile(null);
    setAvatarPreview(null);
  };
  
  const addSuggestedPrompt = () => {
    setFormData({
      ...formData,
      suggested_prompts: [...formData.suggested_prompts, ''],
    });
  };
  
  const updateSuggestedPrompt = (index: number, value: string) => {
    const prompts = [...formData.suggested_prompts];
    prompts[index] = value;
    setFormData({ ...formData, suggested_prompts: prompts });
  };
  
  const removeSuggestedPrompt = (index: number) => {
    setFormData({
      ...formData,
      suggested_prompts: formData.suggested_prompts.filter((_, i) => i !== index),
    });
  };
  
  const toggleKnowledgeStore = (ksId: string) => {
    const ids = formData.knowledge_store_ids.includes(ksId)
      ? formData.knowledge_store_ids.filter(id => id !== ksId)
      : [...formData.knowledge_store_ids, ksId];
    setFormData({ ...formData, knowledge_store_ids: ids });
  };
  
  // Handle avatar file selection
  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        setError('Image must be less than 5MB');
        return;
      }
      setAvatarFile(file);
      setAvatarPreview(URL.createObjectURL(file));
    }
  };
  
  // Category options - fetched from API
  const [categories, setCategories] = useState<{ value: string; label: string; icon?: string }[]>([
    { value: 'general', label: 'General' }, // Fallback default
  ]);
  
  // Fetch categories on mount
  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const res = await api.get('/assistants/categories');
        if (res.data && res.data.length > 0) {
          setCategories(res.data.map((c: { value: string; label: string; icon: string }) => ({
            value: c.value,
            label: c.label,
            icon: c.icon,
          })));
        }
      } catch (err) {
        console.error('Failed to fetch categories:', err);
        // Keep fallback categories
      }
    };
    fetchCategories();
  }, []);
  
  // Filter and sort public GPTs
  const filteredAndSortedGpts = (() => {
    let filtered = publicGpts.filter(gpt => {
      // Search filter
      const matchesSearch = !searchQuery || 
        gpt.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        gpt.tagline?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        gpt.description?.toLowerCase().includes(searchQuery.toLowerCase());
      
      // Category filter
      const matchesCategory = categoryFilter === 'all' || gpt.category === categoryFilter;
      
      return matchesSearch && matchesCategory;
    });
    
    // Sort
    switch (sortBy) {
      case 'popularity':
        filtered.sort((a, b) => b.conversation_count - a.conversation_count);
        break;
      case 'updated':
        filtered.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
        break;
      case 'alphabetical':
        filtered.sort((a, b) => a.name.localeCompare(b.name));
        break;
    }
    
    return filtered;
  })();
  
  // Pagination
  const totalPages = Math.ceil(filteredAndSortedGpts.length / itemsPerPage);
  const paginatedGpts = filteredAndSortedGpts.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );
  
  // Reset to page 1 when filters change
  const handleFilterChange = (filter: string) => {
    setCategoryFilter(filter);
    setCurrentPage(1);
  };
  
  const handleSortChange = (sort: 'popularity' | 'updated' | 'alphabetical') => {
    setSortBy(sort);
    setCurrentPage(1);
  };
  
  const handleSearchChange = (query: string) => {
    setSearchQuery(query);
    setCurrentPage(1);
  };
  
  const iconOptions = ['AI', 'ML', 'BOT', 'PRO', 'DOC', 'DEV', 'BIZ', 'ART', 'SCI', 'ENG', 'OPS', 'FIN'];
  
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-[var(--color-text)]">Custom GPTs</h1>
            <p className="text-sm text-[var(--color-text-secondary)]">
              Create and discover AI assistants with custom knowledge and behavior
            </p>
          </div>
          <button
            onClick={() => setShowForm(true)}
            className="flex items-center gap-2 px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Create GPT
          </button>
        </div>
        
        {/* Tabs */}
        <div className="flex gap-1 mb-6 border-b border-[var(--color-border)]">
          <button
            onClick={() => setActiveTab('my-gpts')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === 'my-gpts'
                ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            My GPTs ({myGpts.length})
          </button>
          <button
            onClick={() => setActiveTab('explore')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === 'explore'
                ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            Explore Marketplace
          </button>
        </div>
        
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-[var(--color-error)]/10 border border-[var(--color-error)]/30 text-[var(--color-error)] text-sm flex justify-between">
            {error}
            <button onClick={() => setError(null)}>✕</button>
          </div>
        )}
        
        {/* My GPTs Tab */}
        {activeTab === 'my-gpts' && (
          <>
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <svg className="animate-spin h-8 w-8 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
            ) : myGpts.length === 0 ? (
              <div className="text-center py-12 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
                <svg className="w-12 h-12 mx-auto mb-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <p className="text-[var(--color-text-secondary)]">No Custom GPTs yet</p>
                <p className="text-zinc-500/60 text-sm mt-1">Create one to get started</p>
                <button
                  onClick={() => setShowForm(true)}
                  className="mt-4 px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
                >
                  Create Your First GPT
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {myGpts.map((gpt) => (
                  <div
                    key={gpt.id}
                    className="bg-[var(--color-surface)] rounded-xl p-5 border border-[var(--color-border)] hover:border-[var(--color-text-secondary)]/30 transition-all group"
                  >
                    <div className="flex items-start gap-4">
                      {gpt.avatar_url ? (
                        <img
                          src={gpt.avatar_url}
                          alt={gpt.name}
                          className="w-14 h-14 rounded-xl object-cover flex-shrink-0"
                        />
                      ) : (
                        <div
                          className="w-14 h-14 rounded-xl flex items-center justify-center text-3xl flex-shrink-0"
                          style={{ backgroundColor: `${gpt.color}20` }}
                        >
                          {gpt.icon}
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-[var(--color-text)] truncate">{gpt.name}</h3>
                          {gpt.is_public && (
                            <span className="px-2 py-0.5 text-xs rounded-full bg-[var(--color-success)]/10 text-[var(--color-success)]">
                              Public
                            </span>
                          )}
                        </div>
                        {gpt.tagline && (
                          <p className="text-sm text-[var(--color-text-secondary)] truncate">{gpt.tagline}</p>
                        )}
                        <div className="flex items-center gap-3 mt-2 text-xs text-[var(--color-text-secondary)]">
                          <span className="px-2 py-0.5 rounded-full bg-[var(--color-background)] capitalize">{gpt.category || 'general'}</span>
                          <span>{gpt.conversation_count} chats</span>
                          {gpt.average_rating > 0 && (
                            <span>⭐ {gpt.average_rating.toFixed(1)}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2 mt-4 pt-4 border-t border-[var(--color-border)]">
                      <button
                        onClick={() => handleStartChat(gpt)}
                        className="flex-1 px-3 py-1.5 text-sm rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90"
                      >
                        Chat
                      </button>
                      <button
                        onClick={() => startEditing(gpt)}
                        className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-background)] text-[var(--color-text)] hover:bg-[var(--color-border)]"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete(gpt.id)}
                        className="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-error)] hover:bg-red-500/10"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
        
        {/* Explore Marketplace Tab */}
        {activeTab === 'explore' && (
          <>
            {/* Search and Filters */}
            <div className="mb-6 space-y-4">
              {/* Search */}
              <div className="relative">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  placeholder="Search Custom GPTs..."
                  className="w-full pl-10 pr-4 py-3 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
              
              {/* Categories and Sort */}
              <div className="flex flex-wrap items-center gap-3">
                {/* Category Filter */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-[var(--color-text-secondary)]">Category:</span>
                  <select
                    value={categoryFilter}
                    onChange={(e) => handleFilterChange(e.target.value)}
                    className="px-3 py-1.5 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  >
                    <option value="all">All Categories</option>
                    {categories.map(cat => (
                      <option key={cat.value} value={cat.value}>{cat.label}</option>
                    ))}
                  </select>
                </div>
                
                {/* Sort */}
                <div className="flex items-center gap-2 ml-auto">
                  <span className="text-sm text-[var(--color-text-secondary)]">Sort:</span>
                  <div className="flex bg-[var(--color-background)] rounded-lg p-1 border border-[var(--color-border)]">
                    <button
                      onClick={() => handleSortChange('popularity')}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        sortBy === 'popularity'
                          ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      }`}
                    >
                      Popular
                    </button>
                    <button
                      onClick={() => handleSortChange('updated')}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        sortBy === 'updated'
                          ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      }`}
                    >
                      Recent
                    </button>
                    <button
                      onClick={() => handleSortChange('alphabetical')}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        sortBy === 'alphabetical'
                          ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      }`}
                    >
                      A-Z
                    </button>
                  </div>
                </div>
              </div>
              
              {/* Results count */}
              <div className="text-sm text-[var(--color-text-secondary)]">
                {filteredAndSortedGpts.length} {filteredAndSortedGpts.length === 1 ? 'GPT' : 'GPTs'} found
              </div>
            </div>
            
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <svg className="animate-spin h-8 w-8 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
            ) : paginatedGpts.length === 0 ? (
              <div className="text-center py-12 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
                <svg className="w-12 h-12 mx-auto mb-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                </svg>
                <p className="text-[var(--color-text-secondary)]">No GPTs found</p>
                <p className="text-zinc-500/60 text-sm mt-1">Try adjusting your filters or be the first to publish one!</p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {paginatedGpts.map((gpt) => (
                    <div
                      key={gpt.id}
                      className="bg-[var(--color-surface)] rounded-xl p-5 border border-[var(--color-border)] hover:border-[var(--color-text-secondary)]/30 transition-all"
                    >
                      <div className="flex items-start gap-3 mb-3">
                        {gpt.avatar_url ? (
                          <img
                            src={gpt.avatar_url}
                            alt={gpt.name}
                            className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
                          />
                        ) : (
                          <div
                            className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl flex-shrink-0"
                            style={{ backgroundColor: `${gpt.color}20` }}
                          >
                            {gpt.icon}
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <h3 className="font-semibold text-[var(--color-text)] truncate">{gpt.name}</h3>
                          {gpt.tagline && (
                            <p className="text-sm text-[var(--color-text-secondary)] truncate">{gpt.tagline}</p>
                          )}
                        </div>
                      </div>
                      
                      {gpt.description && (
                        <p className="text-sm text-[var(--color-text-secondary)] line-clamp-2 mb-3">
                          {gpt.description}
                        </p>
                      )}
                      
                      <div className="flex items-center justify-between text-xs text-[var(--color-text-secondary)] mb-4">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded-full bg-[var(--color-background)] capitalize">{gpt.category || 'general'}</span>
                          <span>{gpt.conversation_count} chats</span>
                        </div>
                        {gpt.average_rating > 0 && (
                          <span>⭐ {gpt.average_rating.toFixed(1)}</span>
                        )}
                      </div>
                      
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleStartChat(gpt)}
                          className="flex-1 px-3 py-2 text-sm rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90"
                        >
                          Chat
                        </button>
                        {subscribedIds.has(gpt.id) ? (
                          <button
                            onClick={() => handleUnsubscribe(gpt)}
                            className="px-3 py-2 text-sm rounded-lg bg-[var(--color-background)] text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:bg-[var(--color-border)]"
                          >
                            ✓ Subscribed
                          </button>
                        ) : (
                          <button
                            onClick={() => handleSubscribe(gpt)}
                          className="px-3 py-2 text-sm rounded-lg bg-[var(--color-success)]/10 text-[var(--color-success)] border border-[var(--color-success)]/30 hover:bg-[var(--color-success)]/20"
                        >
                          + Subscribe
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-8 flex items-center justify-center gap-2">
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[var(--color-border)]"
                  >
                    Previous
                  </button>
                  
                  <div className="flex items-center gap-1">
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      let pageNum: number;
                      if (totalPages <= 5) {
                        pageNum = i + 1;
                      } else if (currentPage <= 3) {
                        pageNum = i + 1;
                      } else if (currentPage >= totalPages - 2) {
                        pageNum = totalPages - 4 + i;
                      } else {
                        pageNum = currentPage - 2 + i;
                      }
                      return (
                        <button
                          key={pageNum}
                          onClick={() => setCurrentPage(pageNum)}
                          className={`w-10 h-10 rounded-lg text-sm font-medium transition-colors ${
                            currentPage === pageNum
                              ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                              : 'bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-border)]'
                          }`}
                        >
                          {pageNum}
                        </button>
                      );
                    })}
                  </div>
                  
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[var(--color-border)]"
                  >
                    Next
                  </button>
                </div>
              )}
              </>
            )}
          </>
        )}
        
        {/* Create/Edit Form Modal */}
        {showForm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={resetForm}>
            <div
              className="bg-[var(--color-surface)] rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto border border-[var(--color-border)]"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 bg-[var(--color-surface)] border-b border-[var(--color-border)] p-4 flex justify-between items-center">
                <h2 className="text-lg font-semibold text-[var(--color-text)]">
                  {editingGpt ? 'Edit Custom GPT' : 'Create Custom GPT'}
                </h2>
                <button onClick={resetForm} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)]">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="p-6 space-y-6">
                {/* Basic Info */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2 flex gap-4">
                    {/* Avatar Upload */}
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Avatar</label>
                      <div className="relative">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleAvatarChange}
                          className="hidden"
                          id="avatar-upload"
                        />
                        <label
                          htmlFor="avatar-upload"
                          className="w-16 h-16 rounded-xl flex items-center justify-center cursor-pointer border-2 border-dashed border-[var(--color-border)] hover:border-[var(--color-primary)] transition-colors overflow-hidden"
                          style={{ backgroundColor: avatarPreview ? undefined : `${formData.color}20` }}
                        >
                          {avatarPreview ? (
                            <img src={avatarPreview} alt="Avatar preview" className="w-full h-full object-cover" />
                          ) : (
                            <span className="text-2xl">{formData.icon}</span>
                          )}
                        </label>
                        {avatarPreview && (
                          <button
                            type="button"
                            onClick={() => { setAvatarFile(null); setAvatarPreview(null); }}
                            className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white text-xs flex items-center justify-center"
                          >
                            ×
                          </button>
                        )}
                      </div>
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">64×64px</p>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Icon (fallback)</label>
                      <div className="flex flex-wrap gap-1">
                        {iconOptions.map((icon) => (
                          <button
                            key={icon}
                            type="button"
                            onClick={() => setFormData({ ...formData, icon })}
                            className={`w-9 h-9 rounded-lg text-lg flex items-center justify-center ${
                              formData.icon === icon
                                ? 'bg-[var(--color-button)] ring-2 ring-[var(--color-border)] text-[var(--color-button-text)]'
                                : 'bg-[var(--color-background)]'
                            }`}
                          >
                            {icon}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Color</label>
                      <input
                        type="color"
                        value={formData.color}
                        onChange={(e) => setFormData({ ...formData, color: e.target.value })}
                        className="w-9 h-9 rounded-lg cursor-pointer"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Name *</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => handleNameChange(e.target.value)}
                      placeholder="My Custom GPT"
                      className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Category</label>
                    <select
                      value={formData.category}
                      onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                    >
                      {categories.map(cat => (
                        <option key={cat.value} value={cat.value}>{cat.label}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="col-span-2">
                    <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Tagline</label>
                    <input
                      type="text"
                      value={formData.tagline}
                      onChange={(e) => setFormData({ ...formData, tagline: e.target.value })}
                      placeholder="A helpful assistant for..."
                      className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Description</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Describe what this GPT does..."
                    rows={2}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  />
                </div>
                
                {/* System Prompt */}
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">System Prompt *</label>
                  <textarea
                    value={formData.system_prompt}
                    onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
                    placeholder="You are a helpful assistant that..."
                    rows={4}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] font-mono text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Welcome Message</label>
                  <textarea
                    value={formData.welcome_message}
                    onChange={(e) => setFormData({ ...formData, welcome_message: e.target.value })}
                    placeholder="Hello! I'm here to help you with..."
                    rows={2}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  />
                </div>
                
                {/* Suggested Prompts */}
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Suggested Prompts</label>
                  <div className="space-y-2">
                    {formData.suggested_prompts.map((prompt, index) => (
                      <div key={index} className="flex gap-2">
                        <input
                          type="text"
                          value={prompt}
                          onChange={(e) => updateSuggestedPrompt(index, e.target.value)}
                          placeholder="Example prompt..."
                          className="flex-1 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                        />
                        {formData.suggested_prompts.length > 1 && (
                          <button
                            onClick={() => removeSuggestedPrompt(index)}
                            className="p-2 text-[var(--color-error)] hover:bg-red-500/10 rounded"
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        )}
                      </div>
                    ))}
                    <button
                      onClick={addSuggestedPrompt}
                      className="text-sm text-[var(--color-primary)] hover:underline"
                    >
                      + Add another prompt
                    </button>
                  </div>
                </div>
                
                {/* Knowledge Stores */}
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-2">
                    Knowledge Bases {isLoading && <span className="text-xs">(loading...)</span>}
                  </label>
                  {knowledgeStores.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2">
                      {knowledgeStores.map((ks) => (
                        <button
                          key={ks.id}
                          onClick={() => toggleKnowledgeStore(ks.id)}
                          className={`flex items-center gap-2 p-3 rounded-lg border text-left transition-all ${
                            formData.knowledge_store_ids.includes(ks.id)
                              ? 'border-[var(--color-button)] bg-[var(--color-button)]/20'
                              : 'border-[var(--color-border)] hover:border-[var(--color-text-secondary)]'
                          }`}
                        >
                          <span className="text-xl">{ks.icon}</span>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-[var(--color-text)] truncate text-sm">{ks.name}</p>
                            <p className="text-xs text-[var(--color-text-secondary)]">{ks.document_count} docs</p>
                          </div>
                          {formData.knowledge_store_ids.includes(ks.id) && (
                            <svg className="w-5 h-5 text-[var(--color-text)]" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                            </svg>
                          )}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="p-4 rounded-lg border border-dashed border-[var(--color-border)] text-center">
                      <p className="text-sm text-[var(--color-text-secondary)] mb-2">No knowledge bases available</p>
                      <a
                        href="/documents"
                        className="text-sm text-[var(--color-primary)] hover:underline"
                      >
                        Create a Knowledge Base →
                      </a>
                    </div>
                  )}
                </div>
                
                {/* Model Settings */}
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Model</label>
                  <select
                    value={formData.model}
                    onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  >
                    {models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name || model.id}
                      </option>
                    ))}
                    {models.length === 0 && (
                      <option value={defaultModel || 'default'}>{defaultModel || 'Default Model'}</option>
                    )}
                  </select>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    Temperature and context window are inherited from system defaults
                  </p>
                </div>
                
                {/* Visibility */}
                <div className="flex items-center gap-6">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.is_public}
                      onChange={(e) => setFormData({ ...formData, is_public: e.target.checked })}
                      className="rounded"
                    />
                    <span className="text-sm text-[var(--color-text)]">Public</span>
                  </label>
                  
                  {formData.is_public && (
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={formData.is_discoverable}
                        onChange={(e) => setFormData({ ...formData, is_discoverable: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm text-[var(--color-text)]">Show in Marketplace</span>
                    </label>
                  )}
                </div>
              </div>
              
              <div className="sticky bottom-0 bg-[var(--color-surface)] border-t border-[var(--color-border)] p-4 flex justify-end gap-3">
                <button
                  onClick={resetForm}
                  className="px-4 py-2 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-background)]"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={!formData.name.trim() || !formData.system_prompt.trim()}
                  className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                  {editingGpt ? 'Save Changes' : 'Create GPT'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
