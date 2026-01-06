import { useState, useEffect, useMemo, lazy, Suspense } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import api from '../lib/api';

// Import admin tab components and types
import { 
  SystemTab, 
  OAuthTab, 
  FeaturesTab, 
  CategoriesTab,
  AssistantModesTab,
  SystemSettings,
  OAuthSettings,
  LLMSettings,
  LLMProvider,
  FeatureFlags,
  APIRateLimits,
  TierConfig,
  UserListItem,
  ChatListItem,
  ChatDetail,
  ToolConfig,
  ToolUsageStats,
  ToolUsageStat,
  FilterConfig,
  FilterChainDef,
  FilterChainStep,
  FilterChainSchema,
  GlobalKBStore,
  GPTCategory,
  TabId,
  formatHours,
} from '../components/admin';

// Lazy load FlowEditor for better performance
const FlowEditor = lazy(() => import('../components/FlowEditor'));

export default function Admin() {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  
  // Settings state
  const [systemSettings, setSystemSettings] = useState<SystemSettings | null>(null);
  const [oauthSettings, setOAuthSettings] = useState<OAuthSettings | null>(null);
  const [llmSettings, setLLMSettings] = useState<LLMSettings | null>(null);
  const [llmProviders, setLLMProviders] = useState<LLMProvider[]>([]);
  const [editingProvider, setEditingProvider] = useState<LLMProvider | null>(null);
  const [showProviderForm, setShowProviderForm] = useState(false);
  const [testingProvider, setTestingProvider] = useState<string | null>(null);
  const [featureFlags, setFeatureFlags] = useState<FeatureFlags | null>(null);
  const [apiRateLimits, setApiRateLimits] = useState<APIRateLimits | null>(null);
  const [tiers, setTiers] = useState<TierConfig[]>([]);
  
  // Dev settings (stored in localStorage)
  const [debugVoiceMode, setDebugVoiceMode] = useState(() => {
    return localStorage.getItem('nexus-debug-voice-mode') === 'true';
  });
  
  // Dev settings (stored in backend)
  const [debugTokenResets, setDebugTokenResets] = useState(false);
  const [debugDocumentQueue, setDebugDocumentQueue] = useState(false);
  const [debugRag, setDebugRag] = useState(false);
  const [debugFilterChains, setDebugFilterChains] = useState(false);
  const [lastTokenReset, setLastTokenReset] = useState<string | null>(null);
  const [tokenRefillHours, setTokenRefillHours] = useState(720);
  const [debugSettingsLoading, setDebugSettingsLoading] = useState(false);
  
  // UI state
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | object | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('system');
  const [editingTier, setEditingTier] = useState<string | null>(null);
  const [newFeature, setNewFeature] = useState<string>('');
  
  // User management state
  const [users, setUsers] = useState<UserListItem[]>([]);
  const [usersTotal, setUsersTotal] = useState(0);
  const [usersPage, setUsersPage] = useState(1);
  const [usersSearch, setUsersSearch] = useState('');
  const [usersLoading, setUsersLoading] = useState(false);
  const [selectedUser, setSelectedUser] = useState<UserListItem | null>(null);
  const [showUserModal, setShowUserModal] = useState(false);
  const [userAction, setUserAction] = useState<'upgrade' | 'refund' | 'chats' | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  
  // Chat viewing state
  const [userChats, setUserChats] = useState<ChatListItem[]>([]);
  const [userChatsTotal, setUserChatsTotal] = useState(0);
  const [userChatsPage, setUserChatsPage] = useState(1);
  const [allChats, setAllChats] = useState<ChatListItem[]>([]);
  const [allChatsTotal, setAllChatsTotal] = useState(0);
  const [allChatsPage, setAllChatsPage] = useState(1);
  const [allChatsSearch, setAllChatsSearch] = useState('');
  const [chatsLoading, setChatsLoading] = useState(false);
  const [selectedChat, setSelectedChat] = useState<ChatDetail | null>(null);
  const [showChatModal, setShowChatModal] = useState(false);
  
  // Tools state
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [toolsLoading, setToolsLoading] = useState(false);
  const [toolUsageStats, setToolUsageStats] = useState<ToolUsageStats | null>(null);
  const [showToolModal, setShowToolModal] = useState(false);
  const [editingTool, setEditingTool] = useState<ToolConfig | null>(null);
  const [toolForm, setToolForm] = useState({
    name: '',
    description: '',
    tool_type: 'mcp' as 'mcp' | 'openapi',
    url: '',
    api_key: '',
    is_public: false,
    auth_type: 'bearer',
  });
  const [probeResult, setProbeResult] = useState<{ success: boolean; message: string; tools?: any[] } | null>(null);
  const [probing, setProbing] = useState(false);
  
  // Filter state
  const [filters, setFilters] = useState<FilterConfig[]>([]);
  const [filtersLoading, setFiltersLoading] = useState(false);
  const [showFilterModal, setShowFilterModal] = useState(false);
  const [editingFilter, setEditingFilter] = useState<FilterConfig | null>(null);
  const [filterForm, setFilterForm] = useState<FilterConfig>({
    name: '',
    description: null,
    filter_type: 'to_llm',
    priority: 'medium',
    enabled: true,
    filter_mode: 'pattern',
    pattern: null,
    replacement: null,
    word_list: null,
    case_sensitive: false,
    action: 'modify',
    block_message: null,
    code: null,
    llm_prompt: null,
    config: null,
    is_global: true,
  });
  const [filterTestContent, setFilterTestContent] = useState('');
  const [filterTestResult, setFilterTestResult] = useState<{ original: string; result: string; modified: boolean; blocked: boolean; block_reason?: string } | null>(null);
  const [filterTesting, setFilterTesting] = useState(false);
  
  // Filter Chains state (configurable agentic flows)
  const [filterChains, setFilterChains] = useState<FilterChainDef[]>([]);
  const [filterChainsLoading, setFilterChainsLoading] = useState(false);
  const [filterChainSchema, setFilterChainSchema] = useState<FilterChainSchema | null>(null);
  const [showFilterChainModal, setShowFilterChainModal] = useState(false);
  const [editingFilterChain, setEditingFilterChain] = useState<FilterChainDef | null>(null);
  const [filterChainForm, setFilterChainForm] = useState<FilterChainDef>({
    name: '',
    description: '',
    enabled: true,
    priority: 100,
    retain_history: true,
    bidirectional: false,
    max_iterations: 10,
    debug: false,
    skip_if_rag_hit: true,
    // Export as tool defaults
    export_as_tool: false,
    tool_name: '',
    tool_label: '',
    advertise_to_llm: false,
    advertise_text: '',
    trigger_pattern: '',
    trigger_source: 'both',
    erase_from_display: true,
    keep_in_history: true,
    button_enabled: false,
    button_icon: '',
    button_location: 'response',
    button_trigger_mode: 'immediate',
    tool_variables: [],
    definition: { steps: [] },
  });
  const [filterChainJsonMode, setFilterChainJsonMode] = useState(false);
  const [filterChainJson, setFilterChainJson] = useState('');
  const [previewJsonChain, setPreviewJsonChain] = useState<FilterChainDef | null>(null);
  
  // Global Knowledge Store state
  const [globalKBStores, setGlobalKBStores] = useState<GlobalKBStore[]>([]);
  const [allKBStores, setAllKBStores] = useState<GlobalKBStore[]>([]);
  const [globalKBLoading, setGlobalKBLoading] = useState(false);
  const [globalKBEnabled, setGlobalKBEnabled] = useState(true);
  
  // GPT Categories state
  const [gptCategories, setGptCategories] = useState<GPTCategory[]>([]);
  const [categoriesLoading, setCategoriesLoading] = useState(false);
  const [showCategoryModal, setShowCategoryModal] = useState(false);
  const [editingCategory, setEditingCategory] = useState<GPTCategory | null>(null);
  const [categoryForm, setCategoryForm] = useState({
    value: '',
    label: '',
    icon: '📁',
    description: '',
    sort_order: 0,
  });
  
  // LLM test state
  const [llmTestResult, setLLMTestResult] = useState<{ success: boolean; message: string; models?: string[] } | null>(null);
  const [llmTesting, setLLMTesting] = useState(false);
  
  const PAGE_SIZE = 10;
  
  // Redirect non-admins
  useEffect(() => {
    if (user && !user.is_admin) {
      navigate('/');
    }
  }, [user, navigate]);
  
  // Fetch initial data
  useEffect(() => {
    fetchData();
  }, []);
  
  // Fetch debug settings when dev tab is activated
  useEffect(() => {
    if (activeTab === 'dev') {
      fetchDebugSettings();
    }
  }, [activeTab]);
  
  // Fetch global KB stores when that tab is activated
  useEffect(() => {
    if (activeTab === 'global_kb') {
      fetchGlobalKBStores();
    }
  }, [activeTab]);
  
  // Fetch GPT categories when that tab is activated
  useEffect(() => {
    if (activeTab === 'categories') {
      fetchCategories();
    }
  }, [activeTab]);
  
  const fetchGlobalKBStores = async () => {
    setGlobalKBLoading(true);
    try {
      // Fetch all stores
      const allRes = await api.get('/knowledge-stores/admin/all');
      setAllKBStores(allRes.data);
      
      // Filter to global ones
      setGlobalKBStores(allRes.data.filter((s: GlobalKBStore) => s.is_global));
      
      // Get global KB enabled setting (using raw settings endpoint)
      const settingsRes = await api.get('/admin/settings/raw');
      const globalEnabled = settingsRes.data.find((s: any) => s.key === 'global_knowledge_store_enabled');
      setGlobalKBEnabled(globalEnabled?.value === 'true');
    } catch (err: any) {
      console.error('Failed to load global KB stores:', err);
      setError(err.response?.data?.detail || 'Failed to load global knowledge stores');
    } finally {
      setGlobalKBLoading(false);
    }
  };
  
  const toggleGlobalKBEnabled = async () => {
    try {
      const newValue = !globalKBEnabled;
      await api.put('/admin/setting', {
        key: 'global_knowledge_store_enabled',
        value: String(newValue),
      });
      setGlobalKBEnabled(newValue);
      setSuccess(`Global knowledge stores ${newValue ? 'enabled' : 'disabled'}`);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update setting');
    }
  };
  
  const toggleStoreGlobal = async (store: GlobalKBStore, makeGlobal: boolean) => {
    try {
      await api.post(`/knowledge-stores/admin/global/${store.id}`, null, {
        params: {
          is_global: makeGlobal,
          min_score: store.global_min_score || 0.7,
          max_results: store.global_max_results || 3,
        },
      });
      await fetchGlobalKBStores();
      setSuccess(`Store "${store.name}" ${makeGlobal ? 'added to' : 'removed from'} global stores`);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update store');
    }
  };
  
  const updateGlobalStoreSettings = async (store: GlobalKBStore, minScore: number, maxResults: number) => {
    try {
      await api.post(`/knowledge-stores/admin/global/${store.id}`, null, {
        params: {
          is_global: true,
          min_score: minScore,
          max_results: maxResults,
        },
      });
      await fetchGlobalKBStores();
      setSuccess('Global store settings updated');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update store settings');
    }
  };
  
  // GPT Category functions
  const fetchCategories = async () => {
    setCategoriesLoading(true);
    try {
      const res = await api.get('/assistants/categories', { params: { include_inactive: true } });
      setGptCategories(res.data || []);
    } catch (err: any) {
      console.error('Failed to load categories:', err);
      setError(err.response?.data?.detail || 'Failed to load categories');
    } finally {
      setCategoriesLoading(false);
    }
  };
  
  const resetCategoryForm = () => {
    setCategoryForm({
      value: '',
      label: '',
      icon: '📁',
      description: '',
      sort_order: gptCategories.length,
    });
    setEditingCategory(null);
    setShowCategoryModal(false);
  };
  
  const startEditCategory = (cat: GPTCategory) => {
    setEditingCategory(cat);
    setCategoryForm({
      value: cat.value,
      label: cat.label,
      icon: cat.icon,
      description: cat.description || '',
      sort_order: cat.sort_order,
    });
    setShowCategoryModal(true);
  };
  
  const saveCategory = async () => {
    if (!categoryForm.value.trim() || !categoryForm.label.trim()) {
      setError('Value and label are required');
      return;
    }
    
    // Validate value format (lowercase, alphanumeric, hyphens)
    if (!/^[a-z0-9-]+$/.test(categoryForm.value)) {
      setError('Value must be lowercase letters, numbers, and hyphens only');
      return;
    }
    
    try {
      if (editingCategory) {
        await api.patch(`/assistants/categories/${editingCategory.id}`, {
          label: categoryForm.label,
          icon: categoryForm.icon,
          description: categoryForm.description || null,
          sort_order: categoryForm.sort_order,
        });
        setSuccess('Category updated');
      } else {
        await api.post('/assistants/categories', categoryForm);
        setSuccess('Category created');
      }
      setTimeout(() => setSuccess(null), 3000);
      await fetchCategories();
      resetCategoryForm();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save category');
    }
  };
  
  const toggleCategoryActive = async (cat: GPTCategory) => {
    try {
      await api.patch(`/assistants/categories/${cat.id}`, {
        is_active: !cat.is_active,
      });
      await fetchCategories();
      setSuccess(`Category ${cat.is_active ? 'disabled' : 'enabled'}`);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update category');
    }
  };
  
  const deleteCategory = async (cat: GPTCategory) => {
    if (cat.value === 'general') {
      setError('Cannot delete the general category');
      return;
    }
    if (!confirm(`Delete category "${cat.label}"? GPTs using this category will be moved to "General".`)) {
      return;
    }
    try {
      await api.delete(`/assistants/categories/${cat.id}`);
      await fetchCategories();
      setSuccess('Category deleted');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete category');
    }
  };
  
  const fetchDebugSettings = async () => {
    setDebugSettingsLoading(true);
    try {
      const res = await api.get('/admin/debug-settings');
      setDebugTokenResets(res.data.debug_token_resets);
      setDebugDocumentQueue(res.data.debug_document_queue);
      setDebugRag(res.data.debug_rag);
      setDebugFilterChains(res.data.debug_filter_chains);
      setLastTokenReset(res.data.last_token_reset_timestamp);
      setTokenRefillHours(res.data.token_refill_interval_hours);
    } catch (err: any) {
      console.error('Failed to load debug settings:', err);
    } finally {
      setDebugSettingsLoading(false);
    }
  };
  
  const saveDebugTokenResets = async (value: boolean) => {
    try {
      await api.put('/admin/debug-settings', { debug_token_resets: value });
      setDebugTokenResets(value);
      setSuccess('Debug setting saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save debug setting');
    }
  };
  
  const saveDebugDocumentQueue = async (value: boolean) => {
    try {
      await api.put('/admin/debug-settings', { debug_document_queue: value });
      setDebugDocumentQueue(value);
      setSuccess('Debug setting saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save debug setting');
    }
  };
  
  const saveDebugRag = async (value: boolean) => {
    try {
      await api.put('/admin/debug-settings', { debug_rag: value });
      setDebugRag(value);
      setSuccess('Debug setting saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save debug setting');
    }
  };
  
  const saveDebugFilterChains = async (value: boolean) => {
    try {
      await api.put('/admin/debug-settings', { debug_filter_chains: value });
      setDebugFilterChains(value);
      setSuccess('Debug setting saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save debug setting');
    }
  };
  
  const fetchData = async () => {
    try {
      const [systemRes, oauthRes, llmRes, featuresRes, tiersRes, rateLimitsRes] = await Promise.all([
        api.get('/admin/settings'),
        api.get('/admin/oauth-settings'),
        api.get('/admin/llm-settings'),
        api.get('/admin/feature-flags'),
        api.get('/admin/tiers'),
        api.get('/admin/api-rate-limits'),
      ]);
      setSystemSettings(systemRes.data);
      setOAuthSettings(oauthRes.data);
      setLLMSettings(llmRes.data);
      setFeatureFlags(featuresRes.data);
      setTiers(tiersRes.data.tiers);
      setApiRateLimits(rateLimitsRes.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load settings');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Save functions for each settings type
  const saveSystemSettings = async () => {
    if (!systemSettings) return;
    setIsSaving(true);
    setError(null);
    try {
      await api.put('/admin/settings', systemSettings);
      setSuccess('System settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };
  
  const saveOAuthSettings = async () => {
    if (!oauthSettings) return;
    setIsSaving(true);
    setError(null);
    try {
      await api.put('/admin/oauth-settings', oauthSettings);
      setSuccess('OAuth settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save OAuth settings');
    } finally {
      setIsSaving(false);
    }
  };
  
  const saveLLMSettings = async () => {
    if (!llmSettings) return;
    setIsSaving(true);
    setError(null);
    try {
      await api.put('/admin/llm-settings', llmSettings);
      setSuccess('LLM settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save LLM settings');
    } finally {
      setIsSaving(false);
    }
  };
  
  const testLLMConnection = async () => {
    setLLMTesting(true);
    setLLMTestResult(null);
    try {
      const response = await api.post('/admin/llm-settings/test');
      setLLMTestResult(response.data);
    } catch (err: any) {
      setLLMTestResult({
        success: false,
        message: err.response?.data?.detail || 'Connection test failed',
      });
    } finally {
      setLLMTesting(false);
    }
  };
  
  // LLM Provider management
  const loadLLMProviders = async () => {
    try {
      const response = await api.get('/admin/llm-providers');
      setLLMProviders(response.data.providers || []);
    } catch (err) {
      console.error('Failed to load LLM providers:', err);
    }
  };
  
  const saveProvider = async (provider: Partial<LLMProvider>) => {
    setIsSaving(true);
    setError(null);
    try {
      if (editingProvider?.id) {
        await api.put(`/admin/llm-providers/${editingProvider.id}`, provider);
        setSuccess('Provider updated successfully');
      } else {
        await api.post('/admin/llm-providers', provider);
        setSuccess('Provider created successfully');
      }
      setTimeout(() => setSuccess(null), 3000);
      setShowProviderForm(false);
      setEditingProvider(null);
      loadLLMProviders();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save provider');
    } finally {
      setIsSaving(false);
    }
  };
  
  const deleteProvider = async (id: string) => {
    if (!confirm('Are you sure you want to delete this provider?')) return;
    try {
      await api.delete(`/admin/llm-providers/${id}`);
      setSuccess('Provider deleted successfully');
      setTimeout(() => setSuccess(null), 3000);
      loadLLMProviders();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete provider');
    }
  };
  
  const testProvider = async (id: string) => {
    setTestingProvider(id);
    try {
      const response = await api.post(`/admin/llm-providers/${id}/test`);
      if (response.data.status === 'ok') {
        setSuccess(`Connection OK: ${response.data.message}`);
      } else {
        setError(response.data.message || 'Test failed');
      }
      setTimeout(() => { setSuccess(null); setError(null); }, 5000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Connection test failed');
    } finally {
      setTestingProvider(null);
    }
  };
  
  const saveFeatureFlags = async () => {
    if (!featureFlags) return;
    setIsSaving(true);
    setError(null);
    try {
      await api.put('/admin/feature-flags', featureFlags);
      setSuccess('Feature flags saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save feature flags');
    } finally {
      setIsSaving(false);
    }
  };
  
  const saveApiRateLimits = async () => {
    if (!apiRateLimits) return;
    setIsSaving(true);
    setError(null);
    try {
      await api.put('/admin/api-rate-limits', apiRateLimits);
      setSuccess('API rate limits saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save API rate limits');
    } finally {
      setIsSaving(false);
    }
  };
  
  const saveTiers = async () => {
    setIsSaving(true);
    setError(null);
    try {
      await api.put('/admin/tiers', { tiers });
      setSuccess('Tiers saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save tiers');
    } finally {
      setIsSaving(false);
    }
  };
  
  // Tier management
  const updateTier = (tierId: string, updates: Partial<TierConfig>) => {
    setTiers(tiers.map(t => t.id === tierId ? { ...t, ...updates } : t));
  };
  
  const addFeatureToTier = (tierId: string) => {
    if (!newFeature.trim()) return;
    const tier = tiers.find(t => t.id === tierId);
    if (tier) {
      updateTier(tierId, { features: [...tier.features, newFeature.trim()] });
      setNewFeature('');
    }
  };
  
  const removeFeatureFromTier = (tierId: string, index: number) => {
    const tier = tiers.find(t => t.id === tierId);
    if (tier) {
      updateTier(tierId, { features: tier.features.filter((_, i) => i !== index) });
    }
  };
  
  // User management
  const fetchUsers = async (page = 1, search = '') => {
    setUsersLoading(true);
    try {
      const response = await api.get('/admin/users', {
        params: { page, page_size: PAGE_SIZE, search: search || undefined },
      });
      setUsers(response.data.users);
      setUsersTotal(response.data.total);
      setUsersPage(page);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load users');
    } finally {
      setUsersLoading(false);
    }
  };
  
  useEffect(() => {
    if (activeTab === 'users') {
      fetchUsers(1, usersSearch);
    }
  }, [activeTab]);
  
  const handleBanUser = async (targetUser: UserListItem) => {
    if (!confirm(`${targetUser.is_active ? 'Ban' : 'Unban'} ${targetUser.email}?`)) return;
    setActionLoading(true);
    try {
      await api.patch(`/admin/users/${targetUser.id}`, { is_active: !targetUser.is_active });
      setSuccess(`User ${targetUser.is_active ? 'banned' : 'unbanned'}`);
      fetchUsers(usersPage, usersSearch);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update user');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  const handleResetTokens = async (targetUser: UserListItem) => {
    if (!confirm(`Reset token usage for ${targetUser.email}?`)) return;
    setActionLoading(true);
    try {
      await api.post(`/admin/users/${targetUser.id}/reset-tokens`);
      setSuccess(`Tokens reset for ${targetUser.email}`);
      fetchUsers(usersPage, usersSearch);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to reset tokens');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  const handleUpgradeUser = async (targetUser: UserListItem, newTier: string) => {
    setActionLoading(true);
    try {
      await api.patch(`/admin/users/${targetUser.id}`, { tier: newTier });
      setSuccess(`${targetUser.email} upgraded to ${newTier}`);
      setShowUserModal(false);
      setUserAction(null);
      fetchUsers(usersPage, usersSearch);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upgrade user');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  const handleRefundTokens = async (targetUser: UserListItem, amount: number) => {
    setActionLoading(true);
    try {
      const newLimit = targetUser.tokens_limit + amount;
      await api.patch(`/admin/users/${targetUser.id}`, { tokens_limit: newLimit });
      setSuccess(`Refunded ${formatTokens(amount)} tokens to ${targetUser.email}`);
      setShowUserModal(false);
      setUserAction(null);
      fetchUsers(usersPage, usersSearch);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to refund tokens');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  const handleToggleAdmin = async (targetUser: UserListItem) => {
    if (targetUser.id === user?.id) {
      setError("Cannot modify your own admin status");
      return;
    }
    if (!confirm(`${targetUser.is_admin ? 'Remove admin from' : 'Make admin'} ${targetUser.email}?`)) return;
    setActionLoading(true);
    try {
      await api.patch(`/admin/users/${targetUser.id}`, { is_admin: !targetUser.is_admin });
      setSuccess(`Admin status updated for ${targetUser.email}`);
      fetchUsers(usersPage, usersSearch);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update user');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  // Chat viewing
  const fetchUserChats = async (userId: string, page = 1) => {
    setChatsLoading(true);
    try {
      const response = await api.get(`/admin/users/${userId}/chats`, {
        params: { page, page_size: PAGE_SIZE },
      });
      setUserChats(response.data.chats);
      setUserChatsTotal(response.data.total);
      setUserChatsPage(page);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load chats');
    } finally {
      setChatsLoading(false);
    }
  };
  
  const fetchAllChats = async (page = 1, search = '') => {
    setChatsLoading(true);
    try {
      const response = await api.get('/admin/all-chats', {
        params: { page, page_size: PAGE_SIZE, search: search || undefined },
      });
      setAllChats(response.data.chats);
      setAllChatsTotal(response.data.total);
      setAllChatsPage(page);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load chats');
    } finally {
      setChatsLoading(false);
    }
  };
  
  useEffect(() => {
    if (activeTab === 'chats') {
      fetchAllChats(1, allChatsSearch);
    }
  }, [activeTab]);
  
  const viewChatDetail = async (chatId: string) => {
    setChatsLoading(true);
    try {
      const response = await api.get(`/admin/chats/${chatId}`);
      setSelectedChat(response.data);
      setShowChatModal(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load chat');
    } finally {
      setChatsLoading(false);
    }
  };
  
  // Tools management
  const fetchTools = async () => {
    setToolsLoading(true);
    try {
      const [toolsResponse, statsResponse] = await Promise.all([
        api.get('/tools'),
        api.get('/tools/usage/stats'),
      ]);
      setTools(toolsResponse.data);
      setToolUsageStats(statsResponse.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load tools');
    } finally {
      setToolsLoading(false);
    }
  };
  
  const resetToolUsage = async (toolId?: string) => {
    const confirmMsg = toolId 
      ? 'Reset usage statistics for this tool?' 
      : 'Reset ALL tool usage statistics? This cannot be undone.';
    if (!confirm(confirmMsg)) return;
    
    try {
      if (toolId) {
        await api.delete(`/tools/usage/stats/${toolId}`);
      } else {
        await api.delete('/tools/usage/stats');
      }
      fetchTools(); // Refresh stats
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to reset tool usage');
    }
  };
  
  const getToolUsageStats = (toolId: string): ToolUsageStat | undefined => {
    return toolUsageStats?.tools.find(t => t.tool_id === toolId);
  };
  
  useEffect(() => {
    if (activeTab === 'tools') {
      fetchTools();
    }
  }, [activeTab]);
  
  // Load LLM providers when LLM tab is active
  useEffect(() => {
    if (activeTab === 'llm') {
      loadLLMProviders();
    }
  }, [activeTab]);
  
  const resetToolForm = () => {
    setToolForm({
      name: '',
      description: '',
      tool_type: 'mcp',
      url: '',
      api_key: '',
      is_public: false,
      auth_type: 'bearer',
    });
    setEditingTool(null);
    setProbeResult(null);
  };
  
  const handleProbeUrl = async () => {
    setProbing(true);
    setProbeResult(null);
    try {
      const response = await api.post('/tools/probe', {
        url: toolForm.url,
        tool_type: toolForm.tool_type,
        api_key: toolForm.api_key || undefined,
        auth_type: toolForm.auth_type,
      });
      setProbeResult(response.data);
    } catch (err: any) {
      setProbeResult({
        success: false,
        message: err.response?.data?.detail || 'Failed to probe URL',
      });
    } finally {
      setProbing(false);
    }
  };
  
  const handleCreateTool = async () => {
    setActionLoading(true);
    try {
      if (editingTool) {
        await api.put(`/tools/${editingTool.id}`, {
          name: toolForm.name,
          description: toolForm.description || undefined,
          url: toolForm.url,
          api_key: toolForm.api_key || undefined,
          is_public: toolForm.is_public,
          config: { auth_type: toolForm.auth_type },
        });
        setSuccess('Tool updated');
      } else {
        await api.post('/tools', {
          name: toolForm.name,
          description: toolForm.description || undefined,
          tool_type: toolForm.tool_type,
          url: toolForm.url,
          api_key: toolForm.api_key || undefined,
          is_public: toolForm.is_public,
          config: { auth_type: toolForm.auth_type },
        });
        setSuccess('Tool created');
      }
      setShowToolModal(false);
      resetToolForm();
      fetchTools();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save tool');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  const handleDeleteTool = async (toolId: string) => {
    if (!confirm('Delete this tool?')) return;
    try {
      await api.delete(`/tools/${toolId}`);
      setSuccess('Tool deleted');
      fetchTools();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete tool');
    }
    setTimeout(() => setSuccess(null), 3000);
  };
  
  // Filter management
  const fetchFilters = async () => {
    setFiltersLoading(true);
    try {
      const response = await api.get('/admin/filters');
      setFilters(response.data.filters);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load filters');
    } finally {
      setFiltersLoading(false);
    }
  };
  
  useEffect(() => {
    if (activeTab === 'filters') {
      fetchFilters();
    }
  }, [activeTab]);
  
  const resetFilterForm = () => {
    setFilterForm({
      name: '',
      description: null,
      filter_type: 'to_llm',
      priority: 'medium',
      enabled: true,
      filter_mode: 'pattern',
      pattern: null,
      replacement: null,
      word_list: null,
      case_sensitive: false,
      action: 'modify',
      block_message: null,
      code: null,
      llm_prompt: null,
      config: null,
      is_global: true,
    });
    setEditingFilter(null);
    setFilterTestResult(null);
    setFilterTestContent('');
  };
  
  const handleCreateFilter = async () => {
    setActionLoading(true);
    try {
      if (editingFilter?.id) {
        await api.patch(`/admin/filters/${editingFilter.id}`, {
          name: filterForm.name,
          description: filterForm.description || undefined,
          priority: filterForm.priority,
          enabled: filterForm.enabled,
          filter_mode: filterForm.filter_mode,
          pattern: filterForm.pattern || undefined,
          replacement: filterForm.replacement || undefined,
          word_list: filterForm.word_list || undefined,
          case_sensitive: filterForm.case_sensitive,
          action: filterForm.action,
          block_message: filterForm.block_message || undefined,
          code: filterForm.code || undefined,
          llm_prompt: filterForm.llm_prompt || undefined,
          config: filterForm.config || undefined,
          is_global: filterForm.is_global,
        });
        setSuccess('Filter updated');
      } else {
        await api.post('/admin/filters', {
          name: filterForm.name,
          description: filterForm.description || undefined,
          filter_type: filterForm.filter_type,
          priority: filterForm.priority,
          enabled: filterForm.enabled,
          filter_mode: filterForm.filter_mode,
          pattern: filterForm.pattern || undefined,
          replacement: filterForm.replacement || undefined,
          word_list: filterForm.word_list || undefined,
          case_sensitive: filterForm.case_sensitive,
          action: filterForm.action,
          block_message: filterForm.block_message || undefined,
          code: filterForm.code || undefined,
          llm_prompt: filterForm.llm_prompt || undefined,
          config: filterForm.config || undefined,
          is_global: filterForm.is_global,
        });
        setSuccess('Filter created');
      }
      setShowFilterModal(false);
      resetFilterForm();
      fetchFilters();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save filter');
    } finally {
      setActionLoading(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };
  
  const handleDeleteFilter = async (filterId: string) => {
    if (!confirm('Delete this filter?')) return;
    try {
      await api.delete(`/admin/filters/${filterId}`);
      setSuccess('Filter deleted');
      fetchFilters();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete filter');
    }
    setTimeout(() => setSuccess(null), 3000);
  };
  
  const handleToggleFilter = async (filterId: string) => {
    try {
      await api.post(`/admin/filters/${filterId}/toggle`);
      fetchFilters();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to toggle filter');
    }
  };
  
  const handleTestFilter = async () => {
    if (!editingFilter?.id || !filterTestContent) return;
    setFilterTesting(true);
    setFilterTestResult(null);
    try {
      const response = await api.post('/admin/filters/test', null, {
        params: { filter_id: editingFilter.id, content: filterTestContent },
      });
      setFilterTestResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to test filter');
    } finally {
      setFilterTesting(false);
    }
  };
  
  const getFilterTypeLabel = (type: string) => {
    switch (type) {
      case 'to_llm': return 'User → LLM';
      case 'from_llm': return 'LLM → User';
      case 'to_tools': return 'To Tools';
      case 'from_tools': return 'From Tools';
      default: return type;
    }
  };
  
  const getFilterTypeColor = (type: string) => {
    switch (type) {
      case 'to_llm': return 'bg-blue-500/20 text-blue-400';
      case 'from_llm': return 'bg-green-500/20 text-green-400';
      case 'to_tools': return 'bg-orange-500/20 text-orange-400';
      case 'from_tools': return 'bg-purple-500/20 text-purple-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };
  
  // Filter Chains management
  const fetchFilterChains = async () => {
    setFilterChainsLoading(true);
    try {
      const [chainsRes, schemaRes, toolsRes] = await Promise.all([
        api.get('/admin/filter-chains'),
        api.get('/admin/filter-chains/schema'),
        api.get('/tools'),
      ]);
      setFilterChains(chainsRes.data);
      setFilterChainSchema(schemaRes.data);
      setTools(toolsRes.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load filter chains');
    } finally {
      setFilterChainsLoading(false);
    }
  };
  
  // Build flat list of available tools for filter chain step dropdown
  const availableTools = useMemo(() => {
    // If schema has available_tools, use those (includes MCP/OpenAPI from backend)
    if (filterChainSchema?.available_tools) {
      return filterChainSchema.available_tools as Array<{ value: string; label: string; category: string }>;
    }
    
    // Fallback: build from tools state
    const toolList: Array<{ value: string; label: string; category: string }> = [
      // Built-in tools that are always available
      { value: 'web_search', label: '🌐 Web Search', category: 'Built-in' },
      { value: 'web_fetch', label: '📄 Fetch Web Page', category: 'Built-in' },
      { value: 'calculator', label: '🔢 Calculator', category: 'Built-in' },
      { value: 'code_interpreter', label: '💻 Run Code', category: 'Built-in' },
    ];
    
    // Add MCP/OpenAPI tools and their sub-tools
    for (const tool of tools) {
      if (!tool.is_enabled) continue;
      
      if (tool.schema_cache && tool.schema_cache.length > 0) {
        // Tool has sub-tools from schema_cache
        for (const subTool of tool.schema_cache) {
          const icon = tool.tool_type === 'mcp' ? '🔌' : '🔗';
          toolList.push({
            value: `${tool.name}:${subTool.name}`,
            label: `${icon} ${tool.name} → ${subTool.name}`,
            category: tool.name,
          });
        }
      } else {
        // Tool without sub-tools (single operation)
        const icon = tool.tool_type === 'mcp' ? '🔌' : '🔗';
        toolList.push({
          value: tool.name,
          label: `${icon} ${tool.name}`,
          category: 'External',
        });
      }
    }
    
    return toolList;
  }, [tools, filterChainSchema]);
  
  useEffect(() => {
    if (activeTab === 'filter_chains') {
      fetchFilterChains();
    }
  }, [activeTab]);
  
  const resetFilterChainForm = () => {
    setFilterChainForm({
      name: '',
      description: '',
      enabled: true,
      priority: 100,
      retain_history: true,
      bidirectional: false,
      max_iterations: 10,
      debug: false,
      skip_if_rag_hit: true,
      // Export as tool defaults
      export_as_tool: false,
      tool_name: '',
      tool_label: '',
      advertise_to_llm: false,
      advertise_text: '',
      trigger_pattern: '',
      trigger_source: 'both',
      erase_from_display: true,
      keep_in_history: true,
      button_enabled: false,
      button_icon: '',
      button_location: 'response',
      button_trigger_mode: 'immediate',
      tool_variables: [],
      definition: { steps: [] },
    });
    setEditingFilterChain(null);
    setFilterChainJsonMode(false);
    setFilterChainJson('');
  };
  
  const handleCreateFilterChain = async () => {
    try {
      let definition = filterChainForm.definition;
      
      // Parse JSON if in JSON mode
      if (filterChainJsonMode && filterChainJson) {
        try {
          definition = JSON.parse(filterChainJson);
        } catch {
          setError('Invalid JSON in definition');
          return;
        }
      }
      
      const payload = {
        name: filterChainForm.name,
        description: filterChainForm.description || undefined,
        enabled: filterChainForm.enabled,
        priority: filterChainForm.priority,
        retain_history: filterChainForm.retain_history,
        bidirectional: filterChainForm.bidirectional,
        outbound_chain_id: filterChainForm.outbound_chain_id || undefined,
        max_iterations: filterChainForm.max_iterations,
        debug: filterChainForm.debug,
        skip_if_rag_hit: filterChainForm.skip_if_rag_hit,
        // Export as tool fields
        export_as_tool: filterChainForm.export_as_tool,
        tool_name: filterChainForm.tool_name || undefined,
        tool_label: filterChainForm.tool_label || undefined,
        advertise_to_llm: filterChainForm.advertise_to_llm,
        advertise_text: filterChainForm.advertise_text || undefined,
        trigger_pattern: filterChainForm.trigger_pattern || undefined,
        trigger_source: filterChainForm.trigger_source,
        erase_from_display: filterChainForm.erase_from_display,
        keep_in_history: filterChainForm.keep_in_history,
        button_enabled: filterChainForm.button_enabled,
        button_icon: filterChainForm.button_icon || undefined,
        button_location: filterChainForm.button_location,
        button_trigger_mode: filterChainForm.button_trigger_mode,
        tool_variables: filterChainForm.tool_variables,
        definition,
      };
      
      if (editingFilterChain?.id) {
        await api.put(`/admin/filter-chains/${editingFilterChain.id}`, payload);
        setSuccess('Filter chain updated');
      } else {
        await api.post('/admin/filter-chains', payload);
        setSuccess('Filter chain created');
      }
      
      setShowFilterChainModal(false);
      resetFilterChainForm();
      fetchFilterChains();
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      if (typeof detail === 'string') {
        setError(detail);
      } else if (detail && typeof detail === 'object') {
        setError(JSON.stringify(detail));
      } else {
        setError('Failed to save filter chain');
      }
    }
    setTimeout(() => setSuccess(null), 3000);
  };
  
  const handleDeleteFilterChain = async (chainId: string) => {
    if (!confirm('Delete this filter chain?')) return;
    try {
      await api.delete(`/admin/filter-chains/${chainId}`);
      setSuccess('Filter chain deleted');
      fetchFilterChains();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete filter chain');
    }
    setTimeout(() => setSuccess(null), 3000);
  };
  
  const handleToggleFilterChain = async (chainId: string) => {
    const chain = filterChains.find(c => c.id === chainId);
    if (!chain) return;
    try {
      await api.put(`/admin/filter-chains/${chainId}`, { enabled: !chain.enabled });
      fetchFilterChains();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to toggle filter chain');
    }
  };
  
  const addStep = (stepType: string) => {
    const newStep: FilterChainStep = {
      id: `step_${Date.now()}`,
      type: stepType,
      name: filterChainSchema?.step_types[stepType]?.label || stepType,
      enabled: true,
      config: {},
    };
    setFilterChainForm(prev => ({
      ...prev,
      definition: {
        steps: [...prev.definition.steps, newStep],
      },
    }));
  };
  
  const updateStep = (stepId: string, updates: Partial<FilterChainStep>) => {
    setFilterChainForm(prev => ({
      ...prev,
      definition: {
        steps: prev.definition.steps.map(s => 
          s.id === stepId ? { ...s, ...updates } : s
        ),
      },
    }));
  };
  
  const removeStep = (stepId: string) => {
    setFilterChainForm(prev => ({
      ...prev,
      definition: {
        steps: prev.definition.steps.filter(s => s.id !== stepId),
      },
    }));
  };
  
  const moveStep = (stepId: string, direction: 'up' | 'down') => {
    setFilterChainForm(prev => {
      const steps = [...prev.definition.steps];
      const idx = steps.findIndex(s => s.id === stepId);
      if (idx === -1) return prev;
      if (direction === 'up' && idx > 0) {
        [steps[idx - 1], steps[idx]] = [steps[idx], steps[idx - 1]];
      } else if (direction === 'down' && idx < steps.length - 1) {
        [steps[idx], steps[idx + 1]] = [steps[idx + 1], steps[idx]];
      }
      return { ...prev, definition: { steps } };
    });
  };
  
  const getStepTypeColor = (category: string) => {
    switch (category) {
      case 'llm': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      case 'tool': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'context': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'flow': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'variable': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
      case 'debug': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };
  
  // Utility functions
  const formatTokens = (tokens: number) => {
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(0)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`;
    return tokens.toString();
  };
  
  const AdminTabIcon = ({ type }: { type: string }) => {
    const iconClass = "w-4 h-4";
    switch (type) {
      case 'system':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
      case 'oauth':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>;
      case 'llm':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>;
      case 'features':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>;
      case 'filters':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" /></svg>;
      case 'tiers':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>;
      case 'users':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>;
      case 'chats':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>;
      case 'tools':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
      case 'filter_chains':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 4v6m0 0v6m0-6h6m-6 0h-6" transform="translate(-3, 0) scale(0.5)" /></svg>;
      case 'global_kb':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
      case 'categories':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>;
      case 'modes':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>;
      case 'dev':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>;
      default:
        return null;
    }
  };

  const tabs: { id: TabId; label: string; iconType: string }[] = [
    { id: 'system', label: 'System', iconType: 'system' },
    { id: 'oauth', label: 'OAuth', iconType: 'oauth' },
    { id: 'llm', label: 'LLM', iconType: 'llm' },
    { id: 'features', label: 'Features', iconType: 'features' },
    { id: 'filters', label: 'Filters', iconType: 'filters' },
    { id: 'filter_chains', label: 'Filter Chains', iconType: 'filter_chains' },
    { id: 'global_kb', label: 'Global KB', iconType: 'global_kb' },
    { id: 'categories', label: 'GPT Categories', iconType: 'categories' },
    { id: 'modes', label: 'Assistant Modes', iconType: 'modes' },
    { id: 'tiers', label: 'Tiers', iconType: 'tiers' },
    { id: 'users', label: 'Users', iconType: 'users' },
    { id: 'chats', label: 'Chats', iconType: 'chats' },
    { id: 'tools', label: 'Tools', iconType: 'tools' },
    { id: 'dev', label: 'Site Dev', iconType: 'dev' },
  ];
  
  if (!user?.is_admin) {
    return null;
  }
  
  return (
    <div className="h-full overflow-y-auto bg-[var(--color-background)]">
      <div className="max-w-6xl mx-auto p-6 pb-12">
        <h1 className="text-2xl font-bold text-[var(--color-text)] mb-6">Admin Panel</h1>
        
        {/* Status Messages */}
        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            {typeof error === 'string' ? error : JSON.stringify(error)}
            <button onClick={() => setError(null)} className="ml-2 text-red-300 hover:text-red-100">×</button>
          </div>
        )}
        {success && (
          <div className="mb-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg text-green-400">
            {success}
          </div>
        )}
        
        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-6 border-b border-[var(--color-border)] pb-4">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                  : 'bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]'
              }`}
            >
              <AdminTabIcon type={tab.iconType} />
              {tab.label}
            </button>
          ))}
        </div>
        
        {isLoading ? (
          <div className="text-center py-8 text-[var(--color-text-secondary)]">Loading...</div>
        ) : (
          <>
            {/* SYSTEM SETTINGS TAB */}
            {activeTab === 'system' && systemSettings && (
              <div className="space-y-6">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Prompts</h2>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Default System Prompt</label>
                      <textarea
                        value={systemSettings.default_system_prompt}
                        onChange={(e) => setSystemSettings({ ...systemSettings, default_system_prompt: e.target.value })}
                        rows={3}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                        All Models Prompt
                        <span className="ml-2 text-xs text-[var(--color-text-muted)]">(appended to ALL system prompts including Custom GPTs)</span>
                      </label>
                      <textarea
                        value={systemSettings.all_models_prompt || ''}
                        onChange={(e) => setSystemSettings({ ...systemSettings, all_models_prompt: e.target.value })}
                        rows={4}
                        placeholder="Instructions that apply to all models and Custom GPTs..."
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Title Generation Prompt</label>
                      <textarea
                        value={systemSettings.title_generation_prompt}
                        onChange={(e) => setSystemSettings({ ...systemSettings, title_generation_prompt: e.target.value })}
                        rows={2}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">RAG Context Prompt (Legacy/Default)</label>
                      <textarea
                        value={systemSettings.rag_context_prompt}
                        onChange={(e) => setSystemSettings({ ...systemSettings, rag_context_prompt: e.target.value })}
                        rows={2}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">Fallback prompt if specific prompts below are empty</p>
                    </div>
                    
                    <div className="col-span-full border-t border-[var(--color-border)] pt-4 mt-2">
                      <h3 className="text-sm font-semibold text-[var(--color-text)] mb-3">Source-Specific RAG Prompts</h3>
                      <p className="text-xs text-[var(--color-text-secondary)] mb-4">Use {'{context}'} placeholder for the retrieved content. Use {'{sources}'} for source names (Global KB only).</p>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Global Knowledge Base Prompt</label>
                      <textarea
                        value={systemSettings.rag_prompt_global_kb || ''}
                        onChange={(e) => setSystemSettings({ ...systemSettings, rag_prompt_global_kb: e.target.value })}
                        rows={4}
                        placeholder="Leave empty for default authoritative prompt. Use {context} and {sources} placeholders."
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Custom GPT Knowledge Base Prompt</label>
                      <textarea
                        value={systemSettings.rag_prompt_gpt_kb || ''}
                        onChange={(e) => setSystemSettings({ ...systemSettings, rag_prompt_gpt_kb: e.target.value })}
                        rows={4}
                        placeholder="Leave empty to use legacy prompt. Use {context} placeholder."
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">User Documents Prompt</label>
                      <textarea
                        value={systemSettings.rag_prompt_user_docs || ''}
                        onChange={(e) => setSystemSettings({ ...systemSettings, rag_prompt_user_docs: e.target.value })}
                        rows={4}
                        placeholder="Leave empty to use legacy prompt. Use {context} placeholder."
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Chat History Knowledge Prompt</label>
                      <textarea
                        value={systemSettings.rag_prompt_chat_history || ''}
                        onChange={(e) => setSystemSettings({ ...systemSettings, rag_prompt_chat_history: e.target.value })}
                        rows={4}
                        placeholder="Leave empty for default chat history prompt. Use {context} placeholder."
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                  </div>
                </div>
                
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Token Pricing & Limits</h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Input Token Price (per 1M)</label>
                      <input
                        type="number"
                        step="0.01"
                        value={systemSettings.input_token_price}
                        onChange={(e) => setSystemSettings({ ...systemSettings, input_token_price: parseFloat(e.target.value) || 0 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Output Token Price (per 1M)</label>
                      <input
                        type="number"
                        step="0.01"
                        value={systemSettings.output_token_price}
                        onChange={(e) => setSystemSettings({ ...systemSettings, output_token_price: parseFloat(e.target.value) || 0 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                        Token Refill Interval (hours)
                        <span className="ml-2 text-xs opacity-70">= {formatHours(systemSettings.token_refill_interval_hours)}</span>
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={systemSettings.token_refill_interval_hours}
                        onChange={(e) => setSystemSettings({ ...systemSettings, token_refill_interval_hours: parseInt(e.target.value) || 1 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Free Tier Tokens</label>
                      <input
                        type="number"
                        min="0"
                        value={systemSettings.free_tier_tokens}
                        onChange={(e) => setSystemSettings({ ...systemSettings, free_tier_tokens: parseInt(e.target.value) || 0 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Pro Tier Tokens</label>
                      <input
                        type="number"
                        min="0"
                        value={systemSettings.pro_tier_tokens}
                        onChange={(e) => setSystemSettings({ ...systemSettings, pro_tier_tokens: parseInt(e.target.value) || 0 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Enterprise Tier Tokens</label>
                      <input
                        type="number"
                        min="0"
                        value={systemSettings.enterprise_tier_tokens}
                        onChange={(e) => setSystemSettings({ ...systemSettings, enterprise_tier_tokens: parseInt(e.target.value) || 0 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                  </div>
                </div>
                
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Storage Limits</h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max File Upload Size (MB)</label>
                      <input
                        type="number"
                        min="1"
                        max="10000"
                        value={systemSettings.max_upload_size_mb}
                        onChange={(e) => setSystemSettings({ ...systemSettings, max_upload_size_mb: parseInt(e.target.value) || 100 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum size per uploaded file</p>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max Knowledge Store Size (MB)</label>
                      <input
                        type="number"
                        min="1"
                        max="100000"
                        value={systemSettings.max_knowledge_store_size_mb}
                        onChange={(e) => setSystemSettings({ ...systemSettings, max_knowledge_store_size_mb: parseInt(e.target.value) || 500 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">Total size limit per knowledge base</p>
                    </div>
                    
                    <div className="col-span-full">
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-2">Max Knowledge Stores per Tier</label>
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Free</label>
                          <input
                            type="number"
                            min="0"
                            value={systemSettings.max_knowledge_stores_free}
                            onChange={(e) => setSystemSettings({ ...systemSettings, max_knowledge_stores_free: parseInt(e.target.value) || 0 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Pro</label>
                          <input
                            type="number"
                            min="0"
                            value={systemSettings.max_knowledge_stores_pro}
                            onChange={(e) => setSystemSettings({ ...systemSettings, max_knowledge_stores_pro: parseInt(e.target.value) || 0 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Enterprise</label>
                          <input
                            type="number"
                            min="0"
                            value={systemSettings.max_knowledge_stores_enterprise}
                            onChange={(e) => setSystemSettings({ ...systemSettings, max_knowledge_stores_enterprise: parseInt(e.target.value) || 0 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={saveSystemSettings}
                  disabled={isSaving}
                  className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save System Settings'}
                </button>
              </div>
            )}
            
            {/* OAUTH SETTINGS TAB */}
            {activeTab === 'oauth' && oauthSettings && (
              <div className="space-y-6">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">🔷 Google OAuth</h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <input
                        type="checkbox"
                        id="google-enabled"
                        checked={oauthSettings.google_oauth_enabled}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, google_oauth_enabled: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="google-enabled" className="text-sm text-[var(--color-text)]">Enable Google OAuth</label>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client ID</label>
                      <input
                        type="text"
                        value={oauthSettings.google_client_id}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, google_client_id: e.target.value })}
                        placeholder="xxxxx.apps.googleusercontent.com"
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client Secret</label>
                      <input
                        type="password"
                        value={oauthSettings.google_client_secret}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, google_client_secret: e.target.value })}
                        placeholder="GOCSPX-xxxxx"
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Timeout (seconds)</label>
                      <input
                        type="number"
                        min="5"
                        max="300"
                        value={oauthSettings.google_oauth_timeout}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, google_oauth_timeout: parseInt(e.target.value) || 30 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                  </div>
                </div>
                
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">⬛ GitHub OAuth</h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <input
                        type="checkbox"
                        id="github-enabled"
                        checked={oauthSettings.github_oauth_enabled}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, github_oauth_enabled: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="github-enabled" className="text-sm text-[var(--color-text)]">Enable GitHub OAuth</label>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client ID</label>
                      <input
                        type="text"
                        value={oauthSettings.github_client_id}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, github_client_id: e.target.value })}
                        placeholder="Iv1.xxxxx"
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client Secret</label>
                      <input
                        type="password"
                        value={oauthSettings.github_client_secret}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, github_client_secret: e.target.value })}
                        placeholder="xxxxx"
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Timeout (seconds)</label>
                      <input
                        type="number"
                        min="5"
                        max="300"
                        value={oauthSettings.github_oauth_timeout}
                        onChange={(e) => setOAuthSettings({ ...oauthSettings, github_oauth_timeout: parseInt(e.target.value) || 30 })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={saveOAuthSettings}
                  disabled={isSaving}
                  className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save OAuth Settings'}
                </button>
              </div>
            )}
            
            {/* LLM SETTINGS TAB */}
            {activeTab === 'llm' && llmSettings && (
              <div className="space-y-6">
                {/* LLM PROVIDERS SECTION */}
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h2 className="text-lg font-semibold text-[var(--color-text)]">LLM Providers</h2>
                      <p className="text-xs text-[var(--color-text-secondary)]">Configure multiple models with hybrid routing (smart text + vision)</p>
                    </div>
                    <button
                      onClick={() => { setEditingProvider(null); setShowProviderForm(true); }}
                      className="px-3 py-1.5 bg-[var(--color-primary)] text-white rounded-lg text-sm hover:opacity-90"
                    >
                      + Add Provider
                    </button>
                  </div>
                  
                  {llmProviders.length === 0 ? (
                    <div className="text-center py-8 text-[var(--color-text-secondary)]">
                      <p>No providers configured. Add a provider or use legacy settings below.</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {llmProviders.map((provider) => (
                        <div key={provider.id} className="p-4 bg-[var(--color-background)] rounded-lg border border-[var(--color-border)]">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-[var(--color-text)]">{provider.name}</span>
                                {provider.is_default && (
                                  <span className="px-2 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded">Default</span>
                                )}
                                {provider.is_vision_default && (
                                  <span className="px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">Vision</span>
                                )}
                                {provider.is_multimodal && (
                                  <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded">MM</span>
                                )}
                                {!provider.is_enabled && (
                                  <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded">Disabled</span>
                                )}
                              </div>
                              <p className="text-sm text-[var(--color-text-secondary)] mt-1">{provider.model_id}</p>
                              <p className="text-xs text-[var(--color-text-secondary)]">{provider.base_url}</p>
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={() => testProvider(provider.id)}
                                disabled={testingProvider === provider.id}
                                className="px-2 py-1 text-xs bg-[var(--color-surface)] border border-[var(--color-border)] rounded hover:bg-[var(--color-background)]"
                              >
                                {testingProvider === provider.id ? '...' : 'Test'}
                              </button>
                              <button
                                onClick={() => { setEditingProvider(provider); setShowProviderForm(true); }}
                                className="px-2 py-1 text-xs bg-[var(--color-surface)] border border-[var(--color-border)] rounded hover:bg-[var(--color-background)]"
                              >
                                Edit
                              </button>
                              <button
                                onClick={() => deleteProvider(provider.id)}
                                className="px-2 py-1 text-xs bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                              >
                                Delete
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* Hybrid routing explanation */}
                  {llmProviders.length > 0 && (
                    <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-sm">
                      <p className="text-blue-400 font-medium">Hybrid Routing</p>
                      <p className="text-[var(--color-text-secondary)] text-xs mt-1">
                        Set one model as <b>Default</b> (text/reasoning) and another as <b>Vision</b> (image understanding).
                        When images are sent, the Vision model describes them, then the Default model responds.
                      </p>
                    </div>
                  )}
                </div>
                
                {/* Provider Form Modal */}
                {showProviderForm && (
                  <div className="bg-[var(--color-surface)] rounded-xl p-6 border-2 border-[var(--color-primary)]">
                    <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                      {editingProvider ? 'Edit Provider' : 'Add Provider'}
                    </h3>
                    <form onSubmit={(e) => {
                      e.preventDefault();
                      const form = e.target as HTMLFormElement;
                      const formData = new FormData(form);
                      saveProvider({
                        name: formData.get('name') as string,
                        base_url: formData.get('base_url') as string,
                        api_key: (formData.get('api_key') as string) || undefined,
                        model_id: formData.get('model_id') as string,
                        is_multimodal: formData.get('is_multimodal') === 'on',
                        supports_tools: formData.get('supports_tools') === 'on',
                        supports_streaming: formData.get('supports_streaming') === 'on',
                        is_default: formData.get('is_default') === 'on',
                        is_vision_default: formData.get('is_vision_default') === 'on',
                        is_enabled: formData.get('is_enabled') === 'on',
                        timeout: parseInt(formData.get('timeout') as string) || 300,
                        max_tokens: parseInt(formData.get('max_tokens') as string) || 8192,
                        context_size: parseInt(formData.get('context_size') as string) || 128000,
                        temperature: formData.get('temperature') as string || '0.7',
                        vision_prompt: (formData.get('vision_prompt') as string) || undefined,
                      });
                    }} className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Display Name *</label>
                          <input
                            name="name"
                            required
                            defaultValue={editingProvider?.name || ''}
                            placeholder="GPT-4o, Llama 70B, etc."
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Model ID *</label>
                          <input
                            name="model_id"
                            required
                            defaultValue={editingProvider?.model_id || ''}
                            placeholder="gpt-4o, llama3.1:70b, etc."
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">API Base URL *</label>
                        <input
                          name="base_url"
                          required
                          defaultValue={editingProvider?.base_url || ''}
                          placeholder="http://localhost:11434/v1"
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">API Key</label>
                        <input
                          name="api_key"
                          type="password"
                          defaultValue=""
                          placeholder="Leave empty for local models"
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        />
                        {editingProvider?.api_key && (
                          <p className="text-xs text-[var(--color-text-secondary)] mt-1">Current: {editingProvider.api_key}</p>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Timeout (s)</label>
                          <input
                            name="timeout"
                            type="number"
                            defaultValue={editingProvider?.timeout || 300}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max Tokens</label>
                          <input
                            name="max_tokens"
                            type="number"
                            defaultValue={editingProvider?.max_tokens || 8192}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Context Size</label>
                          <input
                            name="context_size"
                            type="number"
                            defaultValue={editingProvider?.context_size || 128000}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Temperature</label>
                          <input
                            name="temperature"
                            type="text"
                            defaultValue={editingProvider?.temperature || '0.7'}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <label className="flex items-center gap-2 p-3 bg-[var(--color-background)] rounded-lg cursor-pointer">
                          <input name="is_enabled" type="checkbox" defaultChecked={editingProvider?.is_enabled ?? true} className="w-4 h-4" />
                          <span className="text-sm text-[var(--color-text)]">Enabled</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-[var(--color-background)] rounded-lg cursor-pointer">
                          <input name="is_default" type="checkbox" defaultChecked={editingProvider?.is_default ?? false} className="w-4 h-4" />
                          <span className="text-sm text-[var(--color-text)]">Default Model</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-[var(--color-background)] rounded-lg cursor-pointer">
                          <input name="is_vision_default" type="checkbox" defaultChecked={editingProvider?.is_vision_default ?? false} className="w-4 h-4" />
                          <span className="text-sm text-[var(--color-text)]">Vision Model</span>
                        </label>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <label className="flex items-center gap-2 p-3 bg-[var(--color-background)] rounded-lg cursor-pointer">
                          <input name="is_multimodal" type="checkbox" defaultChecked={editingProvider?.is_multimodal ?? false} className="w-4 h-4" />
                          <span className="text-sm text-[var(--color-text)]">Multimodal (Vision)</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-[var(--color-background)] rounded-lg cursor-pointer">
                          <input name="supports_tools" type="checkbox" defaultChecked={editingProvider?.supports_tools ?? true} className="w-4 h-4" />
                          <span className="text-sm text-[var(--color-text)]">Supports Tools</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-[var(--color-background)] rounded-lg cursor-pointer">
                          <input name="supports_streaming" type="checkbox" defaultChecked={editingProvider?.supports_streaming ?? true} className="w-4 h-4" />
                          <span className="text-sm text-[var(--color-text)]">Supports Streaming</span>
                        </label>
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Vision Prompt (optional)</label>
                        <textarea
                          name="vision_prompt"
                          rows={3}
                          defaultValue={editingProvider?.vision_prompt || ''}
                          placeholder="Custom prompt for image descriptions. Use {user_message} placeholder."
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                        />
                      </div>
                      
                      <div className="flex gap-3">
                        <button
                          type="submit"
                          disabled={isSaving}
                          className="px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:opacity-90 disabled:opacity-50"
                        >
                          {isSaving ? 'Saving...' : editingProvider ? 'Update Provider' : 'Add Provider'}
                        </button>
                        <button
                          type="button"
                          onClick={() => { setShowProviderForm(false); setEditingProvider(null); }}
                          className="px-4 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] rounded-lg hover:bg-[var(--color-background)]"
                        >
                          Cancel
                        </button>
                      </div>
                    </form>
                  </div>
                )}
                
                {/* LEGACY LLM CONNECTION */}
                <details className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
                  <summary className="p-4 cursor-pointer text-[var(--color-text)] font-medium">
                    Legacy LLM Settings (fallback if no providers configured)
                  </summary>
                  <div className="p-6 pt-0">
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">API Base URL</label>
                        <input
                          type="text"
                          value={llmSettings.llm_api_base_url}
                          onChange={(e) => setLLMSettings({ ...llmSettings, llm_api_base_url: e.target.value })}
                          placeholder="http://localhost:11434/v1"
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                        />
                        <p className="text-xs text-[var(--color-text-secondary)] mt-1">OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, etc.)</p>
                      </div>
                    
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">API Key</label>
                        <input
                          type="password"
                          value={llmSettings.llm_api_key}
                          onChange={(e) => setLLMSettings({ ...llmSettings, llm_api_key: e.target.value })}
                          placeholder="sk-xxxxx or leave empty"
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                        />
                      </div>
                    
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Default Model</label>
                        <input
                          type="text"
                          value={llmSettings.llm_model}
                          onChange={(e) => setLLMSettings({ ...llmSettings, llm_model: e.target.value })}
                          placeholder="default"
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                        />
                      </div>
                    
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Timeout (seconds)</label>
                          <input
                            type="number"
                            min="10"
                            max="600"
                            value={llmSettings.llm_timeout}
                            onChange={(e) => setLLMSettings({ ...llmSettings, llm_timeout: parseInt(e.target.value) || 300 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                        </div>
                      
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max Tokens (default)</label>
                          <input
                            type="number"
                            min="1"
                            value={llmSettings.llm_max_tokens}
                            onChange={(e) => setLLMSettings({ ...llmSettings, llm_max_tokens: parseInt(e.target.value) || 4096 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                        </div>
                      
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Temperature (default)</label>
                          <input
                            type="number"
                            min="0"
                            max="2"
                            step="0.1"
                            value={llmSettings.llm_temperature}
                            onChange={(e) => setLLMSettings({ ...llmSettings, llm_temperature: parseFloat(e.target.value) || 0.7 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="llm-stream"
                        checked={llmSettings.llm_stream_default}
                        onChange={(e) => setLLMSettings({ ...llmSettings, llm_stream_default: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="llm-stream" className="text-sm text-[var(--color-text)]">Enable streaming by default</label>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="llm-multimodal"
                        checked={llmSettings.llm_multimodal}
                        onChange={(e) => setLLMSettings({ ...llmSettings, llm_multimodal: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="llm-multimodal" className="text-sm text-[var(--color-text)]">
                        Model supports vision/images (multimodal)
                      </label>
                    </div>
                    
                    {/* History Compression Settings */}
                    <div className="border-t border-[var(--color-border)] pt-4 mt-4">
                      <h4 className="text-sm font-medium text-[var(--color-text)] mb-3 flex items-center gap-2">
                        📦 History Compression
                        <span className="text-xs font-normal text-[var(--color-text-secondary)]">
                          (reduces context usage for long conversations)
                        </span>
                      </h4>
                      
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="history-compression-enabled"
                            checked={llmSettings.history_compression_enabled}
                            onChange={(e) => setLLMSettings({ ...llmSettings, history_compression_enabled: e.target.checked })}
                            className="rounded"
                          />
                          <label htmlFor="history-compression-enabled" className="text-sm text-[var(--color-text)]">
                            Enable automatic history compression
                          </label>
                        </div>
                        
                        {llmSettings.history_compression_enabled && (
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pl-6">
                            <div>
                              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
                                Compression Threshold (messages)
                              </label>
                              <input
                                type="number"
                                min="5"
                                max="100"
                                value={llmSettings.history_compression_threshold}
                                onChange={(e) => setLLMSettings({ ...llmSettings, history_compression_threshold: parseInt(e.target.value) || 20 })}
                                className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                              />
                              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Compress after this many messages</p>
                            </div>
                            
                            <div>
                              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
                                Keep Recent (message pairs)
                              </label>
                              <input
                                type="number"
                                min="2"
                                max="20"
                                value={llmSettings.history_compression_keep_recent}
                                onChange={(e) => setLLMSettings({ ...llmSettings, history_compression_keep_recent: parseInt(e.target.value) || 6 })}
                                className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                              />
                              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Recent turns kept verbatim</p>
                            </div>
                            
                            <div>
                              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
                                Target Tokens
                              </label>
                              <input
                                type="number"
                                min="2000"
                                max="128000"
                                step="1000"
                                value={llmSettings.history_compression_target_tokens}
                                onChange={(e) => setLLMSettings({ ...llmSettings, history_compression_target_tokens: parseInt(e.target.value) || 8000 })}
                                className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                              />
                              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Compress if exceeds this</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Test Connection */}
                    <div className="border-t border-[var(--color-border)] pt-4 mt-4">
                      <button
                        onClick={testLLMConnection}
                        disabled={llmTesting}
                        className="px-4 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] rounded-lg hover:bg-[var(--color-background)] disabled:opacity-50"
                      >
                        {llmTesting ? 'Testing...' : '🔌 Test Connection'}
                      </button>
                      
                      {llmTestResult && (
                        <div className={`mt-3 p-3 rounded-lg text-sm ${
                          llmTestResult.success
                            ? 'bg-green-500/10 border border-green-500/30 text-green-400'
                            : 'bg-red-500/10 border border-red-500/30 text-red-400'
                        }`}>
                          <p className="font-medium">{llmTestResult.success ? '✓ Connected' : '✗ Failed'}</p>
                          <p className="text-xs mt-1 opacity-80">{llmTestResult.message}</p>
                          {llmTestResult.success && llmTestResult.models && llmTestResult.models.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {llmTestResult.models.map((m, i) => (
                                <span key={i} className="px-2 py-0.5 rounded text-xs bg-[var(--color-background)]">{m}</span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={saveLLMSettings}
                    disabled={isSaving}
                    className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50 mt-4"
                  >
                    {isSaving ? 'Saving...' : 'Save Legacy Settings'}
                  </button>
                </details>
              </div>
            )}
            
            {/* FEATURE FLAGS TAB */}
            {activeTab === 'features' && featureFlags && (
              <div className="space-y-6">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Feature Flags</h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-[var(--color-background)] rounded-lg">
                      <div>
                        <p className="text-[var(--color-text)] font-medium">Enable Registration</p>
                        <p className="text-xs text-[var(--color-text-secondary)]">Allow new users to create accounts</p>
                      </div>
                      <input
                        type="checkbox"
                        checked={featureFlags.enable_registration}
                        onChange={(e) => setFeatureFlags({ ...featureFlags, enable_registration: e.target.checked })}
                        className="w-5 h-5 rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-[var(--color-background)] rounded-lg">
                      <div>
                        <p className="text-[var(--color-text)] font-medium">Enable Billing</p>
                        <p className="text-xs text-[var(--color-text-secondary)]">Show billing page and enforce tier limits</p>
                      </div>
                      <input
                        type="checkbox"
                        checked={featureFlags.enable_billing}
                        onChange={(e) => setFeatureFlags({ ...featureFlags, enable_billing: e.target.checked })}
                        className="w-5 h-5 rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-[var(--color-background)] rounded-lg border-2 border-yellow-500/30">
                      <div>
                        <p className="text-[var(--color-text)] font-medium">🎉 Free For All Mode</p>
                        <p className="text-xs text-[var(--color-text-secondary)]">Disable ALL token limits - unlimited usage for everyone</p>
                      </div>
                      <input
                        type="checkbox"
                        checked={featureFlags.freeforall}
                        onChange={(e) => setFeatureFlags({ ...featureFlags, freeforall: e.target.checked })}
                        className="w-5 h-5 rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-[var(--color-background)] rounded-lg border-2 border-red-500/30">
                      <div>
                        <p className="text-[var(--color-text)] font-medium">🛡️ Safety Filters</p>
                        <p className="text-xs text-[var(--color-text-secondary)]">Enable prompt injection detection and content moderation filters</p>
                        <p className="text-xs text-yellow-500/80 mt-1">⚠️ May cause false positives with code files containing patterns like "system:", "user:", etc.</p>
                      </div>
                      <input
                        type="checkbox"
                        checked={featureFlags.enable_safety_filters}
                        onChange={(e) => setFeatureFlags({ ...featureFlags, enable_safety_filters: e.target.checked })}
                        className="w-5 h-5 rounded"
                      />
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={saveFeatureFlags}
                  disabled={isSaving}
                  className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save Feature Flags'}
                </button>
                
                {/* API Rate Limits */}
                {apiRateLimits && (
                  <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                    <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">API Rate Limits</h2>
                    <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                      Configure rate limits for the OpenAI-compatible /v1 API endpoints (requests per minute per API key)
                    </p>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                          Chat Completions (/v1/chat/completions)
                        </label>
                        <input
                          type="number"
                          value={apiRateLimits.api_rate_limit_completions}
                          onChange={(e) => setApiRateLimits({ ...apiRateLimits, api_rate_limit_completions: parseInt(e.target.value) || 60 })}
                          min={1}
                          max={1000}
                          className="w-full px-3 py-2 bg-[var(--color-background)] text-[var(--color-text)] border border-[var(--color-border)] rounded-lg"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                          Embeddings (/v1/embeddings)
                        </label>
                        <input
                          type="number"
                          value={apiRateLimits.api_rate_limit_embeddings}
                          onChange={(e) => setApiRateLimits({ ...apiRateLimits, api_rate_limit_embeddings: parseInt(e.target.value) || 200 })}
                          min={1}
                          max={1000}
                          className="w-full px-3 py-2 bg-[var(--color-background)] text-[var(--color-text)] border border-[var(--color-border)] rounded-lg"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                          Image Generations (/v1/images/generations)
                        </label>
                        <input
                          type="number"
                          value={apiRateLimits.api_rate_limit_images}
                          onChange={(e) => setApiRateLimits({ ...apiRateLimits, api_rate_limit_images: parseInt(e.target.value) || 10 })}
                          min={1}
                          max={100}
                          className="w-full px-3 py-2 bg-[var(--color-background)] text-[var(--color-text)] border border-[var(--color-border)] rounded-lg"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                          Models List (/v1/models)
                        </label>
                        <input
                          type="number"
                          value={apiRateLimits.api_rate_limit_models}
                          onChange={(e) => setApiRateLimits({ ...apiRateLimits, api_rate_limit_models: parseInt(e.target.value) || 100 })}
                          min={1}
                          max={1000}
                          className="w-full px-3 py-2 bg-[var(--color-background)] text-[var(--color-text)] border border-[var(--color-border)] rounded-lg"
                        />
                      </div>
                    </div>
                    
                    <button
                      onClick={saveApiRateLimits}
                      disabled={isSaving}
                      className="mt-4 px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                    >
                      {isSaving ? 'Saving...' : 'Save Rate Limits'}
                    </button>
                  </div>
                )}
              </div>
            )}
            
            {/* FILTERS TAB */}
            {activeTab === 'filters' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-[var(--color-text)]">Message Filters</h2>
                  <button
                    onClick={() => {
                      resetFilterForm();
                      setShowFilterModal(true);
                    }}
                    className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
                  >
                    + Add Filter
                  </button>
                </div>
                
                {/* Filter type legend */}
                <div className="flex flex-wrap gap-3 text-sm">
                  <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-400">User → LLM</span>
                  <span className="px-2 py-1 rounded bg-green-500/20 text-green-400">LLM → User</span>
                  <span className="px-2 py-1 rounded bg-orange-500/20 text-orange-400">To Tools</span>
                  <span className="px-2 py-1 rounded bg-purple-500/20 text-purple-400">From Tools</span>
                </div>
                
                {filtersLoading ? (
                  <div className="text-center py-8 text-[var(--color-text-secondary)]">Loading filters...</div>
                ) : filters.length === 0 ? (
                  <div className="text-center py-8 text-[var(--color-text-secondary)]">
                    No filters configured. Create one to get started.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Group filters by type */}
                    {(['to_llm', 'from_llm', 'to_tools', 'from_tools'] as const).map(filterType => {
                      const typeFilters = filters.filter(f => f.filter_type === filterType);
                      if (typeFilters.length === 0) return null;
                      
                      return (
                        <div key={filterType} className="space-y-2">
                          <h3 className={`text-sm font-medium ${getFilterTypeColor(filterType)} px-2 py-1 rounded inline-block`}>
                            {getFilterTypeLabel(filterType)}
                          </h3>
                          <div className="grid gap-3">
                            {typeFilters.map(filter => (
                              <div
                                key={filter.id}
                                className={`bg-[var(--color-surface)] rounded-lg p-4 border ${
                                  filter.enabled ? 'border-[var(--color-border)]' : 'border-red-500/30 opacity-60'
                                }`}
                              >
                                <div className="flex items-start justify-between gap-4">
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                      <span className="font-medium text-[var(--color-text)]">{filter.name}</span>
                                      <span className={`px-2 py-0.5 rounded text-xs ${
                                        filter.action === 'block' ? 'bg-red-500/20 text-red-400' :
                                        filter.action === 'modify' ? 'bg-yellow-500/20 text-yellow-400' :
                                        filter.action === 'log' ? 'bg-blue-500/20 text-blue-400' :
                                        'bg-gray-500/20 text-gray-400'
                                      }`}>
                                        {filter.action}
                                      </span>
                                      <span className="px-2 py-0.5 rounded text-xs bg-[var(--color-background)] text-[var(--color-text-secondary)]">
                                        {filter.priority}
                                      </span>
                                    </div>
                                    {filter.description && (
                                      <p className="text-sm text-[var(--color-text-secondary)] mt-1">{filter.description}</p>
                                    )}
                                    <div className="text-xs text-[var(--color-text-secondary)] mt-2">
                                      Mode: {filter.filter_mode}
                                      {filter.pattern && <span className="ml-2">| Pattern: <code className="bg-[var(--color-background)] px-1 rounded">{filter.pattern.substring(0, 30)}...</code></span>}
                                      {filter.word_list && filter.word_list.length > 0 && <span className="ml-2">| Words: {filter.word_list.length}</span>}
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <button
                                      onClick={() => handleToggleFilter(filter.id!)}
                                      className={`px-3 py-1 rounded text-sm ${
                                        filter.enabled
                                          ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                                          : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                      }`}
                                    >
                                      {filter.enabled ? 'Enabled' : 'Disabled'}
                                    </button>
                                    <button
                                      onClick={() => {
                                        setEditingFilter(filter);
                                        setFilterForm(filter);
                                        setShowFilterModal(true);
                                      }}
                                      className="px-3 py-1 bg-[var(--color-background)] text-[var(--color-text)] rounded text-sm hover:bg-[var(--color-border)]"
                                    >
                                      Edit
                                    </button>
                                    <button
                                      onClick={() => handleDeleteFilter(filter.id!)}
                                      className="px-3 py-1 bg-red-500/20 text-red-400 rounded text-sm hover:bg-red-500/30"
                                    >
                                      Delete
                                    </button>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            
            {/* FILTER CHAINS TAB */}
            {activeTab === 'filter_chains' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-[var(--color-text)]">Configurable Filter Chains</h2>
                    <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                      Build agentic flows with LLM decisions, tool calls, conditionals, and loops
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      resetFilterChainForm();
                      setShowFilterChainModal(true);
                    }}
                    className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
                  >
                    + New Chain
                  </button>
                </div>
                
                {filterChainsLoading ? (
                  <div className="text-center py-8 text-[var(--color-text-secondary)]">Loading filter chains...</div>
                ) : filterChains.length === 0 ? (
                  <div className="text-center py-8 text-[var(--color-text-secondary)]">
                    No filter chains configured. Create one to get started.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {filterChains.map(chain => (
                      <div
                        key={chain.id}
                        className={`bg-[var(--color-surface)] rounded-lg p-4 border ${
                          chain.enabled ? 'border-[var(--color-border)]' : 'border-red-500/30 opacity-60'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-[var(--color-text)]">{chain.name}</span>
                              <span className="px-2 py-0.5 rounded text-xs bg-[var(--color-background)] text-[var(--color-text-secondary)]">
                                Priority: {chain.priority}
                              </span>
                              {chain.bidirectional && (
                                <span className="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400">
                                  Bidirectional
                                </span>
                              )}
                              {chain.retain_history && (
                                <span className="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400">
                                  Hidden
                                </span>
                              )}
                              {chain.debug && (
                                <span className="px-2 py-0.5 rounded text-xs bg-yellow-500/20 text-yellow-400">
                                  🐛 Debug
                                </span>
                              )}
                              {chain.skip_if_rag_hit && (
                                <span className="px-2 py-0.5 rounded text-xs bg-green-500/20 text-green-400">
                                  Skip if RAG
                                </span>
                              )}
                            </div>
                            {chain.description && (
                              <p className="text-sm text-[var(--color-text-secondary)] mt-1">{chain.description}</p>
                            )}
                            <div className="text-xs text-[var(--color-text-secondary)] mt-2">
                              {chain.definition?.steps?.length || 0} steps
                              {chain.max_iterations && <span className="ml-2">| Max iterations: {chain.max_iterations}</span>}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => handleToggleFilterChain(chain.id!)}
                              className={`px-3 py-1 rounded text-sm ${
                                chain.enabled
                                  ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                                  : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                              }`}
                            >
                              {chain.enabled ? 'Enabled' : 'Disabled'}
                            </button>
                            <button
                              onClick={() => {
                                setEditingFilterChain(chain);
                                setFilterChainForm(chain);
                                setFilterChainJson(JSON.stringify(chain.definition, null, 2));
                                setShowFilterChainModal(true);
                              }}
                              className="px-3 py-1 bg-[var(--color-background)] text-[var(--color-text)] rounded text-sm hover:bg-[var(--color-border)]"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => setPreviewJsonChain(chain)}
                              className="px-3 py-1 bg-[var(--color-background)] text-[var(--color-text)] rounded text-sm hover:bg-[var(--color-border)]"
                            >
                              Preview JSON
                            </button>
                            <button
                              onClick={() => handleDeleteFilterChain(chain.id!)}
                              className="px-3 py-1 bg-red-500/20 text-red-400 rounded text-sm hover:bg-red-500/30"
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {/* TIERS TAB */}
            {activeTab === 'tiers' && (
              <div className="space-y-6">
                {tiers.map((tier) => (
                  <div key={tier.id} className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-[var(--color-text)]">{tier.name}</h3>
                      <button
                        onClick={() => setEditingTier(editingTier === tier.id ? null : tier.id)}
                        className="text-sm text-[var(--color-primary)] hover:underline"
                      >
                        {editingTier === tier.id ? 'Done' : 'Edit'}
                      </button>
                    </div>
                    
                    {editingTier === tier.id ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Name</label>
                            <input
                              type="text"
                              value={tier.name}
                              onChange={(e) => updateTier(tier.id, { name: e.target.value })}
                              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                            />
                          </div>
                          <div>
                            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Price ($/month)</label>
                            <input
                              type="number"
                              value={tier.price}
                              onChange={(e) => updateTier(tier.id, { price: parseFloat(e.target.value) || 0 })}
                              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                            />
                          </div>
                          <div>
                            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Tokens</label>
                            <input
                              type="number"
                              value={tier.tokens}
                              onChange={(e) => updateTier(tier.id, { tokens: parseInt(e.target.value) || 0 })}
                              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={tier.popular}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setTiers(tiers.map(t => ({ ...t, popular: t.id === tier.id })));
                                } else {
                                  updateTier(tier.id, { popular: false });
                                }
                              }}
                              className="rounded"
                            />
                            <label className="text-sm text-[var(--color-text)]">Popular (highlighted)</label>
                          </div>
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Features</label>
                          <div className="space-y-2">
                            {tier.features.map((feature, i) => (
                              <div key={i} className="flex items-center gap-2">
                                <span className="flex-1 px-3 py-1 bg-[var(--color-background)] rounded text-sm text-[var(--color-text)]">{feature}</span>
                                <button
                                  onClick={() => removeFeatureFromTier(tier.id, i)}
                                  className="text-red-400 hover:text-red-300"
                                >×</button>
                              </div>
                            ))}
                            <div className="flex gap-2">
                              <input
                                type="text"
                                value={newFeature}
                                onChange={(e) => setNewFeature(e.target.value)}
                                placeholder="Add feature..."
                                className="flex-1 px-3 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                                onKeyDown={(e) => e.key === 'Enter' && addFeatureToTier(tier.id)}
                              />
                              <button
                                onClick={() => addFeatureToTier(tier.id)}
                                className="px-3 py-1 bg-[var(--color-button)] text-[var(--color-button-text)] rounded text-sm"
                              >Add</button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <p className="text-2xl font-bold text-[var(--color-text)]">${tier.price}<span className="text-sm font-normal text-[var(--color-text-secondary)]">/mo</span></p>
                        <p className="text-sm text-[var(--color-text-secondary)] mb-2">{formatTokens(tier.tokens)} tokens</p>
                        <ul className="text-sm text-[var(--color-text-secondary)]">
                          {tier.features.map((f, i) => (
                            <li key={i}>• {f}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
                
                <button
                  onClick={saveTiers}
                  disabled={isSaving}
                  className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save Tiers'}
                </button>
              </div>
            )}
            
            {/* USERS TAB */}
            {activeTab === 'users' && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={usersSearch}
                    onChange={(e) => setUsersSearch(e.target.value)}
                    placeholder="Search users..."
                    className="flex-1 px-4 py-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)]"
                    onKeyDown={(e) => e.key === 'Enter' && fetchUsers(1, usersSearch)}
                  />
                  <button
                    onClick={() => fetchUsers(1, usersSearch)}
                    className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg"
                  >Search</button>
                </div>
                
                <div className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-[var(--color-background)]">
                      <tr>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">User</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Tier</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Tokens</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Chats</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {usersLoading ? (
                        <tr><td colSpan={5} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">Loading...</td></tr>
                      ) : users.length === 0 ? (
                        <tr><td colSpan={5} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">No users found</td></tr>
                      ) : users.map((u) => (
                        <tr key={u.id} className="border-t border-[var(--color-border)]">
                          <td className="px-4 py-3">
                            <div>
                              <p className="text-[var(--color-text)] font-medium">
                                {u.username}
                                {u.is_admin && <span className="ml-2 text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded">Admin</span>}
                                {!u.is_active && <span className="ml-2 text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">Banned</span>}
                              </p>
                              <p className="text-xs text-[var(--color-text-secondary)]">{u.email}</p>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-[var(--color-text)]">{u.tier}</td>
                          <td className="px-4 py-3 text-[var(--color-text)]">
                            {formatTokens(u.tokens_used)} / {formatTokens(u.tokens_limit)}
                          </td>
                          <td className="px-4 py-3">
                            <button
                              onClick={() => {
                                setSelectedUser(u);
                                setUserAction('chats');
                                fetchUserChats(u.id, 1);
                                setShowUserModal(true);
                              }}
                              className="text-[var(--color-primary)] hover:underline"
                            >
                              {u.chat_count} chats
                            </button>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex gap-2">
                              <button
                                onClick={() => { setSelectedUser(u); setUserAction('upgrade'); setShowUserModal(true); }}
                                className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded hover:bg-blue-500/30"
                              >Tier</button>
                              <button
                                onClick={() => { setSelectedUser(u); setUserAction('refund'); setShowUserModal(true); }}
                                className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded hover:bg-green-500/30"
                              >+Tokens</button>
                              <button
                                onClick={() => handleResetTokens(u)}
                                className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded hover:bg-yellow-500/30"
                              >Reset</button>
                              <button
                                onClick={() => handleToggleAdmin(u)}
                                disabled={u.id === user?.id}
                                className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded hover:bg-purple-500/30 disabled:opacity-50"
                              >{u.is_admin ? 'Demote' : 'Admin'}</button>
                              <button
                                onClick={() => handleBanUser(u)}
                                className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                              >{u.is_active ? 'Ban' : 'Unban'}</button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                {/* Pagination */}
                {usersTotal > PAGE_SIZE && (
                  <div className="flex justify-center gap-2">
                    <button
                      onClick={() => fetchUsers(usersPage - 1, usersSearch)}
                      disabled={usersPage === 1}
                      className="px-3 py-1 bg-[var(--color-surface)] rounded disabled:opacity-50"
                    >←</button>
                    <span className="px-3 py-1 text-[var(--color-text-secondary)]">
                      Page {usersPage} of {Math.ceil(usersTotal / PAGE_SIZE)}
                    </span>
                    <button
                      onClick={() => fetchUsers(usersPage + 1, usersSearch)}
                      disabled={usersPage >= Math.ceil(usersTotal / PAGE_SIZE)}
                      className="px-3 py-1 bg-[var(--color-surface)] rounded disabled:opacity-50"
                    >→</button>
                  </div>
                )}
              </div>
            )}
            
            {/* CHATS TAB */}
            {activeTab === 'chats' && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={allChatsSearch}
                    onChange={(e) => setAllChatsSearch(e.target.value)}
                    placeholder="Search chats by title..."
                    className="flex-1 px-4 py-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)]"
                    onKeyDown={(e) => e.key === 'Enter' && fetchAllChats(1, allChatsSearch)}
                  />
                  <button
                    onClick={() => fetchAllChats(1, allChatsSearch)}
                    className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg"
                  >Search</button>
                </div>
                
                <div className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-[var(--color-background)]">
                      <tr>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Title</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Owner</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Model</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Messages</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Updated</th>
                        <th className="px-4 py-3 text-left text-[var(--color-text-secondary)]">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chatsLoading ? (
                        <tr><td colSpan={6} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">Loading...</td></tr>
                      ) : allChats.length === 0 ? (
                        <tr><td colSpan={6} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">No chats found</td></tr>
                      ) : allChats.map((chat) => (
                        <tr key={chat.id} className="border-t border-[var(--color-border)]">
                          <td className="px-4 py-3 text-[var(--color-text)]">{chat.title}</td>
                          <td className="px-4 py-3 text-[var(--color-text-secondary)]">{chat.owner_email || 'Unknown'}</td>
                          <td className="px-4 py-3 text-[var(--color-text-secondary)]">{chat.model || 'default'}</td>
                          <td className="px-4 py-3 text-[var(--color-text)]">{chat.message_count}</td>
                          <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                            {chat.updated_at ? new Date(chat.updated_at).toLocaleDateString() : '-'}
                          </td>
                          <td className="px-4 py-3">
                            <button
                              onClick={() => viewChatDetail(chat.id)}
                              className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded hover:bg-blue-500/30"
                            >View</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                {/* Pagination */}
                {allChatsTotal > PAGE_SIZE && (
                  <div className="flex justify-center gap-2">
                    <button
                      onClick={() => fetchAllChats(allChatsPage - 1, allChatsSearch)}
                      disabled={allChatsPage === 1}
                      className="px-3 py-1 bg-[var(--color-surface)] rounded disabled:opacity-50"
                    >←</button>
                    <span className="px-3 py-1 text-[var(--color-text-secondary)]">
                      Page {allChatsPage} of {Math.ceil(allChatsTotal / PAGE_SIZE)}
                    </span>
                    <button
                      onClick={() => fetchAllChats(allChatsPage + 1, allChatsSearch)}
                      disabled={allChatsPage >= Math.ceil(allChatsTotal / PAGE_SIZE)}
                      className="px-3 py-1 bg-[var(--color-surface)] rounded disabled:opacity-50"
                    >→</button>
                  </div>
                )}
              </div>
            )}
            
            {/* TOOLS TAB */}
            {activeTab === 'tools' && (
              <div className="space-y-4">
                {/* Header with stats summary */}
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => { resetToolForm(); setShowToolModal(true); }}
                    className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg"
                  >+ Add Tool</button>
                  
                  {toolUsageStats && toolUsageStats.total_calls > 0 && (
                    <div className="flex items-center gap-4">
                      <div className="text-sm text-[var(--color-text-secondary)]">
                        Total: <span className="text-[var(--color-text)] font-medium">{toolUsageStats.total_calls.toLocaleString()}</span> calls
                        <span className="mx-2">•</span>
                        <span className="text-green-400">{toolUsageStats.successful_calls.toLocaleString()}</span> success
                        <span className="mx-2">•</span>
                        <span className="text-red-400">{toolUsageStats.failed_calls.toLocaleString()}</span> failed
                      </div>
                      <button
                        onClick={() => resetToolUsage()}
                        className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                      >Reset All Stats</button>
                    </div>
                  )}
                </div>
                
                <div className="grid gap-4">
                  {toolsLoading ? (
                    <div className="text-center py-8 text-[var(--color-text-secondary)]">Loading...</div>
                  ) : tools.length === 0 ? (
                    <div className="text-center py-8 text-[var(--color-text-secondary)]">No tools configured</div>
                  ) : (
                    tools.map((tool) => {
                      const usage = getToolUsageStats(tool.id);
                      return (
                        <div key={tool.id} className="bg-[var(--color-surface)] rounded-xl p-4 border border-[var(--color-border)]">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h3 className="text-[var(--color-text)] font-medium">{tool.name}</h3>
                              <p className="text-sm text-[var(--color-text-secondary)]">{tool.url}</p>
                              <div className="flex flex-wrap gap-2 mt-2">
                                <span className={`text-xs px-2 py-0.5 rounded ${tool.tool_type === 'mcp' ? 'bg-blue-500/20 text-blue-400' : 'bg-green-500/20 text-green-400'}`}>
                                  {tool.tool_type.toUpperCase()}
                                </span>
                                {tool.is_public && <span className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">Public</span>}
                                {tool.has_api_key && <span className="text-xs px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400">API Key</span>}
                              </div>
                              
                              {/* Usage Stats */}
                              {usage && usage.total_calls > 0 && (
                                <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
                                  <div className="flex flex-wrap gap-4 text-xs">
                                    <div>
                                      <span className="text-[var(--color-text-secondary)]">Calls:</span>
                                      <span className="ml-1 text-[var(--color-text)] font-medium">{usage.total_calls.toLocaleString()}</span>
                                    </div>
                                    <div>
                                      <span className="text-[var(--color-text-secondary)]">Success:</span>
                                      <span className="ml-1 text-green-400 font-medium">{usage.successful_calls.toLocaleString()}</span>
                                      <span className="text-[var(--color-text-secondary)] ml-1">
                                        ({Math.round((usage.successful_calls / usage.total_calls) * 100)}%)
                                      </span>
                                    </div>
                                    <div>
                                      <span className="text-[var(--color-text-secondary)]">Failed:</span>
                                      <span className="ml-1 text-red-400 font-medium">{usage.failed_calls.toLocaleString()}</span>
                                    </div>
                                    <div>
                                      <span className="text-[var(--color-text-secondary)]">Avg Time:</span>
                                      <span className="ml-1 text-[var(--color-text)]">{usage.avg_duration_ms.toFixed(0)}ms</span>
                                    </div>
                                    <div>
                                      <span className="text-[var(--color-text-secondary)]">Users:</span>
                                      <span className="ml-1 text-[var(--color-text)]">{usage.unique_users}</span>
                                    </div>
                                    {usage.last_used && (
                                      <div>
                                        <span className="text-[var(--color-text-secondary)]">Last Used:</span>
                                        <span className="ml-1 text-[var(--color-text)]">
                                          {new Date(usage.last_used).toLocaleDateString()}
                                        </span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                              {(!usage || usage.total_calls === 0) && (
                                <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
                                  <span className="text-xs text-[var(--color-text-secondary)]">No usage recorded</span>
                                </div>
                              )}
                            </div>
                            <div className="flex flex-col gap-2 ml-4">
                              <button
                                onClick={() => {
                                  setEditingTool(tool);
                                  setToolForm({
                                    name: tool.name,
                                    description: tool.description || '',
                                    tool_type: tool.tool_type,
                                    url: tool.url,
                                    api_key: '',
                                    is_public: tool.is_public,
                                    auth_type: 'bearer',
                                  });
                                  setShowToolModal(true);
                                }}
                                className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded"
                              >Edit</button>
                              {usage && usage.total_calls > 0 && (
                                <button
                                  onClick={() => resetToolUsage(tool.id)}
                                  className="text-xs px-2 py-1 bg-orange-500/20 text-orange-400 rounded"
                                >Reset Stats</button>
                              )}
                              <button
                                onClick={() => handleDeleteTool(tool.id)}
                                className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded"
                              >Delete</button>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            )}
            
            {/* MODALS */}
            
            {/* User Action Modal */}
            {showUserModal && selectedUser && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto border border-[var(--color-border)]">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                    {userAction === 'upgrade' ? 'Change Tier' : userAction === 'refund' ? 'Add Tokens' : 'User Chats'}
                    <span className="text-sm font-normal text-[var(--color-text-secondary)] ml-2">— {selectedUser.email}</span>
                  </h3>
                  
                  {userAction === 'upgrade' && (
                    <div className="flex flex-col gap-2">
                      {['free', 'pro', 'enterprise'].map((tier) => (
                        <button
                          key={tier}
                          onClick={() => handleUpgradeUser(selectedUser, tier)}
                          disabled={actionLoading || selectedUser.tier === tier}
                          className={`px-4 py-2 rounded-lg text-sm font-medium ${
                            selectedUser.tier === tier ? 'bg-[var(--color-border)] text-[var(--color-text-secondary)]' : 'bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90'
                          } disabled:opacity-50`}
                        >
                          {tier.charAt(0).toUpperCase() + tier.slice(1)}
                          {selectedUser.tier === tier && ' (current)'}
                        </button>
                      ))}
                    </div>
                  )}
                  
                  {userAction === 'refund' && (
                    <div className="flex flex-wrap gap-2">
                      {[10000, 50000, 100000, 500000, 1000000].map((amount) => (
                        <button
                          key={amount}
                          onClick={() => handleRefundTokens(selectedUser, amount)}
                          disabled={actionLoading}
                          className="px-3 py-2 rounded-lg text-sm font-medium bg-green-500/20 text-green-400 hover:bg-green-500/30 disabled:opacity-50"
                        >
                          +{formatTokens(amount)}
                        </button>
                      ))}
                    </div>
                  )}
                  
                  {userAction === 'chats' && (
                    <div className="space-y-2">
                      {chatsLoading ? (
                        <div className="text-center py-4 text-[var(--color-text-secondary)]">Loading...</div>
                      ) : userChats.length === 0 ? (
                        <div className="text-center py-4 text-[var(--color-text-secondary)]">No chats</div>
                      ) : userChats.map((chat) => (
                        <div key={chat.id} className="flex items-center justify-between p-3 bg-[var(--color-background)] rounded-lg">
                          <div>
                            <p className="text-[var(--color-text)] font-medium">{chat.title}</p>
                            <p className="text-xs text-[var(--color-text-secondary)]">{chat.message_count} messages • {chat.model || 'default'}</p>
                          </div>
                          <button
                            onClick={() => viewChatDetail(chat.id)}
                            className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded"
                          >View</button>
                        </div>
                      ))}
                      
                      {userChatsTotal > PAGE_SIZE && (
                        <div className="flex justify-center gap-2 mt-4">
                          <button
                            onClick={() => fetchUserChats(selectedUser.id, userChatsPage - 1)}
                            disabled={userChatsPage === 1}
                            className="px-3 py-1 bg-[var(--color-surface)] rounded disabled:opacity-50"
                          >←</button>
                          <span className="px-3 py-1 text-[var(--color-text-secondary)]">
                            {userChatsPage} / {Math.ceil(userChatsTotal / PAGE_SIZE)}
                          </span>
                          <button
                            onClick={() => fetchUserChats(selectedUser.id, userChatsPage + 1)}
                            disabled={userChatsPage >= Math.ceil(userChatsTotal / PAGE_SIZE)}
                            className="px-3 py-1 bg-[var(--color-surface)] rounded disabled:opacity-50"
                          >→</button>
                        </div>
                      )}
                    </div>
                  )}
                  
                  <div className="mt-6 flex justify-end">
                    <button
                      onClick={() => { setShowUserModal(false); setUserAction(null); setSelectedUser(null); }}
                      className="px-4 py-2 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                    >Close</button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Chat Detail Modal */}
            {showChatModal && selectedChat && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-4xl max-h-[80vh] overflow-y-auto border border-[var(--color-border)]">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-[var(--color-text)]">{selectedChat.title}</h3>
                      <p className="text-sm text-[var(--color-text-secondary)]">
                        {selectedChat.owner_username} ({selectedChat.owner_email}) • {selectedChat.model || 'default'}
                      </p>
                    </div>
                    <button
                      onClick={() => { setShowChatModal(false); setSelectedChat(null); }}
                      className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)] text-2xl"
                    >×</button>
                  </div>
                  
                  <div className="space-y-4">
                    {selectedChat.messages.map((msg) => (
                      <div key={msg.id} className={`p-4 rounded-lg ${msg.role === 'user' ? 'bg-blue-500/10 border-l-2 border-blue-500' : 'bg-[var(--color-background)]'}`}>
                        <p className="text-xs text-[var(--color-text-secondary)] mb-1 uppercase">{msg.role}</p>
                        <p className="text-[var(--color-text)] whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {/* Tool Modal */}
            {showToolModal && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-md border border-[var(--color-border)]">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                    {editingTool ? 'Edit Tool' : 'Add Tool'}
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Name</label>
                      <input
                        type="text"
                        value={toolForm.name}
                        onChange={(e) => setToolForm({ ...toolForm, name: e.target.value })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Type</label>
                      <select
                        value={toolForm.tool_type}
                        onChange={(e) => setToolForm({ ...toolForm, tool_type: e.target.value as 'mcp' | 'openapi' })}
                        disabled={!!editingTool}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      >
                        <option value="mcp">MCP</option>
                        <option value="openapi">OpenAPI</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">URL</label>
                      <input
                        type="text"
                        value={toolForm.url}
                        onChange={(e) => setToolForm({ ...toolForm, url: e.target.value })}
                        placeholder="http://localhost:3000 or https://api.example.com/openapi.json"
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">API Key (optional)</label>
                      <input
                        type="password"
                        value={toolForm.api_key}
                        onChange={(e) => setToolForm({ ...toolForm, api_key: e.target.value })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      />
                    </div>
                    
                    <div>
                      <button
                        onClick={handleProbeUrl}
                        disabled={probing || !toolForm.url}
                        className="w-full px-4 py-2 bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] rounded-lg disabled:opacity-50"
                      >
                        {probing ? 'Testing...' : 'Test Connection'}
                      </button>
                      
                      {probeResult && (
                        <div className={`mt-2 p-3 rounded-lg text-sm ${probeResult.success ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
                          {probeResult.message}
                        </div>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="tool-public"
                        checked={toolForm.is_public}
                        onChange={(e) => setToolForm({ ...toolForm, is_public: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="tool-public" className="text-sm text-[var(--color-text)]">Make available to all users</label>
                    </div>
                  </div>
                  
                  <div className="mt-6 flex justify-end gap-2">
                    <button
                      onClick={() => { setShowToolModal(false); resetToolForm(); }}
                      className="px-4 py-2 text-sm text-[var(--color-text-secondary)]"
                    >Cancel</button>
                    <button
                      onClick={handleCreateTool}
                      disabled={actionLoading || !toolForm.name || !toolForm.url}
                      className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg disabled:opacity-50"
                    >
                      {actionLoading ? 'Saving...' : (editingTool ? 'Update' : 'Create')}
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Filter Modal */}
            {showFilterModal && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-2xl border border-[var(--color-border)] max-h-[90vh] overflow-y-auto">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                    {editingFilter ? 'Edit Filter' : 'Add Filter'}
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Name *</label>
                        <input
                          type="text"
                          value={filterForm.name}
                          onChange={(e) => setFilterForm({ ...filterForm, name: e.target.value })}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          placeholder="my-filter"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Filter Type *</label>
                        <select
                          value={filterForm.filter_type}
                          onChange={(e) => setFilterForm({ ...filterForm, filter_type: e.target.value as any })}
                          disabled={!!editingFilter}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        >
                          <option value="to_llm">User → LLM (ToLLMFromChat)</option>
                          <option value="from_llm">LLM → User (FromLLMToChat)</option>
                          <option value="to_tools">To Tools</option>
                          <option value="from_tools">From Tools</option>
                        </select>
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Description</label>
                      <input
                        type="text"
                        value={filterForm.description || ''}
                        onChange={(e) => setFilterForm({ ...filterForm, description: e.target.value || null })}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        placeholder="What does this filter do?"
                      />
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Priority</label>
                        <select
                          value={filterForm.priority}
                          onChange={(e) => setFilterForm({ ...filterForm, priority: e.target.value as any })}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        >
                          <option value="highest">Highest (runs first)</option>
                          <option value="high">High</option>
                          <option value="medium">Medium</option>
                          <option value="low">Low</option>
                          <option value="least">Least (runs last)</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Action</label>
                        <select
                          value={filterForm.action}
                          onChange={(e) => setFilterForm({ ...filterForm, action: e.target.value as any })}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        >
                          <option value="modify">Modify (replace content)</option>
                          <option value="block">Block (stop message)</option>
                          <option value="log">Log (passthrough + log)</option>
                          <option value="passthrough">Passthrough (no change)</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Filter Mode</label>
                        <select
                          value={filterForm.filter_mode}
                          onChange={(e) => setFilterForm({ ...filterForm, filter_mode: e.target.value as any })}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                        >
                          <option value="pattern">Pattern/Regex</option>
                          <option value="code">Custom Code</option>
                          <option value="llm">LLM-based</option>
                        </select>
                      </div>
                    </div>
                    
                    {filterForm.filter_mode === 'pattern' && (
                      <>
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Regex Pattern</label>
                          <input
                            type="text"
                            value={filterForm.pattern || ''}
                            onChange={(e) => setFilterForm({ ...filterForm, pattern: e.target.value || null })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] font-mono"
                            placeholder="\\b(badword1|badword2)\\b"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Replacement Text (for modify action)</label>
                          <input
                            type="text"
                            value={filterForm.replacement || ''}
                            onChange={(e) => setFilterForm({ ...filterForm, replacement: e.target.value || null })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                            placeholder="[FILTERED]"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Word List (comma-separated, alternative to regex)</label>
                          <input
                            type="text"
                            value={filterForm.word_list?.join(', ') || ''}
                            onChange={(e) => setFilterForm({ 
                              ...filterForm, 
                              word_list: e.target.value ? e.target.value.split(',').map(w => w.trim()).filter(Boolean) : null 
                            })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                            placeholder="word1, word2, word3"
                          />
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="case-sensitive"
                            checked={filterForm.case_sensitive}
                            onChange={(e) => setFilterForm({ ...filterForm, case_sensitive: e.target.checked })}
                            className="rounded"
                          />
                          <label htmlFor="case-sensitive" className="text-sm text-[var(--color-text)]">Case sensitive matching</label>
                        </div>
                      </>
                    )}
                    
                    {filterForm.filter_mode === 'code' && (
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Python Code (advanced)</label>
                        <textarea
                          value={filterForm.code || ''}
                          onChange={(e) => setFilterForm({ ...filterForm, code: e.target.value || null })}
                          rows={6}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] font-mono text-sm"
                          placeholder="# content and context variables are available\nreturn content.replace('foo', 'bar')"
                        />
                      </div>
                    )}
                    
                    {filterForm.filter_mode === 'llm' && (
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">LLM Prompt</label>
                        <textarea
                          value={filterForm.llm_prompt || ''}
                          onChange={(e) => setFilterForm({ ...filterForm, llm_prompt: e.target.value || null })}
                          rows={4}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          placeholder="Review this content and return a sanitized version..."
                        />
                      </div>
                    )}
                    
                    {filterForm.action === 'block' && (
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Block Message</label>
                        <input
                          type="text"
                          value={filterForm.block_message || ''}
                          onChange={(e) => setFilterForm({ ...filterForm, block_message: e.target.value || null })}
                          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          placeholder="Your message was blocked because..."
                        />
                      </div>
                    )}
                    
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="filter-enabled"
                          checked={filterForm.enabled}
                          onChange={(e) => setFilterForm({ ...filterForm, enabled: e.target.checked })}
                          className="rounded"
                        />
                        <label htmlFor="filter-enabled" className="text-sm text-[var(--color-text)]">Enabled</label>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="filter-global"
                          checked={filterForm.is_global}
                          onChange={(e) => setFilterForm({ ...filterForm, is_global: e.target.checked })}
                          className="rounded"
                        />
                        <label htmlFor="filter-global" className="text-sm text-[var(--color-text)]">Apply globally (all chats)</label>
                      </div>
                    </div>
                    
                    {/* Test section for existing filters */}
                    {editingFilter?.id && (
                      <div className="border-t border-[var(--color-border)] pt-4 mt-4">
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Test Filter</label>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={filterTestContent}
                            onChange={(e) => setFilterTestContent(e.target.value)}
                            className="flex-1 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                            placeholder="Enter test content..."
                          />
                          <button
                            onClick={handleTestFilter}
                            disabled={filterTesting || !filterTestContent}
                            className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
                          >
                            {filterTesting ? 'Testing...' : 'Test'}
                          </button>
                        </div>
                        {filterTestResult && (
                          <div className={`mt-2 p-3 rounded-lg text-sm ${
                            filterTestResult.blocked ? 'bg-red-500/20 text-red-400' :
                            filterTestResult.modified ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-green-500/20 text-green-400'
                          }`}>
                            {filterTestResult.blocked ? (
                              <>Blocked: {filterTestResult.block_reason}</>
                            ) : filterTestResult.modified ? (
                              <>Modified: "{filterTestResult.result}"</>
                            ) : (
                              <>Passed through unchanged</>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-6 flex justify-end gap-2">
                    <button
                      onClick={() => { setShowFilterModal(false); resetFilterForm(); }}
                      className="px-4 py-2 text-sm text-[var(--color-text-secondary)]"
                    >Cancel</button>
                    <button
                      onClick={handleCreateFilter}
                      disabled={actionLoading || !filterForm.name}
                      className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg disabled:opacity-50"
                    >
                      {actionLoading ? 'Saving...' : (editingFilter ? 'Update' : 'Create')}
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            
            {/* Filter Chain Modal - Visual Node-Based Builder */}
            {showFilterChainModal && (
              <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
                <div className="bg-[var(--color-surface)] rounded-xl w-[95vw] h-[95vh] border border-[var(--color-border)] flex flex-col">
                  {/* Header */}
                  <div className="px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between shrink-0">
                    <div className="flex items-center gap-4">
                      <h3 className="text-lg font-semibold text-[var(--color-text)]">
                        {editingFilterChain ? 'Edit Filter Chain' : 'Create Filter Chain'}
                      </h3>
                      <input
                        type="text"
                        value={filterChainForm.name}
                        onChange={(e) => setFilterChainForm({ ...filterChainForm, name: e.target.value })}
                        className="px-3 py-1.5 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm w-48"
                        placeholder="Chain name..."
                      />
                      <input
                        type="text"
                        value={filterChainForm.description || ''}
                        onChange={(e) => setFilterChainForm({ ...filterChainForm, description: e.target.value })}
                        className="px-3 py-1.5 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm flex-1"
                        placeholder="Description (optional)..."
                      />
                      <select
                        value={filterChainForm.priority}
                        onChange={(e) => setFilterChainForm({ ...filterChainForm, priority: parseInt(e.target.value) })}
                        className="px-2 py-1.5 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                      >
                        <option value={10}>Priority: First (10)</option>
                        <option value={50}>Priority: Early (50)</option>
                        <option value={100}>Priority: Normal (100)</option>
                        <option value={200}>Priority: Late (200)</option>
                        <option value={500}>Priority: Last (500)</option>
                      </select>
                    </div>
                    <div className="flex items-center gap-3 flex-wrap">
                      <label className="flex items-center gap-2 cursor-pointer text-sm">
                        <input
                          type="checkbox"
                          checked={filterChainForm.enabled}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, enabled: e.target.checked })}
                          className="rounded"
                        />
                        <span className="text-[var(--color-text)]">Enabled</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer text-sm" title="Preserve conversation history (don't modify messages)">
                        <input
                          type="checkbox"
                          checked={filterChainForm.retain_history}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, retain_history: e.target.checked })}
                          className="rounded"
                        />
                        <span className="text-[var(--color-text)]">Retain History</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer text-sm" title="Process both incoming and outgoing messages">
                        <input
                          type="checkbox"
                          checked={filterChainForm.bidirectional}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, bidirectional: e.target.checked })}
                          className="rounded"
                        />
                        <span className="text-[var(--color-text)]">Bidirectional</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer text-sm" title="Skip this filter chain if RAG (knowledge base) found results">
                        <input
                          type="checkbox"
                          checked={filterChainForm.skip_if_rag_hit}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, skip_if_rag_hit: e.target.checked })}
                          className="rounded"
                        />
                        <span className="text-[var(--color-text)]">Skip if RAG hit</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer text-sm">
                        <input
                          type="checkbox"
                          checked={filterChainForm.debug}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, debug: e.target.checked })}
                          className="rounded"
                        />
                        <span className="text-[var(--color-text)]">🐛 Debug</span>
                      </label>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-[var(--color-text-secondary)]">Max Iterations:</span>
                        <input
                          type="number"
                          value={filterChainForm.max_iterations}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, max_iterations: parseInt(e.target.value) || 10 })}
                          className="w-16 px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                          min={1}
                          max={100}
                        />
                      </div>
                      {filterChainForm.bidirectional && (
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-[var(--color-text-secondary)]">Outbound Chain:</span>
                          <select
                            value={filterChainForm.outbound_chain_id || ''}
                            onChange={(e) => setFilterChainForm({ ...filterChainForm, outbound_chain_id: e.target.value || undefined })}
                            className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                          >
                            <option value="">Same chain</option>
                            {filterChains.filter(c => c.id !== filterChainForm.id).map(c => (
                              <option key={c.id} value={c.id}>{c.name}</option>
                            ))}
                          </select>
                        </div>
                      )}
                      <button
                        onClick={() => setFilterChainJsonMode(!filterChainJsonMode)}
                        className={`px-3 py-1.5 rounded text-sm ${
                          filterChainJsonMode
                            ? 'bg-yellow-500/20 text-yellow-400'
                            : 'bg-[var(--color-background)] text-[var(--color-text-secondary)]'
                        }`}
                      >
                        {filterChainJsonMode ? '← Visual' : 'JSON →'}
                      </button>
                    </div>
                  </div>
                  
                  {/* Export as Tool Section */}
                  <div className="px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-background)]/50">
                    <div className="flex items-center gap-4">
                      <label className="flex items-center gap-2 cursor-pointer text-sm font-medium">
                        <input
                          type="checkbox"
                          checked={filterChainForm.export_as_tool}
                          onChange={(e) => setFilterChainForm({ ...filterChainForm, export_as_tool: e.target.checked })}
                          className="rounded"
                        />
                        <span className="text-[var(--color-text)]">🔧 Export as Dynamic Tool</span>
                      </label>
                      
                      {filterChainForm.export_as_tool && (
                        <>
                          <input
                            type="text"
                            value={filterChainForm.tool_name || ''}
                            onChange={(e) => setFilterChainForm({ ...filterChainForm, tool_name: e.target.value })}
                            className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm w-32"
                            placeholder="Tool ID (e.g., WebSearch)"
                          />
                          <input
                            type="text"
                            value={filterChainForm.tool_label || ''}
                            onChange={(e) => setFilterChainForm({ ...filterChainForm, tool_label: e.target.value })}
                            className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm w-40"
                            placeholder="Label (e.g., 🔍 Search)"
                          />
                        </>
                      )}
                    </div>
                    
                    {filterChainForm.export_as_tool && (
                      <div className="mt-3 grid grid-cols-2 gap-4">
                        {/* Left column - Advertising & Trigger */}
                        <div className="space-y-3">
                          <div className="text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Advertising & Trigger</div>
                          
                          <label className="flex items-center gap-2 cursor-pointer text-sm">
                            <input
                              type="checkbox"
                              checked={filterChainForm.advertise_to_llm}
                              onChange={(e) => setFilterChainForm({ ...filterChainForm, advertise_to_llm: e.target.checked })}
                              className="rounded"
                            />
                            <span className="text-[var(--color-text)]">Advertise to LLM (include in system prompt)</span>
                          </label>
                          
                          {filterChainForm.advertise_to_llm && (
                            <textarea
                              value={filterChainForm.advertise_text || ''}
                              onChange={(e) => setFilterChainForm({ ...filterChainForm, advertise_text: e.target.value })}
                              className="w-full px-2 py-1.5 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm h-16 resize-none"
                              placeholder='e.g., Use $WebSearch="query" to search the web'
                            />
                          )}
                          
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-[var(--color-text-secondary)]">Trigger Pattern:</span>
                            <input
                              type="text"
                              value={filterChainForm.trigger_pattern || ''}
                              onChange={(e) => setFilterChainForm({ ...filterChainForm, trigger_pattern: e.target.value })}
                              className="flex-1 px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm font-mono"
                              placeholder='e.g., \$WebSearch="([^"]+)"'
                            />
                          </div>
                          
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-[var(--color-text-secondary)]">Trigger Source:</span>
                            <select
                              value={filterChainForm.trigger_source}
                              onChange={(e) => setFilterChainForm({ ...filterChainForm, trigger_source: e.target.value as 'llm' | 'user' | 'both' })}
                              className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                            >
                              <option value="both">Both (LLM & User)</option>
                              <option value="llm">LLM Only</option>
                              <option value="user">User Only</option>
                            </select>
                          </div>
                          
                          <div className="flex items-center gap-4">
                            <label className="flex items-center gap-2 cursor-pointer text-sm">
                              <input
                                type="checkbox"
                                checked={filterChainForm.erase_from_display}
                                onChange={(e) => setFilterChainForm({ ...filterChainForm, erase_from_display: e.target.checked })}
                                className="rounded"
                              />
                              <span className="text-[var(--color-text)]">Erase from display</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer text-sm">
                              <input
                                type="checkbox"
                                checked={filterChainForm.keep_in_history}
                                onChange={(e) => setFilterChainForm({ ...filterChainForm, keep_in_history: e.target.checked })}
                                className="rounded"
                              />
                              <span className="text-[var(--color-text)]">Keep in history</span>
                            </label>
                          </div>
                        </div>
                        
                        {/* Right column - UI Button */}
                        <div className="space-y-3">
                          <div className="text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">UI Button (User Hints)</div>
                          
                          <label className="flex items-center gap-2 cursor-pointer text-sm">
                            <input
                              type="checkbox"
                              checked={filterChainForm.button_enabled}
                              onChange={(e) => setFilterChainForm({ ...filterChainForm, button_enabled: e.target.checked })}
                              className="rounded"
                            />
                            <span className="text-[var(--color-text)]">Show button in chat UI</span>
                          </label>
                          
                          {filterChainForm.button_enabled && (
                            <>
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-[var(--color-text-secondary)]">Icon:</span>
                                <input
                                  type="text"
                                  value={filterChainForm.button_icon || ''}
                                  onChange={(e) => setFilterChainForm({ ...filterChainForm, button_icon: e.target.value })}
                                  className="w-20 px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm text-center"
                                  placeholder="🔍"
                                />
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-[var(--color-text-secondary)]">Location:</span>
                                <select
                                  value={filterChainForm.button_location}
                                  onChange={(e) => setFilterChainForm({ ...filterChainForm, button_location: e.target.value as 'response' | 'query' | 'both' })}
                                  className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                                >
                                  <option value="response">On Responses</option>
                                  <option value="query">On Queries</option>
                                  <option value="both">Both</option>
                                </select>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-[var(--color-text-secondary)]">Trigger Mode:</span>
                                <select
                                  value={filterChainForm.button_trigger_mode}
                                  onChange={(e) => setFilterChainForm({ ...filterChainForm, button_trigger_mode: e.target.value as 'immediate' | 'modal' | 'selection' })}
                                  className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
                                >
                                  <option value="immediate">Immediate</option>
                                  <option value="selection">On Selection</option>
                                  <option value="modal">Open Modal</option>
                                </select>
                              </div>
                            </>
                          )}
                          
                          {/* Variables section */}
                          <div className="text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider mt-4">Tool Variables</div>
                          <div className="text-xs text-[var(--color-text-secondary)]">
                            Variables available: $Selected, $Query, $1, $2 (capture groups), $Var[name]
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Content - Flow Editor or JSON */}
                  <div className="flex-1 overflow-hidden">
                    {filterChainJsonMode ? (
                      <div className="h-full p-4">
                        <textarea
                          value={filterChainJson || JSON.stringify(filterChainForm.definition, null, 2)}
                          onChange={(e) => {
                            setFilterChainJson(e.target.value);
                            try {
                              const parsed = JSON.parse(e.target.value);
                              if (parsed.steps && Array.isArray(parsed.steps)) {
                                setFilterChainForm(prev => ({ ...prev, definition: parsed }));
                              }
                            } catch {
                              // Invalid JSON, don't update
                            }
                          }}
                          className="w-full h-full font-mono text-sm bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg p-4 text-[var(--color-text)] resize-none"
                          placeholder='{"steps": []}'
                        />
                      </div>
                    ) : (
                      <Suspense fallback={
                        <div className="h-full flex items-center justify-center text-[var(--color-text-secondary)]">
                          Loading visual editor...
                        </div>
                      }>
                        <FlowEditor
                          definition={filterChainForm.definition}
                          onChange={(newDef) => setFilterChainForm(prev => ({ ...prev, definition: newDef }))}
                          availableTools={availableTools}
                          filterChains={filterChains
                            .filter(c => c.id && c.id !== editingFilterChain?.id)
                            .map(c => ({ id: c.id!, name: c.name }))
                          }
                        />
                      </Suspense>
                    )}
                  </div>
                  
                  {/* Footer */}
                  <div className="px-4 py-3 border-t border-[var(--color-border)] flex justify-between shrink-0">
                    <div className="flex items-center gap-4 text-sm text-[var(--color-text-secondary)]">
                      <span>{filterChainForm.definition?.steps?.length || 0} steps</span>
                      <input
                        type="text"
                        value={filterChainForm.description || ''}
                        onChange={(e) => setFilterChainForm({ ...filterChainForm, description: e.target.value })}
                        className="px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm w-64"
                        placeholder="Description (optional)..."
                      />
                    </div>
                    <div className="flex gap-3">
                      <button
                        onClick={() => { setShowFilterChainModal(false); resetFilterChainForm(); }}
                        className="px-4 py-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleCreateFilterChain}
                        disabled={!filterChainForm.name}
                        className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50 font-medium"
                      >
                        {editingFilterChain ? 'Save Changes' : 'Create Chain'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* JSON Preview Modal */}
            {previewJsonChain && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
                <div className="bg-[var(--color-surface)] rounded-xl w-full max-w-4xl border border-[var(--color-border)] max-h-[90vh] flex flex-col">
                  <div className="p-4 border-b border-[var(--color-border)] flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-[var(--color-text)]">
                      {previewJsonChain.name} - JSON Definition
                    </h3>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(JSON.stringify(previewJsonChain, null, 2));
                        }}
                        className="px-3 py-1 bg-[var(--color-background)] text-[var(--color-text)] rounded text-sm hover:bg-[var(--color-border)]"
                      >
                        Copy All
                      </button>
                      <button
                        onClick={() => setPreviewJsonChain(null)}
                        className="p-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                  <div className="p-4 overflow-auto flex-1">
                    <pre className="text-sm text-[var(--color-text)] bg-[var(--color-background)] p-4 rounded-lg overflow-auto font-mono whitespace-pre-wrap">
                      {JSON.stringify(previewJsonChain, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )}
            
            {/* GLOBAL KNOWLEDGE STORES TAB */}
            {activeTab === 'global_kb' && (
              <div className="space-y-6">
                {/* Feature Toggle */}
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-[var(--color-text)]">Global Knowledge Stores</h3>
                      <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                        Global stores are automatically searched on every query. Results are only included if they meet the relevance threshold.
                      </p>
                    </div>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <span className="text-sm text-[var(--color-text-secondary)]">
                        {globalKBEnabled ? 'Enabled' : 'Disabled'}
                      </span>
                      <div 
                        className={`relative w-12 h-6 rounded-full transition-colors cursor-pointer ${
                          globalKBEnabled ? 'bg-green-500' : 'bg-gray-600'
                        }`}
                        onClick={toggleGlobalKBEnabled}
                      >
                        <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                          globalKBEnabled ? 'translate-x-7' : 'translate-x-1'
                        }`} />
                      </div>
                    </label>
                  </div>
                </div>
                
                {/* Current Global Stores */}
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                    Active Global Stores ({globalKBStores.length})
                  </h3>
                  
                  {globalKBLoading ? (
                    <div className="text-[var(--color-text-secondary)]">Loading...</div>
                  ) : globalKBStores.length === 0 ? (
                    <div className="text-[var(--color-text-secondary)] py-8 text-center">
                      No global stores configured. Add a store from the list below.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {globalKBStores.map(store => (
                        <div 
                          key={store.id}
                          className="flex items-center justify-between p-4 bg-[var(--color-background)] rounded-lg border border-green-500/30"
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">{store.icon}</span>
                            <div>
                              <div className="font-medium text-[var(--color-text)]">{store.name}</div>
                              <div className="text-xs text-[var(--color-text-secondary)]">
                                Owner: {store.owner_username} • {store.document_count} docs
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2">
                              <label className="text-xs text-[var(--color-text-secondary)]">Min Score:</label>
                              <input
                                type="number"
                                min="0"
                                max="1"
                                step="0.05"
                                value={store.global_min_score}
                                onChange={(e) => {
                                  const newScore = parseFloat(e.target.value);
                                  if (!isNaN(newScore) && newScore >= 0 && newScore <= 1) {
                                    updateGlobalStoreSettings(store, newScore, store.global_max_results);
                                  }
                                }}
                                className="w-16 px-2 py-1 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[var(--color-text)]"
                              />
                            </div>
                            <div className="flex items-center gap-2">
                              <label className="text-xs text-[var(--color-text-secondary)]">Max Results:</label>
                              <input
                                type="number"
                                min="1"
                                max="20"
                                value={store.global_max_results}
                                onChange={(e) => {
                                  const newMax = parseInt(e.target.value);
                                  if (!isNaN(newMax) && newMax >= 1 && newMax <= 20) {
                                    updateGlobalStoreSettings(store, store.global_min_score, newMax);
                                  }
                                }}
                                className="w-16 px-2 py-1 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[var(--color-text)]"
                              />
                            </div>
                            <button
                              onClick={() => toggleStoreGlobal(store, false)}
                              className="px-3 py-1 text-sm bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                            >
                              Remove
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                
                {/* All Available Stores */}
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                    All Knowledge Stores ({allKBStores.length})
                  </h3>
                  <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                    Select stores to make them global. Global stores are searched on every query for all users.
                  </p>
                  
                  {globalKBLoading ? (
                    <div className="text-[var(--color-text-secondary)]">Loading...</div>
                  ) : allKBStores.filter(s => !s.is_global).length === 0 ? (
                    <div className="text-[var(--color-text-secondary)] py-8 text-center">
                      All stores are already global, or no stores exist.
                    </div>
                  ) : (
                    <div className="grid gap-3 max-h-96 overflow-y-auto">
                      {allKBStores.filter(s => !s.is_global).map(store => (
                        <div 
                          key={store.id}
                          className="flex items-center justify-between p-4 bg-[var(--color-background)] rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">{store.icon}</span>
                            <div>
                              <div className="font-medium text-[var(--color-text)]">{store.name}</div>
                              <div className="text-xs text-[var(--color-text-secondary)]">
                                Owner: {store.owner_username} • {store.document_count} docs
                                {store.description && ` • ${store.description.slice(0, 50)}${store.description.length > 50 ? '...' : ''}`}
                              </div>
                            </div>
                          </div>
                          <button
                            onClick={() => toggleStoreGlobal(store, true)}
                            className="px-3 py-1 text-sm bg-green-500/20 text-green-400 rounded hover:bg-green-500/30"
                          >
                            Make Global
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* GPT CATEGORIES TAB */}
            {activeTab === 'categories' && (
              <div className="space-y-6">
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-[var(--color-text)]">GPT Categories</h3>
                      <p className="text-sm text-[var(--color-text-secondary)]">
                        Manage categories for organizing Custom GPTs in the marketplace
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        resetCategoryForm();
                        setShowCategoryModal(true);
                      }}
                      className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
                    >
                      + Add Category
                    </button>
                  </div>
                  
                  {categoriesLoading ? (
                    <div className="text-[var(--color-text-secondary)]">Loading...</div>
                  ) : gptCategories.length === 0 ? (
                    <div className="text-center py-8 text-[var(--color-text-secondary)]">
                      No categories yet. Click "Add Category" to create one.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {gptCategories.map(cat => (
                        <div
                          key={cat.id}
                          className={`flex items-center justify-between p-4 rounded-lg transition-colors ${
                            cat.is_active 
                              ? 'bg-[var(--color-background)]' 
                              : 'bg-[var(--color-background)]/50 opacity-60'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-2xl w-10 text-center">{cat.icon}</span>
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-[var(--color-text)]">{cat.label}</span>
                                <code className="text-xs px-2 py-0.5 rounded bg-[var(--color-surface)] text-[var(--color-text-secondary)]">
                                  {cat.value}
                                </code>
                                {!cat.is_active && (
                                  <span className="text-xs px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                                    Disabled
                                  </span>
                                )}
                              </div>
                              {cat.description && (
                                <div className="text-sm text-[var(--color-text-secondary)]">{cat.description}</div>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-[var(--color-text-secondary)] mr-2">
                              Order: {cat.sort_order}
                            </span>
                            <button
                              onClick={() => startEditCategory(cat)}
                              className="px-3 py-1 text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] rounded hover:bg-[var(--color-border)]"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => toggleCategoryActive(cat)}
                              className={`px-3 py-1 text-sm rounded ${
                                cat.is_active
                                  ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30'
                                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                              }`}
                            >
                              {cat.is_active ? 'Disable' : 'Enable'}
                            </button>
                            {cat.value !== 'general' && (
                              <button
                                onClick={() => deleteCategory(cat)}
                                className="px-3 py-1 text-sm bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                              >
                                Delete
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                
                {/* Category Modal */}
                {showCategoryModal && (
                  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-md border border-[var(--color-border)]">
                      <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                        {editingCategory ? 'Edit Category' : 'New Category'}
                      </h3>
                      
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                            Value (URL-friendly) {!editingCategory && '*'}
                          </label>
                          <input
                            type="text"
                            value={categoryForm.value}
                            onChange={(e) => setCategoryForm({ ...categoryForm, value: e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '') })}
                            placeholder="e.g., data-science"
                            disabled={!!editingCategory}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] disabled:opacity-50"
                          />
                          <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                            Lowercase letters, numbers, and hyphens only
                          </p>
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Label *</label>
                          <input
                            type="text"
                            value={categoryForm.label}
                            onChange={(e) => setCategoryForm({ ...categoryForm, label: e.target.value })}
                            placeholder="e.g., Data Science"
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Icon (emoji)</label>
                          <input
                            type="text"
                            value={categoryForm.icon}
                            onChange={(e) => setCategoryForm({ ...categoryForm, icon: e.target.value })}
                            placeholder="📁"
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Description</label>
                          <input
                            type="text"
                            value={categoryForm.description}
                            onChange={(e) => setCategoryForm({ ...categoryForm, description: e.target.value })}
                            placeholder="Optional description..."
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Sort Order</label>
                          <input
                            type="number"
                            value={categoryForm.sort_order}
                            onChange={(e) => setCategoryForm({ ...categoryForm, sort_order: parseInt(e.target.value) || 0 })}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                          />
                          <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                            Lower numbers appear first
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex justify-end gap-3 mt-6">
                        <button
                          onClick={resetCategoryForm}
                          className="px-4 py-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={saveCategory}
                          className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
                        >
                          {editingCategory ? 'Save Changes' : 'Create Category'}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* ASSISTANT MODES TAB */}
            {activeTab === 'modes' && (
              <AssistantModesTab />
            )}
            
            {/* SITE DEV TAB */}
            {activeTab === 'dev' && (
              <div className="space-y-6">
                {/* Local Debug Settings */}
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">Local Debug Settings</h3>
                  <p className="text-sm text-[var(--color-text-secondary)] mb-6">
                    These settings are stored locally in your browser.
                  </p>
                  
                  <div className="space-y-4">
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={debugVoiceMode}
                        onChange={(e) => {
                          const value = e.target.checked;
                          setDebugVoiceMode(value);
                          localStorage.setItem('nexus-debug-voice-mode', value ? 'true' : 'false');
                        }}
                        className="w-5 h-5 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
                      />
                      <div>
                        <div className="text-[var(--color-text)] font-medium">Debug Talk To Me Session Data</div>
                        <div className="text-sm text-[var(--color-text-secondary)]">
                          Shows message IDs, parent chain, and leaf tracking info in the voice mode overlay
                        </div>
                      </div>
                    </label>
                  </div>
                </div>
                
                {/* Server Debug Settings */}
                <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">Server Debug Settings</h3>
                  <p className="text-sm text-[var(--color-text-secondary)] mb-6">
                    These settings are stored on the server and affect backend logging.
                  </p>
                  
                  {debugSettingsLoading ? (
                    <div className="text-[var(--color-text-secondary)]">Loading...</div>
                  ) : (
                    <div className="space-y-4">
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={debugTokenResets}
                          onChange={(e) => saveDebugTokenResets(e.target.checked)}
                          className="w-5 h-5 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
                        />
                        <div>
                          <div className="text-[var(--color-text)] font-medium">Debug Token Resets</div>
                          <div className="text-sm text-[var(--color-text-secondary)]">
                            Logs detailed token counts for all users when token reset check occurs (hourly)
                          </div>
                        </div>
                      </label>
                      
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={debugDocumentQueue}
                          onChange={(e) => saveDebugDocumentQueue(e.target.checked)}
                          className="w-5 h-5 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
                        />
                        <div>
                          <div className="text-[var(--color-text)] font-medium">Debug Document Queue</div>
                          <div className="text-sm text-[var(--color-text-secondary)]">
                            Logs document processing queue activity (task added, processing, completed)
                          </div>
                        </div>
                      </label>
                      
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={debugRag}
                          onChange={(e) => saveDebugRag(e.target.checked)}
                          className="w-5 h-5 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
                        />
                        <div>
                          <div className="text-[var(--color-text)] font-medium">Debug RAG</div>
                          <div className="text-sm text-[var(--color-text-secondary)]">
                            Logs all knowledge base queries: search terms, matched documents, relevance scores, and retrieved context
                          </div>
                        </div>
                      </label>
                      
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={debugFilterChains}
                          onChange={(e) => saveDebugFilterChains(e.target.checked)}
                          className="w-5 h-5 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
                        />
                        <div>
                          <div className="text-[var(--color-text)] font-medium">Debug Filter Chains</div>
                          <div className="text-sm text-[var(--color-text-secondary)]">
                            Logs detailed filter chain execution: step name, input values, output values, and execution time for each node
                          </div>
                        </div>
                      </label>
                      
                      {/* Token Reset Info */}
                      <div className="mt-4 p-4 bg-[var(--color-background)] rounded-lg border border-[var(--color-border)]">
                        <h4 className="text-sm font-medium text-[var(--color-text)] mb-2">Token Reset Status</h4>
                        <div className="space-y-1 text-sm text-[var(--color-text-secondary)]">
                          <div>
                            <span className="font-medium">Refill Interval:</span> {tokenRefillHours} hours ({Math.round(tokenRefillHours / 24)} days)
                          </div>
                          <div>
                            <span className="font-medium">Last Reset:</span>{' '}
                            {lastTokenReset 
                              ? new Date(lastTokenReset).toLocaleString() 
                              : 'Never (will reset on next check)'}
                          </div>
                          {lastTokenReset && (
                            <div>
                              <span className="font-medium">Next Reset:</span>{' '}
                              {(() => {
                                const lastResetDate = new Date(lastTokenReset);
                                const nextReset = new Date(lastResetDate.getTime() + tokenRefillHours * 60 * 60 * 1000);
                                const now = new Date();
                                const hoursUntil = Math.max(0, (nextReset.getTime() - now.getTime()) / (1000 * 60 * 60));
                                return `${nextReset.toLocaleString()} (${Math.round(hoursUntil)} hours from now)`;
                              })()}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
