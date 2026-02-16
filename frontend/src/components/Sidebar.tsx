import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Link, useNavigate, useParams, useLocation } from 'react-router-dom';
import { useChatStore } from '../stores/chatStore';
import { useAuthStore } from '../stores/authStore';
import { useModelsStore } from '../stores/modelsStore';
import { formatRelativeTime } from '../lib/formatters';
import api from '../lib/api';
import type { Chat } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  isMobile?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ isOpen, onToggle, isMobile = false, onClose }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { chatId } = useParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [importResult, setImportResult] = useState<{ success: number; failed: number } | null>(null);
  const [showSortDropdown, setShowSortDropdown] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['Today', 'Local', 'All Chats', 'Search Results']));
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sortDropdownRef = useRef<HTMLDivElement>(null);
  
  const { user } = useAuthStore();
  const {
    chats,
    isLoadingChats,
    hasMoreChats,
    chatSearchQuery,
    chatSortBy,
    chatGroupCounts,
    groupHasMore,
    groupLoading,
    sidebarReloadTrigger,
    createChat,
    deleteChat,
    updateChatTitle,
    setCurrentChat,
    fetchChats,
    fetchGroupChats,
    setChatSortBy,
    deleteGroupChats,
  } = useChatStore();
  
  const { getDisplayName } = useModelsStore();
  const chatListRef = useRef<HTMLDivElement>(null);
  
  // Reload sidebar when trigger changes (chat created, deleted, or updated)
  useEffect(() => {
    if (sidebarReloadTrigger === 0) return; // Skip initial render
    const loadChats = async () => {
      // DON'T pass sortBy - we just want to refresh counts without clearing chats
      // The chats array already has the new/updated chat from createChat/deleteChat
      await fetchChats(false, searchQuery);
    };
    loadChats();
  }, [sidebarReloadTrigger]);
  
  // Load chats when sort changes
  // For grouped views: fetch counts, then fetch each group
  // For alphabetical: fetchChats returns the chats directly
  useEffect(() => {
    const loadChats = async () => {
      // Pass sortBy explicitly so fetchChats knows to clear chats for grouped views
      await fetchChats(false, searchQuery, chatSortBy);
    };
    loadChats();
  }, [chatSortBy]);
  
  // NC-0.8.0.27: Debounced search — trigger fetchChats when searchQuery changes
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    searchTimeoutRef.current = setTimeout(() => {
      fetchChats(false, searchQuery, chatSortBy);
    }, 300);
    return () => {
      if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
    };
  }, [searchQuery]);
  
  // When group counts arrive, fetch chats for each group (grouped views only)
  useEffect(() => {
    if (!chatGroupCounts) return;
    if (chatSortBy === 'alphabetical') return; // alphabetical doesn't use groups
    
    const groups = Object.keys(chatGroupCounts).filter(g => chatGroupCounts[g] > 0);
    if (groups.length === 0) return;
    
    // Expand first group
    setExpandedSections(new Set([groups[0]]));
    
    // Fetch each group's chats
    const dateField = chatSortBy === 'created' ? 'created_at' : 'updated_at';
    const groupType = chatSortBy === 'source' ? 'source' : 'period';
    
    groups.forEach(group => {
      fetchGroupChats(groupType, group, false, groupType === 'period' ? dateField : undefined);
    });
  }, [chatGroupCounts, chatSortBy]);
  
  // Handle infinite scroll
  const handleScroll = useCallback(() => {
    const container = chatListRef.current;
    if (!container || isLoadingChats || !hasMoreChats) return;
    
    // Load more when scrolled near bottom (within 100px)
    const { scrollTop, scrollHeight, clientHeight } = container;
    if (scrollHeight - scrollTop - clientHeight < 100) {
      fetchChats(true, chatSearchQuery);
    }
  }, [isLoadingChats, hasMoreChats, fetchChats, chatSearchQuery]);
  
  // Attach scroll listener
  useEffect(() => {
    const container = chatListRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll]);
  
  // Use chats directly - backend does the filtering now
  const safeChats = Array.isArray(chats) ? chats : [];
  
  // Get current chat to know last used model
  const { currentChat } = useChatStore.getState();
  const { defaultModel } = useModelsStore.getState();
  
  const handleNewChat = async () => {
    // Get the last used model from current chat, or fall back to default
    const lastModel = currentChat?.assistant_id 
      ? `gpt:${currentChat.assistant_id}`  // Custom GPT
      : currentChat?.model || defaultModel;
    
    try {
      // Create new chat with last used model
      const newChat = await createChat(lastModel);
      if (newChat?.id) {
        navigate(`/chat/${newChat.id}`);
      }
    } catch (err) {
      console.error('Failed to create new chat:', err);
      // Fallback to empty state on error
      navigate('/chat');
    }
    onClose?.();
  };
  
  const handleImportClick = () => {
    fileInputRef.current?.click();
  };
  
  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Warn if file is very large
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > 100) {
      const proceed = confirm(`This file is ${fileSizeMB.toFixed(0)}MB. Large imports may take several minutes. Continue?`);
      if (!proceed) {
        if (fileInputRef.current) fileInputRef.current.value = '';
        return;
      }
    }
    
    setIsImporting(true);
    setImportResult(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // Get auth token
      const authData = localStorage.getItem('nexus-auth');
      let token = '';
      if (authData) {
        try {
          const { state } = JSON.parse(authData);
          token = state?.accessToken || '';
        } catch {}
      }
      
      // Use AbortController with 10 minute timeout for large imports
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10 * 60 * 1000); // 10 minutes
      
      // Use native fetch to avoid axios default Content-Type header issues
      const response = await fetch('/api/chats/import', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      setImportResult({ success: result.total_imported, failed: result.total_failed });
      
      // Refresh chat list
      await fetchChats();
      
      // Navigate to the first imported chat if any
      if (result.results?.length > 0 && result.results[0].chat_id) {
        navigate(`/chat/${result.results[0].chat_id}`);
      }
      
      // Clear the result after 5 seconds
      setTimeout(() => setImportResult(null), 5000);
    } catch (err: any) {
      console.error('Import failed:', err);
      let errorMessage = 'Failed to import chats';
      
      if (err.name === 'AbortError') {
        errorMessage = 'Import timed out. The file may be too large or the server is busy. Try again or import a smaller file.';
      } else if (err.message === 'Failed to fetch') {
        errorMessage = 'Connection lost during import. The server may still be processing - check your chat list in a few minutes.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      alert(errorMessage);
    } finally {
      setIsImporting(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };
  
  const handleSelectChat = (chat: Chat) => {
    setCurrentChat(chat);
    navigate(`/chat/${chat.id}`);
    onClose?.();
  };
  
  const handleDeleteChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this chat?')) {
      await deleteChat(id);
      if (id === chatId) {
        navigate('/');
      }
    }
  };
  
  const startEditing = (e: React.MouseEvent, chat: Chat) => {
    e.stopPropagation();
    setEditingId(chat.id);
    setEditTitle(chat.title);
  };
  
  const saveTitle = async (id: string) => {
    if (editTitle.trim()) {
      await updateChatTitle(id, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };
  
  // Close sort dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sortDropdownRef.current && !sortDropdownRef.current.contains(e.target as Node)) {
        setShowSortDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  // Get source from chat's source field
  const getSource = (chat: { source?: string }): string => {
    switch (chat.source) {
      case 'chatgpt': return 'ChatGPT';
      case 'grok': return 'Grok';
      case 'claude': return 'Claude';
      default: return 'Native';
    }
  };
  
  // Sort and group chats - NO FRONTEND FILTERING
  // Each accordion fetches its own data from backend via fetchGroupChats
  // groupedChats just organizes what's already been fetched
  const { groupedChats, groupOrder } = useMemo(() => {
    const groups: Record<string, Chat[]> = {};
    let order: string[] = [];
    
    // NC-0.8.0.27: When searching, show flat results regardless of sort mode
    if (searchQuery) {
      groups['Search Results'] = [...safeChats].sort((a, b) => 
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
      );
      order = ['Search Results'];
      const activeOrder = order.filter(g => groups[g] && groups[g].length > 0);
      return { groupedChats: groups, groupOrder: activeOrder };
    }
    
    if (chatSortBy === 'alphabetical') {
      groups['All Chats'] = [...safeChats].sort((a, b) => a.title.localeCompare(b.title));
      order = ['All Chats'];
    } else if (chatSortBy === 'source') {
      order = ['Native', 'ChatGPT', 'Grok', 'Claude'];
      // Initialize empty arrays - data comes from fetchGroupChats
      order.forEach(g => { groups[g] = []; });
      // Place loaded chats by their source
      safeChats.forEach(chat => {
        const source = getSource(chat);
        if (groups[source]) groups[source].push(chat);
      });
      // Sort within each group
      order.forEach(g => {
        groups[g].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
      });
    } else {
      // Date-based (modified or created)
      order = ['Today', 'Last 7 Days', 'Last 30 Days', 'Older'];
      // Initialize empty arrays - data comes from fetchGroupChats
      order.forEach(g => { groups[g] = []; });
      // Place loaded chats by their period (using _assignedPeriod tag from fetchGroupChats)
      safeChats.forEach(chat => {
        const period = (chat as Chat & { _assignedPeriod?: string })._assignedPeriod;
        if (period && groups[period]) {
          groups[period].push(chat);
        }
      });
      // Sort within each group by the appropriate date field
      const dateField = chatSortBy === 'created' ? 'created_at' : 'updated_at';
      order.forEach(g => {
        groups[g].sort((a, b) => new Date(b[dateField]).getTime() - new Date(a[dateField]).getTime());
      });
    }
    
    // Filter to only groups with chats OR with backend counts
    const activeOrder = order.filter(g => 
      (groups[g] && groups[g].length > 0) || 
      (chatGroupCounts && chatGroupCounts[g] > 0)
    );
    
    return { groupedChats: groups, groupOrder: activeOrder };
  }, [safeChats, chatSortBy, chatGroupCounts, searchQuery]);
  
  const sortOptions = [
    { value: 'modified' as const, label: 'Date Modified' },
    { value: 'created' as const, label: 'Date Created' },
    { value: 'alphabetical' as const, label: 'Alphabetical' },
    { value: 'source' as const, label: 'Source' },
  ];

  const isActive = (path: string) => location.pathname === path;
  
  const handleNavClick = () => {
    onClose?.();
  };
  
  if (!isOpen) {
    return null;
  }
  
  // Mobile: slide-out panel with overlay
  // Desktop: static sidebar
  const sidebarClasses = isMobile
    ? 'fixed left-0 top-0 h-full w-80 bg-[var(--color-surface)] border-r border-[var(--color-border)] flex flex-col z-50 transform transition-transform duration-300 ease-in-out'
    : 'w-72 flex-shrink-0 bg-[var(--color-surface)] border-r border-[var(--color-border)] flex flex-col';
  
  return (
    <aside className={sidebarClasses}>
      {/* Header */}
      <div className="p-4 border-b border-[var(--color-border)]">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-lg md:text-base text-[var(--color-text)]">Chats</h2>
          <button
            onClick={onToggle}
            className="p-2 md:p-1.5 rounded-lg hover:bg-zinc-700/30 transition-colors"
          >
            <svg className="w-6 h-6 md:w-5 md:h-5 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {isMobile ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              )}
            </svg>
          </button>
        </div>
        
        {/* New chat button */}
        <div className="flex gap-2">
          <button
            onClick={handleNewChat}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 md:py-2.5 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 transition-opacity text-base md:text-sm"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>
          
          {/* Import button */}
          <button
            onClick={handleImportClick}
            disabled={isImporting}
            title="Import chats from ChatGPT, Grok, Claude, etc."
            className="px-3 py-3 md:py-2.5 bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] rounded-lg hover:bg-[var(--color-background)] transition-colors disabled:opacity-50"
          >
            {isImporting ? (
              <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImportFile}
            className="hidden"
          />
        </div>
        
        {/* Import result notification */}
        {importResult && (
          <div className="mt-2 px-3 py-2 rounded-lg bg-green-500/20 border border-green-500/30 text-green-400 text-sm">
            ✓ Imported {importResult.success} chat{importResult.success !== 1 ? 's' : ''}
            {importResult.failed > 0 && ` (${importResult.failed} failed)`}
          </div>
        )}
        
        {/* Search */}
        <div className="mt-3 flex items-center gap-2">
          <div className="flex-1 relative">
            <input
              type="text"
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-3 md:py-2 pr-8 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 text-base md:text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          <svg className="w-5 h-5 md:w-4 md:h-4 text-[var(--color-text-secondary)] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        
        {/* Sort dropdown */}
        <div className="mt-2 relative" ref={sortDropdownRef}>
          <button
            onClick={() => setShowSortDropdown(!showSortDropdown)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-sm hover:bg-[var(--color-surface)] transition-colors"
          >
            <span className="text-[var(--color-text-secondary)]">
              {sortOptions.find(o => o.value === chatSortBy)?.label}
            </span>
            <svg className={`w-4 h-4 text-[var(--color-text-secondary)] transition-transform ${showSortDropdown ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {showSortDropdown && (
            <div className="absolute z-10 w-full mt-1 py-1 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-lg">
              {sortOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => {
                    setChatSortBy(option.value);
                    setShowSortDropdown(false);
                  }}
                  className={`w-full px-3 py-2 text-left text-sm transition-colors ${
                    chatSortBy === option.value
                      ? 'bg-[var(--color-button)]/50 text-[var(--color-text)]'
                      : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-background)] hover:text-[var(--color-text)]'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* Chat list */}
      <div ref={chatListRef} className="flex-1 overflow-y-auto py-2 min-h-0">
        {isLoadingChats && chats.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <svg className="animate-spin h-6 w-6 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        ) : safeChats.length === 0 ? (
          <div className="text-center py-8 px-4">
            <p className="text-[var(--color-text-secondary)] text-base md:text-sm">
              {searchQuery ? 'No chats found' : 'No chats yet'}
            </p>
            {!searchQuery && (
              <p className="text-zinc-500/60 text-sm md:text-xs mt-1">
                Start a new conversation!
              </p>
            )}
          </div>
        ) : (
          groupOrder.map((group) => {
            const groupType = chatSortBy === 'source' ? 'source' : 'period';
            const groupKey = `${groupType}:${group}`;
            const totalInGroup = chatGroupCounts?.[group] ?? groupedChats[group]?.length ?? 0;
            const loadedInGroup = groupedChats[group]?.length ?? 0;
            const hasMoreInGroup = groupHasMore[groupKey] !== false && loadedInGroup < totalInGroup;
            const isLoadingGroup = groupLoading[groupKey];
            
            return (
            <div key={group} className="mb-2">
              <div className="group/header w-full px-4 py-2 flex items-center justify-between hover:bg-zinc-700/20 rounded-lg transition-colors">
                <button
                  onClick={() => {
                    const wasExpanded = expandedSections.has(group);
                    
                    setExpandedSections(prev => {
                      const next = new Set(prev);
                      if (next.has(group)) {
                        next.delete(group);
                      } else {
                        next.add(group);
                      }
                      return next;
                    });
                    
                    // When expanding, fetch from backend if we haven't loaded this group yet
                    if (!wasExpanded && loadedInGroup === 0 && !isLoadingGroup) {
                      const dateField = chatSortBy === 'created' ? 'created_at' : 'updated_at';
                      fetchGroupChats(groupType, group, false, groupType === 'period' ? dateField : undefined);
                    }
                  }}
                  className="flex items-center gap-2 flex-1"
                >
                  <svg 
                    className={`w-3 h-3 text-[var(--color-text-secondary)] transition-transform ${expandedSections.has(group) ? 'rotate-90' : ''}`} 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  <span className="text-sm md:text-xs font-medium text-[var(--color-text-secondary)]">
                    {group}
                  </span>
                  <span className="text-xs text-[var(--color-text-secondary)]/60">
                    {totalInGroup}
                  </span>
                </button>
                {/* Delete group button */}
                {totalInGroup > 0 && (
                  <button
                    onClick={async (e) => {
                      e.stopPropagation();
                      if (confirm(`Delete all ${totalInGroup} chats in "${group}"?`)) {
                        try {
                          await deleteGroupChats(groupType, group);
                        } catch {
                          alert('Failed to delete chats');
                        }
                      }
                    }}
                    className="p-1 rounded hover:bg-red-500/10 text-[var(--color-text-secondary)]/40 hover:text-[var(--color-error)] opacity-0 group-hover/header:opacity-100 transition-opacity"
                    title={`Delete all ${group} chats`}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                )}
              </div>
              {expandedSections.has(group) && (
                <div 
                  className="max-h-[400px] overflow-y-auto"
                  onScroll={(e) => {
                    const container = e.currentTarget;
                    const { scrollTop, scrollHeight, clientHeight } = container;
                    // Load more when scrolled near bottom
                    if (scrollHeight - scrollTop - clientHeight < 100) {
                      if (chatSortBy === 'alphabetical') {
                        // Alphabetical uses fetchChats with pagination
                        if (hasMoreChats && !isLoadingChats) {
                          fetchChats(true, chatSearchQuery);
                        }
                      } else if (hasMoreInGroup && !isLoadingGroup) {
                        // Source/period use fetchGroupChats
                        const dateField = chatSortBy === 'created' ? 'created_at' : 'updated_at';
                        fetchGroupChats(groupType, group, true, groupType === 'period' ? dateField : undefined);
                      }
                    }
                  }}
                >
                  {(groupedChats[group] || []).map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => handleSelectChat(chat)}
                  className={`group mx-2 px-3 py-3 md:py-2.5 rounded-lg cursor-pointer transition-colors ${
                    chatId === chat.id
                      ? 'bg-[var(--color-button)]/80 border border-[var(--color-border)]'
                      : 'hover:bg-zinc-700/30 active:bg-zinc-700/50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    {editingId === chat.id ? (
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onBlur={() => saveTitle(chat.id)}
                        onKeyDown={(e) => e.key === 'Enter' && saveTitle(chat.id)}
                        onClick={(e) => e.stopPropagation()}
                        className="flex-1 px-2 py-1 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-base md:text-sm text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                        autoFocus
                      />
                    ) : (
                      <div className="flex items-center gap-1.5 flex-1 min-w-0">
                        {chat.is_knowledge_indexed && (
                          <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" title="Indexed in knowledge base" />
                        )}
                        <span className="text-base md:text-sm text-[var(--color-text)] truncate">
                          {chat.title}
                        </span>
                      </div>
                    )}
                    
                    {/* Actions - always visible on mobile */}
                    <div className={`flex items-center gap-1 ${isMobile ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'} transition-opacity`}>
                      <button
                        onClick={(e) => startEditing(e, chat)}
                        className="p-2 md:p-1 rounded hover:bg-zinc-700/50 text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                      >
                        <svg className="w-4 h-4 md:w-3.5 md:h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                        </svg>
                      </button>
                      <button
                        onClick={(e) => handleDeleteChat(e, chat.id)}
                        className="p-2 md:p-1 rounded hover:bg-red-500/10 text-[var(--color-text-secondary)] hover:text-[var(--color-error)]"
                      >
                        <svg className="w-4 h-4 md:w-3.5 md:h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-sm md:text-xs text-zinc-500/60">
                      {chat.model ? getDisplayName(chat.model) : 'Default'}
                    </span>
                    <span className="text-sm md:text-xs text-zinc-500/40">•</span>
                    <span className="text-sm md:text-xs text-zinc-500/60">
                      {((chat.total_input_tokens ?? 0) + (chat.total_output_tokens ?? 0)).toLocaleString()} tokens
                    </span>
                  </div>
                </div>
                  ))}
                  {/* Load more indicator for this group */}
                  {(isLoadingGroup || (chatSortBy === 'alphabetical' && isLoadingChats && chats.length > 0)) && (
                    <div className="flex items-center justify-center py-2">
                      <svg className="animate-spin h-4 w-4 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                    </div>
                  )}
                  {/* Load more button for this group */}
                  {chatSortBy === 'alphabetical' ? (
                    // Alphabetical uses fetchChats
                    hasMoreChats && !isLoadingChats && (
                      <button
                        onClick={() => fetchChats(true, chatSearchQuery)}
                        className="w-full py-2 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/20 rounded"
                      >
                        Load more
                      </button>
                    )
                  ) : (
                    // Source/period use fetchGroupChats
                    hasMoreInGroup && !isLoadingGroup && (
                      <button
                        onClick={() => {
                          const dateField = chatSortBy === 'created' ? 'created_at' : 'updated_at';
                          fetchGroupChats(groupType, group, true, groupType === 'period' ? dateField : undefined);
                        }}
                        className="w-full py-2 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/20 rounded"
                      >
                        Load more ({totalInGroup - loadedInGroup} remaining)
                      </button>
                    )
                  )}
                </div>
              )}
            </div>
          );
          })
        )}
        
        {/* Loading more indicator */}
        {isLoadingChats && chats.length > 0 && (
          <div className="flex items-center justify-center py-4">
            <svg className="animate-spin h-5 w-5 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        )}
      </div>
      
      {/* Bottom Navigation */}
      <div className="border-t border-[var(--color-border)] p-2">
        <nav className="space-y-1">
          <Link
            to="/gpts"
            onClick={handleNavClick}
            className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
              isActive('/gpts')
                ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            }`}
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            Custom GPTs
          </Link>
          
          <Link
            to="/agents"
            onClick={handleNavClick}
            className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
              isActive('/agents')
                ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            }`}
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Agent Flows
          </Link>
          
          <Link
            to="/vibe"
            onClick={handleNavClick}
            className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
              isActive('/vibe')
                ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            }`}
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            Vibe Code
          </Link>
          
          <Link
            to="/documents"
            onClick={handleNavClick}
            className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
              isActive('/documents')
                ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            }`}
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Documents
          </Link>
          
          <Link
            to="/billing"
            onClick={handleNavClick}
            className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
              isActive('/billing')
                ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            }`}
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
            </svg>
            Billing
          </Link>
          
          <Link
            to="/settings"
            onClick={handleNavClick}
            className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
              isActive('/settings')
                ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            }`}
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Settings
          </Link>
          
          {user?.is_admin && (
            <Link
              to="/admin"
              onClick={handleNavClick}
              className={`flex items-center gap-3 px-3 py-3 md:py-2 rounded-lg text-base md:text-sm transition-colors ${
                isActive('/admin')
                  ? 'bg-[var(--color-warning)]/10 text-[var(--color-warning)]'
                  : 'text-[var(--color-warning)] hover:bg-[var(--color-warning)]/10'
              }`}
            >
              <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              Admin
            </Link>
          )}
        </nav>
      </div>
    </aside>
  );
}
