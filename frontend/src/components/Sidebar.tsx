import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Link, useNavigate, useParams, useLocation } from 'react-router-dom';
import { useChatStore } from '../stores/chatStore';
import { useAuthStore } from '../stores/authStore';
import { useModelsStore } from '../stores/modelsStore';
import { formatRelativeTime } from '../lib/formatters';
import api from '../lib/api';
import type { Chat } from '../types';

type SortOption = 'modified' | 'created' | 'alphabetical' | 'source';

interface GroupCounts {
  groups: string[];
  counts: Record<string, number>;
  total: number;
}

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
  const [sortBy, setSortBy] = useState<SortOption>('modified');
  const [showSortDropdown, setShowSortDropdown] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['Today']));
  const [groupCounts, setGroupCounts] = useState<GroupCounts | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sortDropdownRef = useRef<HTMLDivElement>(null);
  
  const { user } = useAuthStore();
  const {
    chats,
    isLoadingChats,
    hasMoreChats,
    chatSearchQuery,
    createChat,
    deleteChat,
    updateChatTitle,
    setCurrentChat,
    fetchChats,
    addImportedChats,
  } = useChatStore();
  
  const { getDisplayName } = useModelsStore();
  const chatListRef = useRef<HTMLDivElement>(null);
  
  // Debounced search - triggers backend search
  useEffect(() => {
    // Clear any existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    // Debounce search by 300ms
    searchTimeoutRef.current = setTimeout(() => {
      fetchChats(false, searchQuery);
    }, 300);
    
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery, fetchChats]);
  
  // Fetch group counts from backend (lightweight query for accurate counts)
  // Re-fetches when sortBy changes or when chats are added/deleted
  useEffect(() => {
    if (sortBy === 'alphabetical' || searchQuery) {
      // No counts needed for alphabetical or when searching
      setGroupCounts(null);
      return;
    }
    
    const fetchCounts = async () => {
      try {
        const res = await api.get('/chats/counts', { 
          params: { sort_by: sortBy } 
        });
        setGroupCounts(res.data);
      } catch (err) {
        console.error('Failed to fetch chat counts:', err);
      }
    };
    
    fetchCounts();
  }, [sortBy, searchQuery, chats.length]); // chats.length triggers refetch after add/delete
  
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
      
      // Use native fetch to avoid axios default Content-Type header issues
      const response = await fetch('/api/chats/import', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      setImportResult({ success: result.total_imported, failed: result.total_failed });
      
      // Add imported chats to store without replacing existing ones
      if (result.imported_chats && result.imported_chats.length > 0) {
        addImportedChats(result.imported_chats);
      }
      
      // Refresh counts from server
      if (sortBy !== 'alphabetical') {
        try {
          const countsRes = await api.get('/chats/counts', { 
            params: { sort_by: sortBy } 
          });
          setGroupCounts(countsRes.data);
        } catch (err) {
          console.error('Failed to refresh chat counts:', err);
        }
      }
      
      // Navigate to the first imported chat if any
      if (result.results?.length > 0 && result.results[0].chat_id) {
        navigate(`/chat/${result.results[0].chat_id}`);
      }
      
      // Clear the result after 5 seconds
      setTimeout(() => setImportResult(null), 5000);
    } catch (err: any) {
      console.error('Import failed:', err);
      alert(err.message || 'Failed to import chats');
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
  
  // Get date category for grouping - uses calendar days, not hours
  const getDateCategory = (dateStr: string): string => {
    const date = new Date(dateStr);
    const now = new Date();
    
    // Reset times to midnight for proper calendar day comparison
    const dateDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    
    const diffDays = Math.floor((today.getTime() - dateDay.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays > 0 && diffDays < 7) return 'This Week';
    if (diffDays >= 7 && diffDays < 30) return 'Last 30 Days';
    return 'Older';
  };
  
  // Get source from chat title prefix
  const getSource = (title: string): string => {
    if (title.startsWith('ChatGPT: ')) return 'ChatGPT';
    if (title.startsWith('Grok: ')) return 'Grok';
    return 'Local';
  };
  
  // Sort and group chats based on selected sort option
  const { sortedChats, groupedChats, dateGroups } = useMemo(() => {
    let sorted = [...safeChats];
    
    // Sort chats
    switch (sortBy) {
      case 'created':
        sorted.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        break;
      case 'modified':
        sorted.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
        break;
      case 'alphabetical':
        sorted.sort((a, b) => a.title.localeCompare(b.title));
        break;
      case 'source':
        // Sort by source, then by date
        sorted.sort((a, b) => {
          const sourceA = getSource(a.title);
          const sourceB = getSource(b.title);
          if (sourceA !== sourceB) {
            // Order: Local, ChatGPT, Grok
            const order = ['Local', 'ChatGPT', 'Grok'];
            return order.indexOf(sourceA) - order.indexOf(sourceB);
          }
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        });
        break;
    }
    
    // Group by date category (only for date-based sorts) or source
    if (sortBy === 'alphabetical') {
      return { sortedChats: sorted, groupedChats: {}, dateGroups: [] };
    }
    
    if (sortBy === 'source') {
      const groups: Record<string, Chat[]> = {};
      const groupOrder = ['Local', 'ChatGPT', 'Grok'];
      
      sorted.forEach(chat => {
        const source = getSource(chat.title);
        if (!groups[source]) groups[source] = [];
        groups[source].push(chat);
      });
      
      const activeGroups = groupOrder.filter(g => groups[g] && groups[g].length > 0);
      return { sortedChats: sorted, groupedChats: groups, dateGroups: activeGroups };
    }
    
    const dateField = sortBy === 'created' ? 'created_at' : 'updated_at';
    const groups: Record<string, Chat[]> = {};
    const groupOrder = ['Today', 'This Week', 'Last 30 Days', 'Older'];
    
    sorted.forEach(chat => {
      const category = getDateCategory(chat[dateField]);
      if (!groups[category]) groups[category] = [];
      groups[category].push(chat);
    });
    
    // Only include groups that have chats
    const activeGroups = groupOrder.filter(g => groups[g] && groups[g].length > 0);
    
    return { sortedChats: sorted, groupedChats: groups, dateGroups: activeGroups };
  }, [safeChats, sortBy]);
  
  // Get display groups - prefer API counts, fall back to loaded chats
  const displayGroups = useMemo(() => {
    if (sortBy === 'alphabetical') return [];
    // Use API groups if available (they have accurate counts even before data is loaded)
    if (groupCounts && groupCounts.groups) {
      return groupCounts.groups.filter(g => (groupCounts.counts[g] || 0) > 0);
    }
    // Fall back to groups from loaded chats
    return dateGroups;
  }, [sortBy, groupCounts, dateGroups]);
  
  // Get count for a group - prefer API counts
  const getGroupCount = (group: string): number => {
    if (groupCounts && groupCounts.counts) {
      return groupCounts.counts[group] || 0;
    }
    return groupedChats[group]?.length || 0;
  };
  
  // Toggle accordion section
  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };
  
  // Export chats in a section
  const handleExportSection = async (e: React.MouseEvent, group: string) => {
    e.stopPropagation();
    const chatsToExport = groupedChats[group] || [];
    if (chatsToExport.length === 0) return;
    
    try {
      // Export as JSON
      const exportData = {
        exported_at: new Date().toISOString(),
        group: group,
        chat_count: chatsToExport.length,
        chats: chatsToExport
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `nuechat-${group.toLowerCase().replace(/\s+/g, '-')}-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export section:', err);
      alert('Failed to export chats');
    }
  };
  
  // Refetch counts from API
  const refetchCounts = async () => {
    if (sortBy === 'alphabetical') return;
    try {
      const res = await api.get('/chats/counts', { 
        params: { sort_by: sortBy } 
      });
      setGroupCounts(res.data);
    } catch (err) {
      console.error('Failed to refetch chat counts:', err);
    }
  };
  
  // Delete all chats in a section
  const handleDeleteSection = async (e: React.MouseEvent, group: string) => {
    e.stopPropagation();
    const count = getGroupCount(group);
    if (count === 0) return;
    
    // First confirmation
    if (!confirm(`Are you sure you want to delete all ${count} chats in "${group}"? This cannot be undone.`)) {
      return;
    }
    
    // Second confirmation - require typing
    const confirmText = prompt(`Type "DELETE ${group.toUpperCase()}" to confirm:`);
    if (confirmText !== `DELETE ${group.toUpperCase()}`) {
      alert('Deletion cancelled. Text did not match.');
      return;
    }
    
    try {
      // Delete each chat that's loaded in this group
      const chatsToDelete = groupedChats[group] || [];
      for (const chat of chatsToDelete) {
        await deleteChat(chat.id);
      }
      
      // Refresh counts from server
      await refetchCounts();
      
      // If currently viewing a deleted chat, navigate home
      if (chatId && chatsToDelete.some(c => c.id === chatId)) {
        navigate('/');
      }
    } catch (err) {
      console.error('Failed to delete section:', err);
      alert('Failed to delete some chats');
    }
  };
  
  const sortOptions: { value: SortOption; label: string }[] = [
    { value: 'modified', label: 'Date Modified' },
    { value: 'created', label: 'Date Created' },
    { value: 'alphabetical', label: 'Alphabetical' },
    { value: 'source', label: 'Source' },
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
        
        {/* Search and Sort */}
        <div className="mt-3 flex items-center gap-2">
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 px-3 py-3 md:py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 text-base md:text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent"
          />
          <svg className="w-5 h-5 md:w-4 md:h-4 text-[var(--color-text-secondary)] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        
        {/* Sort dropdown */}
        <div className="mt-2 relative" ref={sortDropdownRef}>
          <button
            onClick={() => setShowSortDropdown(!showSortDropdown)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm hover:bg-[var(--color-surface)] transition-colors"
          >
            <span className="text-[var(--color-text-secondary)]">Sort by:</span>
            <span className="flex items-center gap-1">
              {sortOptions.find(o => o.value === sortBy)?.label}
              <svg className={`w-4 h-4 transition-transform ${showSortDropdown ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </span>
          </button>
          
          {showSortDropdown && (
            <div className="absolute top-full left-0 right-0 mt-1 py-1 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-lg z-10">
              {sortOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => {
                    setSortBy(option.value);
                    setShowSortDropdown(false);
                    // Set default expanded section based on sort type
                    if (option.value === 'source') {
                      setExpandedSections(new Set(['Local']));
                    } else if (option.value !== 'alphabetical') {
                      setExpandedSections(new Set(['Today']));
                    } else {
                      setExpandedSections(new Set());
                    }
                  }}
                  className={`w-full px-3 py-2 text-left text-sm hover:bg-[var(--color-background)] transition-colors ${
                    sortBy === option.value ? 'text-[var(--color-primary)]' : 'text-[var(--color-text)]'
                  }`}
                >
                  {option.label}
                  {sortBy === option.value && (
                    <svg className="inline-block w-4 h-4 ml-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* Chat list */}
      <div ref={chatListRef} className="flex-1 overflow-y-auto py-2 pr-[30px]">
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
        ) : sortBy === 'alphabetical' ? (
          // Alphabetical: flat list, no accordions
          <div className="space-y-1">
            {sortedChats.map((chat) => (
              <div
                key={chat.id}
                onClick={() => handleSelectChat(chat)}
                className={`group ml-2 mr-3 px-3 py-3 md:py-2.5 rounded-lg cursor-pointer transition-colors ${
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
          </div>
        ) : (
          // Date-based and source sorts: accordion groups
          <div className="space-y-1">
            {displayGroups.map((group) => (
              <div key={group} className="border-b border-[var(--color-border)]/50 last:border-b-0">
                {/* Accordion header with hover actions - no nested buttons */}
                <div className="group/header relative flex items-center justify-between px-4 py-3 hover:bg-[var(--color-background)]/50 transition-colors">
                  {/* Toggle area - clickable div, not button */}
                  <div 
                    onClick={() => toggleSection(group)}
                    className="flex items-center gap-2 cursor-pointer flex-1"
                  >
                    <svg
                      className={`w-4 h-4 text-[var(--color-text-secondary)] transition-transform ${
                        expandedSections.has(group) ? 'rotate-90' : ''
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                    <span className="text-sm font-medium text-[var(--color-text)]">{group}</span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {/* Section action buttons - appear on hover */}
                    <div className="flex items-center gap-1 opacity-0 group-hover/header:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleExportSection(e, group);
                        }}
                        className="p-1 rounded hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                        title={`Export all chats in ${group}`}
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSection(e, group);
                        }}
                        className="p-1 rounded hover:bg-red-500/10 text-[var(--color-text-secondary)] hover:text-[var(--color-error)]"
                        title={`Delete all chats in ${group}`}
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                    <span className="text-xs text-[var(--color-text-secondary)]">
                      {getGroupCount(group)}
                    </span>
                  </div>
                </div>
                
                {/* Accordion content */}
                {expandedSections.has(group) && (
                  <div className="pb-2">
                    {(!groupedChats[group] || groupedChats[group].length === 0) && getGroupCount(group) > 0 ? (
                      <div className="ml-2 mr-3 px-3 py-3 text-sm text-[var(--color-text-secondary)]">
                        {isLoadingChats ? (
                          <div className="flex items-center gap-2">
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            <span>Loading...</span>
                          </div>
                        ) : (
                          <span>Scroll down to load more chats</span>
                        )}
                      </div>
                    ) : null}
                    {groupedChats[group]?.map((chat) => (
                      <div
                        key={chat.id}
                        onClick={() => handleSelectChat(chat)}
                        className={`group ml-2 mr-3 px-3 py-3 md:py-2.5 rounded-lg cursor-pointer transition-colors ${
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
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        
        {/* Loading more indicator */}
        {isLoadingChats && chats.length > 0 && (
          <div className="flex items-center justify-center gap-2 py-4 text-[var(--color-text-secondary)]">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span className="text-xs">Loading chats...</span>
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
