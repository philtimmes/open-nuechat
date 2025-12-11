import { useState } from 'react';
import { Link, useNavigate, useParams, useLocation } from 'react-router-dom';
import { useChatStore } from '../stores/chatStore';
import { useAuthStore } from '../stores/authStore';
import { useModelsStore } from '../stores/modelsStore';
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
  
  const { user } = useAuthStore();
  const {
    chats,
    isLoadingChats,
    createChat,
    deleteChat,
    updateChatTitle,
    setCurrentChat,
  } = useChatStore();
  
  const { getDisplayName } = useModelsStore();
  
  // Defensive: ensure chats is an array before filtering
  const safeChats = Array.isArray(chats) ? chats : [];
  const filteredChats = safeChats.filter((chat) =>
    chat.title?.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  const handleNewChat = async () => {
    const chat = await createChat();
    navigate(`/chat/${chat.id}`);
    onClose?.();
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
  
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };
  
  // Group chats by date
  const groupedChats = filteredChats.reduce((groups, chat) => {
    const date = formatDate(chat.updated_at);
    if (!groups[date]) groups[date] = [];
    groups[date].push(chat);
    return groups;
  }, {} as Record<string, Chat[]>);

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
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 md:py-2.5 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 transition-opacity text-base md:text-sm"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Chat
        </button>
        
        {/* Search */}
        <div className="mt-3 relative">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 md:w-4 md:h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 md:pl-9 pr-3 py-3 md:py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 text-base md:text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent"
          />
        </div>
      </div>
      
      {/* Chat list */}
      <div className="flex-1 overflow-y-auto py-2">
        {isLoadingChats ? (
          <div className="flex items-center justify-center py-8">
            <svg className="animate-spin h-6 w-6 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        ) : filteredChats.length === 0 ? (
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
          Object.entries(groupedChats).map(([date, dateChats]) => (
            <div key={date} className="mb-4">
              <div className="px-4 py-1">
                <span className="text-sm md:text-xs font-medium text-[var(--color-text-secondary)]">
                  {date}
                </span>
              </div>
              {dateChats.map((chat) => (
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
                      <span className="text-base md:text-sm text-[var(--color-text)] truncate flex-1">
                        {chat.title}
                      </span>
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
          ))
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
