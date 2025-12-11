import { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useChatStore } from '../stores/chatStore';
import { useBrandingStore, useFeatureFlags } from '../stores/brandingStore';
import { useMobile } from '../hooks/useMobile';
import { useChatShortcuts } from '../hooks/useKeyboardShortcuts';
import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { isMobile } = useMobile();
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const { user, logout } = useAuthStore();
  const { fetchChats, createChat, setShowArtifacts } = useChatStore();
  const { config } = useBrandingStore();
  const features = useFeatureFlags();
  const navigate = useNavigate();
  const location = useLocation();
  
  const appName = config?.app_name || 'Open-NueChat';
  
  // Keyboard shortcuts handlers
  const handleNewChat = useCallback(async () => {
    const chat = await createChat();
    if (chat) {
      navigate(`/chat/${chat.id}`);
    }
  }, [createChat, navigate]);
  
  const handleToggleSidebar = useCallback(() => {
    setSidebarOpen(prev => !prev);
  }, []);
  
  const handleToggleArtifacts = useCallback(() => {
    setShowArtifacts((prev: boolean) => !prev);
  }, [setShowArtifacts]);
  
  const handleFocusInput = useCallback(() => {
    // Find and focus the chat input
    const input = document.querySelector('[data-chat-input]') as HTMLTextAreaElement;
    if (input) {
      input.focus();
    }
  }, []);
  
  const handleSearch = useCallback(() => {
    // Focus search if present, or show a search modal
    const search = document.querySelector('[data-search-input]') as HTMLInputElement;
    if (search) {
      search.focus();
    }
  }, []);
  
  const handleSettings = useCallback(() => {
    navigate('/settings');
  }, [navigate]);
  
  // Register keyboard shortcuts
  useChatShortcuts({
    onNewChat: handleNewChat,
    onToggleSidebar: handleToggleSidebar,
    onToggleArtifacts: handleToggleArtifacts,
    onFocusInput: handleFocusInput,
    onSearch: handleSearch,
    onSettings: handleSettings,
  });
  
  // Listen for toggle-sidebar custom event from ChatPage
  useEffect(() => {
    const handleToggleSidebarEvent = () => {
      setSidebarOpen(prev => !prev);
    };
    window.addEventListener('toggle-sidebar', handleToggleSidebarEvent);
    return () => window.removeEventListener('toggle-sidebar', handleToggleSidebarEvent);
  }, []);
  
  // Close sidebar on mobile when route changes
  useEffect(() => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  }, [location.pathname, isMobile]);
  
  // Update sidebar state when switching between mobile/desktop
  useEffect(() => {
    setSidebarOpen(!isMobile);
  }, [isMobile]);
  
  useEffect(() => {
    fetchChats();
  }, [fetchChats]);
  
  const handleLogout = () => {
    logout();
    navigate('/login');
  };
  
  const closeSidebar = () => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  };
  
  return (
    <div className="flex h-screen bg-[var(--color-background)] overflow-hidden">
      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 transition-opacity"
          onClick={closeSidebar}
        />
      )}
      
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        isMobile={isMobile}
        onClose={closeSidebar}
      />
      
      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top header */}
        <header className="flex items-center justify-between px-3 md:px-4 py-3 bg-[var(--color-surface)] border-b border-[var(--color-border)]">
          <div className="flex items-center gap-2 md:gap-3">
            {/* Hamburger menu - always show on mobile, show when sidebar closed on desktop */}
            {(isMobile || !sidebarOpen) && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 rounded-lg hover:bg-zinc-700/30 transition-colors"
              >
                <svg className="w-6 h-6 md:w-5 md:h-5 text-[var(--color-text)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}
            <Link to="/" className="flex items-center gap-2">
              {config?.logo_url ? (
                <img src={config.logo_url} alt={appName} className="h-7 md:h-8" />
              ) : (
                <span className="text-base md:text-lg font-semibold text-[var(--color-text)]">
                  {appName}
                </span>
              )}
            </Link>
          </div>
          
          <div className="flex items-center gap-2">
            {/* User menu */}
            <div className="relative group">
              <button className="flex items-center gap-2 px-2 md:px-3 py-1.5 rounded-lg hover:bg-zinc-700/30 transition-colors">
                <div className="w-9 h-9 md:w-8 md:h-8 rounded-full bg-[var(--color-button)] flex items-center justify-center text-[var(--color-button-text)] text-sm font-medium">
                  {user?.username?.charAt(0).toUpperCase() || 'U'}
                </div>
                <span className="hidden sm:block text-sm text-[var(--color-text)]">
                  {user?.username || 'User'}
                </span>
                <svg className="hidden sm:block w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {/* Dropdown menu */}
              <div className="absolute right-0 mt-2 w-48 py-2 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                <div className="px-4 py-2 border-b border-[var(--color-border)]">
                  <p className="text-sm font-medium text-[var(--color-text)]">{user?.username}</p>
                  <p className="text-xs text-[var(--color-text-secondary)]">{user?.email}</p>
                  <span className="inline-block mt-1 px-2 py-0.5 rounded text-xs bg-[var(--color-button)]/80 text-[var(--color-button-text)] capitalize">
                    {user?.tier || 'free'}
                  </span>
                </div>
                <Link
                  to="/settings"
                  className="block px-4 py-2 text-sm text-[var(--color-text)] hover:bg-zinc-700/30"
                >
                  Settings
                </Link>
                {features.billing && (
                  <Link
                    to="/billing"
                    className="block px-4 py-2 text-sm text-[var(--color-text)] hover:bg-zinc-700/30"
                  >
                    Billing
                  </Link>
                )}
                <button
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2 text-sm text-[var(--color-error)] hover:bg-red-500/10"
                >
                  Sign out
                </button>
              </div>
            </div>
          </div>
        </header>
        
        {/* Page content */}
        <main className="flex-1 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
