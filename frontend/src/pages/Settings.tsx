import { useState, useEffect } from 'react';
import { useAuthStore } from '../stores/authStore';
import { useThemeStore } from '../stores/themeStore';
import { useModelsStore } from '../stores/modelsStore';
import { useVoiceStore } from '../stores/voiceStore';
import api from '../lib/api';
import type { Theme } from '../types';

interface SubscriptionStats {
  id: string;
  name: string;
  slug: string;
  icon: string;
  color: string;
  tagline: string | null;
  owner_username: string;
  conversation_count: number;
  message_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export default function Settings() {
  const { user } = useAuthStore();
  const { themes, currentTheme, applyTheme, applyThemeToAccount, isLoading } = useThemeStore();
  const { models, subscribedAssistants, defaultModel, selectedModel, setSelectedModel, fetchModels, getDisplayName } = useModelsStore();
  const { 
    ttsEnabled, ttsMethod, selectedVoice, selectedLocalVoice, autoReadResponses,
    sttEnabled, selectedLanguage,
    availableVoices, localVoices, availableLanguages,
    ttsAvailable, sttAvailable, localTtsAvailable,
    isLoadingVoices, isLoadingLanguages,
    setTtsEnabled, setTtsMethod, setSelectedVoice, setSelectedLocalVoice, setAutoReadResponses,
    setSttEnabled, setSelectedLanguage,
    fetchVoices, fetchLocalVoices, fetchLanguages, checkServiceStatus
  } = useVoiceStore();
  const [activeTab, setActiveTab] = useState<'appearance' | 'account' | 'subscriptions' | 'preferences' | 'voice'>('appearance');
  const [subscriptionStats, setSubscriptionStats] = useState<SubscriptionStats[]>([]);
  const [loadingStats, setLoadingStats] = useState(false);
  const [unsubscribing, setUnsubscribing] = useState<string | null>(null);
  
  // Fetch models on mount
  useEffect(() => {
    fetchModels(true); // Force refresh to get latest subscriptions
  }, [fetchModels]);
  
  // Fetch subscription stats when tab is active
  useEffect(() => {
    if (activeTab === 'subscriptions') {
      loadSubscriptionStats();
    }
  }, [activeTab]);
  
  // Fetch voice options when voice tab is active
  useEffect(() => {
    if (activeTab === 'voice') {
      checkServiceStatus();
      fetchVoices();
      fetchLocalVoices();
      fetchLanguages();
    }
  }, [activeTab, checkServiceStatus, fetchVoices, fetchLocalVoices, fetchLanguages]);
  
  const loadSubscriptionStats = async () => {
    setLoadingStats(true);
    try {
      const res = await api.get('/assistants/subscribed/stats');
      setSubscriptionStats(res.data || []);
    } catch (err) {
      console.error('Failed to load subscription stats:', err);
    } finally {
      setLoadingStats(false);
    }
  };
  
  const handleUnsubscribe = async (assistantId: string) => {
    setUnsubscribing(assistantId);
    try {
      await api.delete(`/assistants/${assistantId}/subscribe`);
      setSubscriptionStats(prev => prev.filter(s => s.id !== assistantId));
      fetchModels(true); // Refresh models to remove from dropdown
    } catch (err) {
      console.error('Failed to unsubscribe:', err);
    } finally {
      setUnsubscribing(null);
    }
  };
  
  const formatTokens = (tokens: number) => {
    if (tokens >= 1000000) {
      return `${(tokens / 1000000).toFixed(1)}M`;
    }
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(1)}K`;
    }
    return tokens.toString();
  };
  
  const handleThemeSelect = async (theme: Theme) => {
    applyTheme(theme);
    try {
      await applyThemeToAccount(theme.id);
    } catch (err) {
      console.error('Failed to save theme preference:', err);
    }
  };
  
  const tabs = [
    { id: 'appearance', label: 'Appearance', iconType: 'appearance' },
    { id: 'account', label: 'Account', iconType: 'account' },
    { id: 'voice', label: 'Voice', iconType: 'voice' },
    { id: 'subscriptions', label: 'Subscriptions', iconType: 'subscriptions' },
    { id: 'preferences', label: 'Preferences', iconType: 'preferences' },
  ];
  
  const TabIcon = ({ type }: { type: string }) => {
    const iconClass = "w-4 h-4";
    switch (type) {
      case 'appearance':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" /></svg>;
      case 'account':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>;
      case 'voice':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>;
      case 'subscriptions':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>;
      case 'preferences':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
      default:
        return null;
    }
  };
  
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-2xl font-bold text-[var(--color-text)] mb-6">Settings</h1>
        
        {/* Tabs */}
        <div className="flex gap-2 mb-8 border-b border-[var(--color-border)]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px ${
                activeTab === tab.id
                  ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                  : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
              }`}
            >
              <TabIcon type={tab.iconType} />
              {tab.label}
            </button>
          ))}
        </div>
        
        {/* Appearance tab */}
        {activeTab === 'appearance' && (
          <div className="space-y-8">
            {/* Theme selection */}
            <section>
              <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Theme</h2>
              <p className="text-[var(--color-text-secondary)] text-sm mb-4">
                Choose a theme that matches your style. Your selection will be saved to your account.
              </p>
              
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <svg className="animate-spin h-6 w-6 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {themes.filter(t => t.is_system).map((theme) => (
                    <button
                      key={theme.id}
                      onClick={() => handleThemeSelect(theme)}
                      className={`group relative p-4 rounded-xl border-2 transition-all ${
                        currentTheme?.id === theme.id
                          ? 'border-[var(--color-primary)] ring-2 ring-[var(--color-primary)]'
                          : 'border-[var(--color-border)] hover:border-zinc-500/50'
                      }`}
                    >
                      {/* Theme preview */}
                      <div
                        className="h-24 rounded-lg mb-3 overflow-hidden"
                        style={{ backgroundColor: theme.colors.background }}
                      >
                        <div
                          className="h-6 flex items-center px-2 gap-1"
                          style={{ backgroundColor: theme.colors.surface }}
                        >
                          <div
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: theme.colors.error }}
                          />
                          <div
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: theme.colors.warning }}
                          />
                          <div
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: theme.colors.success }}
                          />
                        </div>
                        <div className="p-2 space-y-1">
                          <div
                            className="h-2 w-3/4 rounded"
                            style={{ backgroundColor: theme.colors.primary }}
                          />
                          <div
                            className="h-2 w-1/2 rounded"
                            style={{ backgroundColor: theme.colors.secondary }}
                          />
                          <div
                            className="h-2 w-2/3 rounded"
                            style={{ backgroundColor: theme.colors.text_secondary }}
                          />
                        </div>
                      </div>
                      
                      <div className="text-left">
                        <h3 className="font-medium text-[var(--color-text)]">{theme.name}</h3>
                        <p className="text-xs text-[var(--color-text-secondary)] mt-0.5 line-clamp-2">
                          {theme.description}
                        </p>
                      </div>
                      
                      {/* Selected indicator */}
                      {currentTheme?.id === theme.id && (
                        <div className="absolute top-2 right-2 w-6 h-6 rounded-full bg-[var(--color-button)] flex items-center justify-center">
                          <svg className="w-4 h-4 text-[var(--color-button-text)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </section>
            
            {/* Custom themes */}
            {themes.filter(t => !t.is_system).length > 0 && (
              <section>
                <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Custom Themes</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {themes.filter(t => !t.is_system).map((theme) => (
                    <button
                      key={theme.id}
                      onClick={() => handleThemeSelect(theme)}
                      className={`group relative p-4 rounded-xl border-2 transition-all ${
                        currentTheme?.id === theme.id
                          ? 'border-[var(--color-primary)] ring-2 ring-[var(--color-primary)]'
                          : 'border-[var(--color-border)] hover:border-zinc-500/50'
                      }`}
                    >
                      <div
                        className="h-16 rounded-lg mb-2"
                        style={{
                          background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary})`,
                        }}
                      />
                      <h3 className="font-medium text-[var(--color-text)]">{theme.name}</h3>
                    </button>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}
        
        {/* Account tab */}
        {activeTab === 'account' && (
          <div className="space-y-6">
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Account Information</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Email</label>
                  <p className="text-[var(--color-text)]">{user?.email}</p>
                </div>
                
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Username</label>
                  <p className="text-[var(--color-text)]">{user?.username}</p>
                </div>
                
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Plan</label>
                  <div className="flex items-center gap-2">
                    <span className="px-3 py-1 rounded-full bg-[var(--color-button)]/80 text-[var(--color-button-text)] text-sm font-medium capitalize">
                      {user?.tier || 'free'}
                    </span>
                    <a
                      href="/billing"
                      className="text-sm text-[var(--color-accent)] hover:underline"
                    >
                      Upgrade
                    </a>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Member since</label>
                  <p className="text-[var(--color-text)]">
                    {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                  </p>
                </div>
              </div>
            </section>
            
            {/* Danger zone */}
            <section className="bg-red-500/5 rounded-xl p-6 border border-red-500/20">
              <h2 className="text-lg font-semibold text-[var(--color-error)] mb-2">Danger Zone</h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                Once you delete your account, there is no going back. Please be certain.
              </p>
              <button className="px-4 py-2 rounded-lg border border-[var(--color-error)] text-[var(--color-error)] hover:bg-red-500/10 transition-colors">
                Delete Account
              </button>
            </section>
          </div>
        )}
        
        {/* Subscriptions tab */}
        {activeTab === 'subscriptions' && (
          <div className="space-y-6">
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">Custom GPT Subscriptions</h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-6">
                Manage your subscribed Custom GPTs and view your usage statistics.
              </p>
              
              {loadingStats ? (
                <div className="flex items-center justify-center py-8">
                  <svg className="animate-spin h-6 w-6 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
              ) : subscriptionStats.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-[var(--color-text-secondary)] mb-4">
                    You haven't subscribed to any Custom GPTs yet.
                  </p>
                  <a
                    href="/gpts"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                    </svg>
                    Explore Marketplace
                  </a>
                </div>
              ) : (
                <div className="space-y-4">
                  {subscriptionStats.map((sub) => (
                    <div
                      key={sub.id}
                      className="flex items-start gap-4 p-4 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]"
                    >
                      {/* Icon */}
                      <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl shrink-0"
                        style={{ backgroundColor: `${sub.color}20` }}
                      >
                        {sub.icon}
                      </div>
                      
                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <h3 className="font-medium text-[var(--color-text)]">{sub.name}</h3>
                            <p className="text-sm text-[var(--color-text-secondary)]">
                              by {sub.owner_username}
                            </p>
                            {sub.tagline && (
                              <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                                {sub.tagline}
                              </p>
                            )}
                          </div>
                          <button
                            onClick={() => handleUnsubscribe(sub.id)}
                            disabled={unsubscribing === sub.id}
                            className="shrink-0 px-3 py-1.5 text-sm rounded-lg border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-error)]/10 hover:border-[var(--color-error)]/30 hover:text-[var(--color-error)] transition-colors disabled:opacity-50"
                          >
                            {unsubscribing === sub.id ? 'Unsubscribing...' : 'Unsubscribe'}
                          </button>
                        </div>
                        
                        {/* Usage Stats */}
                        <div className="flex flex-wrap gap-4 mt-3 text-sm">
                          <div className="flex items-center gap-1.5">
                            <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                            </svg>
                            <span className="text-[var(--color-text)]">{sub.conversation_count}</span>
                            <span className="text-[var(--color-text-secondary)]">chats</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span className="text-[var(--color-text)]">{sub.message_count}</span>
                            <span className="text-[var(--color-text-secondary)]">messages</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                            </svg>
                            <span className="text-[var(--color-text)]">{formatTokens(sub.total_tokens)}</span>
                            <span className="text-[var(--color-text-secondary)]">tokens</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
            
            <div className="text-center">
              <a
                href="/gpts"
                className="text-sm text-[var(--color-accent)] hover:underline"
              >
                Browse more Custom GPTs in the Marketplace →
              </a>
            </div>
          </div>
        )}
        
        {/* Voice tab */}
        {activeTab === 'voice' && (
          <div className="space-y-6">
            {/* Service Status */}
            <div className="flex gap-4 mb-4">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                ttsAvailable 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-red-500/20 text-red-400'
              }`}>
                <span className={`w-2 h-2 rounded-full ${ttsAvailable ? 'bg-green-400' : 'bg-red-400'}`}></span>
                TTS {ttsAvailable ? 'Available' : 'Unavailable'}
              </div>
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                sttAvailable 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-red-500/20 text-red-400'
              }`}>
                <span className={`w-2 h-2 rounded-full ${sttAvailable ? 'bg-green-400' : 'bg-red-400'}`}></span>
                STT {sttAvailable ? 'Available' : 'Unavailable'}
              </div>
            </div>

            {/* Text-to-Speech Settings */}
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <div className="flex items-center gap-3 mb-4">
                <svg className="w-5 h-5 text-[var(--color-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
                <h2 className="text-lg font-semibold text-[var(--color-text)]">Text-to-Speech (TTS)</h2>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Enable TTS</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Allow reading responses aloud
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={ttsEnabled}
                      onChange={(e) => setTtsEnabled(e.target.checked)}
                      className="sr-only peer" 
                    />
                    <div className="w-11 h-6 bg-[var(--color-border)] peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-[var(--color-primary)]/30 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--color-button)]"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">TTS Method</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      {ttsMethod === 'natural' ? 'High-quality server-side voices' : 'Device/browser voices'}
                    </p>
                  </div>
                  <select 
                    value={ttsMethod}
                    onChange={(e) => setTtsMethod(e.target.value as 'natural' | 'local')}
                    disabled={!ttsEnabled}
                    className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 min-w-[200px]"
                  >
                    <option value="natural" disabled={!ttsAvailable}>
                      Natural Voice {!ttsAvailable && '(Unavailable)'}
                    </option>
                    <option value="local" disabled={!localTtsAvailable}>
                      OS Specific {!localTtsAvailable && '(Unavailable)'}
                    </option>
                  </select>
                </div>
                
                {/* Natural Voice Selection (Kokoro) */}
                {ttsMethod === 'natural' && (
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-[var(--color-text)] font-medium">Natural Voice</h3>
                      <p className="text-sm text-[var(--color-text-secondary)]">
                        High-quality AI-generated voices
                      </p>
                    </div>
                    <select 
                      value={selectedVoice}
                      onChange={(e) => setSelectedVoice(e.target.value)}
                      disabled={!ttsEnabled || isLoadingVoices || !ttsAvailable}
                      className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 min-w-[200px]"
                    >
                      {isLoadingVoices ? (
                        <option>Loading voices...</option>
                      ) : availableVoices.length > 0 ? (
                        <>
                          <optgroup label="Female Voices">
                            {availableVoices.filter(v => v.gender === 'female').map((voice) => (
                              <option key={voice.id} value={voice.id}>
                                {voice.name}
                              </option>
                            ))}
                          </optgroup>
                          <optgroup label="Male Voices">
                            {availableVoices.filter(v => v.gender === 'male').map((voice) => (
                              <option key={voice.id} value={voice.id}>
                                {voice.name}
                              </option>
                            ))}
                          </optgroup>
                        </>
                      ) : (
                        <option value="af_heart">Heart (Female)</option>
                      )}
                    </select>
                  </div>
                )}
                
                {/* OS Specific Voice Selection (Web Speech API) */}
                {ttsMethod === 'local' && (
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-[var(--color-text)] font-medium">OS Voice</h3>
                      <p className="text-sm text-[var(--color-text-secondary)]">
                        {localVoices.length > 0 
                          ? `${localVoices.length} voices available on your device`
                          : 'No local voices detected'}
                      </p>
                    </div>
                    <select 
                      value={selectedLocalVoice}
                      onChange={(e) => setSelectedLocalVoice(e.target.value)}
                      disabled={!ttsEnabled || !localTtsAvailable}
                      className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 min-w-[200px] max-w-[300px]"
                    >
                      {localVoices.length > 0 ? (
                        <>
                          {/* Group by language */}
                          {(() => {
                            const byLang: Record<string, typeof localVoices> = {};
                            localVoices.forEach(v => {
                              const lang = v.lang.split('-')[0].toUpperCase();
                              if (!byLang[lang]) byLang[lang] = [];
                              byLang[lang].push(v);
                            });
                            return Object.entries(byLang).map(([lang, voices]) => (
                              <optgroup key={lang} label={lang}>
                                {voices.map((voice) => (
                                  <option key={voice.id} value={voice.id}>
                                    {voice.name}
                                  </option>
                                ))}
                              </optgroup>
                            ));
                          })()}
                        </>
                      ) : (
                        <option value="">No voices available</option>
                      )}
                    </select>
                  </div>
                )}
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Auto-Read Responses</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Automatically read new AI responses aloud
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={autoReadResponses}
                      onChange={(e) => setAutoReadResponses(e.target.checked)}
                      disabled={!ttsEnabled}
                      className="sr-only peer" 
                    />
                    <div className="w-11 h-6 bg-[var(--color-border)] peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-[var(--color-primary)]/30 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--color-button)] peer-disabled:opacity-50"></div>
                  </label>
                </div>
              </div>
            </section>

            {/* Speech-to-Text Settings */}
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <div className="flex items-center gap-3 mb-4">
                <svg className="w-5 h-5 text-[var(--color-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                <h2 className="text-lg font-semibold text-[var(--color-text)]">Speech-to-Text (STT)</h2>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Enable STT</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Allow voice input for messages
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={sttEnabled}
                      onChange={(e) => setSttEnabled(e.target.checked)}
                      className="sr-only peer" 
                    />
                    <div className="w-11 h-6 bg-[var(--color-border)] peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-[var(--color-primary)]/30 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--color-button)]"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Language</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Language for speech recognition (auto-detect if not set)
                    </p>
                  </div>
                  <select 
                    value={selectedLanguage}
                    onChange={(e) => setSelectedLanguage(e.target.value)}
                    disabled={!sttEnabled || isLoadingLanguages}
                    className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 min-w-[200px]"
                  >
                    <option value="">Auto-detect</option>
                    {isLoadingLanguages ? (
                      <option>Loading languages...</option>
                    ) : (
                      availableLanguages.slice(0, 30).map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.name}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              </div>
            </section>

            {/* Talk to Me Mode Info */}
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <div className="flex items-center gap-3 mb-4">
                <svg className="w-5 h-5 text-[var(--color-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <h2 className="text-lg font-semibold text-[var(--color-text)]">Talk to Me Mode</h2>
              </div>
              
              <p className="text-[var(--color-text-secondary)] text-sm mb-4">
                Enable voice conversation mode by clicking the microphone button next to the send button. 
                In this mode, you speak your messages and the AI will read its responses aloud automatically.
              </p>
              
              <div className="bg-[var(--color-background)] rounded-lg p-4 text-sm">
                <p className="text-[var(--color-text)] font-medium mb-2">How it works:</p>
                <ul className="text-[var(--color-text-secondary)] space-y-1 list-disc list-inside">
                  <li>Click the microphone button to start voice mode</li>
                  <li>Speak your message - it will be transcribed and sent</li>
                  <li>The AI response will be read aloud automatically</li>
                  <li>Say "STOP" or click the stop button to exit voice mode</li>
                </ul>
              </div>
            </section>
          </div>
        )}
        
        {/* Preferences tab */}
        {activeTab === 'preferences' && (
          <div className="space-y-6">
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Chat Preferences</h2>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Default Model</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Choose the default AI model for new chats
                    </p>
                  </div>
                  <select 
                    value={selectedModel || defaultModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  >
                    {/* Regular models */}
                    {models.length > 0 ? (
                      models.map((model) => (
                        <option key={model.id} value={model.id}>
                          {getDisplayName(model.id)}
                        </option>
                      ))
                    ) : (
                      <option value={defaultModel}>{getDisplayName(defaultModel)}</option>
                    )}
                    
                    {/* Subscribed assistants */}
                    {subscribedAssistants.length > 0 && (
                      <>
                        <option disabled>──────────</option>
                        {subscribedAssistants.map((assistant) => (
                          <option key={assistant.id} value={assistant.id}>
                            {assistant.name}
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Stream Responses</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Show responses as they're being generated
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" defaultChecked className="sr-only peer" />
                    <div className="w-11 h-6 bg-[var(--color-border)] peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-[var(--color-primary)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--color-button)]"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Sound Effects</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Play sounds for notifications and events
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" className="sr-only peer" />
                    <div className="w-11 h-6 bg-[var(--color-border)] peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-[var(--color-primary)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--color-button)]"></div>
                  </label>
                </div>
              </div>
            </section>
            
            <section className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Data & Privacy</h2>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--color-text)] font-medium">Chat History</h3>
                    <p className="text-sm text-[var(--color-text-secondary)]">
                      Save chat history for future reference
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" defaultChecked className="sr-only peer" />
                    <div className="w-11 h-6 bg-[var(--color-border)] peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-[var(--color-primary)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--color-button)]"></div>
                  </label>
                </div>
                
                <button className="px-4 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-zinc-700/30 transition-colors">
                  Export All Data
                </button>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
