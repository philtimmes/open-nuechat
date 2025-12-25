import { useEffect, useState } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { useBrandingStore } from '../stores/brandingStore';
import { useAuthStore } from '../stores/authStore';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';

interface SharedAttachment {
  type: 'image' | 'file';
  name?: string;
  url?: string;
  data?: string;
  mime_type?: string;
}

interface SharedMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  parent_id?: string | null;
  input_tokens?: number;
  output_tokens?: number;
  sibling_count?: number;
  attachments?: SharedAttachment[];
}

interface SharedChatData {
  id: string;
  title: string;
  model: string;
  assistant_id?: string;
  assistant_name?: string;
  owner_name?: string | null;  // null if anonymous
  created_at: string;
  messages: SharedMessage[];
  all_messages?: SharedMessage[];
  has_branches?: boolean;
}

export default function SharedChat() {
  const { shareId } = useParams<{ shareId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [chat, setChat] = useState<SharedChatData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAllMessages, setShowAllMessages] = useState(searchParams.get('view') === 'all');
  const [cloning, setCloning] = useState(false);
  
  const { config, loadConfig, isLoaded: brandingLoaded } = useBrandingStore();
  const { isAuthenticated, accessToken } = useAuthStore();
  
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);
  
  const handleContinueChat = async () => {
    if (!shareId) return;
    
    // If not logged in, redirect to login with return URL
    if (!isAuthenticated || !accessToken) {
      // Store the intended action in sessionStorage
      sessionStorage.setItem('continueSharedChat', shareId);
      navigate('/login');
      return;
    }
    
    // Clone the chat
    setCloning(true);
    try {
      const response = await fetch(`/api/shared/${shareId}/clone`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to clone chat');
      }
      
      const data = await response.json();
      // Navigate to the new chat
      navigate(`/?chat=${data.chat_id}`);
    } catch (err) {
      console.error('Failed to clone chat:', err);
      alert('Failed to continue chat. Please try again.');
    } finally {
      setCloning(false);
    }
  };
  
  useEffect(() => {
    const fetchSharedChat = async () => {
      if (!shareId) return;
      
      try {
        const response = await fetch(`/api/shared/${shareId}`);
        if (!response.ok) {
          if (response.status === 404) {
            setError('This shared chat could not be found.');
          } else {
            setError('Failed to load shared chat.');
          }
          return;
        }
        
        const data = await response.json();
        setChat(data);
      } catch (err) {
        console.error('Failed to fetch shared chat:', err);
        setError('Failed to load shared chat.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchSharedChat();
  }, [shareId]);
  
  const appName = config?.app_name || 'Open-NueChat';
  // Use chat's assistant_name (Custom GPT name) first, then model name
  const assistantDisplayName = chat?.assistant_name || chat?.model || 'Assistant';
  
  // Wait for both chat and branding to load
  if (loading || !brandingLoaded) {
    return (
      <div className="min-h-screen bg-[var(--color-background)] flex items-center justify-center">
        <div className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Loading...
        </div>
      </div>
    );
  }
  
  if (error || !chat) {
    return (
      <div className="min-h-screen bg-[var(--color-background)] flex items-center justify-center">
        <div className="text-center p-8">
          <h1 className="text-2xl font-bold text-[var(--color-text)] mb-2">Chat Not Found</h1>
          <p className="text-[var(--color-text-secondary)]">{error || 'This shared chat could not be found.'}</p>
          <a
            href="/"
            className="inline-block mt-4 px-4 py-2 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90 transition-opacity"
          >
            Go Home
          </a>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-[var(--color-background)] flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--color-border)] px-4 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div>
            <a href="/" className="text-lg font-semibold text-[var(--color-text)] hover:text-[var(--color-primary)] transition-colors">
              {config?.logo_url ? (
                <img src={config.logo_url} alt={appName} className="h-8" />
              ) : (
                appName
              )}
            </a>
          </div>
          <div className="text-sm text-[var(--color-text-secondary)]">
            Shared Chat
          </div>
        </div>
      </header>
      
      {/* Chat title */}
      <div className="border-b border-[var(--color-border)] px-4 py-2">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-lg font-medium text-[var(--color-text)]">{chat.title}</h1>
          <div className="flex items-center justify-between">
            <p className="text-xs text-[var(--color-text-secondary)]">
              {new Date(chat.created_at).toLocaleDateString()} • {assistantDisplayName}
            </p>
            {chat.has_branches && chat.all_messages && (
              <button
                onClick={() => setShowAllMessages(!showAllMessages)}
                className="text-xs px-2 py-1 rounded bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] border border-[var(--color-border)] transition-colors"
              >
                {showAllMessages ? '🌲 Show Branch' : '📋 Show All Messages'}
              </button>
            )}
          </div>
          {showAllMessages && chat.has_branches && (
            <p className="text-xs text-[var(--color-text-secondary)] mt-1 italic">
              Showing all {chat.all_messages?.length || 0} messages chronologically
            </p>
          )}
        </div>
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto divide-y divide-[var(--color-border)]/50">
          {(showAllMessages && chat.all_messages ? chat.all_messages : chat.messages).map((message) => {
            // Determine user display name: owner_name, or "Anonymous" if null
            const userDisplayName = chat.owner_name || 'Anonymous';
            
            return (
            <div key={message.id} className="py-4 px-4">
              <div className="flex items-start gap-3">
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-sm font-medium ${
                  message.role === 'user'
                    ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                    : 'bg-[var(--color-surface)] text-[var(--color-text)] border border-[var(--color-border)]'
                }`}>
                  {message.role === 'user' ? userDisplayName.charAt(0).toUpperCase() : 'AI'}
                </div>
                
                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-[var(--color-text)]">
                      {message.role === 'user' ? userDisplayName : assistantDisplayName}
                    </span>
                    {message.output_tokens && (
                      <span className="text-xs text-[var(--color-text-secondary)]">
                        • {message.output_tokens} tokens
                      </span>
                    )}
                  </div>
                  
                  {message.role === 'user' ? (
                    // User messages: preserve whitespace/newlines exactly as typed
                    <div className="max-w-none text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                      {message.content}
                    </div>
                  ) : (
                  <div className="max-w-none text-[var(--color-text)] leading-relaxed">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        code({ className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          const isInline = !match && !className;
                          
                          if (isInline) {
                            return (
                              <code className="px-1.5 py-0.5 rounded bg-[var(--color-surface)] text-[var(--color-secondary)] text-sm" {...props}>
                                {children}
                              </code>
                            );
                          }
                          
                          return (
                            <SyntaxHighlighter
                              style={oneDark}
                              language={match?.[1] || 'text'}
                              PreTag="div"
                              customStyle={{
                                margin: '0.5rem 0',
                                borderRadius: '0.5rem',
                                fontSize: '0.875rem',
                              }}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          );
                        },
                        p({ children }) {
                          return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>;
                        },
                        ul({ children }) {
                          return <ul className="list-disc list-outside ml-4 mb-2 space-y-0.5">{children}</ul>;
                        },
                        ol({ children }) {
                          return <ol className="list-decimal list-outside ml-4 mb-2 space-y-0.5">{children}</ol>;
                        },
                        li({ children }) {
                          return <li className="text-[var(--color-text)] pl-1">{children}</li>;
                        },
                        table({ children }) {
                          return (
                            <div className="overflow-x-auto my-4 rounded-lg border border-[var(--color-border)]">
                              <table className="min-w-full divide-y divide-[var(--color-border)]">
                                {children}
                              </table>
                            </div>
                          );
                        },
                        thead({ children }) {
                          return <thead className="bg-[var(--color-surface)]">{children}</thead>;
                        },
                        tbody({ children }) {
                          return <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-background)]">{children}</tbody>;
                        },
                        tr({ children }) {
                          return <tr className="hover:bg-[var(--color-surface)]/50 transition-colors">{children}</tr>;
                        },
                        th({ children }) {
                          return (
                            <th className="px-3 py-2 text-left text-xs font-semibold text-[var(--color-text)] uppercase tracking-wider whitespace-nowrap">
                              {children}
                            </th>
                          );
                        },
                        td({ children }) {
                          return (
                            <td className="px-3 py-2 text-sm text-[var(--color-text-secondary)] whitespace-normal">
                              {children}
                            </td>
                          );
                        },
                        a({ href, children }) {
                          return (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[var(--color-accent)] hover:underline"
                            >
                              {children}
                            </a>
                          );
                        },
                        blockquote({ children }) {
                          return (
                            <blockquote className="border-l-2 border-[var(--color-border)] pl-4 italic text-[var(--color-text-secondary)]">
                              {children}
                            </blockquote>
                          );
                        },
                        h1({ children }) {
                          return <h1 className="text-2xl font-bold mt-6 mb-3 text-[var(--color-text)]">{children}</h1>;
                        },
                        h2({ children }) {
                          return <h2 className="text-xl font-bold mt-5 mb-2 text-[var(--color-text)]">{children}</h2>;
                        },
                        h3({ children }) {
                          return <h3 className="text-lg font-semibold mt-4 mb-2 text-[var(--color-text)]">{children}</h3>;
                        },
                        h4({ children }) {
                          return <h4 className="text-base font-semibold mt-3 mb-1 text-[var(--color-text)]">{children}</h4>;
                        },
                        h5({ children }) {
                          return <h5 className="text-sm font-semibold mt-2 mb-1 text-[var(--color-text)]">{children}</h5>;
                        },
                        h6({ children }) {
                          return <h6 className="text-sm font-medium mt-2 mb-1 text-[var(--color-text-secondary)]">{children}</h6>;
                        },
                        strong({ children }) {
                          return <strong className="font-bold text-[var(--color-text)]">{children}</strong>;
                        },
                        em({ children }) {
                          return <em className="italic">{children}</em>;
                        },
                        hr() {
                          return <hr className="my-4 border-t border-[var(--color-border)]" />;
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                  )}
                  
                  {/* Image Attachments */}
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {message.attachments.map((att, idx) => (
                        <div key={idx}>
                          {att.type === 'image' ? (
                            <img
                              src={att.url || (att.data ? `data:${att.mime_type || 'image/jpeg'};base64,${att.data}` : '')}
                              alt={att.name || 'Attached image'}
                              className="max-w-sm max-h-64 rounded border border-[var(--color-border)]"
                            />
                          ) : (
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-sm">
                              <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                              <span className="truncate max-w-[200px] text-[var(--color-text)]">{att.name || 'File'}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
          })}
        </div>
      </div>
      
      {/* Footer with Continue Chat button */}
      <footer className="border-t border-[var(--color-border)] px-4 py-4">
        <div className="max-w-4xl mx-auto flex flex-col items-center gap-3">
          <button
            onClick={handleContinueChat}
            disabled={cloning}
            className="px-6 py-2.5 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90 transition-opacity font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {cloning ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Cloning...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Continue this Chat
              </>
            )}
          </button>
          <p className="text-sm text-[var(--color-text-secondary)]">
            Powered by {appName}
          </p>
        </div>
      </footer>
    </div>
  );
}
