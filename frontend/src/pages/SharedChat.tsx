import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useBrandingStore } from '../stores/brandingStore';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface SharedMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  input_tokens?: number;
  output_tokens?: number;
}

interface SharedChatData {
  id: string;
  title: string;
  model: string;
  created_at: string;
  messages: SharedMessage[];
}

export default function SharedChat() {
  const { shareId } = useParams<{ shareId: string }>();
  const [chat, setChat] = useState<SharedChatData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const { config, loadConfig } = useBrandingStore();
  
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);
  
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
  
  if (loading) {
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
          <p className="text-xs text-[var(--color-text-secondary)]">
            {new Date(chat.created_at).toLocaleDateString()} • {chat.model}
          </p>
        </div>
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto divide-y divide-[var(--color-border)]/50">
          {chat.messages.map((message) => (
            <div key={message.id} className="py-4 px-4">
              <div className="flex items-start gap-3">
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-sm font-medium ${
                  message.role === 'user'
                    ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                    : 'bg-[var(--color-surface)] text-[var(--color-text)] border border-[var(--color-border)]'
                }`}>
                  {message.role === 'user' ? 'U' : 'AI'}
                </div>
                
                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-[var(--color-text)]">
                      {message.role === 'user' ? 'User' : chat.model}
                    </span>
                    {message.output_tokens && (
                      <span className="text-xs text-[var(--color-text-secondary)]">
                        • {message.output_tokens} tokens
                      </span>
                    )}
                  </div>
                  
                  <div className="text-[var(--color-text)] prose prose-invert max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
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
                          return <p className="mb-3 last:mb-0">{children}</p>;
                        },
                        ul({ children }) {
                          return <ul className="list-disc pl-4 mb-3 space-y-1">{children}</ul>;
                        },
                        ol({ children }) {
                          return <ol className="list-decimal pl-4 mb-3 space-y-1">{children}</ol>;
                        },
                        table({ children }) {
                          return (
                            <div className="overflow-x-auto my-3">
                              <table className="min-w-full border-collapse border border-[var(--color-border)]">
                                {children}
                              </table>
                            </div>
                          );
                        },
                        th({ children }) {
                          return (
                            <th className="border border-[var(--color-border)] px-3 py-2 bg-[var(--color-surface)] text-left font-medium">
                              {children}
                            </th>
                          );
                        },
                        td({ children }) {
                          return (
                            <td className="border border-[var(--color-border)] px-3 py-2">
                              {children}
                            </td>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Footer */}
      <footer className="border-t border-[var(--color-border)] px-4 py-3">
        <div className="max-w-4xl mx-auto text-center text-sm text-[var(--color-text-secondary)]">
          Powered by {appName}
        </div>
      </footer>
    </div>
  );
}
