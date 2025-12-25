/**
 * VibeChat - Chat panel for AI assistance in code editor
 */

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

type AgentState = 'idle' | 'analyzing' | 'planning' | 'generating' | 'reviewing' | 'applying' | 'testing' | 'complete' | 'error';

interface VibeChatProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  onCreateFile?: (filename: string, content: string, language?: string) => void;
  onInsertCode?: (code: string) => void;
  isStreaming: boolean;
  agentState: AgentState;
}

// Language to file extension mapping
const langToExt: Record<string, string> = {
  typescript: 'ts',
  javascript: 'js',
  python: 'py',
  rust: 'rs',
  go: 'go',
  java: 'java',
  cpp: 'cpp',
  c: 'c',
  csharp: 'cs',
  ruby: 'rb',
  php: 'php',
  swift: 'swift',
  kotlin: 'kt',
  html: 'html',
  css: 'css',
  scss: 'scss',
  json: 'json',
  yaml: 'yaml',
  markdown: 'md',
  sql: 'sql',
  shell: 'sh',
  bash: 'sh',
  tsx: 'tsx',
  jsx: 'jsx',
};

export default function VibeChat({
  messages,
  onSendMessage,
  onCreateFile,
  onInsertCode,
  isStreaming,
  agentState,
}: VibeChatProps) {
  const [input, setInput] = useState('');
  const [copiedIndex, setCopiedIndex] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isStreaming) {
      onSendMessage(input.trim());
      setInput('');
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  const handleCopy = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedIndex(id);
    setTimeout(() => setCopiedIndex(null), 2000);
  };
  
  const handleCreateFileFromCode = (code: string, language: string) => {
    if (!onCreateFile) return;
    
    // Prompt for filename
    const ext = langToExt[language] || 'txt';
    const defaultName = `new_file.${ext}`;
    const filename = window.prompt('Enter filename:', defaultName);
    
    if (filename) {
      onCreateFile(filename, code, language);
    }
  };
  
  const handleInsertCode = (code: string) => {
    if (onInsertCode) {
      onInsertCode(code);
    }
  };
  
  // Quick action buttons
  const quickActions = [
    { label: 'Explain', prompt: 'Explain this code' },
    { label: 'Fix bugs', prompt: 'Find and fix bugs in this code' },
    { label: 'Refactor', prompt: 'Refactor this code to be cleaner' },
    { label: 'Add tests', prompt: 'Write tests for this code' },
    { label: 'Document', prompt: 'Add documentation to this code' },
  ];
  
  // Counter for unique code block IDs
  let codeBlockCounter = 0;
  
  return (
    <div className="flex flex-col h-full bg-[var(--color-background)]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
          <span className="font-medium text-[var(--color-text)]">AI Assistant</span>
        </div>
        {agentState !== 'idle' && (
          <div className="flex items-center gap-2 text-xs text-[var(--color-primary)]">
            <div className="w-2 h-2 rounded-full bg-[var(--color-primary)] animate-pulse" />
            {agentState}
          </div>
        )}
      </div>
      
      {/* Quick actions */}
      <div className="flex gap-1 px-3 py-2 overflow-x-auto border-b border-[var(--color-border)]">
        {quickActions.map((action) => (
          <button
            key={action.label}
            onClick={() => onSendMessage(action.prompt)}
            disabled={isStreaming}
            className="px-2 py-1 text-xs rounded bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-background)] border border-[var(--color-border)] whitespace-nowrap disabled:opacity-50"
          >
            {action.label}
          </button>
        ))}
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-[var(--color-text-secondary)] py-8">
            <div className="text-4xl mb-3">💬</div>
            <div className="text-sm font-medium mb-2">Start a conversation</div>
            <div className="text-xs">
              Ask questions, request code, or describe what you want to build
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div
              className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-sm ${
                message.role === 'user'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'bg-[var(--color-surface)] text-[var(--color-text)] border border-[var(--color-border)]'
              }`}
            >
              {message.role === 'user' ? '👤' : '🤖'}
            </div>
            
            <div
              className={`max-w-[85%] rounded-lg px-3 py-2 ${
                message.role === 'user'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'bg-[var(--color-surface)] text-[var(--color-text)]'
              }`}
            >
              {message.role === 'assistant' ? (
                <div className="prose prose-sm prose-invert max-w-none">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      code({ className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '');
                        const isInline = !match && !className;
                        const code = String(children).replace(/\n$/, '');
                        const language = match?.[1] || 'text';
                        const blockId = `${message.id}-${codeBlockCounter++}`;
                        
                        if (isInline) {
                          return (
                            <code className="px-1 py-0.5 rounded bg-[var(--color-background)] text-[var(--color-secondary)] text-xs" {...props}>
                              {children}
                            </code>
                          );
                        }
                        
                        return (
                          <div className="relative group my-2">
                            {/* Code block header with actions */}
                            <div className="flex items-center justify-between px-3 py-1 bg-[var(--color-background)] border border-[var(--color-border)] border-b-0 rounded-t text-xs">
                              <span className="text-[var(--color-text-secondary)]">{language}</span>
                              <div className="flex items-center gap-1">
                                {/* Copy button */}
                                <button
                                  onClick={() => handleCopy(code, blockId)}
                                  className="p-1 rounded hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                                  title="Copy code"
                                >
                                  {copiedIndex === blockId ? (
                                    <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                  ) : (
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                    </svg>
                                  )}
                                </button>
                                
                                {/* Insert button */}
                                {onInsertCode && (
                                  <button
                                    onClick={() => handleInsertCode(code)}
                                    className="p-1 rounded hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                                    title="Insert at cursor"
                                  >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
                                    </svg>
                                  </button>
                                )}
                                
                                {/* Create file button */}
                                {onCreateFile && (
                                  <button
                                    onClick={() => handleCreateFileFromCode(code, language)}
                                    className="p-1 rounded hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                                    title="Create new file"
                                  >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                  </button>
                                )}
                              </div>
                            </div>
                            
                            <SyntaxHighlighter
                              style={vscDarkPlus}
                              language={language}
                              PreTag="div"
                              customStyle={{
                                margin: 0,
                                borderRadius: '0 0 0.375rem 0.375rem',
                                fontSize: '12px',
                                border: '1px solid var(--color-border)',
                                borderTop: 'none',
                              }}
                            >
                              {code}
                            </SyntaxHighlighter>
                          </div>
                        );
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="text-sm whitespace-pre-wrap">{message.content}</div>
              )}
              
              <div className="text-[10px] opacity-60 mt-1">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        
        {isStreaming && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-[var(--color-surface)] flex items-center justify-center text-sm border border-[var(--color-border)]">
              🤖
            </div>
            <div className="bg-[var(--color-surface)] rounded-lg px-3 py-2">
              <div className="flex gap-1">
                <span className="w-2 h-2 rounded-full bg-[var(--color-text-secondary)] animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 rounded-full bg-[var(--color-text-secondary)] animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 rounded-full bg-[var(--color-text-secondary)] animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input */}
      <div className="border-t border-[var(--color-border)] p-3">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? 'Waiting...' : 'Ask anything or describe what to build...'}
            disabled={isStreaming}
            className="flex-1 px-3 py-2 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-none disabled:opacity-50"
            rows={2}
          />
          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className="px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </form>
        
        <div className="mt-2 text-[10px] text-[var(--color-text-secondary)] text-center">
          Shift+Enter for new line • Tab to accept suggestions
        </div>
      </div>
    </div>
  );
}
