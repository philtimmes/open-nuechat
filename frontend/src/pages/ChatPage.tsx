import { useEffect, useRef, useState, useCallback, memo, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useChatStore } from '../stores/chatStore';
import { useBrandingStore } from '../stores/brandingStore';
import { useModelsStore } from '../stores/modelsStore';
import { useVoiceStore } from '../stores/voiceStore';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useVoice } from '../hooks/useVoice';
import { useChatShortcuts } from '../hooks/useKeyboardShortcuts';
import MessageBubble from '../components/MessageBubble';
import ChatInput from '../components/ChatInput';
import EmptyState from '../components/EmptyState';
import ArtifactsPanel from '../components/ArtifactsPanel';
import SummaryPanel from '../components/SummaryPanel';
import VoiceModeOverlay from '../components/VoiceModeOverlay';
import type { Artifact, Message, GeneratedImage } from '../types';
import { groupArtifactsByFilename, getLatestArtifacts } from '../lib/artifacts';
import { chatApi } from '../lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * Fix nested code fences for proper markdown rendering.
 * 
 * Problem: When LLM generates a file containing code fences (like README.md),
 * the inner ``` closes the outer block prematurely.
 * 
 * Solution: Detect nested fences and increase outer fence to 4+ backticks.
 * CommonMark spec says closing fence must match opening length.
 */
function fixNestedCodeFences(content: string): string {
  const lines = content.split('\n');
  const result: string[] = [];
  
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const openMatch = line.match(/^(`{3,})(\w*)$/);
    
    if (openMatch) {
      const [, backticks, lang] = openMatch;
      const openingLength = backticks.length;
      
      // Scan ahead to find content and detect if it has nested fences
      let hasNestedFences = false;
      let depth = 1;
      let endLine = i + 1;
      
      for (let j = i + 1; j < lines.length && depth > 0; j++) {
        const l = lines[j];
        
        // Check for nested opening fence (has language)
        if (l.match(/^`{3,}\w+$/)) {
          hasNestedFences = true;
          depth++;
        }
        // Check for closing fence (no language)
        else if (l.match(/^`{3,}\s*$/)) {
          depth--;
          if (depth === 0) {
            endLine = j;
          }
        }
      }
      
      // If we found nested fences and the outer fence is only 3 backticks,
      // increase it to 4 backticks so ReactMarkdown parses correctly
      if (hasNestedFences && openingLength === 3) {
        result.push('````' + lang);
        
        // Copy content lines
        for (let j = i + 1; j < endLine; j++) {
          result.push(lines[j]);
        }
        
        // Add closing fence with same length
        result.push('````');
        i = endLine + 1;
      } else {
        // No nested fences or already using 4+ backticks
        result.push(line);
        i++;
      }
    } else {
      result.push(line);
      i++;
    }
  }
  
  return result.join('\n');
}

// Lightweight markdown renderer for streaming - memoized to prevent re-parsing unchanged content
const MessageMarkdown = memo(function MessageMarkdown({ content }: { content: string }) {
  // Fix nested code fences before rendering
  const fixedContent = useMemo(() => fixNestedCodeFences(content), [content]);
  
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, inline, className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || '');
          const code = String(children).replace(/\n$/, '');
          
          if (!inline && match) {
            return (
              <div className="relative my-3">
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    margin: 0,
                    borderRadius: '0.5rem',
                    fontSize: '0.8125rem',
                  }}
                  {...props}
                >
                  {code}
                </SyntaxHighlighter>
              </div>
            );
          }
          
          return (
            <code
              className={`${className || ''} px-1 py-0.5 rounded bg-zinc-800 text-[var(--color-accent)] text-sm`}
              {...props}
            >
              {children}
            </code>
          );
        },
        // Table components for GFM tables
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
          return <th className="px-3 py-2 text-left text-xs font-semibold text-[var(--color-text)] uppercase tracking-wider whitespace-nowrap">{children}</th>;
        },
        td({ children }) {
          return <td className="px-3 py-2 text-sm text-[var(--color-text-secondary)] whitespace-normal">{children}</td>;
        },
        p({ children }) {
          return <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>;
        },
        ul({ children }) {
          return <ul className="list-disc list-outside ml-4 mb-3 space-y-1">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="list-decimal list-outside ml-4 mb-3 space-y-1">{children}</ol>;
        },
        a({ href, children }) {
          return (
            <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--color-accent)] hover:underline">
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
      }}
    >
      {fixedContent}
    </ReactMarkdown>
  );
});

// Memoized message list - only re-renders when messages array reference changes
interface MessageListProps {
  messages: Message[];
  lastAssistantIndex: number;
  generatedImages: Record<string, GeneratedImage>;
  onRetry: (userMessageId: string, userContent: string) => void;
  onEdit: (messageId: string, newContent: string) => void;
  onDelete: (messageId: string) => void;
  onReadAloud?: (content: string) => void;
  onImageRetry?: (prompt: string, width?: number, height?: number, seed?: number) => void;
  onImageEdit?: (prompt: string) => void;
  readingMessageId?: string | null;
  onArtifactClick: (artifact: Artifact) => void;
  onBranchChange: (parentId: string, childId: string) => void;
  assistantName?: string;
  chatId?: string;
  initialSelectedVersions?: Record<string, string>;
  onCurrentLeafChange?: (leafId: string | null) => void;
}

// Build conversation path from tree structure
// Returns messages in display order following the selected branches
interface ConversationNode {
  message: Message;
  siblings: Message[];  // All messages with same parent (including this one)
  siblingIndex: number; // Index of this message among siblings
}

function buildConversationPath(
  messages: Message[],
  selectedVersions: Record<string, string>
): ConversationNode[] {
  if (messages.length === 0) return [];
  
  // Check if any messages have parent_id set (new tree structure)
  const hasTreeStructure = messages.some(m => m.parent_id);
  
  if (!hasTreeStructure) {
    // Legacy mode: no parent_id, show all messages in chronological order
    // This handles existing chats before the migration
    const sorted = [...messages].sort((a, b) => 
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );
    return sorted.map(msg => ({
      message: msg,
      siblings: [msg],
      siblingIndex: 0,
    }));
  }
  
  // Tree mode: build children map and walk the tree
  const childrenByParent = new Map<string, Message[]>();
  
  for (const msg of messages) {
    const parentKey = msg.parent_id || 'root';
    if (!childrenByParent.has(parentKey)) {
      childrenByParent.set(parentKey, []);
    }
    childrenByParent.get(parentKey)!.push(msg);
  }
  
  // Sort children by created_at
  for (const children of childrenByParent.values()) {
    children.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
  }
  
  // Walk the tree from root, following selected branches
  const path: ConversationNode[] = [];
  let currentParent = 'root';
  
  while (true) {
    const children = childrenByParent.get(currentParent);
    if (!children || children.length === 0) break;
    
    // Find selected child or default to last
    let selected = children[children.length - 1]; // Default to newest
    const selectedId = selectedVersions[currentParent];
    if (selectedId) {
      const found = children.find(c => c.id === selectedId);
      if (found) selected = found;
    }
    
    const siblingIndex = children.indexOf(selected);
    path.push({
      message: selected,
      siblings: children,
      siblingIndex,
    });
    
    currentParent = selected.id;
  }
  
  return path;
}

// Get the last assistant message in the current path (for sending new messages)
function getCurrentLeafAssistant(path: ConversationNode[]): Message | null {
  for (let i = path.length - 1; i >= 0; i--) {
    if (path[i].message.role === 'assistant') {
      return path[i].message;
    }
  }
  return null;
}

// Get the user message that the last assistant is responding to (for retry)
function getLastUserMessage(path: ConversationNode[]): Message | null {
  for (let i = path.length - 1; i >= 0; i--) {
    if (path[i].message.role === 'user') {
      return path[i].message;
    }
  }
  return null;
}

const MessageList = memo(function MessageList({
  messages,
  lastAssistantIndex,
  generatedImages,
  onRetry,
  onEdit,
  onDelete,
  onReadAloud,
  onImageRetry,
  onImageEdit,
  readingMessageId,
  onArtifactClick,
  onBranchChange,
  assistantName,
  chatId,
  initialSelectedVersions,
  onCurrentLeafChange,
}: MessageListProps) {
  // Track selected version for each parent (by parent_id -> selected_child_id)
  const [selectedVersions, setSelectedVersions] = useState<Record<string, string>>(
    initialSelectedVersions || {}
  );
  
  // Track message IDs we knew about at last render to detect truly new messages
  const knownMessageIds = useRef<Set<string>>(new Set());
  const isInitialized = useRef(false);
  
  // Build conversation path
  const path = useMemo(() => buildConversationPath(messages, selectedVersions), [messages, selectedVersions]);
  
  // Get current leaf assistant for new message sending
  const currentLeafAssistant = useMemo(() => getCurrentLeafAssistant(path), [path]);
  
  // Notify parent of current leaf changes
  useEffect(() => {
    if (onCurrentLeafChange) {
      onCurrentLeafChange(currentLeafAssistant?.id || null);
    }
  }, [currentLeafAssistant, onCurrentLeafChange]);
  
  // Re-initialize when chat changes
  useEffect(() => {
    setSelectedVersions(initialSelectedVersions || {});
    knownMessageIds.current = new Set(messages.map(m => m.id));
    isInitialized.current = true;
  }, [chatId]); // Only reset on chat change, not on initialSelectedVersions change
  
  // Initialize known messages on first load
  useEffect(() => {
    if (!isInitialized.current) {
      knownMessageIds.current = new Set(messages.map(m => m.id));
      isInitialized.current = true;
    }
  }, [messages]);
  
  // Auto-select ONLY for messages created during this session (retry/edit)
  useEffect(() => {
    if (!isInitialized.current) return;
    
    // Find messages we haven't seen before
    const newMessages = messages.filter(m => !knownMessageIds.current.has(m.id));
    
    // Update known messages
    for (const msg of messages) {
      knownMessageIds.current.add(msg.id);
    }
    
    // Auto-select new messages that are siblings (from retry/edit)
    for (const msg of newMessages) {
      // Use 'root' for messages without parent_id
      const parentKey = msg.parent_id || 'root';
      const siblings = messages.filter(m => (m.parent_id || 'root') === parentKey);
      
      // Only auto-select if this creates a branch (multiple siblings)
      if (siblings.length > 1) {
        setSelectedVersions(prev => {
          const updated = { ...prev, [parentKey]: msg.id };
          // Persist to backend
          if (chatId) {
            chatApi.updateSelectedVersion(chatId, parentKey, msg.id).catch(err => {
              console.error('Failed to save selected version:', err);
            });
          }
          return updated;
        });
      }
    }
  }, [messages, chatId]);
  
  const handleVersionChange = useCallback((parentId: string, newChildId: string) => {
    setSelectedVersions(prev => {
      // Save to backend
      if (chatId) {
        chatApi.updateSelectedVersion(chatId, parentId, newChildId).catch(err => {
          console.error('Failed to save selected version:', err);
        });
      }
      
      onBranchChange(parentId, newChildId);
      return { ...prev, [parentId]: newChildId };
    });
  }, [chatId, onBranchChange]);
  
  return (
    <>
      {path.map((node, index) => {
        const { message, siblings, siblingIndex } = node;
        const hasSiblings = siblings.length > 1;
        const isLastInPath = index === path.length - 1;
        const isLastAssistant = isLastInPath && message.role === 'assistant';
        
        // For retry, we need the user message ID (parent of assistant)
        const parentUserMessage = message.role === 'assistant' ? 
          path.find(n => n.message.id === message.parent_id)?.message : null;
        
        // Build version info for messages with siblings (both user and assistant)
        // For root messages (no parent_id), use 'root' as the parent key
        const parentKey = message.parent_id || 'root';
        const versionInfo = hasSiblings ? {
          current: siblingIndex + 1,
          total: siblings.length,
          onPrev: () => {
            if (siblingIndex > 0) {
              handleVersionChange(parentKey, siblings[siblingIndex - 1].id);
            }
          },
          onNext: () => {
            if (siblingIndex < siblings.length - 1) {
              handleVersionChange(parentKey, siblings[siblingIndex + 1].id);
            }
          },
        } : undefined;
        
        // Debug: check if this message has a generated image
        return (
          <div key={message.id}>
            <MessageBubble
              message={message}
              isLastAssistant={isLastAssistant}
              generatedImage={generatedImages[message.id]}
              onRetry={isLastAssistant && parentUserMessage ? 
                () => onRetry(parentUserMessage.id, parentUserMessage.content) : undefined}
              onEdit={(messageId, newContent) => onEdit(messageId, newContent)}
              onDelete={(messageId) => onDelete(messageId)}
              onReadAloud={message.role === 'assistant' && onReadAloud ? onReadAloud : undefined}
              onImageRetry={onImageRetry}
              onImageEdit={onImageEdit}
              isReadingAloud={readingMessageId === message.id}
              onArtifactClick={onArtifactClick}
              onBranchChange={() => {}}
              assistantName={assistantName}
              versionInfo={versionInfo}
            />
          </div>
        );
      })}
    </>
  );
});

// Streaming message - renders markdown incrementally for better UX
// Re-renders on: code fence close, or every 10 new lines
interface StreamingMessageProps {
  chatId: string;
  content: string;
  toolCall: { name: string; input: string } | null;
  assistantName?: string;
}

function StreamingMessageComponent({
  chatId,
  content,
  toolCall,
  assistantName,
}: StreamingMessageProps) {
  // Track when to re-render markdown
  const lastRenderRef = useRef({ content: '', lineCount: 0 });
  const [renderedContent, setRenderedContent] = useState('');
  
  useEffect(() => {
    if (!content) {
      setRenderedContent('');
      lastRenderRef.current = { content: '', lineCount: 0 };
      return;
    }
    
    const currentLines = content.split('\n').length;
    const lastLines = lastRenderRef.current.lineCount;
    const lastContent = lastRenderRef.current.content;
    
    // Check if we should re-render markdown
    const newContent = content.slice(lastContent.length);
    const closedCodeFence = newContent.includes('```') && 
      (content.match(/```/g)?.length || 0) % 2 === 0; // Even number = all fences closed
    const addedEnoughLines = currentLines - lastLines >= 10;
    
    if (closedCodeFence || addedEnoughLines || content.length < 100) {
      setRenderedContent(content);
      lastRenderRef.current = { content, lineCount: currentLines };
    }
  }, [content]);
  
  // Use rendered content for markdown, but show full content length
  const displayContent = renderedContent || content;
  const hasUnrenderedContent = content.length > renderedContent.length;
  
  return (
    <div className="py-4">
      <div className="max-w-3xl mx-auto px-3 md:px-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm md:text-xs font-medium uppercase tracking-wide text-[var(--color-secondary)]">
            {assistantName || 'Assistant'}
          </span>
        </div>
        
        {/* Tool call indicator */}
        {toolCall && (
          <div className="mb-3 py-2 px-3 rounded border border-[var(--color-border)] bg-[var(--color-surface)]">
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span>Using {toolCall.name}...</span>
            </div>
          </div>
        )}
        
        {/* Markdown content with incremental rendering */}
        {displayContent ? (
          <div className="prose prose-base md:prose-sm max-w-none prose-neutral dark:prose-invert text-[var(--color-text)]">
            <MessageMarkdown content={displayContent} />
            {/* Show unrendered tail as plain text */}
            {hasUnrenderedContent && (
              <span className="whitespace-pre-wrap">{content.slice(renderedContent.length)}</span>
            )}
            <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse ml-0.5 align-middle" />
          </div>
        ) : (
          <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse" />
        )}
      </div>
    </div>
  );
}

const StreamingMessage = memo(StreamingMessageComponent);

export default function ChatPage() {
  const { chatId } = useParams();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Get app name from branding for page title
  const appName = useBrandingStore((state) => state.config?.app_name || 'Open-NueChat');
  
  const {
    currentChat,
    messages: rawMessages,
    isLoadingMessages,
    isSending,
    streamingContent,
    streamingToolCall,
    artifacts: savedArtifacts,
    streamingArtifacts,
    selectedArtifact,
    showArtifacts,
    codeSummary,
    showSummary,
    generatedImages,
    setShowSummary,
    updateCodeSummary,
    fetchCodeSummary,
    setCurrentChat,
    fetchChats,
    createChat,
    retryMessage,
    switchBranch,
    setSelectedArtifact,
    setShowArtifacts,
    collectAllArtifacts,
    zipUploadResult,
    setZipUploadResult,
  } = useChatStore();
  
  // Combine saved artifacts with streaming artifacts for real-time updates
  const artifacts = useMemo(() => {
    return [...savedArtifacts, ...streamingArtifacts];
  }, [savedArtifacts, streamingArtifacts]);
  
  // Chat menu state
  const [showChatMenu, setShowChatMenu] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareCopied, setShareCopied] = useState(false);
  const chatMenuRef = useRef<HTMLDivElement>(null);
  
  // Model selector state
  const [showModelSelector, setShowModelSelector] = useState(false);
  const modelSelectorRef = useRef<HTMLDivElement>(null);
  const { models, subscribedAssistants, defaultModel, getDisplayName: getModelDisplayName } = useModelsStore();
  
  // State for new chat model selection (before chat is created)
  const [newChatModel, setNewChatModel] = useState<string>(defaultModel);
  
  // Zip upload state
  const [isUploadingZip, setIsUploadingZip] = useState(false);
  
  // Voice mode integration - use ref for send function to avoid dependency issues
  const sendMessageRef = useRef<(content: string) => void>(() => {});
  
  const {
    isReading,
    isListening,
    isProcessing,
    isVoiceMode,
    readingMessageId,
    currentParagraph,
    readAloud,
    stopReading,
    toggleVoiceMode,
    stopVoiceMode,
    handleAssistantResponse,
    handleStreamingText,
    resetStreamingTTS,
    ttsEnabled,
    sttEnabled,
  } = useVoice({
    onTranscript: (transcript) => sendMessageRef.current(transcript),
  });
  
  // Input ref for keyboard shortcuts
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const navigate = useNavigate();
  
  // Keyboard shortcuts
  useChatShortcuts({
    onNewChat: () => {
      navigate('/');
      createChat();
    },
    onFocusInput: () => inputRef.current?.focus(),
    onToggleSidebar: () => {
      // Dispatch custom event for Layout to handle
      window.dispatchEvent(new CustomEvent('toggle-sidebar'));
    },
    onDeleteChat: () => {
      if (currentChat) {
        setShowChatMenu(true); // Open menu to show delete option
      }
    },
    onToggleArtifacts: () => setShowArtifacts(!showArtifacts),
    onClosePanel: () => {
      if (showArtifacts) setShowArtifacts(false);
      else if (showSummary) setShowSummary(false);
      else if (showChatMenu) setShowChatMenu(false);
      else if (showModelSelector) setShowModelSelector(false);
      else if (showShareDialog) setShowShareDialog(false);
    },
  });
  
  // Check voice services on mount
  useEffect(() => {
    useVoiceStore.getState().checkServiceStatus();
  }, []);
  
  // Current leaf assistant ID for branching (tracks which assistant message we're "at")
  const [currentLeafAssistantId, setCurrentLeafAssistantId] = useState<string | null>(null);
  
  // Ref to always have the latest leaf ID (avoids stale closure issues)
  const currentLeafAssistantIdRef = useRef<string | null>(null);
  useEffect(() => {
    currentLeafAssistantIdRef.current = currentLeafAssistantId;
  }, [currentLeafAssistantId]);
  
  // Reset state when chat changes (CRITICAL: prevents cross-chat parent_id and stale zip results)
  useEffect(() => {
    setCurrentLeafAssistantId(null);
    setZipUploadResult(null);
    // Clear zip context to prevent sending old manifest to new chat
    useChatStore.getState().setZipContext(null);
  }, [chatId, setZipUploadResult]);
  
  // Open artifacts panel when uploaded files are loaded (from fetchMessages -> fetchUploadedData)
  useEffect(() => {
    if (zipUploadResult?.artifacts && zipUploadResult.artifacts.length > 0) {
      setShowArtifacts(true);
    }
  }, [zipUploadResult, setShowArtifacts]);
  
  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (chatMenuRef.current && !chatMenuRef.current.contains(e.target as Node)) {
        setShowChatMenu(false);
      }
    };
    if (showChatMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showChatMenu]);
  
  // Close model selector when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modelSelectorRef.current && !modelSelectorRef.current.contains(e.target as Node)) {
        setShowModelSelector(false);
      }
    };
    if (showModelSelector) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showModelSelector]);
  
  // Update chat model
  const handleModelChange = async (modelId: string) => {
    if (!currentChat || currentChat.model === modelId) {
      setShowModelSelector(false);
      return;
    }
    
    try {
      await chatApi.update(currentChat.id, { model: modelId });
      // Update local state
      setCurrentChat({ ...currentChat, model: modelId });
      // Refetch chats to update sidebar
      fetchChats();
    } catch (err) {
      console.error('Failed to update chat model:', err);
    }
    setShowModelSelector(false);
  };
  
  // Get model display name for assistant label
  const { getDisplayName } = useModelsStore();
  const assistantName = currentChat?.model ? getDisplayName(currentChat.model) : 'Assistant';
  
  // Defensive: ensure messages is always an array
  const messages = Array.isArray(rawMessages) ? rawMessages : [];
  
  const { sendChatMessage, regenerateMessage, isConnected, stopGeneration } = useWebSocket();
  
  // Load chat when URL changes
  useEffect(() => {
    if (chatId) {
      const { chats } = useChatStore.getState();
      const safeChats = Array.isArray(chats) ? chats : [];
      const chat = safeChats.find((c) => c.id === chatId);
      if (chat) {
        setCurrentChat(chat);
      } else {
        // Fetch chats if not loaded
        fetchChats().then(() => {
          const refreshedChats = useChatStore.getState().chats;
          const safeRefreshed = Array.isArray(refreshedChats) ? refreshedChats : [];
          const foundChat = safeRefreshed.find((c) => c.id === chatId);
          if (foundChat) {
            setCurrentChat(foundChat);
          }
        });
      }
    }
  }, [chatId, setCurrentChat, fetchChats]);
  
  // Fetch code summary when chat changes
  useEffect(() => {
    if (chatId) {
      fetchCodeSummary(chatId);
    }
  }, [chatId, fetchCodeSummary]);
  
  // Scroll to bottom - throttled during streaming
  const lastScrollRef = useRef<number>(0);
  const scrollToBottom = useCallback((force = false) => {
    const now = Date.now();
    // During streaming, only scroll every 300ms unless forced
    if (!force && isSending && now - lastScrollRef.current < 300) {
      return;
    }
    lastScrollRef.current = now;
    messagesEndRef.current?.scrollIntoView({ behavior: isSending ? 'auto' : 'smooth' });
  }, [isSending]);
  
  // Scroll on messages change
  useEffect(() => {
    scrollToBottom(true);
  }, [messages]);
  
  // Scroll on streaming - heavily throttled (handled by scrollToBottom)
  useEffect(() => {
    if (streamingContent) {
      scrollToBottom(false);
    }
  }, [streamingContent, scrollToBottom]);
  
  // Voice mode: Stream TTS as response comes in (paragraph by paragraph)
  const prevIsSendingRef = useRef(isSending);
  const prevStreamingContentRef = useRef<string>('');
  const streamingCompleteHandledRef = useRef(false);
  const lastReadMessageIdRef = useRef<string | null>(null);
  
  useEffect(() => {
    if (!isVoiceMode || !ttsEnabled) {
      prevStreamingContentRef.current = '';
      streamingCompleteHandledRef.current = false;
      return;
    }
    
    // During streaming, send content updates for paragraph detection
    if (isSending && streamingContent) {
      handleStreamingText(streamingContent, false);
      prevStreamingContentRef.current = streamingContent;
      streamingCompleteHandledRef.current = false;
    }
    
    // Detect transition from sending to not sending (response complete)
    if (prevIsSendingRef.current && !isSending && !streamingCompleteHandledRef.current) {
      streamingCompleteHandledRef.current = true;
      
      // Use a longer delay to ensure message state is fully updated
      // The backend sometimes commits the message right before stream_end
      setTimeout(() => {
        // Get the latest message content from the messages array
        const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant');
        const finalContent = lastAssistantMsg?.content || prevStreamingContentRef.current;
        
        if (finalContent && lastAssistantMsg) {
          lastReadMessageIdRef.current = lastAssistantMsg.id;
          handleStreamingText(finalContent, true);
        }
        prevStreamingContentRef.current = '';
      }, 150); // Increased delay to catch late message updates
    }
    
    prevIsSendingRef.current = isSending;
  }, [isSending, isVoiceMode, ttsEnabled, streamingContent, messages, handleStreamingText]);
  
  // Watch for message updates after streaming ends (catches late content updates)
  useEffect(() => {
    if (!isVoiceMode || !ttsEnabled || isSending) return;
    
    const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant');
    if (!lastAssistantMsg) return;
    
    // If we've already read this message but its content changed, read the new content
    if (lastReadMessageIdRef.current === lastAssistantMsg.id) {
      // Content might have been updated - check if there's unread content
      // This is handled by the streaming TTS which tracks lastReadIndex
    }
  }, [messages, isVoiceMode, ttsEnabled, isSending]);
  
  // Reset streaming TTS when chat changes
  useEffect(() => {
    resetStreamingTTS();
    streamingCompleteHandledRef.current = false;
    lastReadMessageIdRef.current = null;
  }, [chatId, resetStreamingTTS]);
  
  // Track scroll position for scroll-to-bottom button
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom);
    };
    
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Update page title when chat title changes
  useEffect(() => {
    const chatTitle = currentChat?.title;
    if (chatTitle && chatTitle !== 'New Chat') {
      document.title = `${appName} - ${chatTitle}`;
    } else if (chatId) {
      // Chat exists but title not loaded yet - wait for it
      document.title = appName;
    } else {
      document.title = appName;
    }
    
    // Reset to app name when unmounting
    return () => {
      document.title = appName;
    };
  }, [currentChat?.title, appName, chatId]);
  
  const handleSendMessage = async (content: string, _attachments?: File[]) => {
    let targetChatId = currentChat?.id;
    
    // Calculate parentId directly from store to avoid stale closure issues
    // This is critical for voice mode which calls via ref
    let parentId: string | null = null;
    if (targetChatId) {
      const { messages: storeMessages } = useChatStore.getState();
      // Filter to current chat and sort by created_at
      const chatMessages = storeMessages
        .filter(m => m.chat_id === targetChatId)
        .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
      // Find last assistant message - that's the leaf we continue from
      const lastAssistant = [...chatMessages].reverse().find(m => m.role === 'assistant');
      parentId = lastAssistant?.id || null;
    }
    
    // Create new chat if none exists
    if (!targetChatId) {
      // Use selected model for new chat (newChatModel or default)
      const newChat = await createChat(newChatModel || defaultModel);
      targetChatId = newChat.id;
      // Update URL without navigation
      window.history.pushState({}, '', `/chat/${newChat.id}`);
      // New chat has no parent - first message is root
      parentId = null;
    }
    
    // Pass the current leaf assistant ID as parent_id for linear conversation
    // This tells the backend which conversation path we're continuing from
    sendChatMessage(targetChatId, content, undefined, parentId);
  };
  
  // Update voice ref with send function
  // Note: handleSendMessage calculates parentId from store directly, so no stale closure issues
  useEffect(() => {
    sendMessageRef.current = handleSendMessage;
  }, [currentChat, newChatModel, defaultModel]);
  
  // Handle retry - receives user message ID and content from MessageList
  const handleRetry = useCallback((userMessageId: string, userContent: string) => {
    if (!currentChat) return;
    
    // Regenerate with the user message ID as parent
    // This creates a sibling assistant response
    regenerateMessage(currentChat.id, userContent, userMessageId);
  }, [currentChat, regenerateMessage]);
  
  // Handle retry from voice mode - finds the last user message and retries
  const handleVoiceModeRetry = useCallback(() => {
    if (!currentChat || !messages.length) return;
    
    // Find the last user message
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUserMessage) {
      console.log('No user message found to retry');
      return;
    }
    
    console.log('Voice mode retry - regenerating response to:', lastUserMessage.content.substring(0, 50));
    
    // Stop any current reading
    stopReading();
    
    // Regenerate the response
    regenerateMessage(currentChat.id, lastUserMessage.content, lastUserMessage.id);
  }, [currentChat, messages, regenerateMessage, stopReading]);
  
  const handleBranchChange = useCallback((parentId: string, childId: string) => {
    // Branch change is handled by MessageList state
    // Could persist to server here if needed
    console.log('Branch changed:', parentId, '->', childId);
  }, []);
  
  // Handle message edit - creates a new branch with edited content
  // For user messages, also regenerates the AI response
  const handleEditMessage = useCallback(async (messageId: string, newContent: string) => {
    if (!currentChat) return;
    
    // Find the message being edited to check its role
    const messageToEdit = messages.find(m => m.id === messageId);
    if (!messageToEdit) {
      console.error('Message not found:', messageId);
      return;
    }
    
    const isUserMessage = messageToEdit.role === 'user';
    
    try {
      // Call API to create edited branch (new sibling message)
      const response = await chatApi.editMessage(currentChat.id, messageId, newContent, isUserMessage);
      const newMessage = response.data;
      
      console.log('Message edited, new branch created:', newMessage.id, 'isUser:', isUserMessage);
      
      // For user messages, trigger WebSocket regeneration to generate AI response
      if (isUserMessage) {
        // The new user message needs an AI response
        // regenerateMessage will create the assistant message with parent_id = newMessage.id
        regenerateMessage(currentChat.id, newContent, newMessage.id);
      }
      
      // Refresh messages to get updated tree
      await useChatStore.getState().fetchMessages(currentChat.id);
    } catch (error) {
      console.error('Failed to edit message:', error);
      alert('Failed to edit message. Please try again.');
    }
  }, [currentChat, messages, regenerateMessage]);
  
  // Handle message delete - removes message and all descendants
  const handleDeleteMessage = useCallback(async (messageId: string) => {
    if (!currentChat) return;
    
    try {
      // Call API to delete message branch
      const response = await chatApi.deleteMessage(currentChat.id, messageId);
      const result = response.data;
      
      // Refresh messages to get updated tree
      useChatStore.getState().fetchMessages(currentChat.id);
      
      console.log(`Message deleted: ${result.total_deleted} messages removed`);
    } catch (error) {
      console.error('Failed to delete message:', error);
      alert('Failed to delete message. Please try again.');
    }
  }, [currentChat]);
  
  // Handle image retry - regenerate with optional size and seed
  const handleImageRetry = useCallback((prompt: string, width?: number, height?: number, seed?: number) => {
    if (!currentChat) return;
    
    // Build the message content with size if specified
    let messageContent = `Create an image: ${prompt}`;
    if (width && height) {
      messageContent = `Create a ${width}x${height} image: ${prompt}`;
    }
    
    sendChatMessage(currentChat.id, messageContent);
  }, [currentChat, sendChatMessage]);
  
  // Handle image edit - open input with prompt for editing
  const handleImageEdit = useCallback((prompt: string) => {
    // Focus input and set the prompt for editing
    // The user can modify and submit
    const input = document.querySelector('textarea') as HTMLTextAreaElement;
    if (input) {
      input.value = `Create an image: ${prompt}`;
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    }
  }, []);
  
  // Handle zip file upload - extracts contents and creates artifacts
  const handleZipUpload = useCallback(async (file: File) => {
    let targetChatId = currentChat?.id;
    
    // Create chat if needed with selected model
    if (!targetChatId) {
      const modelToUse = newChatModel || defaultModel;
      const newChat = await createChat(modelToUse);
      targetChatId = newChat.id;
      window.history.pushState({}, '', `/chat/${newChat.id}`);
    }
    
    setIsUploadingZip(true);
    try {
      const response = await chatApi.uploadZip(targetChatId, file);
      const result = response.data as import('../types').ZipUploadResult;
      
      // Store result in chat store (includes artifacts, manifest, etc.)
      setZipUploadResult(result);
      
      // Store the LLM manifest for sending with messages
      if (result.llm_manifest) {
        useChatStore.getState().setZipContext(result.llm_manifest);
        console.log('Stored LLM manifest in chat store', result.llm_manifest.length, 'chars');
      }
      
      // Add artifacts to the main artifacts store
      const now = new Date().toISOString();
      const uploadedArtifacts = result.artifacts.map(artifact => ({
        ...artifact,
        created_at: artifact.created_at || now,
        source: 'upload' as const,
      }));
      
      useChatStore.setState(state => ({
        artifacts: [...state.artifacts.filter(a => a.source !== 'upload'), ...uploadedArtifacts],
        uploadedArtifacts: uploadedArtifacts,
      }));
      
      // Open artifacts panel if we have files
      if (result.artifacts.length > 0) {
        setShowArtifacts(true);
      }
      
      console.log(`Zip uploaded: ${result.total_files} files, ${result.artifacts.length} artifacts`);
    } catch (error) {
      console.error('Failed to upload zip:', error);
      alert('Failed to process zip file. Please try again.');
    } finally {
      setIsUploadingZip(false);
    }
  }, [currentChat, createChat, setShowArtifacts, setZipUploadResult, newChatModel, defaultModel]);
  
  const handleArtifactClick = useCallback((artifact: Artifact) => {
    setSelectedArtifact(artifact);
  }, [setSelectedArtifact]);
  
  // Handle clicking a file in the zip upload card - find and select the artifact
  const handleZipFileClick = useCallback((filepath: string) => {
    const artifact = artifacts.find(a => a.filename === filepath);
    if (artifact) {
      setSelectedArtifact(artifact);
      setShowArtifacts(true);
    }
  }, [artifacts, setSelectedArtifact, setShowArtifacts]);
  
  const handleStop = useCallback(() => {
    if (currentChat) {
      stopGeneration(currentChat.id);
    }
  }, [currentChat, stopGeneration]);
  
  const handleDownloadAll = async () => {
    // Get only the latest version of each unique file
    const latestArtifacts = getLatestArtifacts(artifacts);
    if (latestArtifacts.length === 0) {
      alert('No files to download');
      return;
    }
    
    // Dynamic import of JSZip
    const JSZip = (await import('jszip')).default;
    const zip = new JSZip();
    
    for (const artifact of latestArtifacts) {
      const filename = artifact.filename || `${artifact.title.replace(/\s+/g, '_')}.txt`;
      zip.file(filename, artifact.content);
    }
    
    const blob = await zip.generateAsync({ 
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: { level: 9 }
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentChat?.title || 'chat'}_files.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const handleShareChat = async () => {
    if (!currentChat) return;
    setShowChatMenu(false);
    
    try {
      const response = await chatApi.share(currentChat.id);
      const shareId = response.data.share_id;
      const url = `${window.location.origin}/shared/${shareId}`;
      setShareUrl(url);
      setShowShareDialog(true);
    } catch (err) {
      console.error('Failed to share chat:', err);
      alert('Failed to share chat');
    }
  };
  
  const handleCopyShareUrl = async () => {
    if (!shareUrl) return;
    await navigator.clipboard.writeText(shareUrl);
    setShareCopied(true);
    setTimeout(() => setShareCopied(false), 2000);
  };
  
  const handleExportChat = async () => {
    if (!currentChat) return;
    setShowChatMenu(false);
    
    // Build export object with all versions
    const exportData = {
      id: currentChat.id,
      title: currentChat.title,
      model: currentChat.model,
      created_at: currentChat.created_at,
      updated_at: currentChat.updated_at,
      messages: messages.map(m => ({
        id: m.id,
        role: m.role,
        content: m.content,
        parent_id: m.parent_id,
        created_at: m.created_at,
        input_tokens: m.input_tokens,
        output_tokens: m.output_tokens,
      })),
      artifacts: artifacts.map(a => ({
        id: a.id,
        title: a.title,
        filename: a.filename,
        type: a.type,
        language: a.language,
        content: a.content,
        created_at: a.created_at,
      })),
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentChat.title || 'chat'}_export.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const handleDeleteChat = async () => {
    if (!currentChat) return;
    
    if (!confirm('Are you sure you want to delete this chat?')) {
      return;
    }
    
    setShowChatMenu(false);
    
    try {
      await chatApi.delete(currentChat.id);
      // Refresh chat list and navigate home
      await fetchChats();
      window.location.href = '/';
    } catch (err) {
      console.error('Failed to delete chat:', err);
      alert('Failed to delete chat');
    }
  };
  
  // Get unique file count for display
  const uniqueFileCount = useMemo(() => 
    groupArtifactsByFilename(artifacts).length,
    [artifacts]
  );

  // Find the last assistant message for retry button
  const lastAssistantIndex = messages.map(m => m.role).lastIndexOf('assistant');
  
  // Check if debug voice mode is enabled (set in Admin > Site Dev)
  const debugVoiceModeEnabled = useMemo(() => {
    return localStorage.getItem('nexus-debug-voice-mode') === 'true';
  }, []);
  
  // Build debug info for voice mode overlay (must be before conditional returns)
  const voiceModeDebugInfo = useMemo(() => {
    if (!debugVoiceModeEnabled) return undefined;
    
    const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant');
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
    
    // Calculate what parentId would be at send time (same logic as handleSendMessage)
    const chatMessages = messages
      .filter(m => m.chat_id === currentChat?.id)
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
    const calculatedLeaf = [...chatMessages].reverse().find(m => m.role === 'assistant');
    
    return [
      `Messages: ${messages.length}`,
      `CalcLeaf: ${calculatedLeaf?.id?.substring(0, 8) || 'null'}`,
      `LeafState: ${currentLeafAssistantId?.substring(0, 8) || 'null'}`,
      `LastAsst: ${lastAssistantMsg?.id?.substring(0, 8) || 'none'}`,
      `LastUser: ${lastUserMsg?.id?.substring(0, 8) || 'none'}`,
      `LastUserParent: ${lastUserMsg?.parent_id?.substring(0, 8) || 'null'}`,
      `Chat: ${currentChat?.id?.substring(0, 8) || 'none'}`,
    ].join('\n');
  }, [debugVoiceModeEnabled, messages, currentLeafAssistantId, currentChat?.id]);
  
  // Show empty state if no chat selected
  if (!currentChat && !chatId) {
    return (
      <div className="flex flex-col h-full">
        <EmptyState 
          onSendMessage={handleSendMessage} 
          selectedModel={newChatModel}
          onModelChange={setNewChatModel}
        />
        <div className="border-t border-[var(--color-border)] p-4">
          <ChatInput
            onSend={handleSendMessage}
            onZipUpload={handleZipUpload}
            onVoiceModeToggle={sttEnabled ? toggleVoiceMode : undefined}
            disabled={!isConnected}
            isUploadingZip={isUploadingZip}
            isVoiceMode={isVoiceMode}
            isListening={isListening}
            inputRef={inputRef}
            placeholder={isUploadingZip ? 'Processing zip file...' : 'Start a new conversation...'}
          />
        </div>
      </div>
    );
  }
  
  return (
    <>
      {/* Voice Mode Overlay - full screen when active */}
      <VoiceModeOverlay
        isActive={isVoiceMode}
        isListening={isListening}
        isReading={isReading}
        isProcessing={isProcessing}
        currentText={currentParagraph}
        onInterrupt={stopReading}
        onExit={stopVoiceMode}
        onRetry={handleVoiceModeRetry}
        debugInfo={voiceModeDebugInfo}
      />
      
      <div className="flex h-full">
      {/* Main chat area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header with actions */}
        <div className="flex items-center justify-between px-3 md:px-4 py-2 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-3 min-w-0">
            <h2 className="text-base md:text-sm font-medium text-[var(--color-text)] truncate">
              {currentChat?.title || 'New Chat'}
            </h2>
            
            {/* Model/GPT Selector */}
            <div className="relative" ref={modelSelectorRef}>
              <button
                onClick={() => setShowModelSelector(!showModelSelector)}
                className="flex items-center gap-1.5 px-2 py-1 text-xs rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-background)] transition-colors"
                title="Change model"
              >
                <span className="truncate max-w-[120px]">
                  {currentChat?.model ? getModelDisplayName(currentChat.model) : 'Select Model'}
                </span>
                <svg className={`w-3 h-3 transition-transform ${showModelSelector ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {showModelSelector && (
                <div className="absolute left-0 top-full mt-1 w-64 max-h-80 overflow-y-auto py-1 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] shadow-xl z-50">
                  {/* LLM Models */}
                  {models.length > 0 && (
                    <>
                      <div className="px-3 py-1.5 text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">
                        Models
                      </div>
                      {models.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => handleModelChange(model.id)}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-[var(--color-background)] flex items-center justify-between ${
                            currentChat?.model === model.id ? 'text-[var(--color-primary)]' : 'text-[var(--color-text)]'
                          }`}
                        >
                          <span className="truncate">{getModelDisplayName(model.id)}</span>
                          {currentChat?.model === model.id && (
                            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                          )}
                        </button>
                      ))}
                    </>
                  )}
                  
                  {/* Subscribed Custom GPTs */}
                  {subscribedAssistants.length > 0 && (
                    <>
                      <div className="border-t border-[var(--color-border)] my-1" />
                      <div className="px-3 py-1.5 text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">
                        Custom GPTs
                      </div>
                      {subscribedAssistants.map((assistant) => (
                        <button
                          key={assistant.id}
                          onClick={() => handleModelChange(assistant.id)}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-[var(--color-background)] flex items-center justify-between ${
                            currentChat?.model === assistant.id ? 'text-[var(--color-primary)]' : 'text-[var(--color-text)]'
                          }`}
                        >
                          <span className="flex items-center gap-2 truncate">
                            <span>{assistant.icon || 'AI'}</span>
                            <span className="truncate">{assistant.name}</span>
                          </span>
                          {currentChat?.model === assistant.id && (
                            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                          )}
                        </button>
                      ))}
                    </>
                  )}
                  
                  {models.length === 0 && subscribedAssistants.length === 0 && (
                    <div className="px-3 py-2 text-sm text-[var(--color-text-secondary)]">
                      No models available
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-1 md:gap-2">
            {uniqueFileCount > 0 && (
              <>
                <button
                  onClick={handleDownloadAll}
                  className="hidden md:flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)] transition-colors"
                  title="Download all files as ZIP"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download All
                </button>
                <button
                  onClick={() => setShowArtifacts(!showArtifacts)}
                  className={`flex items-center gap-1 md:gap-1.5 px-2 md:px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    showArtifacts
                      ? 'bg-[var(--color-button)] border-[var(--color-button)] text-[var(--color-button-text)]'
                      : 'bg-[var(--color-surface)] border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]'
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                  </svg>
                  <span className="hidden md:inline">Artifacts</span> ({uniqueFileCount})
                </button>
                <button
                  onClick={() => setShowSummary(!showSummary)}
                  className={`flex items-center gap-1 md:gap-1.5 px-2 md:px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    showSummary
                      ? 'bg-[var(--color-button)] border-[var(--color-button)] text-[var(--color-button-text)]'
                      : 'bg-[var(--color-surface)] border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]'
                  }`}
                  title="Project Summary - Uploads, file changes, and warnings"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                  </svg>
                  <span className="hidden md:inline">Summary</span>
                  {zipUploadResult && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-blue-500/20 text-blue-400">
                      {zipUploadResult.total_files || zipUploadResult.artifacts?.length || 0}
                    </span>
                  )}
                  {codeSummary?.warnings && codeSummary.warnings.length > 0 && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-red-500/20 text-red-400">
                      {codeSummary.warnings.length}
                    </span>
                  )}
                </button>
              </>
            )}
            
            {/* 3-dot menu */}
            <div className="relative" ref={chatMenuRef}>
              <button
                onClick={() => setShowChatMenu(!showChatMenu)}
                className="p-2 rounded-lg text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] transition-colors"
                title="Chat options"
              >
                <span className="text-lg tracking-wider">•••</span>
              </button>
              
              {showChatMenu && (
                <div className="absolute right-0 top-full mt-1 w-44 py-1 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] shadow-xl z-50">
                  <button
                    onClick={handleShareChat}
                    className="w-full px-4 py-2 text-left text-sm text-[var(--color-text)] hover:bg-[var(--color-background)] flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                    </svg>
                    Share
                  </button>
                  <button
                    onClick={handleExportChat}
                    className="w-full px-4 py-2 text-left text-sm text-[var(--color-text)] hover:bg-[var(--color-background)] flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Export Chat
                  </button>
                  <div className="border-t border-[var(--color-border)] my-1" />
                  <button
                    onClick={handleDeleteChat}
                    className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Delete
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Messages container */}
        <div
          ref={containerRef}
          className="flex-1 overflow-y-auto"
        >
          {isLoadingMessages ? (
            <div className="flex items-center justify-center h-32">
              <div className="flex items-center gap-2 text-[var(--color-text-secondary)]">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Loading...
              </div>
            </div>
          ) : messages.length === 0 && !zipUploadResult ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-8">
              <p className="text-[var(--color-text-secondary)]">
                Send a message to start the conversation.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-[var(--color-border)]/50">
              {/* Memoized message list - only re-renders when messages array changes */}
              {messages.length > 0 && (
                <MessageList
                  messages={messages}
                  lastAssistantIndex={lastAssistantIndex}
                  generatedImages={generatedImages}
                  onRetry={handleRetry}
                  onEdit={handleEditMessage}
                  onDelete={handleDeleteMessage}
                  onReadAloud={ttsEnabled ? readAloud : undefined}
                  onImageRetry={handleImageRetry}
                  onImageEdit={handleImageEdit}
                  readingMessageId={readingMessageId}
                  onArtifactClick={handleArtifactClick}
                  onBranchChange={handleBranchChange}
                  assistantName={assistantName}
                  chatId={currentChat?.id}
                  initialSelectedVersions={currentChat?.selected_versions}
                  onCurrentLeafChange={setCurrentLeafAssistantId}
                />
              )}
              
              {/* Streaming message - separate component to avoid re-rendering list */}
              {(streamingContent || streamingToolCall) && (
                <StreamingMessage
                  chatId={currentChat?.id || ''}
                  content={streamingContent}
                  toolCall={streamingToolCall}
                  assistantName={assistantName}
                />
              )}
              
              {/* Sending indicator */}
              {isSending && !streamingContent && !streamingToolCall && (
                <div className="py-4 px-4">
                  <div className="max-w-3xl mx-auto flex items-center gap-2 text-[var(--color-text-secondary)]">
                    <div className="flex gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Scroll to bottom button */}
        {showScrollButton && (
          <button
            onClick={() => scrollToBottom(true)}
            className="absolute bottom-24 right-4 md:right-8 p-2 rounded-full bg-[var(--color-surface)] border border-[var(--color-border)] shadow-lg hover:bg-zinc-700/30 transition-all z-10"
          >
            <svg className="w-5 h-5 text-[var(--color-text)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </button>
        )}
        
        {/* Input area */}
        <div className="border-t border-[var(--color-border)] p-3 md:p-4">
          {/* Voice mode indicator */}
          {isVoiceMode && (
            <div className="mb-3 flex items-center justify-between px-3 py-2 rounded-lg bg-[var(--color-primary)]/10 border border-[var(--color-primary)]/30">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${isListening ? 'bg-red-500 animate-pulse' : isReading ? 'bg-green-500 animate-pulse' : 'bg-[var(--color-primary)]'}`}></div>
                <span className="text-sm font-medium text-[var(--color-text)]">
                  {isListening ? 'Listening...' : isReading ? 'Speaking...' : isProcessing ? 'Processing...' : 'Voice Mode Active'}
                </span>
                <span className="text-xs text-[var(--color-text-secondary)]">
                  Say "STOP" to exit
                </span>
              </div>
              <button
                onClick={toggleVoiceMode}
                className="px-3 py-1 text-sm rounded-lg bg-[var(--color-primary)] text-white hover:opacity-90 transition-opacity"
              >
                Stop
              </button>
            </div>
          )}
          <ChatInput
            onSend={handleSendMessage}
            onStop={handleStop}
            onZipUpload={handleZipUpload}
            onVoiceModeToggle={sttEnabled ? toggleVoiceMode : undefined}
            disabled={!isConnected}
            isStreaming={isSending}
            isUploadingZip={isUploadingZip}
            isVoiceMode={isVoiceMode}
            isListening={isListening}
            inputRef={inputRef}
            placeholder={
              isVoiceMode
                ? isListening
                  ? 'Listening... (say "STOP" to exit)'
                  : isReading
                  ? 'Speaking...'
                  : 'Voice mode active'
                : isUploadingZip
                ? 'Processing zip file...'
                : !isConnected
                ? 'Connecting...'
                : isSending
                ? 'AI is responding...'
                : 'Type your message...'
            }
          />
        </div>
      </div>
      
      {/* Share dialog */}
      {showShareDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] p-6 w-full max-w-md mx-4 shadow-2xl">
            <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">Share Chat</h3>
            <p className="text-sm text-[var(--color-text-secondary)] mb-4">
              Anyone with this link can view this conversation (read-only).
            </p>
            
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={shareUrl || ''}
                readOnly
                className="flex-1 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm"
              />
              <button
                onClick={handleCopyShareUrl}
                className="px-4 py-2 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90 transition-opacity text-sm font-medium"
              >
                {shareCopied ? 'Copied!' : 'Copy URL'}
              </button>
            </div>
            
            <div className="flex justify-end">
              <button
                onClick={() => {
                  setShowShareDialog(false);
                  setShareUrl(null);
                }}
                className="px-4 py-2 rounded-lg text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-background)] transition-colors text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Artifacts panel */}
      {showArtifacts && (
        <ArtifactsPanel
          artifacts={artifacts}
          selectedArtifact={selectedArtifact}
          onSelect={setSelectedArtifact}
          onClose={() => setShowArtifacts(false)}
        />
      )}
      
      {/* Summary panel */}
      {showSummary && (
        <SummaryPanel
          summary={codeSummary}
          zipUploadResult={zipUploadResult}
          onClose={() => setShowSummary(false)}
          onClearWarnings={() => updateCodeSummary(codeSummary?.files || [], [])}
          onFileClick={handleZipFileClick}
        />
      )}
    </div>
    </>
  );
}
