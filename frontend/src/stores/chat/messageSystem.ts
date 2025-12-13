/**
 * Message System v2 - Clean Chat Hierarchy
 * 
 * Core Principles:
 * 1. IDs are assigned BEFORE any action and NEVER change
 * 2. parent_id is immutable once set
 * 3. Messages go through defined states: pending → sending → streaming → committed
 * 4. Commit (database save) only happens after stream_end
 * 5. Frontend is optimistic but respects server as source of truth
 */

import type { Artifact, ToolCall, Attachment, GeneratedImage } from '../../types';

// ============================================
// MESSAGE STATES
// ============================================

/**
 * Message lifecycle states
 */
export type MessageState = 
  | 'pending'    // Created locally, not sent
  | 'sending'    // Sent to server, awaiting confirmation
  | 'streaming'  // Assistant response is being streamed
  | 'committed'  // Saved to database, immutable
  | 'error';     // Failed to send/save

/**
 * Enhanced message type with state tracking
 */
export interface ChatMessage {
  // Immutable identifiers (set once, never change)
  readonly id: string;
  readonly chat_id: string;
  readonly parent_id: string | null;
  readonly role: 'user' | 'assistant' | 'system' | 'tool';
  readonly created_at: string;
  
  // Mutable content (only during streaming)
  content: string;
  
  // State tracking
  state: MessageState;
  
  // Optional fields
  attachments?: Attachment[];
  tool_calls?: ToolCall[];
  artifacts?: Artifact[];
  input_tokens?: number;
  output_tokens?: number;
  
  // Metadata
  metadata?: {
    type?: 'file_content' | 'context';
    path?: string;
    source?: string;
    generated_image?: GeneratedImage;
    image_generation?: {
      status?: 'pending' | 'processing' | 'completed' | 'failed';
      prompt?: string;
      job_id?: string;
    };
  };
  
  // Error info (when state === 'error')
  error?: string;
}

// ============================================
// REQUEST TRACKING
// ============================================

/**
 * Tracks a pending request (user message + expected assistant response)
 */
export interface PendingRequest {
  // Correlation ID for this request
  request_id: string;
  
  // Pre-assigned IDs (generated before sending)
  user_message_id: string;
  assistant_message_id?: string;  // Assigned by server in stream_start
  
  // Request details
  chat_id: string;
  parent_id: string | null;
  content: string;
  attachments?: unknown[];
  
  // Timestamps
  created_at: string;
  sent_at?: string;
  
  // State
  state: 'pending' | 'sent' | 'streaming' | 'complete' | 'error';
  error?: string;
}

// ============================================
// STATE MACHINE TRANSITIONS
// ============================================

/**
 * Valid state transitions for messages
 */
export const MESSAGE_TRANSITIONS: Record<MessageState, MessageState[]> = {
  pending: ['sending', 'error'],
  sending: ['committed', 'error'],  // User messages go directly to committed
  streaming: ['committed', 'error'], // Assistant messages stream then commit
  committed: [],  // Terminal state - no transitions allowed
  error: ['pending'],  // Can retry from error
};

/**
 * Check if a state transition is valid
 */
export function canTransition(from: MessageState, to: MessageState): boolean {
  return MESSAGE_TRANSITIONS[from]?.includes(to) ?? false;
}

/**
 * Assert a transition is valid, throw if not
 */
export function assertTransition(from: MessageState, to: MessageState, context?: string): void {
  if (!canTransition(from, to)) {
    const ctx = context ? ` (${context})` : '';
    throw new Error(`Invalid message state transition: ${from} → ${to}${ctx}`);
  }
}

// ============================================
// ID GENERATION
// ============================================

/**
 * Generate a new message ID
 * Uses crypto.randomUUID() for globally unique IDs
 */
export function generateMessageId(): string {
  return crypto.randomUUID();
}

/**
 * Generate a request correlation ID
 */
export function generateRequestId(): string {
  return `req_${crypto.randomUUID()}`;
}

// ============================================
// MESSAGE FACTORY
// ============================================

/**
 * Create a new user message (pending state)
 */
export function createUserMessage(params: {
  id: string;
  chat_id: string;
  parent_id: string | null;
  content: string;
  attachments?: Attachment[];
}): ChatMessage {
  return {
    id: params.id,
    chat_id: params.chat_id,
    parent_id: params.parent_id,
    role: 'user',
    content: params.content,
    attachments: params.attachments,
    created_at: new Date().toISOString(),
    state: 'pending',
  };
}

/**
 * Create a new assistant message placeholder (streaming state)
 */
export function createAssistantPlaceholder(params: {
  id: string;
  chat_id: string;
  parent_id: string;  // Always the user message ID
}): ChatMessage {
  return {
    id: params.id,
    chat_id: params.chat_id,
    parent_id: params.parent_id,
    role: 'assistant',
    content: '',
    created_at: new Date().toISOString(),
    state: 'streaming',
  };
}

/**
 * Convert server message to ChatMessage (committed state)
 */
export function fromServerMessage(serverMsg: {
  id: string;
  chat_id: string;
  parent_id?: string | null;
  role: string;
  content: string;
  created_at: string;
  artifacts?: Artifact[];
  tool_calls?: ToolCall[];
  input_tokens?: number;
  output_tokens?: number;
  metadata?: Record<string, unknown>;
}): ChatMessage {
  return {
    id: serverMsg.id,
    chat_id: serverMsg.chat_id,
    parent_id: serverMsg.parent_id ?? null,
    role: serverMsg.role as 'user' | 'assistant' | 'system' | 'tool',
    content: serverMsg.content,
    created_at: serverMsg.created_at,
    state: 'committed',
    artifacts: serverMsg.artifacts,
    tool_calls: serverMsg.tool_calls,
    input_tokens: serverMsg.input_tokens,
    output_tokens: serverMsg.output_tokens,
    metadata: serverMsg.metadata as ChatMessage['metadata'],
  };
}

// ============================================
// STREAMING CONTENT ACCUMULATOR
// ============================================

/**
 * Accumulates streaming content with buffering
 */
export class StreamAccumulator {
  private content: string = '';
  private buffer: string = '';
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private lastFlush: number = 0;
  
  private readonly flushInterval = 100; // ms
  private readonly maxBuffer = 500; // characters
  
  constructor(private onUpdate: (content: string) => void) {}
  
  /**
   * Append a chunk to the buffer
   */
  append(chunk: string): void {
    this.buffer += chunk;
    
    // Flush if buffer is large
    if (this.buffer.length >= this.maxBuffer) {
      this.flush();
      return;
    }
    
    // Schedule flush
    if (!this.flushTimer) {
      this.flushTimer = setTimeout(() => {
        this.flushTimer = null;
        this.flush();
      }, this.flushInterval);
    }
  }
  
  /**
   * Flush buffer to content
   */
  flush(): void {
    if (this.buffer.length > 0) {
      this.content += this.buffer;
      this.buffer = '';
      this.lastFlush = performance.now();
      this.onUpdate(this.content);
    }
    
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }
  
  /**
   * Get current accumulated content
   */
  getContent(): string {
    return this.content + this.buffer;
  }
  
  /**
   * Reset accumulator
   */
  reset(): void {
    this.content = '';
    this.buffer = '';
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }
}

// ============================================
// MESSAGE TREE UTILITIES
// ============================================

/**
 * Build message tree from flat list
 */
export function buildMessageTree(messages: ChatMessage[]): Map<string | null, ChatMessage[]> {
  const tree = new Map<string | null, ChatMessage[]>();
  
  for (const msg of messages) {
    const parentId = msg.parent_id;
    if (!tree.has(parentId)) {
      tree.set(parentId, []);
    }
    tree.get(parentId)!.push(msg);
  }
  
  return tree;
}

/**
 * Get linear conversation path from root to leaf
 * Uses selected_versions to choose branches
 */
export function getConversationPath(
  messages: ChatMessage[],
  selectedVersions: Record<string, string> = {}
): ChatMessage[] {
  const tree = buildMessageTree(messages);
  const path: ChatMessage[] = [];
  
  // Start from root messages (no parent)
  let currentLevel = tree.get(null) || [];
  
  while (currentLevel.length > 0) {
    // Sort by created_at
    currentLevel.sort((a, b) => 
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );
    
    // Find selected message for this level, or use first
    let selected = currentLevel[0];
    for (const msg of currentLevel) {
      if (selectedVersions[msg.parent_id || 'root'] === msg.id) {
        selected = msg;
        break;
      }
    }
    
    path.push(selected);
    
    // Move to children of selected message
    currentLevel = tree.get(selected.id) || [];
  }
  
  return path;
}

/**
 * Find siblings (messages with same parent_id)
 */
export function findSiblings(messages: ChatMessage[], messageId: string): ChatMessage[] {
  const target = messages.find(m => m.id === messageId);
  if (!target) return [];
  
  return messages
    .filter(m => m.parent_id === target.parent_id && m.role === target.role)
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
}
