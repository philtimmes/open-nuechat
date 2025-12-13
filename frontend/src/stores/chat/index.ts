/**
 * Chat Store - Composed from domain slices
 * 
 * This store manages all chat-related state including:
 * - Chats (CRUD operations)
 * - Messages (with branching support)
 * - Streaming content
 * - Artifacts and uploaded files
 * - Code summaries
 * 
 * The store is split into focused slices for maintainability:
 * - chatSlice: Chat CRUD operations
 * - messageSlice: Message management and branching
 * - streamSlice: Streaming state
 * - artifactSlice: Artifacts and file uploads
 * - codeSummarySlice: Code tracking for LLM context
 * 
 * NEW (v2): Clean message system with proper state machine
 * - messageSystem.ts: Types and utilities
 * - messageStore.ts: New store implementation
 * See ARCHITECTURE.md for design details
 */
import { create } from 'zustand';
import type { ChatStore } from './types';
import { createChatSlice } from './chatSlice';
import { createMessageSlice } from './messageSlice';
import { createStreamSlice } from './streamSlice';
import { createArtifactSlice } from './artifactSlice';
import { createCodeSummarySlice } from './codeSummarySlice';

/**
 * Main chat store - combines all slices
 * @deprecated Consider using useMessageStore for message operations
 */
export const useChatStore = create<ChatStore>((set, get) => ({
  // Compose all slices
  ...createChatSlice(set, get),
  ...createMessageSlice(set, get),
  ...createStreamSlice(set, get),
  ...createArtifactSlice(set, get),
  ...createCodeSummarySlice(set, get),
}));

// Re-export types for convenience
export type { ChatStore } from './types';

// ============================================
// NEW MESSAGE SYSTEM (v2)
// ============================================

// Export the new message store
export { useMessageStore, useConversation, useIsStreaming, useStreamingContent, useArtifacts } from './messageStore';

// Export message system utilities
export {
  type ChatMessage,
  type MessageState,
  type PendingRequest,
  generateMessageId,
  generateRequestId,
  createUserMessage,
  createAssistantPlaceholder,
  fromServerMessage,
  canTransition,
  assertTransition,
  StreamAccumulator,
  buildMessageTree,
  getConversationPath,
  findSiblings,
} from './messageSystem';
