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
