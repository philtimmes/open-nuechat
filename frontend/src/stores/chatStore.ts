/**
 * Chat Store - Re-exports from modular sliced version
 * 
 * This file maintains backward compatibility with existing imports.
 * The store is now split into focused slices in the chat/ directory.
 * 
 * @see ./chat/index.ts for the composed store
 * @see ./chat/types.ts for type definitions
 */
export { useChatStore } from './chat';
export type { ChatStore } from './chat';

// Re-export individual slice types for consumers who need them
export type {
  ChatSlice,
  MessageSlice,
  StreamSlice,
  ArtifactSlice,
  CodeSummarySlice,
} from './chat/types';
