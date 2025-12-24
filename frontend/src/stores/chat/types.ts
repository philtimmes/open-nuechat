/**
 * Chat store type definitions
 * 
 * Defines all types used by the chat store slices.
 */
import type { Chat, Message, Artifact, CodeSummary, FileChange, SignatureWarning, ZipUploadResult, GeneratedImage, MessageBranch } from '../../types';

/**
 * Chat slice state and actions
 */
export interface ChatSlice {
  chats: Chat[];
  currentChat: Chat | null;
  isLoadingChats: boolean;
  hasMoreChats: boolean;
  chatPage: number;
  chatSearchQuery: string;
  
  fetchChats: (loadMore?: boolean, search?: string) => Promise<void>;
  createChat: (model?: string, systemPrompt?: string, preserveArtifacts?: boolean) => Promise<Chat>;
  setCurrentChat: (chat: Chat | null) => void;
  deleteChat: (chatId: string) => Promise<void>;
  deleteAllChats: () => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  updateChatLocally: (chatId: string, updates: Partial<Chat>) => void;
}

/**
 * Message slice state and actions
 */
export interface MessageSlice {
  messages: Message[];
  isLoadingMessages: boolean;
  
  fetchMessages: (chatId: string) => Promise<void>;
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  replaceMessageId: (tempId: string, realId: string, parentId?: string | null) => void;
  retryMessage: (messageId: string) => Promise<void>;
  switchBranch: (messageId: string, branchIndex: number) => void;
}

/**
 * Streaming slice state and actions
 */
export interface StreamSlice {
  isSending: boolean;
  streamingContent: string;
  streamingToolCall: { name: string; input: string } | null;
  streamingArtifacts: Artifact[];
  error: string | null;
  
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (chunk: string) => void;
  setStreamingToolCall: (toolCall: { name: string; input: string } | null) => void;
  clearStreaming: () => void;
  setIsSending: (isSending: boolean) => void;
  setError: (error: string | null) => void;
  updateStreamingArtifacts: (content: string) => void;
}

/**
 * Artifact slice state and actions
 */
export interface ArtifactSlice {
  artifacts: Artifact[];
  selectedArtifact: Artifact | null;
  showArtifacts: boolean;
  
  // Uploaded files (persisted to backend)
  uploadedArtifacts: Artifact[];
  zipUploadResult: Partial<ZipUploadResult> | null;
  
  // Zip context for LLM
  zipContext: string | null;
  
  // Generated images
  generatedImages: Record<string, GeneratedImage>;
  
  setSelectedArtifact: (artifact: Artifact | null) => void;
  setShowArtifacts: (show: boolean) => void;
  collectAllArtifacts: () => Artifact[];
  addUploadedArtifacts: (artifacts: Artifact[]) => void;
  setZipUploadResult: (result: Partial<ZipUploadResult> | null) => void;
  fetchUploadedData: (chatId: string) => Promise<void>;
  setZipContext: (context: string | null) => void;
  setGeneratedImage: (messageId: string, image: GeneratedImage) => void;
  setArtifacts: (artifacts: Artifact[]) => void;
  updateArtifact: (key: string, artifact: Artifact) => void;
}

/**
 * Code summary slice state and actions
 */
export interface CodeSummarySlice {
  codeSummary: CodeSummary | null;
  showSummary: boolean;
  
  setShowSummary: (show: boolean) => void;
  updateCodeSummary: (files: FileChange[], warnings?: SignatureWarning[]) => void;
  addFileToSummary: (file: FileChange) => void;
  addWarning: (warning: SignatureWarning) => void;
  clearSummary: () => void;
  fetchCodeSummary: (chatId: string) => Promise<void>;
  saveCodeSummary: () => Promise<void>;
}

/**
 * Combined chat store interface
 */
export interface ChatStore extends 
  ChatSlice, 
  MessageSlice, 
  StreamSlice, 
  ArtifactSlice, 
  CodeSummarySlice {}

/**
 * Zustand StateCreator type for slices
 * Using a more permissive type to allow cross-slice access
 */
export type SliceCreator<T> = (
  set: (partial: Partial<ChatStore> | ((state: ChatStore) => Partial<ChatStore>)) => void,
  get: () => ChatStore,
) => T;
