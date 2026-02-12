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
  chatSortBy: 'modified' | 'created' | 'alphabetical' | 'source';
  chatGroupCounts: Record<string, number> | null;  // Actual totals from database
  sidebarReloadTrigger: number;  // Increment to force sidebar reload
  
  // Per-group pagination state
  groupPages: Record<string, number>;  // Current page per group
  groupHasMore: Record<string, boolean>;  // Has more per group
  groupLoading: Record<string, boolean>;  // Loading state per group
  
  fetchChats: (loadMore?: boolean, search?: string, sortBy?: string) => Promise<void>;
  fetchGroupChats: (groupType: 'source' | 'period', groupName: string, loadMore?: boolean, dateField?: 'updated_at' | 'created_at') => Promise<void>;
  setChatSortBy: (sortBy: 'modified' | 'created' | 'alphabetical' | 'source') => void;
  createChat: (model?: string, systemPrompt?: string) => Promise<Chat>;
  setCurrentChat: (chat: Chat | null) => void;
  deleteChat: (chatId: string) => Promise<void>;
  deleteAllChats: () => Promise<void>;
  deleteGroupChats: (groupType: 'source' | 'period', groupName: string) => Promise<number>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  updateChatLocally: (chatId: string, updates: Partial<Chat>) => void;
  triggerSidebarReload: () => void;
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
  toolTimeline: ToolTimelineEvent[];
  error: string | null;
  
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (chunk: string) => void;
  setStreamingToolCall: (toolCall: { name: string; input: string } | null) => void;
  setToolTimelineEvent: (event: ToolTimelineEvent) => void;
  clearToolTimeline: () => void;
  clearStreaming: () => void;
  setIsSending: (isSending: boolean) => void;
  setError: (error: string | null) => void;
  updateStreamingArtifacts: (content: string) => void;
}

export interface ToolTimelineEvent {
  ts: number;
  type: 'tool_start' | 'tool_end';
  tool: string;
  round: number;
  args_summary?: string;
  status?: string;
  duration_ms?: number;
  result_summary?: string;
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
  
  // Pending image context - for standalone image generation before chat starts
  // This image will be included in the first message to a new chat
  pendingImageContext: GeneratedImage | null;
  
  setSelectedArtifact: (artifact: Artifact | null) => void;
  setShowArtifacts: (show: boolean) => void;
  collectAllArtifacts: () => Artifact[];
  addUploadedArtifacts: (artifacts: Artifact[]) => void;
  setZipUploadResult: (result: Partial<ZipUploadResult> | null) => void;
  fetchUploadedData: (chatId: string) => Promise<void>;
  setZipContext: (context: string | null) => void;
  setGeneratedImage: (messageId: string, image: GeneratedImage) => void;
  setPendingImageContext: (image: GeneratedImage | null) => void;
  setArtifacts: (artifacts: Artifact[]) => void;
  updateArtifact: (key: string, artifact: Artifact) => void;
  revertArtifact: (artifactId: string) => void;  // NC-0.8.0.12: Touch old version to make it latest
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
