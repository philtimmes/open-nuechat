// User types
export interface User {
  id: string;
  email: string;
  username: string;
  tier: 'free' | 'pro' | 'enterprise';
  is_admin?: boolean;
  theme?: string;
  created_at: string;
}

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

// Chat types
export interface Chat {
  id: string;
  title: string;
  model: string;
  assistant_id?: string;  // Custom GPT association
  assistant_name?: string;  // Custom GPT name (denormalized)
  system_prompt?: string;
  is_shared: boolean;
  is_knowledge_indexed?: boolean;  // Part of user's chat knowledge base
  // Assistant Mode (NC-0.8.0.0)
  mode_id?: string;
  active_tools?: string[];  // User's tool overrides
  created_at: string;
  updated_at: string;
  total_input_tokens: number;
  total_output_tokens: number;
  message_count?: number;
  // Branch selection: { parent_message_id: selected_child_id }
  selected_versions?: Record<string, string>;
}

export interface Attachment {
  type: 'image' | 'file';
  name: string;
  url?: string;
  data?: string;
  mime_type: string;
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
  output?: string;
}

export interface Artifact {
  id: string;
  type: 'code' | 'document' | 'html' | 'react' | 'svg' | 'mermaid' | 'markdown' | 'json' | 'csv';
  title: string;
  language?: string;
  content: string;
  filename?: string;
  created_at: string;
  // For zip-extracted files
  size?: number;
  signatures?: CodeSignature[];
  source?: 'upload' | 'generated';  // Track origin for filtering
}

export interface CodeSignature {
  name: string;
  kind: string;  // 'function', 'class', 'method', 'variable', 'export', 'import', 'interface', 'type'
  line: number;
  signature?: string;
  docstring?: string;
}

export interface ZipUploadResult {
  filename: string;
  total_files: number;
  total_size: number;
  languages: Record<string, number>;
  file_tree: Record<string, unknown>;
  signature_index: Record<string, CodeSignature[]>;
  artifacts: Artifact[];
  summary: string;  // Human-readable summary
  llm_manifest: string;  // LLM context injection format
}

export interface ZipFileResponse {
  path: string;
  content: string;
  formatted: string;  // Pre-formatted for LLM injection
}

export interface MessageBranch {
  id: string;
  content: string;
  artifacts?: Artifact[];
  tool_calls?: ToolCall[];
  input_tokens?: number;
  output_tokens?: number;
  created_at: string;
}

export interface Message {
  id: string;
  chat_id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  attachments?: Attachment[];
  tool_calls?: ToolCall[];
  artifacts?: Artifact[];
  input_tokens?: number;
  output_tokens?: number;
  created_at: string;
  // Tree structure for conversation branching
  parent_id?: string | null;
  // Legacy branching support (deprecated)
  branches?: MessageBranch[];
  current_branch?: number;
  // Metadata for special message types (file content, generated images, etc.)
  metadata?: {
    type?: 'file_content' | 'context' | 'find_line_result' | 'find_result' | 'search_replace_result';
    path?: string;
    source?: string;
    // Generated image data
    generated_image?: {
      url?: string;
      width?: number;
      height?: number;
      seed?: number;
      prompt?: string;
      job_id?: string;
    };
    // Image generation status
    image_generation?: {
      status?: 'pending' | 'processing' | 'completed' | 'failed';
      prompt?: string;
      width?: number;
      height?: number;
      queue_position?: number;
      error?: string;
      generation_time?: number;
      job_id?: string;
    };
  };
}

// Generated image data (for real-time display)
export interface GeneratedImage {
  base64?: string;  // May not be present when loading from DB
  url?: string;     // URL for persisted images
  width: number;
  height: number;
  seed: number;
  prompt: string;
  generation_time?: number;
  job_id?: string;
}

// Theme types
export interface ThemeColors {
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  text_secondary: string;
  accent: string;
  error: string;
  success: string;
  warning: string;
  border: string;
}

export interface ThemeFonts {
  heading: string;
  body: string;
  code: string;
}

export interface Theme {
  id: string;
  name: string;
  description?: string;
  colors: ThemeColors;
  fonts: ThemeFonts;
  is_system: boolean;
  is_public: boolean;
}

// Billing types
export interface UsageSummary {
  period_start: string;
  period_end: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_cost: number;
  output_cost: number;
  total_cost: number;
  tier_limit: number;
  usage_percentage: number;
}

export interface DailyUsage {
  date: string;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

// Document types (RAG)
export interface Document {
  id: string;
  name: string;
  description?: string;
  file_type: string;
  file_size: number;
  is_processed: boolean;
  chunk_count: number;
  knowledge_store_id?: string;
  created_at: string;
}

export interface SearchResult {
  document_id: string;
  document_name: string;
  content: string;
  similarity: number;
  chunk_index: number;
}

// Knowledge Store types
export interface KnowledgeStore {
  id: string;
  name: string;
  description?: string;
  icon: string;
  color: string;
  is_public: boolean;
  is_discoverable: boolean;
  is_global?: boolean;  // Auto-searched on every query (admin-only setting)
  document_count: number;
  total_chunks: number;
  total_size_bytes: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeStoreShare {
  id: string;
  knowledge_store_id: string;
  shared_with_user_id?: string;
  share_token?: string;
  permission: 'view' | 'use' | 'edit';
  expires_at?: string;
  created_at: string;
}

// Custom Assistant (GPT) types
export interface CustomAssistant {
  id: string;
  owner_id: string;
  name: string;
  slug: string;
  tagline?: string;
  description?: string;
  avatar_url?: string;
  icon: string;
  color: string;
  system_prompt: string;
  welcome_message?: string;
  suggested_prompts: string[];
  model: string;
  temperature: number;
  max_tokens: number;
  enabled_tools: string[];
  is_public: boolean;
  is_discoverable: boolean;
  is_featured: boolean;
  category: string;
  conversation_count: number;
  message_count: number;
  average_rating: number;
  knowledge_store_ids?: string[];
  created_at: string;
  updated_at: string;
}

// WebSocket types
export interface WSMessage {
  type: string;
  payload: unknown;
  timestamp?: string;
}

export interface StreamChunk {
  type: 'stream_start' | 'stream_chunk' | 'stream_end' | 'stream_error' | 'tool_call_start' | 'tool_call_end';
  chat_id?: string;
  message_id?: string;
  content?: string;
  tool_call?: ToolCall;
  error?: string;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}

// Agent Flow types
export interface FlowNodePosition {
  x: number;
  y: number;
}

export interface FlowConnection {
  id: string;
  fromNodeId: string;
  toNodeId: string;
}

export type FlowNodeType = 
  | 'start' 
  | 'return' 
  | 'logic' 
  | 'filter' 
  | 'documents' 
  | 'web' 
  | 'model_request' 
  | 'output';

export interface LogicConfig {
  operator: 'equals' | 'not_equals' | 'contains' | 'greater_than' | 'less_than' | 'and' | 'or' | 'not' | 'if';
  value?: string;
  compareField?: string;
}

export interface FilterConfig {
  type: 'replace' | 'reject' | 'allow';
  pattern: string;
  replacement?: string;
  filterName?: string;  // Reference to existing filter
}

export interface DocumentsConfig {
  source: 'knowledge_base' | 'user_link' | 'google_drive';
  knowledgeStoreId?: string;
  url?: string;
  query?: string;  // Can use variables like {{input}}
}

export interface WebConfig {
  urlSource: 'static' | 'dynamic';
  staticUrl?: string;
  extractContent?: boolean;
}

export interface ModelRequestConfig {
  prompt: string;  // Template with {{input}} variable
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface OutputConfig {
  format?: 'text' | 'json' | 'markdown';
  template?: string;
}

export interface FlowNode {
  id: string;
  type: FlowNodeType;
  position: FlowNodePosition;
  label: string;
  config?: LogicConfig | FilterConfig | DocumentsConfig | WebConfig | ModelRequestConfig | OutputConfig;
}

export interface AgentFlow {
  id: string;
  name: string;
  description?: string;
  nodes: FlowNode[];
  connections: FlowConnection[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

// API response types
export interface ApiError {
  detail: string;
  status_code?: number;
}

// Code Summary types for tracking LLM-generated code
export interface CodeSignatureEntry {
  name: string;
  type: 'function' | 'class' | 'method' | 'variable' | 'interface' | 'type' | 'endpoint';
  signature: string;
  file: string;
  line?: number;
}

export interface FileChange {
  path: string;
  action: 'created' | 'modified' | 'deleted';
  language?: string;
  signatures: CodeSignatureEntry[];
  timestamp: string;
}

export interface SignatureWarning {
  type: 'missing' | 'mismatch' | 'orphan' | 'library_not_found';
  message: string;
  file?: string;
  signature?: string;
  suggestion?: string;
}

export interface CodeSummary {
  id: string;
  chat_id: string;
  files: FileChange[];
  warnings: SignatureWarning[];
  last_updated: string;
  auto_generated: boolean;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

// Assistant Mode types (NC-0.8.0.0)
export interface AssistantMode {
  id: string;
  name: string;
  description?: string;
  icon?: string;
  active_tools: string[];
  advertised_tools: string[];
  filter_chain_id?: string;
  sort_order: number;
  enabled: boolean;
  is_global: boolean;
  created_by?: string;
  created_at?: string;
  updated_at?: string;
}

// Dynamic Tool types (NC-0.8.0.0)
export interface DynamicTool {
  name: string;
  label: string;
  icon?: string;
  location: 'response' | 'query' | 'both';
  trigger_source: 'llm' | 'user' | 'both';
}

export interface UserHint {
  type: 'user_hint';
  label: string;
  icon?: string;
  location: 'response' | 'query' | 'both';
  prompt?: string;
}

export interface UserAction {
  type: 'user_action';
  label: string;
  icon?: string;
  position: 'response' | 'query' | 'input';
  prompt?: string;
}

// Tool icon state
export type ToolState = 'active' | 'inactive' | 'always';

export interface ToolDefinition {
  id: string;
  name: string;
  label: string;
  icon: string;  // SVG path or inline SVG
  category: 'always' | 'mode' | 'toggle';
  description?: string;
}
