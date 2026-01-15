// Admin Panel Type Definitions

export interface SystemSettings {
  default_system_prompt: string;
  all_models_prompt: string;
  title_generation_prompt: string;
  rag_context_prompt: string;
  // Source-specific RAG prompts
  rag_prompt_global_kb: string;
  rag_prompt_gpt_kb: string;
  rag_prompt_user_docs: string;
  rag_prompt_chat_history: string;
  // RAG Thresholds (NC-0.8.0.6)
  rag_threshold_global: number;
  rag_threshold_chat_history: number;
  rag_threshold_local: number;
  input_token_price: number;
  output_token_price: number;
  token_refill_interval_hours: number;
  free_tier_tokens: number;
  pro_tier_tokens: number;
  enterprise_tier_tokens: number;
  // Storage limits
  max_upload_size_mb: number;
  max_knowledge_store_size_mb: number;
  max_knowledge_stores_free: number;
  max_knowledge_stores_pro: number;
  max_knowledge_stores_enterprise: number;
}

export interface OAuthSettings {
  google_client_id: string;
  google_client_secret: string;
  google_oauth_enabled: boolean;
  google_oauth_timeout: number;
  github_client_id: string;
  github_client_secret: string;
  github_oauth_enabled: boolean;
  github_oauth_timeout: number;
}

export interface LLMSettings {
  llm_api_base_url: string;
  llm_api_key: string;
  llm_model: string;
  llm_timeout: number;
  llm_max_tokens: number;
  llm_context_size: number;  // NC-0.8.0.7: Model context window size
  llm_temperature: number;
  llm_stream_default: boolean;
  llm_multimodal: boolean;
  // History compression
  history_compression_enabled: boolean;
  history_compression_threshold: number;
  history_compression_keep_recent: number;
  history_compression_target_tokens: number;
}

export interface LLMProvider {
  id: string;
  name: string;
  base_url: string;
  api_key: string;
  model_id: string;
  is_multimodal: boolean;
  supports_tools: boolean;
  supports_streaming: boolean;
  is_default: boolean;
  is_vision_default: boolean;
  is_enabled: boolean;
  timeout: number;
  max_tokens: number;
  context_size: number;
  temperature: string;
  vision_prompt: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface FeatureFlags {
  enable_registration: boolean;
  enable_billing: boolean;
  freeforall: boolean;
  enable_safety_filters: boolean;
  enable_mermaid_rendering: boolean;
}

export interface APIRateLimits {
  api_rate_limit_completions: number;
  api_rate_limit_embeddings: number;
  api_rate_limit_images: number;
  api_rate_limit_models: number;
}

export interface TierConfig {
  id: string;
  name: string;
  price: number;
  tokens: number;
  features: string[];
  popular: boolean;
}

export interface UserListItem {
  id: string;
  email: string;
  username: string;
  tier: string;
  is_admin: boolean;
  is_active: boolean;
  tokens_limit: number;
  tokens_used: number;
  chat_count: number;
  created_at: string;
}

export interface ChatListItem {
  id: string;
  title: string;
  model: string | null;
  owner_email?: string;
  message_count: number;
  created_at: string;
  updated_at: string | null;
}

export interface ChatDetail {
  id: string;
  title: string;
  model: string | null;
  owner_id: string;
  owner_email: string;
  owner_username: string;
  created_at: string;
  updated_at: string | null;
  messages: Array<{
    id: string;
    role: string;
    content: string;
    created_at: string;
  }>;
}

export interface ToolConfig {
  id: string;
  name: string;
  description: string | null;
  tool_type: 'mcp' | 'openapi';
  url: string;
  has_api_key: boolean;
  is_public: boolean;
  is_enabled: boolean;
  schema_cache?: Array<{
    name: string;
    description?: string;
    parameters?: unknown;
  }>;
}

export interface ToolUsageStat {
  tool_id: string;
  tool_name: string;
  total_calls: number;
  successful_calls: number;
  failed_calls: number;
  total_duration_ms: number;
  avg_duration_ms: number;
  last_used: string | null;
  unique_users: number;
}

export interface ToolUsageStats {
  total_calls: number;
  successful_calls: number;
  failed_calls: number;
  tools: ToolUsageStat[];
}

export interface FilterConfig {
  id?: string;
  name: string;
  description: string | null;
  filter_type: 'to_llm' | 'from_llm' | 'to_tools' | 'from_tools';
  priority: 'highest' | 'high' | 'medium' | 'low' | 'least';
  enabled: boolean;
  filter_mode: 'pattern' | 'code' | 'llm';
  pattern: string | null;
  replacement: string | null;
  word_list: string[] | null;
  case_sensitive: boolean;
  action: 'modify' | 'block' | 'log' | 'passthrough';
  block_message: string | null;
  code: string | null;
  llm_prompt: string | null;
  config: Record<string, unknown> | null;
  is_global: boolean;
}

export interface FilterChainConditional {
  enabled: boolean;
  logic?: string;
  comparisons: Array<{ left: string; operator: string; right: string }>;
  on_true?: FilterChainStep[];
  on_false?: FilterChainStep[];
}

export type StepConfig = Record<string, unknown>;

export interface FilterChainStep {
  id: string;
  type: string;
  name?: string;
  enabled?: boolean;
  config: StepConfig;
  on_error?: string;
  jump_to_step?: string;
  conditional?: FilterChainConditional;
  loop?: {
    enabled: boolean;
    type: string;
    count?: number;
    while?: { left: string; operator: string; right: string };
    max_iterations?: number;
    loop_var?: string;
  };
}

export interface FilterChainDef {
  id?: string;
  name: string;
  description?: string;
  enabled: boolean;
  priority: number;
  retain_history: boolean;
  bidirectional: boolean;
  outbound_chain_id?: string;
  max_iterations: number;
  debug: boolean;
  skip_if_rag_hit: boolean;
  // Export as dynamic tool fields (NC-0.8.0.0)
  export_as_tool: boolean;
  tool_name?: string;
  tool_label?: string;
  advertise_to_llm: boolean;
  advertise_text?: string;
  trigger_pattern?: string;
  trigger_source: 'llm' | 'user' | 'both';
  erase_from_display: boolean;
  keep_in_history: boolean;
  button_enabled: boolean;
  button_icon?: string;
  button_location: 'response' | 'query' | 'both';
  button_trigger_mode: 'immediate' | 'modal' | 'selection';
  tool_variables?: Array<{ name: string; label: string; default?: string; type?: string }>;
  // Definition and metadata
  definition: { steps: FilterChainStep[] };
  created_at?: string;
  updated_at?: string;
}

export interface StepTypeSchema {
  label: string;
  description: string;
  category: string;
  fields: Array<{
    name: string;
    type: string;
    label?: string;
    required?: boolean;
    default?: unknown;
    options?: Array<{ value: string; label: string }>;
  }>;
}

export interface FilterChainSchema {
  step_types: Record<string, StepTypeSchema>;
  comparison_operators: Array<{ value: string; label: string }>;
  builtin_variables: Array<{ value: string; label: string }>;
  available_tools?: Array<{ value: string; label: string; category: string; description?: string }>;
}

export interface GlobalKBStore {
  id: string;
  name: string;
  description: string | null;
  icon: string;
  owner_username: string;
  document_count: number;
  is_global: boolean;
  global_min_score: number;
  global_max_results: number;
  // NC-0.8.0.1: Required keywords filter
  require_keywords_enabled: boolean;
  required_keywords: string[] | null;
  // NC-0.8.0.7: Force trigger keywords
  force_trigger_enabled: boolean;
  force_trigger_keywords: string[] | null;
  force_trigger_max_chunks: number;
}

export interface GPTCategory {
  id: string;
  value: string;
  label: string;
  icon: string;
  description: string | null;
  sort_order: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

// NC-0.8.0.7: Image generation settings
export interface ImageResolutionOption {
  width: number;
  height: number;
  label: string;
}

export interface ImageGenSettings {
  image_gen_default_width: number;
  image_gen_default_height: number;
  image_gen_default_aspect_ratio: string;
  image_gen_available_resolutions: ImageResolutionOption[];
}

export type TabId = 'system' | 'oauth' | 'llm' | 'image_gen' | 'features' | 'tiers' | 'users' | 'chats' | 'tools' | 'filters' | 'filter_chains' | 'global_kb' | 'categories' | 'modes' | 'dev';

// Common props for tab components
export interface TabProps {
  setError: (error: string | null) => void;
  setSuccess: (success: string | null) => void;
}

// Helper function for formatting hours
export function formatHours(hours: number): string {
  if (hours < 24) return `${hours} hour${hours !== 1 ? 's' : ''}`;
  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  if (remainingHours === 0) return `${days} day${days !== 1 ? 's' : ''}`;
  return `${days}d ${remainingHours}h`;
}
