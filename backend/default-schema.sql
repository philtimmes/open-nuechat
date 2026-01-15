-- Default schema and settings for Open-NueChat fresh install
-- This file is executed when the database is first created

-- System Settings (from SETTING_DEFAULTS)
INSERT OR IGNORE INTO system_settings (key, value) VALUES
-- Image Generation
('image_gen_default_width', '1024'),
('image_gen_default_height', '1024'),
('image_gen_default_aspect_ratio', '1:1'),
('image_gen_available_resolutions', '[{"width": 512, "height": 512, "label": "Small Square (512x512)"}, {"width": 768, "height": 768, "label": "Medium Square (768x768)"}, {"width": 1024, "height": 1024, "label": "Large Square (1024x1024)"}, {"width": 768, "height": 1024, "label": "Portrait (768x1024)"}, {"width": 1024, "height": 768, "label": "Landscape (1024x768)"}]'),

-- LLM Settings
('llm_api_base_url', 'http://localhost:8080/v1'),
('llm_api_key', 'not-needed'),
('llm_model', 'default'),
('llm_timeout', '300'),
('llm_max_tokens', '4096'),
('llm_context_size', '200000'),
('llm_temperature', '0.7'),
('llm_stream_default', 'true'),
('llm_multimodal', 'false'),

-- Prompts
('default_system_prompt', 'You are a helpful AI assistant. Be concise, accurate, and helpful.'),
('all_models_prompt', ''),
('title_generation_prompt', 'Generate a short, descriptive title (max 6 words) for a conversation that starts with this message. Return ONLY the title, no quotes or explanation:'),
('rag_context_prompt', 'The following information has been retrieved from the user''s documents to help answer their question:'),

-- Token limits
('free_tier_tokens', '100000'),
('pro_tier_tokens', '1000000'),
('enterprise_tier_tokens', '10000000'),
('token_refill_interval_hours', '720'),

-- Pricing
('input_token_price', '3.0'),
('output_token_price', '15.0'),
('tiers', '[{"id": "free", "name": "Free", "price": 0, "tokens": 100000, "features": ["100K tokens/month", "Basic models", "Community support"], "popular": false}, {"id": "pro", "name": "Pro", "price": 20, "tokens": 1000000, "features": ["1M tokens/month", "All models", "Priority support", "RAG storage: 100MB"], "popular": true}, {"id": "enterprise", "name": "Enterprise", "price": 100, "tokens": 10000000, "features": ["10M tokens/period", "All models", "Dedicated support", "RAG storage: 1GB", "Custom integrations"], "popular": false}]'),

-- OAuth - Google
('google_client_id', ''),
('google_client_secret', ''),
('google_oauth_enabled', 'true'),
('google_oauth_timeout', '30'),

-- OAuth - GitHub
('github_client_id', ''),
('github_client_secret', ''),
('github_oauth_enabled', 'true'),
('github_oauth_timeout', '30'),

-- Feature flags
('enable_registration', 'true'),
('enable_billing', 'true'),
('freeforall', 'false'),
('enable_safety_filters', 'false'),
('enable_mermaid_rendering', 'true'),

-- History compression
('history_compression_enabled', 'true'),
('history_compression_threshold', '20'),
('history_compression_keep_recent', '10'),
('history_compression_target_tokens', '8000'),
('model_context_size', '128000'),

-- API Rate Limits
('api_rate_limit_completions', '60'),
('api_rate_limit_embeddings', '200'),
('api_rate_limit_images', '10'),
('api_rate_limit_models', '100'),

-- Storage limits
('max_upload_size_mb', '100'),
('max_knowledge_store_size_mb', '500'),
('max_knowledge_stores_free', '3'),
('max_knowledge_stores_pro', '20'),
('max_knowledge_stores_enterprise', '100'),

-- Debug settings
('debug_token_resets', 'false'),
('debug_document_queue', 'false'),
('debug_rag', 'false'),
('debug_filter_chains', 'false'),
('debug_tool_advertisements', 'false'),
('debug_tool_calls', 'false'),

-- RAG thresholds
('rag_threshold_global', '0.7'),
('rag_threshold_chat_history', '0.5'),
('rag_threshold_local', '0.4');
