import { SystemSettings, formatHours } from './types';

interface Props {
  settings: SystemSettings;
  setSettings: (settings: SystemSettings) => void;
  onSave: () => void;
  isSaving: boolean;
}

export default function SystemTab({ settings, setSettings, onSave, isSaving }: Props) {
  return (
    <div className="space-y-6">
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Prompts</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Default System Prompt</label>
            <textarea
              value={settings.default_system_prompt}
              onChange={(e) => setSettings({ ...settings, default_system_prompt: e.target.value })}
              rows={3}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
              All Models Prompt
              <span className="ml-2 text-xs text-[var(--color-text-muted)]">(appended to ALL system prompts including Custom GPTs)</span>
            </label>
            <textarea
              value={settings.all_models_prompt || ''}
              onChange={(e) => setSettings({ ...settings, all_models_prompt: e.target.value })}
              rows={4}
              placeholder="Instructions that apply to all models and Custom GPTs..."
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
              Server Timezone
              <span className="ml-2 text-xs text-[var(--color-text-muted)]">(fallback when browser timezone unavailable)</span>
            </label>
            <input
              type="text"
              value={settings.server_timezone || ''}
              onChange={(e) => setSettings({ ...settings, server_timezone: e.target.value })}
              placeholder="America/New_York"
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Title Generation Prompt</label>
            <textarea
              value={settings.title_generation_prompt}
              onChange={(e) => setSettings({ ...settings, title_generation_prompt: e.target.value })}
              rows={2}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">RAG Context Prompt (Legacy/Default)</label>
            <textarea
              value={settings.rag_context_prompt}
              onChange={(e) => setSettings({ ...settings, rag_context_prompt: e.target.value })}
              rows={2}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Fallback prompt if specific prompts below are empty</p>
          </div>
          
          <div className="col-span-full border-t border-[var(--color-border)] pt-4 mt-2">
            <h3 className="text-sm font-semibold text-[var(--color-text)] mb-3">Source-Specific RAG Prompts</h3>
            <p className="text-xs text-[var(--color-text-secondary)] mb-4">Use {'{context}'} placeholder for the retrieved content. Use {'{sources}'} for source names (Global KB only).</p>
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Global Knowledge Base Prompt</label>
            <textarea
              value={settings.rag_prompt_global_kb || ''}
              onChange={(e) => setSettings({ ...settings, rag_prompt_global_kb: e.target.value })}
              rows={4}
              placeholder="Leave empty for default authoritative prompt. Use {context} and {sources} placeholders."
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Custom GPT Knowledge Base Prompt</label>
            <textarea
              value={settings.rag_prompt_gpt_kb || ''}
              onChange={(e) => setSettings({ ...settings, rag_prompt_gpt_kb: e.target.value })}
              rows={4}
              placeholder="Leave empty to use legacy prompt. Use {context} placeholder."
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">User Documents Prompt</label>
            <textarea
              value={settings.rag_prompt_user_docs || ''}
              onChange={(e) => setSettings({ ...settings, rag_prompt_user_docs: e.target.value })}
              rows={4}
              placeholder="Leave empty to use legacy prompt. Use {context} placeholder."
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Chat History Knowledge Prompt</label>
            <textarea
              value={settings.rag_prompt_chat_history || ''}
              onChange={(e) => setSettings({ ...settings, rag_prompt_chat_history: e.target.value })}
              rows={4}
              placeholder="Leave empty for default chat history prompt. Use {context} placeholder."
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          {/* RAG Thresholds (NC-0.8.0.6) */}
          <div className="pt-4 border-t border-[var(--color-border)]">
            <h3 className="text-sm font-semibold text-[var(--color-text)] mb-3">RAG Similarity Thresholds</h3>
            <p className="text-xs text-[var(--color-text-secondary)] mb-3">
              Minimum similarity score (0.0-1.0) for RAG results to be included. Higher = stricter matching. Per-KB thresholds override global.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Global KB Default</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={settings.rag_threshold_global ?? 0.7}
                  onChange={(e) => setSettings({ ...settings, rag_threshold_global: parseFloat(e.target.value) || 0.7 })}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Chat History KB</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={settings.rag_threshold_chat_history ?? 0.5}
                  onChange={(e) => setSettings({ ...settings, rag_threshold_chat_history: parseFloat(e.target.value) || 0.5 })}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Local Chat (Attachments/Overflow)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={settings.rag_threshold_local ?? 0.4}
                  onChange={(e) => setSettings({ ...settings, rag_threshold_local: parseFloat(e.target.value) || 0.4 })}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Token Pricing & Limits</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Input Token Price (per 1M)</label>
            <input
              type="number"
              step="0.01"
              value={settings.input_token_price}
              onChange={(e) => setSettings({ ...settings, input_token_price: parseFloat(e.target.value) || 0 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Output Token Price (per 1M)</label>
            <input
              type="number"
              step="0.01"
              value={settings.output_token_price}
              onChange={(e) => setSettings({ ...settings, output_token_price: parseFloat(e.target.value) || 0 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
              Token Refill Interval (hours)
              <span className="ml-2 text-xs opacity-70">= {formatHours(settings.token_refill_interval_hours)}</span>
            </label>
            <input
              type="number"
              min="1"
              value={settings.token_refill_interval_hours}
              onChange={(e) => setSettings({ ...settings, token_refill_interval_hours: parseInt(e.target.value) || 1 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Free Tier Tokens</label>
            <input
              type="number"
              min="0"
              value={settings.free_tier_tokens}
              onChange={(e) => setSettings({ ...settings, free_tier_tokens: parseInt(e.target.value) || 0 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Pro Tier Tokens</label>
            <input
              type="number"
              min="0"
              value={settings.pro_tier_tokens}
              onChange={(e) => setSettings({ ...settings, pro_tier_tokens: parseInt(e.target.value) || 0 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Enterprise Tier Tokens</label>
            <input
              type="number"
              min="0"
              value={settings.enterprise_tier_tokens}
              onChange={(e) => setSettings({ ...settings, enterprise_tier_tokens: parseInt(e.target.value) || 0 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
        </div>
      </div>
      
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Storage Limits</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max File Upload Size (MB)</label>
            <input
              type="number"
              min="1"
              max="10000"
              value={settings.max_upload_size_mb}
              onChange={(e) => setSettings({ ...settings, max_upload_size_mb: parseInt(e.target.value) || 100 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Maximum size per uploaded file</p>
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max Knowledge Store Size (MB)</label>
            <input
              type="number"
              min="1"
              max="100000"
              value={settings.max_knowledge_store_size_mb}
              onChange={(e) => setSettings({ ...settings, max_knowledge_store_size_mb: parseInt(e.target.value) || 500 })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">Total size limit per knowledge base</p>
          </div>
          
          <div className="col-span-full">
            <label className="block text-sm text-[var(--color-text-secondary)] mb-2">Max Knowledge Stores per Tier</label>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Free</label>
                <input
                  type="number"
                  min="0"
                  value={settings.max_knowledge_stores_free}
                  onChange={(e) => setSettings({ ...settings, max_knowledge_stores_free: parseInt(e.target.value) || 0 })}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Pro</label>
                <input
                  type="number"
                  min="0"
                  value={settings.max_knowledge_stores_pro}
                  onChange={(e) => setSettings({ ...settings, max_knowledge_stores_pro: parseInt(e.target.value) || 0 })}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Enterprise</label>
                <input
                  type="number"
                  min="0"
                  value={settings.max_knowledge_stores_enterprise}
                  onChange={(e) => setSettings({ ...settings, max_knowledge_stores_enterprise: parseInt(e.target.value) || 0 })}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <button
        onClick={onSave}
        disabled={isSaving}
        className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
      >
        {isSaving ? 'Saving...' : 'Save System Settings'}
      </button>
    </div>
  );
}
