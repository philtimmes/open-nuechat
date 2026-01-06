/**
 * AssistantModesTab - NC-0.8.0.0
 * 
 * Admin panel tab for managing assistant mode presets.
 */

import { useState, useEffect } from 'react';
import { assistantModeApi } from '../../lib/api';
import type { AssistantMode } from '../../types';

// Available tools for selection
const AVAILABLE_TOOLS = [
  { id: 'web_search', label: 'Web Search', description: 'Search the internet' },
  { id: 'artifacts', label: 'Artifacts', description: 'Create documents and code' },
  { id: 'image_gen', label: 'Image Generation', description: 'Generate images' },
  { id: 'code_exec', label: 'Code Execution', description: 'Run Python code' },
  { id: 'file_ops', label: 'File Operations', description: 'Read and write files' },
  { id: 'kb_search', label: 'Knowledge Base', description: 'Search knowledge bases' },
  { id: 'citations', label: 'Citations', description: 'Add references' },
];

export function AssistantModesTab() {
  const [modes, setModes] = useState<AssistantMode[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingMode, setEditingMode] = useState<AssistantMode | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  useEffect(() => {
    loadModes();
  }, []);

  const loadModes = async () => {
    setIsLoading(true);
    try {
      const response = await assistantModeApi.list(false); // Include disabled
      setModes(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load assistant modes');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreate = async (data: Partial<AssistantMode>) => {
    try {
      await assistantModeApi.create({
        name: data.name || 'New Mode',
        description: data.description,
        icon: data.icon,
        active_tools: data.active_tools || [],
        advertised_tools: data.advertised_tools || [],
        sort_order: modes.length,
        enabled: true,
        is_global: true,
      });
      await loadModes();
      setIsCreating(false);
    } catch (err) {
      setError('Failed to create mode');
      console.error(err);
    }
  };

  const handleUpdate = async (modeId: string, data: Partial<AssistantMode>) => {
    try {
      await assistantModeApi.update(modeId, data);
      await loadModes();
      setEditingMode(null);
    } catch (err) {
      setError('Failed to update mode');
      console.error(err);
    }
  };

  const handleDelete = async (modeId: string) => {
    if (!confirm('Are you sure you want to delete this mode?')) return;
    
    try {
      await assistantModeApi.delete(modeId);
      await loadModes();
    } catch (err) {
      setError('Failed to delete mode');
      console.error(err);
    }
  };

  const handleDuplicate = async (modeId: string) => {
    try {
      await assistantModeApi.duplicate(modeId);
      await loadModes();
    } catch (err) {
      setError('Failed to duplicate mode');
      console.error(err);
    }
  };

  const handleToggleEnabled = async (modeId: string, enabled: boolean) => {
    try {
      await assistantModeApi.update(modeId, { enabled });
      await loadModes();
    } catch (err) {
      setError('Failed to toggle mode');
      console.error(err);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-[var(--color-text)]">Assistant Modes</h2>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Configure tool presets for different use cases
          </p>
        </div>
        <button
          onClick={() => setIsCreating(true)}
          className="px-4 py-2 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white rounded-lg 
                   transition-colors flex items-center gap-2"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Add Mode
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Mode List */}
      <div className="grid gap-4 md:grid-cols-2">
        {modes.map(mode => (
          <div
            key={mode.id}
            className={`rounded-xl border overflow-hidden transition-all ${
              mode.enabled 
                ? 'bg-[var(--color-surface)] border-[var(--color-border)]' 
                : 'bg-[var(--color-surface)]/50 border-[var(--color-border)]/50 opacity-70'
            }`}
          >
            {/* Card Header */}
            <div className="px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                  mode.enabled ? 'bg-blue-500/20' : 'bg-white/10'
                }`}>
                  <svg className={`w-4 h-4 ${mode.enabled ? 'text-blue-400' : 'text-white/40'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="7" height="7" rx="1" />
                    <rect x="14" y="3" width="7" height="7" rx="1" />
                    <rect x="3" y="14" width="7" height="7" rx="1" />
                    <rect x="14" y="14" width="7" height="7" rx="1" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-medium text-[var(--color-text)]">{mode.name}</h3>
                  {mode.description && (
                    <p className="text-xs text-[var(--color-text-secondary)] line-clamp-1">{mode.description}</p>
                  )}
                </div>
              </div>
              
              {/* Enable Toggle */}
              <button
                onClick={() => handleToggleEnabled(mode.id, !mode.enabled)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  mode.enabled ? 'bg-green-500' : 'bg-white/20'
                }`}
                title={mode.enabled ? 'Enabled - Click to disable' : 'Disabled - Click to enable'}
              >
                <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                  mode.enabled ? 'left-5' : 'left-0.5'
                }`} />
              </button>
            </div>
            
            {/* Card Body */}
            <div className="p-4 space-y-3">
              {/* Active Tools */}
              <div>
                <span className="text-xs text-[var(--color-text-secondary)]">Active Tools</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {(mode.active_tools || []).length > 0 ? (
                    mode.active_tools.map((tool: string) => (
                      <span
                        key={tool}
                        className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded"
                      >
                        {AVAILABLE_TOOLS.find(t => t.id === tool)?.label || tool}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-[var(--color-text-secondary)] italic">None</span>
                  )}
                </div>
              </div>

              {/* Advertised Tools */}
              <div>
                <span className="text-xs text-[var(--color-text-secondary)]">Advertised to LLM</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {(mode.advertised_tools || []).length > 0 ? (
                    mode.advertised_tools.map((tool: string) => (
                      <span
                        key={tool}
                        className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded"
                      >
                        {AVAILABLE_TOOLS.find(t => t.id === tool)?.label || tool}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-[var(--color-text-secondary)] italic">None</span>
                  )}
                </div>
              </div>
            </div>
            
            {/* Card Footer - Actions */}
            <div className="px-4 py-3 border-t border-[var(--color-border)] flex items-center justify-between bg-[var(--color-background)]/30">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setEditingMode(mode)}
                  className="px-3 py-1.5 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] 
                           hover:bg-[var(--color-surface)] rounded-lg transition-colors flex items-center gap-1.5"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                  Edit
                </button>
                <button
                  onClick={() => handleDuplicate(mode.id)}
                  className="px-3 py-1.5 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] 
                           hover:bg-[var(--color-surface)] rounded-lg transition-colors flex items-center gap-1.5"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  Duplicate
                </button>
              </div>
              <button
                onClick={() => handleDelete(mode.id)}
                className="px-3 py-1.5 text-xs text-red-400/70 hover:text-red-400 
                         hover:bg-red-500/10 rounded-lg transition-colors flex items-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Create/Edit Modal */}
      {(isCreating || editingMode) && (
        <ModeEditModal
          mode={editingMode}
          onSave={(data) => {
            if (editingMode) {
              handleUpdate(editingMode.id, data);
            } else {
              handleCreate(data);
            }
          }}
          onClose={() => {
            setIsCreating(false);
            setEditingMode(null);
          }}
        />
      )}
    </div>
  );
}

// Edit/Create Modal
interface ModeEditModalProps {
  mode: AssistantMode | null;
  onSave: (data: Partial<AssistantMode>) => void;
  onClose: () => void;
}

function ModeEditModal({ mode, onSave, onClose }: ModeEditModalProps) {
  const [name, setName] = useState(mode?.name || '');
  const [description, setDescription] = useState(mode?.description || '');
  const [activeTools, setActiveTools] = useState<string[]>(mode?.active_tools || []);
  const [advertisedTools, setAdvertisedTools] = useState<string[]>(mode?.advertised_tools || []);
  const [enabled, setEnabled] = useState(mode?.enabled ?? true);

  const toggleActiveTool = (toolId: string) => {
    if (activeTools.includes(toolId)) {
      setActiveTools(activeTools.filter(t => t !== toolId));
      // Also remove from advertised if present
      setAdvertisedTools(advertisedTools.filter(t => t !== toolId));
    } else {
      setActiveTools([...activeTools, toolId]);
    }
  };

  const toggleAdvertisedTool = (toolId: string) => {
    if (advertisedTools.includes(toolId)) {
      setAdvertisedTools(advertisedTools.filter(t => t !== toolId));
    } else {
      setAdvertisedTools([...advertisedTools, toolId]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({
      name,
      description: description || undefined,
      active_tools: activeTools,
      advertised_tools: advertisedTools,
      enabled,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <h3 className="text-lg font-semibold text-white">
            {mode ? 'Edit Mode' : 'Create Mode'}
          </h3>
          <button
            onClick={onClose}
            className="p-1 text-white/40 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg 
                       text-white placeholder-white/40 focus:outline-none focus:border-white/30"
              placeholder="e.g., Deep Research"
              required
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-1">Description</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg 
                       text-white placeholder-white/40 focus:outline-none focus:border-white/30"
              placeholder="Brief description of this mode"
            />
          </div>

          {/* Active Tools */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-2">
              Active Tools
              <span className="text-white/40 font-normal ml-2">(User can use)</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              {AVAILABLE_TOOLS.map(tool => (
                <label
                  key={tool.id}
                  className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors
                            ${activeTools.includes(tool.id) 
                              ? 'bg-green-500/20 border border-green-500/30' 
                              : 'bg-white/5 border border-transparent hover:bg-white/10'}`}
                >
                  <input
                    type="checkbox"
                    checked={activeTools.includes(tool.id)}
                    onChange={() => toggleActiveTool(tool.id)}
                    className="sr-only"
                  />
                  <span className={`text-sm ${activeTools.includes(tool.id) ? 'text-green-400' : 'text-white/60'}`}>
                    {tool.label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Advertised Tools */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-2">
              Advertised to LLM
              <span className="text-white/40 font-normal ml-2">(LLM knows about)</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              {AVAILABLE_TOOLS.filter(tool => activeTools.includes(tool.id)).map(tool => (
                <label
                  key={tool.id}
                  className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors
                            ${advertisedTools.includes(tool.id) 
                              ? 'bg-blue-500/20 border border-blue-500/30' 
                              : 'bg-white/5 border border-transparent hover:bg-white/10'}`}
                >
                  <input
                    type="checkbox"
                    checked={advertisedTools.includes(tool.id)}
                    onChange={() => toggleAdvertisedTool(tool.id)}
                    className="sr-only"
                  />
                  <span className={`text-sm ${advertisedTools.includes(tool.id) ? 'text-blue-400' : 'text-white/60'}`}>
                    {tool.label}
                  </span>
                </label>
              ))}
              {activeTools.length === 0 && (
                <p className="col-span-2 text-sm text-white/40 italic">
                  Select active tools first
                </p>
              )}
            </div>
          </div>

          {/* Enabled */}
          <div className="flex items-center gap-3">
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer 
                            peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full 
                            peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] 
                            after:start-[2px] after:bg-white after:border-white after:border after:rounded-full 
                            after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600" />
            </label>
            <span className="text-sm text-white/80">Enabled</span>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-white/60 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              {mode ? 'Save Changes' : 'Create Mode'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default AssistantModesTab;
