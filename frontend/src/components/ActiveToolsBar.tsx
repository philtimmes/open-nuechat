/**
 * ActiveToolsBar - NC-0.8.0.0
 * 
 * Visual toolbar showing which tools are active in the current chat.
 * - Line-based SVG icons
 * - Active: Glow effect (brighter, same hue as background)
 * - Inactive: No glow (muted but visible)
 * - Hover: Subtle brightness + tooltip
 * - Mode dropdown for quick switching
 */

import { useState, useEffect } from 'react';
import { assistantModeApi, chatToolsApi } from '../lib/api';
import type { AssistantMode, ToolDefinition } from '../types';

// Default tool definitions with SVG icons
const DEFAULT_TOOLS: ToolDefinition[] = [
  {
    id: 'web_search',
    name: 'web_search',
    label: 'Web Search',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>`,
    category: 'mode',
    description: 'Search the internet for current information',
  },
  {
    id: 'artifacts',
    name: 'artifacts',
    label: 'Artifacts',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>`,
    category: 'mode',
    description: 'Create and edit documents, code, and files',
  },
  {
    id: 'image_gen',
    name: 'image_gen',
    label: 'Image Generation',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
    category: 'mode',
    description: 'Generate images from text descriptions',
  },
  {
    id: 'code_exec',
    name: 'code_exec',
    label: 'Code Execution',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
    category: 'mode',
    description: 'Execute Python code in a sandbox',
  },
  {
    id: 'file_ops',
    name: 'file_ops',
    label: 'File Operations',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
    category: 'mode',
    description: 'Read and write files',
  },
  {
    id: 'kb_search',
    name: 'kb_search',
    label: 'Knowledge Base',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
    category: 'mode',
    description: 'Search assistant knowledge bases',
  },
  {
    id: 'local_rag',
    name: 'local_rag',
    label: 'Chat Documents',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
    category: 'mode',
    description: 'Search documents uploaded to this chat',
  },
  {
    id: 'user_chats_kb',
    name: 'user_chats_kb',
    label: 'Chat History',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
    category: 'mode',
    description: 'Search your indexed chat history',
  },
  {
    id: 'citations',
    name: 'citations',
    label: 'Citations',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3z"/></svg>`,
    category: 'mode',
    description: 'Add citations and references',
  },
];

interface ActiveToolsBarProps {
  chatId: string;
  className?: string;
  compact?: boolean;
}

export function ActiveToolsBar({ chatId, className = '', compact = false }: ActiveToolsBarProps) {
  const [modes, setModes] = useState<AssistantMode[]>([]);
  const [currentMode, setCurrentMode] = useState<AssistantMode | null>(null);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [isCustom, setIsCustom] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showModeDropdown, setShowModeDropdown] = useState(false);

  // Load modes on mount
  useEffect(() => {
    loadModes();
  }, []);

  // Load active tools when chat changes OR when modes finish loading
  useEffect(() => {
    if (chatId) {
      if (modes.length > 0) {
        loadChatTools();
      } else {
        // If modes aren't available yet, still set loading to false
        // so the toolbar renders with default state
        setIsLoading(false);
      }
    }
  }, [chatId, modes.length]);

  const loadModes = async () => {
    try {
      const response = await assistantModeApi.list();
      setModes(response.data);
    } catch (err) {
      console.error('Failed to load assistant modes:', err);
      setIsLoading(false); // Still show toolbar even if modes fail
    }
  };

  const loadChatTools = async () => {
    setIsLoading(true);
    try {
      const response = await chatToolsApi.getActiveTools(chatId);
      const { mode_id, active_tools } = response.data;
      
      if (mode_id) {
        const mode = modes.find(m => m.id === mode_id);
        if (mode) {
          setCurrentMode(mode);
          // Check if user has customized tools
          if (active_tools && JSON.stringify([...active_tools].sort()) !== JSON.stringify([...(mode.active_tools || [])].sort())) {
            setIsCustom(true);
            setActiveTools(active_tools);
          } else {
            setActiveTools(mode.active_tools || []);
          }
        } else {
          // Mode not found, just use the active_tools
          setActiveTools(active_tools || []);
        }
      } else if (active_tools && active_tools.length > 0) {
        setActiveTools(active_tools);
        setIsCustom(true);
      } else {
        // No mode and no tools - set defaults from General mode
        const generalMode = modes.find(m => m.name === 'General');
        if (generalMode) {
          setCurrentMode(generalMode);
          setActiveTools(generalMode.active_tools || []);
        }
      }
    } catch (err) {
      // Chat might not have tools set yet, use defaults
      console.debug('No tools configured for chat, using defaults');
      const generalMode = modes.find(m => m.name === 'General');
      if (generalMode) {
        setCurrentMode(generalMode);
        setActiveTools(generalMode.active_tools || []);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleModeChange = async (mode: AssistantMode) => {
    try {
      await chatToolsApi.updateMode(chatId, mode.id);
      setCurrentMode(mode);
      setActiveTools(mode.active_tools || []);
      setIsCustom(false);
      setShowModeDropdown(false);
    } catch (err) {
      console.error('Failed to update mode:', err);
    }
  };

  const handleToolToggle = async (toolId: string) => {
    const newTools = activeTools.includes(toolId)
      ? activeTools.filter(t => t !== toolId)
      : [...activeTools, toolId];
    
    try {
      await chatToolsApi.updateActiveTools(chatId, newTools);
      setActiveTools(newTools);
      
      // Check if this matches current mode or is custom
      if (currentMode) {
        const modeTools = (currentMode.active_tools || []).sort();
        const newToolsSorted = newTools.sort();
        setIsCustom(JSON.stringify(modeTools) !== JSON.stringify(newToolsSorted));
      } else {
        setIsCustom(true);
      }
    } catch (err) {
      console.error('Failed to update tools:', err);
    }
  };

  if (isLoading) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="h-6 w-32 bg-white/5 animate-pulse rounded" />
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Mode Dropdown */}
      <div className="relative">
        <button
          onClick={() => setShowModeDropdown(!showModeDropdown)}
          className="flex items-center gap-1.5 px-2 py-1 text-xs text-white/60 hover:text-white/80 
                     bg-white/5 hover:bg-white/10 rounded transition-colors"
        >
          <span>{isCustom ? 'Custom' : currentMode?.name || 'General'}</span>
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
        
        {showModeDropdown && (
          <>
            <div 
              className="fixed inset-0 z-40" 
              onClick={() => setShowModeDropdown(false)} 
            />
            <div className="absolute top-full left-0 mt-1 z-50 min-w-[160px] py-1 
                          bg-[#2a2a2a] border border-white/10 rounded-lg shadow-xl">
              {modes.map(mode => (
                <button
                  key={mode.id}
                  onClick={() => handleModeChange(mode)}
                  className={`w-full px-3 py-2 text-left text-sm hover:bg-white/5 transition-colors
                             ${currentMode?.id === mode.id && !isCustom ? 'text-white bg-white/5' : 'text-white/70'}`}
                >
                  <div className="font-medium">{mode.name}</div>
                  {mode.description && (
                    <div className="text-xs text-white/40 mt-0.5">{mode.description}</div>
                  )}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Tool Icons */}
      <div className="flex items-center gap-1">
        {DEFAULT_TOOLS.map(tool => {
          const isActive = activeTools.includes(tool.id);
          
          return (
            <button
              key={tool.id}
              onClick={() => handleToolToggle(tool.id)}
              className={`group relative p-1.5 rounded transition-all duration-150
                         ${isActive 
                           ? 'text-gray-300' 
                           : 'text-gray-500 hover:text-gray-400'}`}
              style={isActive ? {
                filter: 'drop-shadow(0 0 8px rgba(59, 130, 246, 0.6))',
              } : undefined}
              title={tool.label}
            >
              <div 
                className="w-4 h-4"
                dangerouslySetInnerHTML={{ __html: tool.icon }}
              />
              
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 
                            bg-black/90 text-white text-xs rounded whitespace-nowrap
                            opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                {tool.label}
                {tool.description && (
                  <div className="text-white/60 text-[10px] mt-0.5">{tool.description}</div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default ActiveToolsBar;
