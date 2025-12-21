import { useState, useRef, useEffect } from 'react';
import { useBrandingStore } from '../stores/brandingStore';
import { useModelsStore } from '../stores/modelsStore';

interface EmptyStateProps {
  onSendMessage: (prompt: string) => void;
  selectedModel?: string;
  onModelChange?: (modelId: string) => void;
}

export default function EmptyState({ onSendMessage, selectedModel, onModelChange }: EmptyStateProps) {
  const { config } = useBrandingStore();
  const { models, subscribedAssistants, defaultModel, getDisplayName } = useModelsStore();
  const [showModelSelector, setShowModelSelector] = useState(false);
  const modelSelectorRef = useRef<HTMLDivElement>(null);
  
  const currentModel = selectedModel || defaultModel;
  const appName = config?.app_name || 'Chat';
  const welcomeTitle = config?.welcome?.title || `Welcome to ${appName}`;
  const welcomeMessage = config?.welcome?.message || 'Your AI-powered assistant is ready to help.';
  
  // Close selector when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modelSelectorRef.current && !modelSelectorRef.current.contains(e.target as Node)) {
        setShowModelSelector(false);
      }
    };
    if (showModelSelector) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showModelSelector]);
  
  const handleModelSelect = (modelId: string) => {
    console.log('[EmptyState] Model selected:', modelId, 'isAssistant:', modelId.startsWith('gpt:'));
    if (onModelChange) {
      onModelChange(modelId);
    }
    setShowModelSelector(false);
  };
  
  const suggestions = [
    {
      icon: '◇',
      title: 'Brainstorm ideas',
      prompt: 'Help me brainstorm creative ideas for a new project.',
    },
    {
      icon: '✎',
      title: 'Write content',
      prompt: 'Help me write a professional email.',
    },
    {
      icon: '⚙',
      title: 'Debug code',
      prompt: 'Help me debug this code and explain the issue.',
    },
    {
      icon: '▤',
      title: 'Analyze data',
      prompt: 'Help me analyze and understand this data.',
    },
  ];
  
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-4 md:p-8">
      {/* Model selector */}
      {(models.length > 0 || subscribedAssistants.length > 0) && (
        <div className="mb-6 relative" ref={modelSelectorRef}>
          <button
            onClick={() => setShowModelSelector(!showModelSelector)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)] transition-colors"
          >
            <span className="text-sm">Model:</span>
            <span className="font-medium text-sm">{getDisplayName(currentModel)}</span>
            <svg className={`w-4 h-4 transition-transform ${showModelSelector ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {showModelSelector && (
            <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-64 max-h-80 overflow-y-auto py-1 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] shadow-xl z-50">
              {/* LLM Models */}
              {models.length > 0 && (
                <>
                  <div className="px-3 py-1.5 text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">
                    Models
                  </div>
                  {models.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => handleModelSelect(model.id)}
                      className={`w-full px-3 py-2 text-left text-sm hover:bg-[var(--color-background)] flex items-center justify-between ${
                        currentModel === model.id ? 'text-[var(--color-primary)]' : 'text-[var(--color-text)]'
                      }`}
                    >
                      <span className="truncate">{getDisplayName(model.id)}</span>
                      {currentModel === model.id && (
                        <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                  ))}
                </>
              )}
              
              {/* Subscribed Custom GPTs */}
              {subscribedAssistants.length > 0 && (
                <>
                  <div className="border-t border-[var(--color-border)] my-1" />
                  <div className="px-3 py-1.5 text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">
                    Custom GPTs
                  </div>
                  {subscribedAssistants.map((assistant) => (
                    <button
                      key={assistant.id}
                      onClick={() => handleModelSelect(assistant.id)}
                      className={`w-full px-3 py-2 text-left text-sm hover:bg-[var(--color-background)] flex items-center justify-between ${
                        currentModel === assistant.id ? 'text-[var(--color-primary)]' : 'text-[var(--color-text)]'
                      }`}
                    >
                      <span className="flex items-center gap-2 truncate">
                        <span>{assistant.icon || '◈'}</span>
                        <span className="truncate">{assistant.name}</span>
                      </span>
                      {currentModel === assistant.id && (
                        <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* Hero section */}
      <div className="text-center mb-6 md:mb-10">
        <h1 className="text-xl md:text-2xl font-semibold text-[var(--color-text)] mb-2">
          {welcomeTitle}
        </h1>
        <p className="text-base md:text-sm text-[var(--color-text-secondary)] max-w-md mx-auto">
          {welcomeMessage}
        </p>
      </div>
      
      {/* Suggestion prompts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-xl">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSendMessage(suggestion.prompt)}
            className="group p-4 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] hover:border-[var(--color-primary)]/50 active:bg-[var(--color-background)] transition-all text-left"
          >
            <div className="flex items-start gap-3">
              <span className="text-2xl md:text-xl">{suggestion.icon}</span>
              <div>
                <h3 className="font-medium text-[var(--color-text)] text-base md:text-sm group-hover:text-[var(--color-primary)] transition-colors">
                  {suggestion.title}
                </h3>
                <p className="text-sm md:text-xs text-[var(--color-text-secondary)] mt-0.5 line-clamp-2">
                  {suggestion.prompt}
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
