/**
 * TextSelectionBubble - NC-0.8.0.0
 * 
 * Popup menu that appears when user selects text in chat messages.
 * Shows available user_hint actions from filter chains.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { UserHint } from '../types';

// Default hints (can be overridden by filter chains)
const DEFAULT_HINTS: UserHint[] = [
  {
    type: 'user_hint',
    label: 'Explain',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
    location: 'response',
    prompt: 'Can you explain what this means?\n\n{$Selected}',
  },
  {
    type: 'user_hint',
    label: 'Tell me more',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
    location: 'response',
    prompt: 'Elaborate with more details on the following:\n\n{$Selected}',
  },
  {
    type: 'user_hint',
    label: 'Summarize',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="21" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="21" y1="18" x2="3" y2="18"/></svg>`,
    location: 'response',
    prompt: 'Summarize the following concisely:\n\n{$Selected}',
  },
];

interface TextSelectionBubbleProps {
  containerRef: React.RefObject<HTMLElement | null>;
  hints?: UserHint[];
  messageRole?: 'user' | 'assistant';
  onHintSelect: (hint: UserHint, selectedText: string) => void;
}

export function TextSelectionBubble({ 
  containerRef, 
  hints = DEFAULT_HINTS,
  messageRole = 'assistant',
  onHintSelect 
}: TextSelectionBubbleProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [selectedText, setSelectedText] = useState('');
  const bubbleRef = useRef<HTMLDivElement>(null);

  // Filter hints by location
  const filteredHints = hints.filter(hint => {
    if (hint.location === 'both') return true;
    if (hint.location === 'response' && messageRole === 'assistant') return true;
    if (hint.location === 'query' && messageRole === 'user') return true;
    return false;
  });

  const handleSelection = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !containerRef.current) {
      setIsVisible(false);
      return;
    }

    const text = selection.toString().trim();
    if (!text || text.length < 2) {
      setIsVisible(false);
      return;
    }

    // Check if selection is within our container
    const range = selection.getRangeAt(0);
    const container = containerRef.current;
    
    if (!container.contains(range.commonAncestorContainer)) {
      setIsVisible(false);
      return;
    }

    // Get selection position
    const rect = range.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    
    // Position bubble above selection, centered
    const x = rect.left + rect.width / 2 - containerRect.left;
    const y = rect.top - containerRect.top - 8;

    setSelectedText(text);
    setPosition({ x, y });
    setIsVisible(true);
  }, [containerRef]);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (bubbleRef.current && !bubbleRef.current.contains(e.target as Node)) {
      setIsVisible(false);
    }
  }, []);

  useEffect(() => {
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('mousedown', handleClickOutside);
    
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleSelection, handleClickOutside]);

  const handleHintClick = (hint: UserHint) => {
    onHintSelect(hint, selectedText);
    setIsVisible(false);
    // Clear selection
    window.getSelection()?.removeAllRanges();
  };

  if (!isVisible || filteredHints.length === 0) {
    return null;
  }

  return (
    <div
      ref={bubbleRef}
      className="absolute z-50 transform -translate-x-1/2 -translate-y-full animate-in fade-in slide-in-from-bottom-2 duration-150"
      style={{ 
        left: position.x, 
        top: position.y,
      }}
    >
      <div className="flex items-center gap-0.5 px-1 py-1 bg-[#2a2a2a] border border-white/10 rounded-lg shadow-xl">
        {filteredHints.map((hint, index) => (
          <button
            key={index}
            onClick={() => handleHintClick(hint)}
            className="group flex items-center gap-1.5 px-2 py-1.5 text-sm text-white/70 
                     hover:text-white hover:bg-white/10 rounded transition-colors"
            title={hint.label}
          >
            {hint.icon && (
              <div 
                className="w-4 h-4 opacity-70 group-hover:opacity-100"
                dangerouslySetInnerHTML={{ __html: hint.icon }}
              />
            )}
            <span className="text-xs">{hint.label}</span>
          </button>
        ))}
      </div>
      
      {/* Arrow pointing down */}
      <div className="absolute left-1/2 -translate-x-1/2 top-full">
        <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] 
                      border-r-transparent border-t-[6px] border-t-[#2a2a2a]" />
      </div>
    </div>
  );
}

export default TextSelectionBubble;
