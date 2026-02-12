/**
 * StreamingMessageNew - Streaming message component with imperative rendering
 * 
 * Architecture:
 * - Uses streaming parser as single source of truth
 * - Presentation via direct DOM manipulation (no React state per token)
 * - Artifact capture driven by same parser events
 * - Thinking tokens filtered in real-time
 */

import { memo, useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import { createStreamParser, StreamParser, StreamEvent } from '../lib/streamParser';
import type { Artifact } from '../types';

export interface StreamingMessageRef {
  feed: (chunk: string) => void;
  end: () => void;
  reset: () => void;
  getContent: () => string;
  getArtifacts: () => Artifact[];
}

interface StreamingMessageProps {
  assistantName?: string;
  thinkingTokens?: { begin: string; end: string } | null;
  onArtifactComplete?: (artifact: Artifact) => void;
  onContentChange?: (content: string) => void;
}

// Parser state for thinking token filtering
interface ThinkingState {
  isThinking: boolean;
  buffer: string;
}

function StreamingMessageInner(
  { assistantName, thinkingTokens, onArtifactComplete, onContentChange }: StreamingMessageProps,
  ref: React.Ref<StreamingMessageRef>
) {
  // Refs for DOM elements
  const containerRef = useRef<HTMLDivElement>(null);
  const textNodeRef = useRef<Text | null>(null);
  const codeBlockRef = useRef<HTMLPreElement | null>(null);
  const codeContentRef = useRef<HTMLElement | null>(null);
  
  // State refs (not React state - no re-renders!)
  const parserRef = useRef<StreamParser | null>(null);
  const artifactsRef = useRef<Artifact[]>([]);
  const pendingArtifactRef = useRef<{
    tag: string;
    attributes: Record<string, string>;
    content: string;
  } | null>(null);
  const rawContentRef = useRef<string>('');
  const thinkingStateRef = useRef<ThinkingState>({ isThinking: false, buffer: '' });
  const isStreamingRef = useRef(false);

  // Create code block element
  const createCodeBlock = useCallback((tag: string, attrs: Record<string, string>) => {
    const pre = document.createElement('pre');
    pre.className = 'rounded-lg bg-zinc-900 border border-blue-500 p-3 my-2 overflow-x-auto text-sm';
    pre.setAttribute('data-tag', tag);
    
    // Header
    const header = document.createElement('div');
    header.className = 'flex items-center gap-2 mb-2 text-xs text-zinc-400';
    
    const label = document.createElement('span');
    label.className = 'px-2 py-0.5 rounded bg-zinc-800';
    label.textContent = attrs.title || attrs.filename || attrs.name || tag;
    header.appendChild(label);
    
    const indicator = document.createElement('span');
    indicator.className = 'w-2 h-2 rounded-full bg-blue-500 animate-pulse';
    indicator.setAttribute('data-indicator', 'true');
    header.appendChild(indicator);
    
    pre.appendChild(header);
    
    // Code content
    const code = document.createElement('code');
    code.className = 'text-zinc-100 whitespace-pre-wrap break-words';
    pre.appendChild(code);
    
    return pre;
  }, []);

  // Finalize code block
  const finalizeCodeBlock = useCallback((pre: HTMLPreElement) => {
    const indicator = pre.querySelector('[data-indicator]');
    if (indicator) indicator.remove();
    pre.classList.remove('border-blue-500');
    pre.classList.add('border-green-500/50');
  }, []);

  // Append text to container
  const appendText = useCallback((text: string) => {
    if (!containerRef.current) return;
    
    // Apply thinking token filtering
    if (thinkingTokens?.begin && thinkingTokens?.end) {
      const state = thinkingStateRef.current;
      state.buffer += text;
      
      let output = '';
      let remaining = state.buffer;
      
      while (remaining) {
        if (state.isThinking) {
          // Look for end token
          const endIdx = remaining.indexOf(thinkingTokens.end);
          if (endIdx === -1) {
            // Still thinking, consume all
            remaining = '';
          } else {
            // End thinking
            state.isThinking = false;
            remaining = remaining.slice(endIdx + thinkingTokens.end.length);
          }
        } else {
          // Look for begin token
          const beginIdx = remaining.indexOf(thinkingTokens.begin);
          if (beginIdx === -1) {
            // No thinking, output everything except last few chars (in case token is split)
            const safe = remaining.length > 50 ? remaining.slice(0, -50) : '';
            output += safe;
            remaining = remaining.slice(safe.length);
            break;
          } else {
            // Output before thinking, enter thinking state
            output += remaining.slice(0, beginIdx);
            state.isThinking = true;
            remaining = remaining.slice(beginIdx + thinkingTokens.begin.length);
          }
        }
      }
      
      state.buffer = remaining;
      text = output;
    }
    
    if (!text) return;

    // End code block if we're appending text outside it
    if (codeBlockRef.current) {
      codeBlockRef.current = null;
      codeContentRef.current = null;
    }
    
    // Get or create text node
    if (!textNodeRef.current) {
      const span = document.createElement('span');
      span.className = 'whitespace-pre-wrap';
      textNodeRef.current = document.createTextNode('');
      span.appendChild(textNodeRef.current);
      containerRef.current.appendChild(span);
    }
    
    // Append directly to DOM (imperative!)
    textNodeRef.current.textContent += text;
    rawContentRef.current += text;
    
    onContentChange?.(rawContentRef.current);
  }, [thinkingTokens, onContentChange]);

  // Handle parser events
  const handleEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case 'text-delta':
        appendText(event.text);
        break;
        
      case 'tag-open': {
        // End current text node
        textNodeRef.current = null;
        
        // Skip thinking tags in visual output
        if (event.tag === 'thinking') {
          thinkingStateRef.current.isThinking = true;
          break;
        }
        
        // Create code block
        if (containerRef.current) {
          const pre = createCodeBlock(event.tag, event.attributes);
          containerRef.current.appendChild(pre);
          codeBlockRef.current = pre;
          codeContentRef.current = pre.querySelector('code');
        }
        
        // Start artifact capture
        pendingArtifactRef.current = {
          tag: event.tag,
          attributes: event.attributes,
          content: '',
        };
        
        // Add to raw content
        rawContentRef.current += event.rawOpenTag;
        break;
      }
        
      case 'tag-delta': {
        // Skip thinking tag content
        if (event.tag === 'thinking') break;
        
        // Append to code block
        if (codeContentRef.current) {
          codeContentRef.current.textContent += event.delta;
        }
        
        // Update pending artifact
        if (pendingArtifactRef.current) {
          pendingArtifactRef.current.content = event.accumulated;
        }
        
        rawContentRef.current += event.delta;
        break;
      }
        
      case 'tag-close': {
        // Handle thinking tag
        if (event.tag === 'thinking') {
          thinkingStateRef.current.isThinking = false;
          break;
        }
        
        // Finalize code block
        if (codeBlockRef.current) {
          finalizeCodeBlock(codeBlockRef.current);
          codeBlockRef.current = null;
          codeContentRef.current = null;
        }
        
        // Complete artifact
        if (pendingArtifactRef.current) {
          const attrs = event.attributes;
          const artifact: Artifact = {
            id: `artifact-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            title: attrs.title || attrs.filename || attrs.name || event.tag,
            type: (attrs.type as Artifact['type']) || 'code',
            content: event.content,
            language: attrs.language,
            filename: attrs.filename || attrs.name || attrs.path,
            created_at: new Date().toISOString(),
          };
          
          artifactsRef.current.push(artifact);
          onArtifactComplete?.(artifact);
          pendingArtifactRef.current = null;
        }
        
        rawContentRef.current += `</${event.tag}>`;
        onContentChange?.(rawContentRef.current);
        break;
      }
    }
  }, [appendText, createCodeBlock, finalizeCodeBlock, onArtifactComplete, onContentChange]);

  // Initialize parser
  useEffect(() => {
    parserRef.current = createStreamParser(handleEvent);
    return () => {
      parserRef.current = null;
    };
  }, [handleEvent]);

  // Expose methods via ref
  useImperativeHandle(ref, () => ({
    feed: (chunk: string) => {
      if (!isStreamingRef.current) {
        isStreamingRef.current = true;
      }
      parserRef.current?.feed(chunk);
    },
    
    end: () => {
      parserRef.current?.flush();
      isStreamingRef.current = false;
      
      // Flush any remaining thinking buffer
      if (thinkingStateRef.current.buffer && !thinkingStateRef.current.isThinking) {
        appendText(thinkingStateRef.current.buffer);
        thinkingStateRef.current.buffer = '';
      }
    },
    
    reset: () => {
      parserRef.current?.reset();
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
      textNodeRef.current = null;
      codeBlockRef.current = null;
      codeContentRef.current = null;
      artifactsRef.current = [];
      pendingArtifactRef.current = null;
      rawContentRef.current = '';
      thinkingStateRef.current = { isThinking: false, buffer: '' };
      isStreamingRef.current = false;
    },
    
    getContent: () => rawContentRef.current,
    
    getArtifacts: () => [...artifactsRef.current],
  }), [appendText]);

  return (
    <div className="py-4">
      <div className="max-w-3xl mx-auto px-3 md:px-4">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm md:text-xs font-medium uppercase tracking-wide text-[var(--color-secondary)]">
            {assistantName || 'Assistant'}
          </span>
        </div>
        
        {/* Streaming content container - DOM manipulation target */}
        <div 
          ref={containerRef}
          className="prose prose-base md:prose-sm max-w-none prose-neutral dark:prose-invert text-[var(--color-text)]"
        />
        
        {/* Cursor indicator (always visible during streaming) */}
        <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse ml-0.5 align-middle" />
      </div>
    </div>
  );
}

export const StreamingMessageNew = memo(forwardRef(StreamingMessageInner));
