/**
 * Streaming Renderer - Imperative DOM updates for streaming content
 * 
 * Features:
 * - Append-only text rendering (no re-renders per token)
 * - Code block containers for registered tags
 * - Direct DOM manipulation via refs
 * - Separate from React state to avoid render thrashing
 */

import { createStreamParser, StreamEvent, StreamParser } from './streamParser';
import type { Artifact } from '../types';

// Artifact capture state
interface CapturedArtifact {
  tag: string;
  attributes: Record<string, string>;
  content: string;
  startTime: number;
}

// Renderer state
interface RendererState {
  parser: StreamParser;
  containerRef: HTMLDivElement | null;
  currentTextNode: Text | null;
  currentCodeBlock: HTMLPreElement | null;
  currentCodeContent: HTMLElement | null;
  capturedArtifacts: Artifact[];
  pendingArtifact: CapturedArtifact | null;
  isStreaming: boolean;
  onArtifactComplete?: (artifact: Artifact) => void;
}

/**
 * Create a streaming content renderer
 * Uses direct DOM manipulation for performance
 */
export function createStreamingRenderer(options: {
  onArtifactComplete?: (artifact: Artifact) => void;
}) {
  const state: RendererState = {
    parser: null as any, // Will be set below
    containerRef: null,
    currentTextNode: null,
    currentCodeBlock: null,
    currentCodeContent: null,
    capturedArtifacts: [],
    pendingArtifact: null,
    isStreaming: false,
    onArtifactComplete: options.onArtifactComplete,
  };

  // Convert tag attributes to artifact metadata
  function attributesToArtifactMeta(tag: string, attrs: Record<string, string>): Partial<Artifact> {
    const result: Partial<Artifact> = {
      type: 'code',
    };

    // Handle different attribute patterns
    if (attrs.title) result.title = attrs.title;
    if (attrs.type) result.type = attrs.type as Artifact['type'];
    if (attrs.language) result.language = attrs.language;
    if (attrs.filename) result.filename = attrs.filename;
    if (attrs.name) result.filename = attrs.name;
    if (attrs.path) result.filename = attrs.path;

    // Handle artifact=filename pattern
    if (attrs['']) {
      result.filename = attrs[''];
    }

    // Infer from tag name if needed
    if (tag === 'thinking') {
      result.type = 'code'; // thinking is rendered as code block
      result.title = result.title || 'Thinking';
    }

    return result;
  }

  // Create a code block element
  function createCodeBlock(tag: string, attrs: Record<string, string>): HTMLPreElement {
    const pre = document.createElement('pre');
    pre.className = 'streaming-code-block rounded-lg bg-zinc-900 p-4 my-2 overflow-x-auto';
    pre.setAttribute('data-tag', tag);
    
    // Add header with tag info
    const header = document.createElement('div');
    header.className = 'text-xs text-zinc-400 mb-2 flex items-center gap-2';
    
    const tagLabel = document.createElement('span');
    tagLabel.className = 'px-2 py-0.5 rounded bg-zinc-800';
    tagLabel.textContent = attrs.title || attrs.filename || attrs.name || tag;
    header.appendChild(tagLabel);
    
    // Add streaming indicator
    const indicator = document.createElement('span');
    indicator.className = 'streaming-indicator w-2 h-2 rounded-full bg-blue-500 animate-pulse';
    header.appendChild(indicator);
    
    pre.appendChild(header);
    
    // Code content area
    const code = document.createElement('code');
    code.className = 'text-sm text-zinc-100 whitespace-pre-wrap break-words';
    if (attrs.language) {
      code.className += ` language-${attrs.language}`;
    }
    pre.appendChild(code);
    
    return pre;
  }

  // Finalize code block (remove streaming indicator)
  function finalizeCodeBlock(pre: HTMLPreElement): void {
    const indicator = pre.querySelector('.streaming-indicator');
    if (indicator) {
      indicator.remove();
    }
    // Add "completed" styling
    pre.classList.add('streaming-complete');
  }

  // Handle parser events
  function handleEvent(event: StreamEvent): void {
    if (!state.containerRef) return;

    switch (event.type) {
      case 'text-delta': {
        // Append text to current text node or create new one
        if (state.currentCodeBlock) {
          // We're inside a code block but got text - shouldn't happen often
          // Just add to container after code block
          state.currentCodeBlock = null;
          state.currentCodeContent = null;
        }
        
        if (!state.currentTextNode) {
          // Create a new text container
          const span = document.createElement('span');
          span.className = 'streaming-text';
          state.currentTextNode = document.createTextNode('');
          span.appendChild(state.currentTextNode);
          state.containerRef.appendChild(span);
        }
        
        // Append text directly to DOM node (no React re-render!)
        state.currentTextNode.textContent += event.text;
        break;
      }

      case 'tag-open': {
        // End current text node
        state.currentTextNode = null;
        
        // Create code block for this tag
        const pre = createCodeBlock(event.tag, event.attributes);
        state.containerRef.appendChild(pre);
        state.currentCodeBlock = pre;
        state.currentCodeContent = pre.querySelector('code');
        
        // Start artifact capture
        state.pendingArtifact = {
          tag: event.tag,
          attributes: event.attributes,
          content: '',
          startTime: Date.now(),
        };
        break;
      }

      case 'tag-delta': {
        // Append to code block content
        if (state.currentCodeContent) {
          state.currentCodeContent.textContent += event.delta;
        }
        
        // Update pending artifact
        if (state.pendingArtifact) {
          state.pendingArtifact.content = event.accumulated;
        }
        break;
      }

      case 'tag-close': {
        // Finalize code block
        if (state.currentCodeBlock) {
          finalizeCodeBlock(state.currentCodeBlock);
        }
        state.currentCodeBlock = null;
        state.currentCodeContent = null;
        
        // Complete artifact capture
        if (state.pendingArtifact) {
          const meta = attributesToArtifactMeta(event.tag, event.attributes);
          const artifact: Artifact = {
            id: `artifact-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            title: meta.title || event.tag,
            type: meta.type || 'code',
            content: event.content,
            language: meta.language,
            filename: meta.filename,
            created_at: new Date().toISOString(),
          };
          
          state.capturedArtifacts.push(artifact);
          
          // Notify callback
          if (state.onArtifactComplete) {
            state.onArtifactComplete(artifact);
          }
          
          state.pendingArtifact = null;
        }
        break;
      }
    }
  }

  // Create parser with event handler
  state.parser = createStreamParser(handleEvent);

  return {
    /**
     * Set the container element for rendering
     */
    setContainer(container: HTMLDivElement | null): void {
      state.containerRef = container;
    },

    /**
     * Feed a chunk of streaming content
     */
    feed(chunk: string): void {
      if (!state.isStreaming) {
        state.isStreaming = true;
      }
      state.parser.feed(chunk);
    },

    /**
     * End the stream and flush remaining content
     */
    end(): void {
      state.parser.flush();
      state.isStreaming = false;
      state.currentTextNode = null;
      state.currentCodeBlock = null;
      state.currentCodeContent = null;
    },

    /**
     * Reset renderer for new stream
     */
    reset(): void {
      state.parser.reset();
      state.containerRef = null;
      state.currentTextNode = null;
      state.currentCodeBlock = null;
      state.currentCodeContent = null;
      state.capturedArtifacts = [];
      state.pendingArtifact = null;
      state.isStreaming = false;
    },

    /**
     * Clear the container content
     */
    clear(): void {
      if (state.containerRef) {
        state.containerRef.innerHTML = '';
      }
      state.currentTextNode = null;
      state.currentCodeBlock = null;
      state.currentCodeContent = null;
    },

    /**
     * Get captured artifacts
     */
    getArtifacts(): Artifact[] {
      return [...state.capturedArtifacts];
    },

    /**
     * Get raw accumulated content (for final message save)
     */
    getRawContent(): string {
      if (!state.containerRef) return '';
      
      // Extract text content from rendered DOM
      // This preserves the original format including tags
      let content = '';
      const nodes = state.containerRef.childNodes;
      
      for (const node of nodes) {
        if (node.nodeType === Node.TEXT_NODE) {
          content += node.textContent || '';
        } else if (node instanceof HTMLElement) {
          if (node.classList.contains('streaming-text')) {
            content += node.textContent || '';
          } else if (node.classList.contains('streaming-code-block')) {
            const tag = node.getAttribute('data-tag') || 'artifact';
            const code = node.querySelector('code');
            const codeContent = code?.textContent || '';
            // Reconstruct original tag format
            content += `<${tag}>${codeContent}</${tag}>`;
          }
        }
      }
      
      return content;
    },

    /**
     * Check if currently streaming
     */
    isStreaming(): boolean {
      return state.isStreaming;
    },

    /**
     * Check if inside a registered tag
     */
    isInsideTag(): boolean {
      return state.parser.isInsideTag();
    },

    /**
     * Get current tag if inside one
     */
    getCurrentTag(): string | null {
      return state.parser.getCurrentTag();
    },
  };
}

// Type for the renderer instance
export type StreamingRenderer = ReturnType<typeof createStreamingRenderer>;

/**
 * CSS styles for streaming content (add to global CSS or inject)
 */
export const STREAMING_STYLES = `
.streaming-code-block {
  transition: border-color 0.3s ease;
  border: 1px solid transparent;
}

.streaming-code-block:not(.streaming-complete) {
  border-color: rgb(59, 130, 246);
}

.streaming-code-block.streaming-complete {
  border-color: rgb(34, 197, 94);
}

.streaming-indicator {
  animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
`;
