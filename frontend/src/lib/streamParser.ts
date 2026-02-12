/**
 * Streaming Parser - Single source of truth for stream processing
 * 
 * Emits events:
 * - text-delta: Plain text outside registered tags
 * - tag-open: Opening of a registered tag (e.g., <artifact>)
 * - tag-delta: Content inside a registered tag
 * - tag-close: Closing of a registered tag with full captured content
 * 
 * Architecture:
 * - Parser maintains state for whether inside a tag
 * - Does NOT use regex-only parsing
 * - Does NOT assume tags arrive in single chunk
 * - Presentation and protocol are consumers of parser events
 */

// Registered tags that define structural boundaries
export const REGISTERED_TAGS = [
  'artifact',
  'antartifact',  // Common misspelling
  'xaiArtifact',
  'thinking',
  'search_replace',
  'replace_block',
  'find_line',
  'find',
  'request_file',
  'kb_search',
] as const;

export type RegisteredTag = typeof REGISTERED_TAGS[number];

// Event types emitted by the parser
export interface TextDeltaEvent {
  type: 'text-delta';
  text: string;
}

export interface TagOpenEvent {
  type: 'tag-open';
  tag: string;
  attributes: Record<string, string>;
  rawOpenTag: string;
}

export interface TagDeltaEvent {
  type: 'tag-delta';
  tag: string;
  delta: string;
  accumulated: string;
}

export interface TagCloseEvent {
  type: 'tag-close';
  tag: string;
  content: string;
  attributes: Record<string, string>;
}

export type StreamEvent = TextDeltaEvent | TagOpenEvent | TagDeltaEvent | TagCloseEvent;

export type StreamEventHandler = (event: StreamEvent) => void;

// Parser state
interface ParserState {
  buffer: string;
  insideTag: string | null;
  tagAttributes: Record<string, string>;
  tagContent: string;
  rawOpenTag: string;
}

/**
 * Create a streaming parser instance
 * Returns functions to feed chunks and reset state
 */
export function createStreamParser(onEvent: StreamEventHandler) {
  const state: ParserState = {
    buffer: '',
    insideTag: null,
    tagAttributes: {},
    tagContent: '',
    rawOpenTag: '',
  };

  // Check if a string is a registered tag
  function isRegisteredTag(tagName: string): boolean {
    return REGISTERED_TAGS.includes(tagName.toLowerCase() as RegisteredTag);
  }

  // Parse attributes from opening tag
  function parseAttributes(tagStr: string): Record<string, string> {
    const attrs: Record<string, string> = {};
    // Match attr="value" or attr='value' or attr=value
    const attrRegex = /(\w+)=(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g;
    let match;
    while ((match = attrRegex.exec(tagStr)) !== null) {
      const [, name, doubleQuoted, singleQuoted, unquoted] = match;
      attrs[name] = doubleQuoted ?? singleQuoted ?? unquoted ?? '';
    }
    return attrs;
  }

  // Find potential tag start (< followed by registered tag name or /)
  function findPotentialTagStart(text: string, startIndex: number): number {
    for (let i = startIndex; i < text.length; i++) {
      if (text[i] === '<') {
        // Check if this could be a registered tag or closing tag
        const remaining = text.slice(i + 1);
        
        // Check for closing tag
        if (remaining.startsWith('/')) {
          const closeTagMatch = remaining.match(/^\/(\w+)/);
          if (closeTagMatch && isRegisteredTag(closeTagMatch[1])) {
            return i;
          }
        }
        
        // Check for opening tag
        const openTagMatch = remaining.match(/^(\w+)/);
        if (openTagMatch && isRegisteredTag(openTagMatch[1])) {
          return i;
        }
      }
    }
    return -1;
  }

  // Process buffer when outside any tag
  function processOutsideTag(): void {
    const buffer = state.buffer;
    if (!buffer) return;

    // Look for opening tag of registered type
    const tagStartIndex = findPotentialTagStart(buffer, 0);

    if (tagStartIndex === -1) {
      // No potential tags found - emit all but last few chars as text
      // Keep last 50 chars in buffer in case tag is split across chunks
      const safeLength = Math.max(0, buffer.length - 50);
      if (safeLength > 0) {
        onEvent({ type: 'text-delta', text: buffer.slice(0, safeLength) });
        state.buffer = buffer.slice(safeLength);
      }
      return;
    }

    // Emit text before potential tag
    if (tagStartIndex > 0) {
      onEvent({ type: 'text-delta', text: buffer.slice(0, tagStartIndex) });
      state.buffer = buffer.slice(tagStartIndex);
    }

    // Try to parse opening tag
    const remaining = state.buffer;
    
    // Look for complete opening tag: <tagname ... > or <tagname ... />
    const openTagMatch = remaining.match(/^<(\w+)([^>]*?)(\/?)>/);
    
    if (openTagMatch) {
      const [fullMatch, tagName, attrStr, selfClose] = openTagMatch;
      
      if (isRegisteredTag(tagName)) {
        const attributes = parseAttributes(attrStr);
        
        if (selfClose) {
          // Self-closing tag - emit open and immediately close
          onEvent({ type: 'tag-open', tag: tagName, attributes, rawOpenTag: fullMatch });
          onEvent({ type: 'tag-close', tag: tagName, content: '', attributes });
        } else {
          // Opening tag - enter tag state
          state.insideTag = tagName;
          state.tagAttributes = attributes;
          state.tagContent = '';
          state.rawOpenTag = fullMatch;
          onEvent({ type: 'tag-open', tag: tagName, attributes, rawOpenTag: fullMatch });
        }
        
        state.buffer = remaining.slice(fullMatch.length);
        return;
      }
    }

    // Incomplete tag - wait for more data, but check if we have too much buffered
    if (remaining.length > 500) {
      // Probably not a real tag, emit as text and move on
      onEvent({ type: 'text-delta', text: remaining[0] });
      state.buffer = remaining.slice(1);
    }
  }

  // Process buffer when inside a registered tag
  function processInsideTag(): void {
    const buffer = state.buffer;
    const tagName = state.insideTag!;
    const closingTag = `</${tagName}>`;
    
    const closeIndex = buffer.toLowerCase().indexOf(closingTag.toLowerCase());
    
    if (closeIndex === -1) {
      // No closing tag yet - emit delta and keep buffer minimal
      // Keep last 50 chars in case closing tag is split
      const safeLength = Math.max(0, buffer.length - 50);
      if (safeLength > 0) {
        const delta = buffer.slice(0, safeLength);
        state.tagContent += delta;
        onEvent({ 
          type: 'tag-delta', 
          tag: tagName, 
          delta,
          accumulated: state.tagContent 
        });
        state.buffer = buffer.slice(safeLength);
      }
      return;
    }

    // Found closing tag
    const contentBeforeClose = buffer.slice(0, closeIndex);
    state.tagContent += contentBeforeClose;
    
    // Emit final delta if there was content
    if (contentBeforeClose) {
      onEvent({ 
        type: 'tag-delta', 
        tag: tagName, 
        delta: contentBeforeClose,
        accumulated: state.tagContent 
      });
    }
    
    // Emit tag close with full content
    onEvent({ 
      type: 'tag-close', 
      tag: tagName, 
      content: state.tagContent,
      attributes: state.tagAttributes 
    });

    // Reset tag state
    const afterClose = buffer.slice(closeIndex + closingTag.length);
    state.buffer = afterClose;
    state.insideTag = null;
    state.tagAttributes = {};
    state.tagContent = '';
    state.rawOpenTag = '';
  }

  // Main processing function
  function process(): void {
    // Keep processing while buffer has content that can be handled
    let lastBufferLength = -1;
    while (state.buffer.length > 0 && state.buffer.length !== lastBufferLength) {
      lastBufferLength = state.buffer.length;
      
      if (state.insideTag) {
        processInsideTag();
      } else {
        processOutsideTag();
      }
    }
  }

  return {
    /**
     * Feed a chunk of text to the parser
     */
    feed(chunk: string): void {
      state.buffer += chunk;
      process();
    },

    /**
     * Flush remaining buffer (call at end of stream)
     */
    flush(): void {
      if (state.insideTag) {
        // Stream ended while inside a tag - emit what we have
        if (state.buffer) {
          state.tagContent += state.buffer;
          onEvent({ 
            type: 'tag-delta', 
            tag: state.insideTag, 
            delta: state.buffer,
            accumulated: state.tagContent 
          });
        }
        // Emit close with incomplete content
        onEvent({ 
          type: 'tag-close', 
          tag: state.insideTag, 
          content: state.tagContent,
          attributes: state.tagAttributes 
        });
      } else if (state.buffer) {
        // Emit remaining text
        onEvent({ type: 'text-delta', text: state.buffer });
      }
      
      // Reset state
      state.buffer = '';
      state.insideTag = null;
      state.tagAttributes = {};
      state.tagContent = '';
      state.rawOpenTag = '';
    },

    /**
     * Reset parser state (for new stream)
     */
    reset(): void {
      state.buffer = '';
      state.insideTag = null;
      state.tagAttributes = {};
      state.tagContent = '';
      state.rawOpenTag = '';
    },

    /**
     * Get current state (for debugging)
     */
    getState(): Readonly<ParserState> {
      return { ...state };
    },

    /**
     * Check if currently inside a registered tag
     */
    isInsideTag(): boolean {
      return state.insideTag !== null;
    },

    /**
     * Get current tag name if inside one
     */
    getCurrentTag(): string | null {
      return state.insideTag;
    },
  };
}

// Type for the parser instance
export type StreamParser = ReturnType<typeof createStreamParser>;
