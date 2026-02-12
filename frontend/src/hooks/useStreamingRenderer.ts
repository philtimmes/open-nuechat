/**
 * useStreamingRenderer - React hook for streaming content rendering
 * 
 * Provides:
 * - Container ref for the renderer
 * - Functions to control streaming
 * - Access to captured artifacts
 * - Clean integration with React lifecycle
 */

import { useRef, useCallback, useEffect } from 'react';
import { createStreamingRenderer, StreamingRenderer } from '../lib/streamingRenderer';
import type { Artifact } from '../types';

export interface UseStreamingRendererOptions {
  onArtifactComplete?: (artifact: Artifact) => void;
}

export interface UseStreamingRendererResult {
  /**
   * Ref to attach to the container div
   */
  containerRef: React.RefObject<HTMLDivElement | null>;
  
  /**
   * Feed a chunk of content to the renderer
   */
  feed: (chunk: string) => void;
  
  /**
   * End the stream and finalize
   */
  end: () => void;
  
  /**
   * Reset for a new stream
   */
  reset: () => void;
  
  /**
   * Clear rendered content
   */
  clear: () => void;
  
  /**
   * Get captured artifacts
   */
  getArtifacts: () => Artifact[];
  
  /**
   * Get raw content for message save
   */
  getRawContent: () => string;
  
  /**
   * Check if currently streaming
   */
  isStreaming: () => boolean;
  
  /**
   * Check if inside a registered tag
   */
  isInsideTag: () => boolean;
}

export function useStreamingRenderer(
  options: UseStreamingRendererOptions = {}
): UseStreamingRendererResult {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<StreamingRenderer | null>(null);
  const optionsRef = useRef(options);
  
  // Keep options ref updated
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  // Initialize renderer lazily
  const getRenderer = useCallback(() => {
    if (!rendererRef.current) {
      rendererRef.current = createStreamingRenderer({
        onArtifactComplete: (artifact) => {
          optionsRef.current.onArtifactComplete?.(artifact);
        },
      });
    }
    return rendererRef.current;
  }, []);

  // Update container when ref changes
  useEffect(() => {
    const renderer = getRenderer();
    renderer.setContainer(containerRef.current);
    
    return () => {
      renderer.setContainer(null);
    };
  }, [getRenderer]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (rendererRef.current) {
        rendererRef.current.reset();
        rendererRef.current = null;
      }
    };
  }, []);

  const feed = useCallback((chunk: string) => {
    const renderer = getRenderer();
    renderer.setContainer(containerRef.current);
    renderer.feed(chunk);
  }, [getRenderer]);

  const end = useCallback(() => {
    getRenderer().end();
  }, [getRenderer]);

  const reset = useCallback(() => {
    getRenderer().reset();
    if (containerRef.current) {
      getRenderer().setContainer(containerRef.current);
    }
  }, [getRenderer]);

  const clear = useCallback(() => {
    getRenderer().clear();
  }, [getRenderer]);

  const getArtifacts = useCallback(() => {
    return getRenderer().getArtifacts();
  }, [getRenderer]);

  const getRawContent = useCallback(() => {
    return getRenderer().getRawContent();
  }, [getRenderer]);

  const isStreaming = useCallback(() => {
    return rendererRef.current?.isStreaming() ?? false;
  }, []);

  const isInsideTag = useCallback(() => {
    return rendererRef.current?.isInsideTag() ?? false;
  }, []);

  return {
    containerRef,
    feed,
    end,
    reset,
    clear,
    getArtifacts,
    getRawContent,
    isStreaming,
    isInsideTag,
  };
}
