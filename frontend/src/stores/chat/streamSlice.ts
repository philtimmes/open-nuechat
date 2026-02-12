/**
 * Stream slice - Streaming content and sending state
 */
import type { StreamSlice, SliceCreator } from './types';
import { extractArtifacts } from '../../lib/artifacts';

export const createStreamSlice: SliceCreator<StreamSlice> = (set, get) => ({
  isSending: false,
  streamingContent: '',
  streamingToolCall: null,
  streamingArtifacts: [],
  toolTimeline: [],
  error: null,

  setStreamingContent: (content: string) => {
    set({ streamingContent: content });
  },

  appendStreamingContent: (chunk: string) => {
    set((state) => ({
      streamingContent: state.streamingContent + chunk,
    }));
  },

  setStreamingToolCall: (toolCall) => {
    set({ streamingToolCall: toolCall });
  },

  setToolTimelineEvent: (event) => {
    set((state) => ({
      toolTimeline: [...state.toolTimeline, event],
    }));
  },

  clearToolTimeline: () => {
    set({ toolTimeline: [] });
  },

  clearStreaming: () => {
    set({
      streamingContent: '',
      streamingToolCall: null,
      streamingArtifacts: [],
      toolTimeline: [],
      isSending: false,
    });
  },

  setIsSending: (isSending: boolean) => {
    set({ isSending });
  },

  setError: (error: string | null) => {
    set({ error });
  },

  updateStreamingArtifacts: (content: string) => {
    // Extract artifacts from streaming content in real-time
    const { artifacts: streamingArtifacts } = extractArtifacts(content);
    set({ streamingArtifacts });
  },
});
