/**
 * WebSocket message types and type guards
 * 
 * Provides type-safe handling of WebSocket events between
 * the frontend and backend.
 */

// ============ Server -> Client Event Types ============

export interface WSStreamStart {
  type: 'stream_start';
  payload: {
    message_id: string;
    chat_id: string;
  };
}

export interface WSStreamChunk {
  type: 'stream_chunk';
  payload: {
    message_id: string;
    content: string;
  };
}

export interface WSStreamEnd {
  type: 'stream_end';
  payload: {
    message_id: string;
    chat_id: string;
    parent_id?: string;
    usage: {
      input_tokens: number;
      output_tokens: number;
      duration_ms?: number;
      ttft_ms?: number;  // Time to first token
      tokens_per_second?: number;
    };
  };
}

export interface WSStreamError {
  type: 'stream_error';
  payload: {
    message_id?: string;
    error: string;
  };
}

export interface WSToolCall {
  type: 'tool_call';
  payload: {
    message_id: string;
    tool_call: {
      name: string;
      id: string;
      input: Record<string, unknown>;
    };
  };
}

export interface WSToolResult {
  type: 'tool_result';
  payload: {
    message_id: string;
    tool_id: string;
    result: unknown;
  };
}

export interface WSImageGeneration {
  type: 'image_generation';
  payload: {
    message_id: string;
    chat_id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    queue_position?: number;
    image_base64?: string;
    width?: number;
    height?: number;
    seed?: number;
    prompt?: string;
    generation_time?: number;
    error?: string;
  };
}

export interface WSMessageSaved {
  type: 'message_saved';
  payload: {
    temp_id: string;
    real_id: string;
    parent_id?: string;
    chat_id: string;
  };
}

export interface WSPong {
  type: 'pong';
}

export interface WSError {
  type: 'error';
  payload: {
    message: string;
    code?: string;
  };
}

export interface WSSubscribed {
  type: 'subscribed';
  payload: {
    chat_id: string;
  };
}

export interface WSUnsubscribed {
  type: 'unsubscribed';
  payload: {
    chat_id: string;
  };
}

// Union of all server message types
export interface WSBrowserFetchRequest {
  type: 'browser_fetch_request';
  payload: {
    request_id: string;
    url: string;
  };
}

export type ServerMessage =
  | WSStreamStart
  | WSStreamChunk
  | WSStreamEnd
  | WSStreamError
  | WSToolCall
  | WSToolResult
  | WSImageGeneration
  | WSMessageSaved
  | WSPong
  | WSError
  | WSSubscribed
  | WSUnsubscribed
  | WSBrowserFetchRequest;

// ============ Client -> Server Event Types ============

export interface WSSubscribe {
  type: 'subscribe';
  chat_id: string;
}

export interface WSUnsubscribe {
  type: 'unsubscribe';
  chat_id: string;
}

export interface WSChatMessage {
  type: 'chat_message';
  chat_id: string;
  content: string;
  attachments?: Array<{
    type: string;
    data: string;
    filename?: string;
    mime_type?: string;
  }>;
  enable_tools?: boolean;
  enable_rag?: boolean;
  knowledge_store_ids?: string[];
  parent_id?: string;
}

export interface WSStopGeneration {
  type: 'stop_generation';
  chat_id: string;
}

export interface WSPing {
  type: 'ping';
}

export interface WSRegenerateMessage {
  type: 'regenerate';
  chat_id: string;
  message_id: string;
}

// Union of all client message types
export type ClientMessage =
  | WSSubscribe
  | WSUnsubscribe
  | WSChatMessage
  | WSStopGeneration
  | WSPing
  | WSRegenerateMessage;

// ============ Type Guards ============

export function isStreamStart(msg: ServerMessage): msg is WSStreamStart {
  return msg.type === 'stream_start';
}

export function isStreamChunk(msg: ServerMessage): msg is WSStreamChunk {
  return msg.type === 'stream_chunk';
}

export function isStreamEnd(msg: ServerMessage): msg is WSStreamEnd {
  return msg.type === 'stream_end';
}

export function isStreamError(msg: ServerMessage): msg is WSStreamError {
  return msg.type === 'stream_error';
}

export function isToolCall(msg: ServerMessage): msg is WSToolCall {
  return msg.type === 'tool_call';
}

export function isToolResult(msg: ServerMessage): msg is WSToolResult {
  return msg.type === 'tool_result';
}

export function isImageGeneration(msg: ServerMessage): msg is WSImageGeneration {
  return msg.type === 'image_generation';
}

export function isMessageSaved(msg: ServerMessage): msg is WSMessageSaved {
  return msg.type === 'message_saved';
}

export function isPong(msg: ServerMessage): msg is WSPong {
  return msg.type === 'pong';
}

export function isError(msg: ServerMessage): msg is WSError {
  return msg.type === 'error';
}

export function isSubscribed(msg: ServerMessage): msg is WSSubscribed {
  return msg.type === 'subscribed';
}

export function isUnsubscribed(msg: ServerMessage): msg is WSUnsubscribed {
  return msg.type === 'unsubscribed';
}

// ============ Parsing ============

/**
 * Safely parse a WebSocket message from JSON
 */
export function parseServerEvent(data: string): ServerMessage | null {
  try {
    const parsed = JSON.parse(data);
    
    // Validate it has a type field
    if (!parsed || typeof parsed.type !== 'string') {
      console.warn('Invalid WebSocket message: missing type', parsed);
      return null;
    }
    
    // Validate known types
    const knownTypes = [
      'stream_start', 'stream_chunk', 'stream_end', 'stream_error',
      'tool_call', 'tool_result', 'image_generation', 'message_saved',
      'pong', 'error', 'subscribed', 'unsubscribed'
    ];
    
    if (!knownTypes.includes(parsed.type)) {
      console.warn('Unknown WebSocket message type:', parsed.type);
      // Still return it - might be a new event type
    }
    
    return parsed as ServerMessage;
  } catch (e) {
    console.error('Failed to parse WebSocket message:', e, data);
    return null;
  }
}

/**
 * Create a type-safe client message
 */
export function createClientMessage<T extends ClientMessage>(msg: T): string {
  return JSON.stringify(msg);
}

// ============ Helpers ============

/**
 * Check if an event is a streaming event
 */
export function isStreamingEvent(msg: ServerMessage): boolean {
  return ['stream_start', 'stream_chunk', 'stream_end', 'stream_error'].includes(msg.type);
}

/**
 * Check if an event is a tool-related event
 */
export function isToolEvent(msg: ServerMessage): boolean {
  return ['tool_call', 'tool_result'].includes(msg.type);
}

/**
 * Extract error message from any error event
 */
export function extractErrorMessage(msg: ServerMessage): string | null {
  if (isStreamError(msg)) {
    return msg.payload.error;
  }
  if (isError(msg)) {
    return msg.payload.message;
  }
  if (isImageGeneration(msg) && msg.payload.status === 'failed') {
    return msg.payload.error || 'Image generation failed';
  }
  return null;
}
