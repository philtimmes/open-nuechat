/**
 * ToolTimeline - Renders tool activity as a collapsible timeline in the chat.
 * 
 * Live mode: Shows expanding bubbles as tools execute during streaming.
 * Collapsed mode: Shows a compact summary bar after completion.
 * Expands on click to show full timeline.
 */
import React, { useState, useMemo, useRef, useEffect } from 'react';
import type { ToolTimelineEvent } from '../../stores/chat/types';

interface ToolTimelineProps {
  events: ToolTimelineEvent[];
  isLive?: boolean;
}

// All tools now stream their content during generation

/** Extract the "content" field value from partially-streamed JSON args */
function extractStreamedContent(rawJson: string): { filename: string; content: string } | null {
  if (!rawJson) return null;
  
  // Try to extract path/filename
  let filename = '';
  const pathMatch = rawJson.match(/"path"\s*:\s*"([^"]*)/);
  if (pathMatch) filename = pathMatch[1];
  
  // Find the "content" field value — it's the last field usually, and may be incomplete
  const contentStart = rawJson.indexOf('"content"');
  if (contentStart === -1) return filename ? { filename, content: '' } : null;
  
  // Skip past "content": "
  const valueStart = rawJson.indexOf('"', contentStart + 9);
  if (valueStart === -1) return { filename, content: '' };
  
  // Everything after the opening quote is the content (may not have closing quote yet)
  let content = rawJson.slice(valueStart + 1);
  // Remove trailing incomplete JSON (closing "} etc)
  content = content.replace(/"\s*\}\s*$/, '');
  
  // Unescape JSON string escapes
  try {
    content = content
      .replace(/\\n/g, '\n')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\');
  } catch { /* best effort */ }
  
  return { filename, content };
}

// Tool icons/emojis for quick visual identification
const TOOL_ICONS: Record<string, string> = {
  fetch_document: '📄',
  fetch_urls: '🌐',
  fetch_webpage: '🌐',
  web_search: '🔍',
  web_extract: '📋',
  create_file: '📝',
  search_replace: '🔧',
  sed_files: '🔧',
  view_file_lines: '👁',
  view_signature: '📊',
  grep_files: '🔎',
  request_file: '📂',
  execute_python: '🐍',
};

function getToolIcon(tool: string): string {
  return TOOL_ICONS[tool] || '⚡';
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Pair tool_start and tool_end events into completed operations */
function pairEvents(events: ToolTimelineEvent[]) {
  const pairs: Array<{
    tool: string;
    round: number;
    args_summary: string;
    status: string;
    duration_ms: number;
    result_summary: string;
    startTs: number;
    narrator: string;
    toolIndex: number;
  }> = [];

  const pending = new Map<string, ToolTimelineEvent & { _index?: number }>();
  let indexCounter = 0;

  for (const evt of events) {
    if (evt.type === 'tool_start') {
      (evt as any)._index = indexCounter++;
      pending.set(evt.tool + ':' + evt.round, evt);
    } else if (evt.type === 'tool_end') {
      const startEvt = pending.get(evt.tool + ':' + evt.round);
      pairs.push({
        tool: evt.tool,
        round: evt.round,
        args_summary: startEvt?.args_summary || '',
        status: evt.status || 'success',
        duration_ms: evt.duration_ms || 0,
        result_summary: evt.result_summary || '',
        startTs: startEvt?.ts || evt.ts,
        narrator: (startEvt as any)?.narrator || '',
        toolIndex: (startEvt as any)?._index ?? 0,
      });
      pending.delete(evt.tool + ':' + evt.round);
    }
  }

  // Add any still-running tools (have start but no end)
  for (const [, startEvt] of pending) {
    pairs.push({
      tool: startEvt.tool,
      round: startEvt.round,
      args_summary: startEvt.args_summary || '',
      status: (startEvt as any).status || 'running',
      duration_ms: 0,
      result_summary: '',
      startTs: startEvt.ts,
      narrator: (startEvt as any)?.narrator || '',
      toolIndex: (startEvt as any)?._index ?? 0,
    });
  }

  return pairs;
}

export const ToolTimeline: React.FC<ToolTimelineProps> = ({ events, isLive = false }) => {
  const [expanded, setExpanded] = useState(false);

  const pairs = useMemo(() => pairEvents(events), [events]);

  if (pairs.length === 0) return null;

  const totalDuration = pairs.reduce((sum, p) => sum + p.duration_ms, 0);
  const errorCount = pairs.filter(p => p.status === 'error' || p.status === 'empty').length;
  const runningCount = pairs.filter(p => p.status === 'running').length;

  // Compact unique tool names for summary
  const toolCounts: Record<string, number> = {};
  for (const p of pairs) {
    toolCounts[p.tool] = (toolCounts[p.tool] || 0) + 1;
  }
  const toolSummary = Object.entries(toolCounts)
    .map(([t, c]) => c > 1 ? `${t} ×${c}` : t)
    .join(', ');

  // Collapsed summary bar
  if (!expanded && !isLive) {
    return (
      <div
        onClick={() => setExpanded(true)}
        className="flex items-center gap-2 px-3 py-1.5 my-1 rounded-lg 
                   bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700/50
                   cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700/50 
                   transition-colors text-xs text-gray-500 dark:text-gray-400 select-none"
      >
        <span className="text-amber-500">⚡</span>
        <span className="font-medium">{pairs.length} tool{pairs.length !== 1 ? 's' : ''}</span>
        <span className="text-gray-400 dark:text-gray-500">·</span>
        <span className="truncate max-w-[300px]">{toolSummary}</span>
        <span className="text-gray-400 dark:text-gray-500">·</span>
        <span>{formatDuration(totalDuration)}</span>
        {errorCount > 0 && (
          <span className="text-red-500 font-medium">{errorCount} failed</span>
        )}
        <span className="ml-auto text-gray-400">▸</span>
      </div>
    );
  }

  // Expanded timeline (also used for live mode)
  return (
    <div className="my-1 rounded-lg border border-gray-200 dark:border-gray-700/50 
                    bg-gray-50 dark:bg-gray-800/50 overflow-hidden">
      {/* Header */}
      <div
        onClick={() => !isLive && setExpanded(false)}
        className={`flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 dark:text-gray-400 
                    ${!isLive ? 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700/50' : ''} select-none`}
      >
        <span className="text-amber-500">{isLive ? '⏳' : '⚡'}</span>
        <span className="font-medium">
          {isLive ? 'Working' : `${pairs.length} tool${pairs.length !== 1 ? 's' : ''}`}
          {runningCount > 0 && (
            <span className="ml-1 text-blue-500 animate-pulse">({runningCount} running)</span>
          )}
        </span>
        {!isLive && (
          <>
            <span className="text-gray-400 dark:text-gray-500">·</span>
            <span>{formatDuration(totalDuration)}</span>
            <span className="ml-auto text-gray-400">▾</span>
          </>
        )}
      </div>

      {/* Timeline entries - scrollable window showing last ~4 items */}
      <TimelineScroller pairs={pairs} />
    </div>
  );
};

/** Max visible items before scrolling kicks in */
const MAX_VISIBLE = 4;
/** Approximate height per row in px */
const ROW_HEIGHT = 40;

/** Live streaming content preview for a tool */
const FILE_TOOLS = new Set(['create_file', 'search_replace', 'sed_files']);

const ToolContentPreview: React.FC<{ toolName: string; toolIndex: number }> = ({ toolName, toolIndex }) => {
  const scrollRef = useRef<HTMLPreElement>(null);
  const [content, setContent] = useState('');
  const [filename, setFilename] = useState('');
  
  // Poll __toolContentStream for updates
  useEffect(() => {
    const streamKey = `${toolName}:${toolIndex}`;
    let frame: number;
    const poll = () => {
      const raw = (window as any).__toolContentStream?.[streamKey] || '';
      if (raw) {
        if (FILE_TOOLS.has(toolName)) {
          const parsed = extractStreamedContent(raw);
          if (parsed) {
            setContent(parsed.content);
            if (parsed.filename) setFilename(parsed.filename);
          }
        } else {
          // For all other tools, show the raw JSON args as they stream in
          setContent(raw);
        }
      }
      frame = requestAnimationFrame(poll);
    };
    frame = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(frame);
  }, [toolName, toolIndex]);
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content]);
  
  if (!content) return null;
  
  return (
    <pre
      ref={scrollRef}
      className="mt-1 px-2 py-1 rounded bg-gray-100 dark:bg-gray-800/70 border border-gray-200 dark:border-gray-700/40
                 text-[10px] leading-relaxed font-mono text-gray-600 dark:text-gray-300
                 overflow-y-auto overflow-x-hidden whitespace-pre-wrap break-all"
      style={{ maxHeight: 72 }}
    >
      {FILE_TOOLS.has(toolName) && filename && (
        <span className="text-gray-400 dark:text-gray-500">{filename}{'\n'}</span>
      )}
      {content}
      <span className="inline-block w-1 h-3 bg-blue-400 animate-pulse ml-0.5 align-middle" />
    </pre>
  );
};

const TimelineScroller: React.FC<{ pairs: ReturnType<typeof pairEvents> }> = ({ pairs }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  // Auto-scroll to bottom when new items arrive
  useEffect(() => {
    if (pairs.length > prevCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    prevCountRef.current = pairs.length;
  }, [pairs.length]);

  const needsScroll = pairs.length > MAX_VISIBLE;

  return (
    <div
      ref={scrollRef}
      className="px-3 pb-2 space-y-1 overflow-y-auto"
      style={needsScroll ? { maxHeight: ROW_HEIGHT * MAX_VISIBLE } : undefined}
    >
      {pairs.map((pair, i) => (
        <div
          key={`${pair.tool}-${pair.round}-${i}`}
          className={`flex items-start gap-2 text-xs py-1 
                      ${i < pairs.length - 1 ? 'border-b border-gray-100 dark:border-gray-700/30' : ''}`}
        >
          <span className="mt-0.5 flex-shrink-0">
            {pair.status === 'generating' ? (
              <span className="inline-block w-3 h-3 rounded-full border-2 border-blue-400 border-t-transparent animate-spin" />
            ) : pair.status === 'running' ? (
              <span className="inline-block w-3 h-3 rounded-full bg-blue-400 animate-pulse" />
            ) : pair.status === 'error' ? (
              <span className="text-red-500">✗</span>
            ) : pair.status === 'empty' ? (
              <span className="text-amber-500">⚠</span>
            ) : (
              <span className="text-green-500">✓</span>
            )}
          </span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5">
              <span>{getToolIcon(pair.tool)}</span>
              {pair.narrator ? (
                <span className="text-gray-600 dark:text-gray-300 italic">
                  {pair.narrator}
                </span>
              ) : (
                <span className="font-mono font-medium text-gray-700 dark:text-gray-300">
                  {pair.tool}
                </span>
              )}
              {(pair.status === 'running' || pair.status === 'generating') && !pair.narrator && (
                <span className="text-gray-400 dark:text-gray-500 animate-pulse">…</span>
              )}
              {pair.duration_ms > 0 && (
                <span className="text-gray-400 dark:text-gray-500">
                  {formatDuration(pair.duration_ms)}
                </span>
              )}
            </div>
            {pair.narrator && (
              <div className="text-gray-400 dark:text-gray-500 truncate mt-0.5 font-mono text-[10px]">
                {pair.tool}({pair.args_summary})
              </div>
            )}
            {!pair.narrator && pair.args_summary && (
              <div className="text-gray-400 dark:text-gray-500 truncate mt-0.5 font-mono text-[10px]">
                ({pair.args_summary})
              </div>
            )}
            {(pair.status === 'error' || pair.status === 'empty') && pair.result_summary && (
              <div className={`truncate mt-0.5 text-[10px] ${pair.status === 'error' ? 'text-red-400' : 'text-amber-400'}`}>
                {pair.status === 'empty' ? 'No content extracted' : pair.result_summary}
              </div>
            )}
            {/* NC-0.8.0.21: Live content preview for all tools during generation/execution */}
            {(pair.status === 'running' || pair.status === 'generating') && (
              <ToolContentPreview toolName={pair.tool} toolIndex={pair.toolIndex} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ToolTimeline;
