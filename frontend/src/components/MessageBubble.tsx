import React, { useState, memo, useRef, useEffect, useCallback } from 'react';
import type { Message, Artifact, GeneratedImage, UserHint } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import GeneratedImageCard from './GeneratedImageCard';
import TextSelectionBubble from './TextSelectionBubble';
import mermaid from 'mermaid';
import { useBrandingStore } from '../stores/brandingStore';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
});

// Video embed helper functions
interface VideoInfo {
  platform: 'youtube' | 'rumble';
  videoId: string;
  url: string;
}

function extractVideoInfo(url: string): VideoInfo | null {
  // YouTube patterns
  const youtubePatterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]+)/,
  ];
  
  for (const pattern of youtubePatterns) {
    const match = url.match(pattern);
    if (match) {
      return { platform: 'youtube', videoId: match[1], url };
    }
  }
  
  // Rumble patterns
  const rumblePattern = /rumble\.com\/(?:embed\/)?v([a-zA-Z0-9]+)/;
  const rumbleMatch = url.match(rumblePattern);
  if (rumbleMatch) {
    return { platform: 'rumble', videoId: rumbleMatch[1], url };
  }
  
  return null;
}

// Video Embed Component
const VideoEmbed = memo(({ url, className }: { url: string; className?: string }) => {
  const videoInfo = extractVideoInfo(url);
  
  if (!videoInfo) {
    // Not a video URL, render as regular link
    return (
      <a href={url} target="_blank" rel="noopener noreferrer" className="text-[var(--color-primary)] hover:underline">
        {url}
      </a>
    );
  }
  
  const { platform, videoId } = videoInfo;
  
  if (platform === 'youtube') {
    return (
      <div className={`video-embed my-4 ${className || ''}`}>
        <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
          <iframe
            className="absolute top-0 left-0 w-full h-full rounded-lg"
            src={`https://www.youtube.com/embed/${videoId}`}
            title="YouTube video"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      </div>
    );
  }
  
  if (platform === 'rumble') {
    return (
      <div className={`video-embed my-4 ${className || ''}`}>
        <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
          <iframe
            className="absolute top-0 left-0 w-full h-full rounded-lg"
            src={`https://rumble.com/embed/v${videoId}/`}
            title="Rumble video"
            frameBorder="0"
            allowFullScreen
          />
        </div>
      </div>
    );
  }
  
  return null;
});

// Mermaid diagram renderer component
const MermaidDiagram = memo(({ code, id, onError }: { code: string; id: string; onError?: (error: string, code: string) => void }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'preview' | 'code'>('preview');
  const [copied, setCopied] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const errorReportedRef = useRef(false);

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        const { svg } = await mermaid.render(`mermaid-${id}`, code);
        setSvg(svg);
        setError(null);
        errorReportedRef.current = false;
      } catch (err) {
        console.error('Mermaid render error:', err);
        const errorMsg = err instanceof Error ? err.message : 'Failed to render diagram';
        setError(errorMsg);
        setSvg(null);
        
        // Report error to parent (only once per error)
        if (onError && !errorReportedRef.current) {
          errorReportedRef.current = true;
          onError(errorMsg, code);
        }
      }
    };
    renderDiagram();
  }, [code, id, onError]);

  const copyCode = useCallback(() => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  const exportPng = useCallback(async () => {
    if (!svg) return;
    setIsExporting(true);
    
    try {
      // Create a temporary container to get SVG dimensions
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = svg;
      const svgElement = tempDiv.querySelector('svg');
      if (!svgElement) throw new Error('No SVG found');

      // Get dimensions
      const bbox = svgElement.getBBox();
      const width = Math.ceil(bbox.width + 40);
      const height = Math.ceil(bbox.height + 40);

      // Create canvas
      const canvas = document.createElement('canvas');
      canvas.width = width * 2;
      canvas.height = height * 2;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Cannot get canvas context');

      // Draw SVG to canvas
      const img = new Image();
      const svgBlob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);

      img.onload = () => {
        ctx.scale(2, 2);
        ctx.drawImage(img, 0, 0);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'diagram.png';
            a.click();
          }
          URL.revokeObjectURL(url);
          setIsExporting(false);
        }, 'image/png');
      };
      
      img.onerror = () => {
        URL.revokeObjectURL(url);
        setIsExporting(false);
      };
      
      img.src = url;
    } catch (err) {
      console.error('PNG export error:', err);
      setIsExporting(false);
    }
  }, [svg]);

  return (
    <div className="my-3 rounded-lg overflow-hidden border border-zinc-700">
      {/* Header with tabs */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-800 border-b border-zinc-700">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setViewMode('preview')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              viewMode === 'preview'
                ? 'bg-zinc-600 text-white'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Preview
          </button>
          <button
            onClick={() => setViewMode('code')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              viewMode === 'code'
                ? 'bg-zinc-600 text-white'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Code
          </button>
        </div>
        <div className="flex items-center gap-1">
          {viewMode === 'preview' && svg && (
            <button
              onClick={exportPng}
              disabled={isExporting}
              className="px-2 py-0.5 text-xs rounded bg-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-600 transition-colors disabled:opacity-50"
            >
              {isExporting ? 'Exporting...' : 'Export PNG'}
            </button>
          )}
          <button
            onClick={copyCode}
            className="px-2 py-0.5 text-xs rounded bg-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-600 transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>
      
      {/* Content */}
      {viewMode === 'preview' ? (
        <div 
          ref={containerRef}
          className="p-4 bg-zinc-900 overflow-auto"
        >
          {error ? (
            <div className="text-red-400 text-sm">
              <p className="font-medium">Failed to render diagram:</p>
              <p className="text-xs mt-1 opacity-75">{error}</p>
            </div>
          ) : svg ? (
            <div 
              className="flex justify-center [&_svg]:max-w-full"
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          ) : (
            <div className="text-zinc-500 text-sm">Rendering...</div>
          )}
        </div>
      ) : (
        <SyntaxHighlighter
          style={oneDark}
          language="mermaid"
          PreTag="div"
          customStyle={{
            margin: 0,
            borderRadius: 0,
            fontSize: '0.8125rem',
          }}
        >
          {code}
        </SyntaxHighlighter>
      )}
    </div>
  );
});

// Tool citation data
interface ToolCitation {
  id: string;
  tool_name: string;
  operation?: string;
  result_summary?: string;
  result_url?: string;
  success: boolean;
}

/**
 * Unwrap double-nested code fences of the SAME file type.
 * 
 * Problem: LLM sometimes outputs:
 * ```mermaid
 * ```mermaid
 * graph TD
 *   A --> B
 * ```
 * ```
 * 
 * Solution: Detect when outer and inner fences have the same language,
 * and render only the inner content.
 */
function unwrapDoubleNestedSameType(content: string): string {
  const lines = content.split('\n');
  const result: string[] = [];
  
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Match opening fence with language
    const openMatch = line.match(/^(`{3,})(\w+)(?:::.*)?$/);
    
    if (openMatch) {
      const [, outerBackticks, outerLang] = openMatch;
      const outerLen = outerBackticks.length;
      
      // Check if the very next line is another opening fence with the SAME language
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        const innerMatch = nextLine.match(/^(`{3,})(\w+)(?:::.*)?$/);
        
        if (innerMatch) {
          const [, innerBackticks, innerLang] = innerMatch;
          
          // Only unwrap if same language (case-insensitive)
          if (outerLang.toLowerCase() === innerLang.toLowerCase()) {
            // Find the inner closing fence
            let innerCloseIdx = -1;
            let outerCloseIdx = -1;
            let depth = 1;
            
            for (let j = i + 2; j < lines.length; j++) {
              const l = lines[j];
              // Check for another opening fence
              if (l.match(/^`{3,}\w+/)) {
                depth++;
              }
              // Check for closing fence
              else if (l.match(/^`{3,}\s*$/)) {
                depth--;
                if (depth === 0 && innerCloseIdx === -1) {
                  innerCloseIdx = j;
                } else if (depth === -1) {
                  outerCloseIdx = j;
                  break;
                }
              }
            }
            
            // If we found both closing fences, skip the outer wrapper
            if (innerCloseIdx !== -1) {
              // Push the inner opening fence (use original line to preserve any ::filename)
              result.push(nextLine);
              
              // Push content between inner open and inner close
              for (let j = i + 2; j < innerCloseIdx; j++) {
                result.push(lines[j]);
              }
              
              // Push inner closing fence
              result.push(lines[innerCloseIdx]);
              
              // Skip past outer closing fence if found, otherwise just past inner
              if (outerCloseIdx !== -1) {
                i = outerCloseIdx + 1;
              } else {
                i = innerCloseIdx + 1;
              }
              continue;
            }
          }
        }
      }
      
      // No double-nesting or different languages - keep as is
      result.push(line);
      i++;
    } else {
      result.push(line);
      i++;
    }
  }
  
  return result.join('\n');
}

/**
 * Fix nested code fences for proper markdown rendering.
 * 
 * Problem: When LLM generates a file containing code fences (like README.md),
 * the inner ``` closes the outer block prematurely.
 * 
 * Solution: Detect nested fences and increase outer fence to 4+ backticks.
 * CommonMark spec says closing fence must match opening length.
 */
function fixNestedCodeFences(content: string): string {
  const lines = content.split('\n');
  const result: string[] = [];
  
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Only match opening fences that HAVE a language specifier (\w+, not \w*)
    // A bare ``` without language should NOT start a new outer code block
    const openMatch = line.match(/^(`{3,})(\w+)$/);
    
    if (openMatch) {
      const [, backticks, lang] = openMatch;
      const openingLength = backticks.length;
      
      // Scan ahead to find content and detect if it has nested fences
      let hasNestedFences = false;
      let depth = 1;
      let endLine = i + 1;
      
      for (let j = i + 1; j < lines.length && depth > 0; j++) {
        const l = lines[j];
        
        // Check for nested opening fence (has language)
        if (l.match(/^`{3,}\w+$/)) {
          hasNestedFences = true;
          depth++;
        }
        // Check for closing fence (no language, just backticks)
        else if (l.match(/^`{3,}\s*$/)) {
          depth--;
          if (depth === 0) {
            endLine = j;
          }
        }
      }
      
      // If we found nested fences and the outer fence is only 3 backticks,
      // increase it to 4 backticks so ReactMarkdown parses correctly
      if (hasNestedFences && openingLength === 3) {
        result.push('````' + lang);
        
        // Copy content lines
        for (let j = i + 1; j < endLine; j++) {
          result.push(lines[j]);
        }
        
        // Add closing fence with same length
        result.push('````');
        i = endLine + 1;
      } else {
        // No nested fences or already using 4+ backticks
        result.push(line);
        i++;
      }
    } else {
      result.push(line);
      i++;
    }
  }
  
  return result.join('\n');
}

// Preprocess content to extract filenames from lines before code fences
// Transforms:
//   `/path/to/file.js`
//   ```javascript
// Into:
//   ```javascript::/path/to/file.js
//
// Also converts XML-style filename tags to markdown code fences:
//   <Menu.cpp>code</Menu.cpp> → ```cpp::Menu.cpp\ncode\n```
//   <artifact=file.py>code</artifact> → ```python::file.py\ncode\n```
function preprocessContent(content: string): string {
  // NC-0.8.0.6: Strip tool invocation tags from display (keep in history)
  // These are internal tool calls that shouldn't be shown to users
  let processed = content
    // NC-0.8.0.7: Remove image context blocks (keep in history for LLM, hide from user)
    .replace(/\[IMAGE CONTEXT - USER-GENERATED, NOT BY LLM\][\s\S]*?\[END IMAGE CONTEXT\]\s*/g, '')
    // Remove <request_file path="..." offset="..."/> tags
    .replace(/<request_file\s+[^>]*\/?>/gi, '')
    // Remove <find_line path="..." contains="..."/> tags
    .replace(/<find_line\s+[^>]*\/?>/gi, '')
    // Remove <find search="..."/> and <find path="..." search="..."/> tags  
    .replace(/<find\s+[^>]*\/?>/gi, '')
    // Remove <search_replace>...</search_replace> blocks
    .replace(/<search_replace\s+[^>]*>[\s\S]*?<\/search_replace>/gi, '')
    // Remove <replace_line.../> and <replace_block>...</replace_block>
    .replace(/<replace_line\s+[^>]*\/?>/gi, '')
    .replace(/<replace_block\s+[^>]*>[\s\S]*?<\/replace_block>/gi, '')
    // Clean up any leftover empty lines from removed tags
    .replace(/\n{3,}/g, '\n\n');
  
  // First unwrap double-nested same-type code fences (e.g., ```mermaid inside ```mermaid)
  processed = unwrapDoubleNestedSameType(processed);
  
  // Then fix remaining nested code fences
  processed = fixNestedCodeFences(processed);
  
  // Convert XML-style filename tags to markdown code fences
  // Pattern 4a: Direct filename as tag: <Menu.cpp>...</Menu.cpp>
  processed = processed.replace(
    /<([A-Za-z_][\w\-./]*\.(\w+))>\s*([\s\S]*?)\s*<\/\1>/g,
    (match, filepath, ext, code) => {
      const lang = extToMarkdownLang(ext);
      return `\n\`\`\`${lang}::${filepath}\n${code.trim()}\n\`\`\`\n`;
    }
  );
  
  // Pattern 4b: <artifact=filename> or <file=filename> or <code=filename>
  processed = processed.replace(
    /<(?:artifact|file|code)[=:]([A-Za-z_][\w\-./]*\.(\w+))>\s*([\s\S]*?)\s*<\/(?:artifact|file|code|\1)>/gi,
    (match, filepath, ext, code) => {
      const lang = extToMarkdownLang(ext);
      return `\n\`\`\`${lang}::${filepath}\n${code.trim()}\n\`\`\`\n`;
    }
  );
  
  // Pattern 4c: <artifact name="filename"> attribute style
  processed = processed.replace(
    /<(?:artifact|file|code)\s+(?:name|filename|file|path)=["']([A-Za-z_][\w\-./]*\.(\w+))["'][^>]*>\s*([\s\S]*?)\s*<\/(?:artifact|file|code)>/gi,
    (match, filepath, ext, code) => {
      const lang = extToMarkdownLang(ext);
      return `\n\`\`\`${lang}::${filepath}\n${code.trim()}\n\`\`\`\n`;
    }
  );
  
  // Match: optional backticks around filepath, newline, then code fence with language
  // Captures: 1=filepath (with or without backticks), 2=language
  const pattern = /(?:^|\n)`?([^\n`]+\.[a-zA-Z0-9]+)`?\s*\n```(\w+)/g;
  
  processed = processed.replace(pattern, (match, filepath, language) => {
    // Clean up the filepath - remove backticks if present
    const cleanPath = filepath.trim().replace(/^`|`$/g, '');
    // Only treat as filepath if it looks like one (has / or \ or is just a filename with extension)
    if (cleanPath.includes('/') || cleanPath.includes('\\') || /^\w+\.\w+$/.test(cleanPath)) {
      return `\n\`\`\`${language}::${cleanPath}`;
    }
    return match;
  });
  
  // Convert bare video URLs on their own line to markdown links
  // This ensures they get processed by the custom 'a' component
  const videoUrlPattern = /^(https?:\/\/(?:www\.)?(?:youtube\.com\/(?:watch\?v=|shorts\/|embed\/)|youtu\.be\/|rumble\.com\/(?:embed\/)?v)[^\s]+)$/gm;
  processed = processed.replace(videoUrlPattern, (match, url) => {
    return `[${url}](${url})`;
  });
  
  return processed;
}

// Map file extension to markdown language identifier
function extToMarkdownLang(ext: string): string {
  const map: Record<string, string> = {
    'py': 'python',
    'js': 'javascript',
    'ts': 'typescript',
    'tsx': 'tsx',
    'jsx': 'jsx',
    'cpp': 'cpp',
    'cc': 'cpp',
    'cxx': 'cpp',
    'c': 'c',
    'h': 'c',
    'hpp': 'cpp',
    'cs': 'csharp',
    'java': 'java',
    'kt': 'kotlin',
    'go': 'go',
    'rs': 'rust',
    'rb': 'ruby',
    'php': 'php',
    'swift': 'swift',
    'sh': 'bash',
    'bash': 'bash',
    'zsh': 'zsh',
    'ps1': 'powershell',
    'sql': 'sql',
    'html': 'html',
    'htm': 'html',
    'css': 'css',
    'scss': 'scss',
    'less': 'less',
    'json': 'json',
    'xml': 'xml',
    'yaml': 'yaml',
    'yml': 'yaml',
    'toml': 'toml',
    'md': 'markdown',
    'txt': 'text',
  };
  return map[ext.toLowerCase()] || ext.toLowerCase();
}

interface VersionInfo {
  current: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
}

interface MessageProps {
  message: Message;
  isStreaming?: boolean;
  toolCall?: { name: string; input: string } | null;
  toolCitations?: ToolCitation[];
  generatedImage?: GeneratedImage;
  onRetry?: () => void;
  onEdit?: (messageId: string, newContent: string) => void;
  onDelete?: (messageId: string) => void;
  onReadAloud?: (content: string) => void;
  onImageRetry?: (prompt: string, width?: number, height?: number, seed?: number) => void;
  onImageEdit?: (prompt: string) => void;
  isReadingAloud?: boolean;
  isLastAssistant?: boolean;
  onArtifactClick?: (artifact: Artifact) => void;
  onBranchChange?: (branchIndex: number) => void;
  assistantName?: string;
  versionInfo?: VersionInfo;
  // NC-0.8.0.0: Text selection hint handler
  onHintSelect?: (hint: UserHint, selectedText: string) => void;
  // Mermaid error callback for auto-fix
  onMermaidError?: (error: string, code: string, messageId: string) => void;
}

// Memoized message comparison - only re-render if content actually changed
function arePropsEqual(prevProps: MessageProps, nextProps: MessageProps): boolean {
  // Always re-render streaming messages
  if (prevProps.isStreaming || nextProps.isStreaming) return false;
  
  // Check message content
  if (prevProps.message.id !== nextProps.message.id) return false;
  if (prevProps.message.content !== nextProps.message.content) return false;
  if (prevProps.message.current_branch !== nextProps.message.current_branch) return false;
  
  // Check metadata changes (especially for image generation)
  const prevMeta = prevProps.message.metadata;
  const nextMeta = nextProps.message.metadata;
  if (prevMeta?.generated_image !== nextMeta?.generated_image) return false;
  if (prevMeta?.image_generation?.status !== nextMeta?.image_generation?.status) return false;
  
  // Check other props
  if (prevProps.isLastAssistant !== nextProps.isLastAssistant) return false;
  if (prevProps.toolCall !== nextProps.toolCall) return false;
  if (prevProps.assistantName !== nextProps.assistantName) return false;
  if (prevProps.onEdit !== nextProps.onEdit) return false;
  if (prevProps.onDelete !== nextProps.onDelete) return false;
  if (prevProps.isReadingAloud !== nextProps.isReadingAloud) return false;
  
  // Check generated image - re-render when image arrives
  if (prevProps.generatedImage !== nextProps.generatedImage) return false;
  
  // Check tool citations
  const prevCitations = prevProps.toolCitations?.length ?? 0;
  const nextCitations = nextProps.toolCitations?.length ?? 0;
  if (prevCitations !== nextCitations) return false;
  
  // Check version info
  if (prevProps.versionInfo?.current !== nextProps.versionInfo?.current) return false;
  if (prevProps.versionInfo?.total !== nextProps.versionInfo?.total) return false;
  
  // Check hint handler (NC-0.8.0.0)
  if (prevProps.onHintSelect !== nextProps.onHintSelect) return false;
  
  return true;
}

function MessageBubbleInner({ 
  message, 
  isStreaming, 
  toolCall, 
  toolCitations,
  generatedImage,
  onRetry,
  onEdit,
  onDelete,
  onReadAloud,
  onImageRetry,
  onImageEdit,
  isReadingAloud,
  isLastAssistant,
  onArtifactClick,
  onBranchChange,
  assistantName,
  versionInfo,
  onHintSelect,
  onMermaidError,
}: MessageProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  
  // Get mermaid rendering setting from branding store
  const mermaidEnabled = useBrandingStore((state) => state.config?.features?.mermaid_rendering ?? true);
  
  // Ref for text selection bubble
  const contentRef = useRef<HTMLDivElement>(null);
  
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const isTool = message.role === 'tool';
  const isAssistant = message.role === 'assistant';
  const isFileContent = message.metadata?.type === 'file_content';
  
  // Branch info
  const hasBranches = message.branches && message.branches.length > 0;
  const totalBranches = hasBranches ? message.branches!.length + 1 : 1;
  const currentBranch = message.current_branch ?? (hasBranches ? message.branches!.length : 0);
  
  const copyToClipboard = async (code: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const copyMessage = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handlePrevBranch = () => {
    if (onBranchChange && currentBranch > 0) {
      onBranchChange(currentBranch - 1);
    }
  };
  
  const handleNextBranch = () => {
    if (onBranchChange && currentBranch < totalBranches - 1) {
      onBranchChange(currentBranch + 1);
    }
  };
  
  const handleStartEdit = () => {
    setEditContent(message.content);
    setIsEditing(true);
  };
  
  const handleCancelEdit = () => {
    setEditContent(message.content);
    setIsEditing(false);
  };
  
  const handleSaveEdit = () => {
    if (onEdit && editContent.trim() && editContent !== message.content) {
      onEdit(message.id, editContent.trim());
    }
    setIsEditing(false);
  };
  
  const handleDelete = () => {
    if (onDelete) {
      onDelete(message.id);
    }
    setShowDeleteConfirm(false);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancelEdit();
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSaveEdit();
    }
  };
  
  // Find artifact references in content like [📦 Artifact: Title]
  const renderContentWithArtifacts = (content: string) => {
    const artifactPattern = /\[📦 Artifact: ([^\]]+)\]/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;
    let key = 0;
    
    while ((match = artifactPattern.exec(content)) !== null) {
      // Add text before the artifact reference
      if (match.index > lastIndex) {
        parts.push(content.slice(lastIndex, match.index));
      }
      
      // Add artifact button
      const artifactTitle = match[1];
      const artifact = message.artifacts?.find(a => a.title === artifactTitle);
      
      parts.push(
        <button
          key={`artifact-${key++}`}
          onClick={() => artifact && onArtifactClick?.(artifact)}
          className="inline-flex items-center gap-1.5 px-2 py-1 my-1 rounded-lg bg-[var(--color-button)]/80 border border-[var(--color-border)] text-[var(--color-button-text)] text-sm hover:bg-[var(--color-button)] transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
          </svg>
          <span className="font-medium">{artifactTitle}</span>
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </button>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < content.length) {
      parts.push(content.slice(lastIndex));
    }
    
    return parts.length > 0 ? parts : content;
  };
  
  if (isSystem) {
    return (
      <div className="py-2 px-4 text-center">
        <span className="text-xs text-[var(--color-text-secondary)] italic">
          {message.content}
        </span>
      </div>
    );
  }
  
  if (isTool) {
    return (
      <div className="py-2 pl-6 border-l-2 border-[var(--color-border)] mx-4 max-w-3xl">
        <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)] mb-1">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          Tool Result
        </div>
        <pre className="text-xs text-[var(--color-text-secondary)] whitespace-pre-wrap font-mono">
          {message.content}
        </pre>
      </div>
    );
  }
  
  // Check if content has artifact references
  const hasArtifactRefs = /\[📦 Artifact: [^\]]+\]/.test(message.content);
  const contentWithoutArtifacts = hasArtifactRefs 
    ? message.content.replace(/\[📦 Artifact: [^\]]+\]/g, '') 
    : message.content;
  
  // Preprocess content: fix nested code fences and extract filenames
  const processedContent = preprocessContent(contentWithoutArtifacts);
  
  return (
    <div className={`group py-4 md:py-4 ${isUser && !isFileContent ? 'bg-[var(--color-surface)]/30' : ''} ${isFileContent ? 'bg-blue-500/5 border-l-2 border-blue-500/50' : ''}`}>
      <div className="max-w-3xl mx-auto px-3 md:px-4">
        {/* Role label with branch navigation */}
        <div className="flex items-center gap-2 mb-2">
          <span className={`flex items-center gap-1.5 text-sm md:text-xs font-medium uppercase tracking-wide ${
            isFileContent ? 'text-blue-400' : 
            isUser ? 'text-[var(--color-primary)]' : 'text-[var(--color-secondary)]'
          }`}>
            {isFileContent && (
              <svg className="w-4 h-4 md:w-3.5 md:h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            )}
            {isFileContent ? `File: ${message.metadata?.path || 'Unknown'}` : 
             isUser ? 'You' : (assistantName || 'Assistant')}
          </span>
          {message.input_tokens != null && message.output_tokens != null && (
            <span className="text-sm md:text-xs text-[var(--color-text-secondary)]">
              · {message.input_tokens + message.output_tokens} tokens
            </span>
          )}
          
          {/* Branch navigation */}
          {hasBranches && totalBranches > 1 && (
            <div className="flex items-center gap-1 ml-2">
              <button
                onClick={handlePrevBranch}
                disabled={currentBranch === 0}
                className="p-0.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span className="text-xs text-[var(--color-text-secondary)] min-w-[40px] text-center">
                {currentBranch + 1}/{totalBranches}
              </span>
              <button
                onClick={handleNextBranch}
                disabled={currentBranch === totalBranches - 1}
                className="p-0.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
        </div>
        
        {/* Tool call indicator */}
        {toolCall && (
          <div className="mb-3 py-2 px-3 rounded border border-[var(--color-border)] bg-[var(--color-surface)]">
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Using: {toolCall.name}
            </div>
          </div>
        )}
        
        {/* Artifact references from content */}
        {hasArtifactRefs && message.artifacts && message.artifacts.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {message.artifacts.map((artifact) => (
              <button
                key={artifact.id}
                onClick={() => onArtifactClick?.(artifact)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--color-button)]/80 border border-[var(--color-border)] text-[var(--color-button-text)] text-sm hover:bg-[var(--color-button)] transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
                <span className="font-medium">{artifact.title}</span>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>
            ))}
          </div>
        )}
        
        {/* Edit mode */}
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full min-h-[100px] p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] resize-y focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              autoFocus
              placeholder="Edit your message..."
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--color-text-secondary)]">
                Ctrl+Enter to save, Esc to cancel
              </span>
              <div className="flex gap-2">
                <button
                  onClick={handleCancelEdit}
                  className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveEdit}
                  disabled={!editContent.trim() || editContent === message.content}
                  className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Save & Create Branch
                </button>
              </div>
            </div>
          </div>
        ) : processedContent ? (
          isStreaming ? (
            // During streaming: plain text for performance (skip expensive markdown parsing)
            <div className="prose prose-base md:prose-sm max-w-none prose-neutral dark:prose-invert text-[var(--color-text)] whitespace-pre-wrap">
              {processedContent}
              <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse ml-0.5 align-middle" />
            </div>
          ) : (
            // After streaming complete: full markdown rendering with text selection support
            <div ref={contentRef} className="relative prose prose-base md:prose-sm max-w-none prose-neutral dark:prose-invert text-[var(--color-text)]">
              {/* Text Selection Bubble - NC-0.8.0.0 */}
              {onHintSelect && (
                <TextSelectionBubble
                  containerRef={contentRef}
                  messageRole={message.role as 'user' | 'assistant'}
                  onHintSelect={onHintSelect}
                />
              )}
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, inline, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    const code = String(children).replace(/\n$/, '');
                    
                    // Extract filename from meta or first line comment
                    // Pattern: ```lang::/path/to/file or ```lang filename="/path/to/file"
                    let filename: string | null = null;
                    const metaMatch = className?.match(/language-\w+::(.+)/);
                    if (metaMatch) {
                      filename = metaMatch[1];
                    }
                    
                    if (!inline && match) {
                      const language = match[1];
                      
                      // Handler for mermaid errors
                      const handleMermaidError = (error: string, mermaidCode: string) => {
                        if (onMermaidError) {
                          onMermaidError(error, mermaidCode, message.id);
                        }
                      };
                      
                      // Special handling for mermaid diagrams (if enabled)
                      if (mermaidEnabled && (language === 'mermaid' || language === 'mmd')) {
                        const diagramId = `${message.id}-${Math.random().toString(36).substr(2, 9)}`;
                        return <MermaidDiagram code={code} id={diagramId} onError={handleMermaidError} />;
                      }
                      
                      // Detect mermaid syntax even without explicit language tag (if enabled)
                      const mermaidKeywords = /^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie|gitGraph|mindmap|timeline|quadrantChart|requirementDiagram|C4Context)\b/m;
                      if (mermaidEnabled && mermaidKeywords.test(code.trim())) {
                        const diagramId = `${message.id}-${Math.random().toString(36).substr(2, 9)}`;
                        return <MermaidDiagram code={code} id={diagramId} onError={handleMermaidError} />;
                      }
                      
                      return (
                        <div className="relative group/code my-3">
                          {/* Filename header */}
                          {filename && (
                            <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-800 border-b border-zinc-700 rounded-t-lg">
                              <span className="text-xs text-zinc-400 font-mono truncate">{filename}</span>
                              <button
                                onClick={() => copyToClipboard(code)}
                                className="px-2 py-0.5 text-xs rounded bg-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-600 transition-colors"
                              >
                                {copiedCode === code ? 'Copied!' : 'Copy'}
                              </button>
                            </div>
                          )}
                          {/* Copy button for blocks without filename */}
                          {!filename && (
                            <div className="absolute right-2 top-2 opacity-0 group-hover/code:opacity-100 transition-opacity z-10">
                              <button
                                onClick={() => copyToClipboard(code)}
                                className="px-2 py-1 text-xs rounded bg-zinc-700 text-zinc-300 hover:text-white"
                              >
                                {copiedCode === code ? 'Copied!' : 'Copy'}
                              </button>
                            </div>
                          )}
                          <SyntaxHighlighter
                            style={oneDark}
                            language={language}
                            PreTag="div"
                            customStyle={{
                              margin: 0,
                              borderRadius: filename ? '0 0 0.5rem 0.5rem' : '0.5rem',
                              fontSize: '0.8125rem',
                            }}
                            {...props}
                          >
                            {code}
                          </SyntaxHighlighter>
                        </div>
                      );
                    }
                    
                    return (
                      <code
                        className={`${className} px-1 py-0.5 rounded bg-zinc-800 text-[var(--color-accent)] text-sm`}
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  // Table components for GFM tables
                  table({ children }) {
                    return (
                      <div className="overflow-x-auto my-4 rounded-lg border border-[var(--color-border)]">
                        <table className="min-w-full divide-y divide-[var(--color-border)]">
                          {children}
                        </table>
                      </div>
                    );
                  },
                  thead({ children }) {
                    return (
                      <thead className="bg-[var(--color-surface)]">
                        {children}
                      </thead>
                    );
                  },
                  tbody({ children }) {
                    return (
                      <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-background)]">
                        {children}
                      </tbody>
                    );
                  },
                  tr({ children }) {
                    return (
                      <tr className="hover:bg-[var(--color-surface)]/50 transition-colors">
                        {children}
                      </tr>
                    );
                  },
                  th({ children }) {
                    return (
                      <th className="px-3 py-2 text-left text-xs font-semibold text-[var(--color-text)] uppercase tracking-wider whitespace-nowrap">
                        {children}
                      </th>
                    );
                  },
                  td({ children }) {
                    return (
                      <td className="px-3 py-2 text-sm text-[var(--color-text-secondary)] whitespace-normal">
                        {children}
                      </td>
                    );
                  },
                  p({ children }) {
                    return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>;
                  },
                  ul({ children }) {
                    return <ul className="list-disc list-outside ml-4 mb-2 space-y-0.5">{children}</ul>;
                  },
                  ol({ children }) {
                    return <ol className="list-decimal list-outside ml-4 mb-2 space-y-0.5">{children}</ol>;
                  },
                  a({ href, children }) {
                    // Check if this is a video URL that should be embedded
                    if (href) {
                      const videoInfo = extractVideoInfo(href);
                      if (videoInfo) {
                        // Render video embed instead of link
                        return <VideoEmbed url={href} />;
                      }
                    }
                    // Regular link
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[var(--color-accent)] hover:underline"
                      >
                        {children}
                      </a>
                    );
                  },
                  blockquote({ children }) {
                    return (
                      <blockquote className="border-l-2 border-[var(--color-border)] pl-4 italic text-[var(--color-text-secondary)]">
                        {children}
                      </blockquote>
                    );
                  },
                }}
              >
                {processedContent}
              </ReactMarkdown>
            </div>
          )
        ) : isStreaming ? (
          <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse" />
        ) : null}
        
        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.attachments.map((att, idx) => (
              <div key={idx}>
                {att.type === 'image' ? (
                  <img
                    src={att.url || `data:${att.mime_type};base64,${att.data}`}
                    alt={att.name}
                    className="max-w-sm max-h-64 rounded border border-[var(--color-border)]"
                  />
                ) : (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-sm">
                    <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span className="truncate max-w-[200px] text-[var(--color-text)]">{att.name}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        
        {/* Generated Image */}
        {/* Check both prop and message metadata for generated image */}
        {(() => {
          const img = generatedImage || (message.metadata?.generated_image as GeneratedImage | undefined);
          const imageGenStatus = message.metadata?.image_generation;
          
          // Show generating indicator if status is pending/processing
          if (imageGenStatus && (imageGenStatus.status === 'pending' || imageGenStatus.status === 'processing')) {
            return (
              <div className="mt-3 p-4 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)]">
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-6 w-6 border-2 border-[var(--color-primary)] border-t-transparent"></div>
                  <div>
                    <p className="text-sm font-medium text-[var(--color-text)]">
                      Generating image...
                    </p>
                    {imageGenStatus.queue_position && imageGenStatus.queue_position > 1 && (
                      <p className="text-xs text-[var(--color-text-secondary)]">
                        Queue position: {imageGenStatus.queue_position}
                      </p>
                    )}
                    {imageGenStatus.width && imageGenStatus.height && (
                      <p className="text-xs text-[var(--color-text-secondary)]">
                        {imageGenStatus.width} × {imageGenStatus.height}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          }
          
          // Image must have either base64 or url to render
          if (!img || (!img.base64 && !img.url)) return null;
          return (
            <GeneratedImageCard
              image={img}
              messageId={message.id}
              onRetry={onImageRetry}
              onEdit={onImageEdit}
            />
          );
        })()}
        
        {/* Tool Citations */}
        {toolCitations && toolCitations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
            <p className="text-xs text-[var(--color-text-secondary)] mb-2 flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              Sources
            </p>
            <div className="flex flex-wrap gap-2">
              {toolCitations.map((citation) => (
                <div
                  key={citation.id}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs ${
                    citation.success
                      ? 'bg-[var(--color-button)]/80 text-[var(--color-button-text)]'
                      : 'bg-red-500/10 text-red-400'
                  }`}
                >
                  <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span className="font-medium">{citation.tool_name}</span>
                  {citation.operation && citation.operation !== citation.tool_name && (
                    <span className="text-[var(--color-text-secondary)]">→ {citation.operation}</span>
                  )}
                  {citation.result_url && (
                    <a
                      href={citation.result_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-1 hover:underline text-[var(--color-secondary)]"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Delete confirmation modal */}
        {showDeleteConfirm && (
          <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
            <p className="text-sm text-[var(--color-text)] mb-2">
              Delete this message and all its replies?
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-3 py-1 text-sm rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="px-3 py-1 text-sm rounded bg-red-500 text-white hover:bg-red-600"
              >
                Delete Branch
              </button>
            </div>
          </div>
        )}
        
        {/* Action buttons - show for user and assistant messages */}
        {(isUser || isAssistant) && !isStreaming && !isEditing && (
          <div className="flex items-center justify-between mt-3">
            {/* Action buttons - always visible on mobile, hover-visible on desktop */}
            <div className="flex items-center gap-1 md:gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity">
              {/* Copy button */}
              <button
                onClick={copyMessage}
                className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] transition-colors touch-manipulation"
                title="Copy message"
              >
                {copied ? (
                  <svg className="w-5 h-5 md:w-4 md:h-4 text-[var(--color-success)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                )}
              </button>
              
              {/* Edit button */}
              {onEdit && (
                <button
                  onClick={handleStartEdit}
                  className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] transition-colors touch-manipulation"
                  title="Edit message (creates new branch)"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                </button>
              )}
              
              {/* Regenerate button (assistant only) */}
              {isAssistant && onRetry && (
                <button
                  onClick={onRetry}
                  className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] transition-colors touch-manipulation"
                  title="Regenerate response"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              )}
              
              {/* Read Aloud button (assistant only) */}
              {isAssistant && onReadAloud && (
                <button
                  onClick={() => onReadAloud(message.content)}
                  className={`p-2 md:p-1.5 rounded transition-colors touch-manipulation ${
                    isReadingAloud 
                      ? 'text-[var(--color-primary)] bg-[var(--color-primary)]/10' 
                      : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)]'
                  }`}
                  title={isReadingAloud ? "Stop reading" : "Read aloud"}
                >
                  {isReadingAloud ? (
                    <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    </svg>
                  )}
                </button>
              )}
              
              {/* Delete button */}
              {onDelete && (
                <button
                  onClick={() => setShowDeleteConfirm(true)}
                  className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-red-400 hover:bg-red-500/10 active:text-red-400 active:bg-red-500/10 transition-colors touch-manipulation"
                  title="Delete message and replies"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              )}
            </div>
            
            {/* Version pagination */}
            {versionInfo && versionInfo.total > 1 && (
              <div className="flex items-center gap-1 text-sm text-[var(--color-text-secondary)]">
                <button
                  onClick={versionInfo.onPrev}
                  disabled={versionInfo.current <= 1}
                  className="p-2 md:p-1 rounded hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors touch-manipulation"
                  title="Previous version"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <span className="px-1 min-w-[3rem] text-center">
                  {versionInfo.current}/{versionInfo.total}
                </span>
                <button
                  onClick={versionInfo.onNext}
                  disabled={versionInfo.current >= versionInfo.total}
                  className="p-2 md:p-1 rounded hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors touch-manipulation"
                  title="Next version"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Export memoized version
const MessageBubble = memo(MessageBubbleInner, arePropsEqual);
export default MessageBubble;
export type { ToolCitation };
