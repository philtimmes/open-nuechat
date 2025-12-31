/**
 * CodeEditor - Monaco-like code editor component
 * 
 * Features:
 * - Syntax highlighting (via CSS classes)
 * - Line numbers
 * - AI suggestions overlay
 * - Lint error markers
 * - Tab to accept suggestions
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface LintError {
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

interface CodeEditorProps {
  content: string;
  language: string;
  onChange: (content: string) => void;
  onCursorChange: (position: { line: number; column: number }) => void;
  suggestions: string[];
  showSuggestions: boolean;
  onAcceptSuggestion: (suggestion: string) => void;
  onDismissSuggestions: () => void;
  lintErrors: LintError[];
}

export default function CodeEditor({
  content,
  language,
  onChange,
  onCursorChange,
  suggestions,
  showSuggestions,
  onAcceptSuggestion,
  onDismissSuggestions,
  lintErrors,
}: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLSpanElement>(null);
  const [selectedSuggestion, setSelectedSuggestion] = useState(0);
  const [cursorPosition, setCursorPosition] = useState({ top: 0, left: 0 });
  const [charWidth, setCharWidth] = useState(8.4);
  
  // Map language names
  const languageMap: Record<string, string> = {
    'typescript': 'tsx',
    'javascript': 'jsx',
    'python': 'python',
    'rust': 'rust',
    'go': 'go',
    'java': 'java',
    'cpp': 'cpp',
    'c': 'c',
    'css': 'css',
    'scss': 'scss',
    'html': 'html',
    'json': 'json',
    'yaml': 'yaml',
    'markdown': 'markdown',
    'sql': 'sql',
    'shell': 'bash',
    'plaintext': 'text',
  };
  
  const highlightLanguage = languageMap[language] || 'text';
  
  // Measure actual character width on mount
  useEffect(() => {
    if (measureRef.current) {
      const width = measureRef.current.getBoundingClientRect().width / 10; // 10 chars
      if (width > 0) {
        setCharWidth(width);
      }
    }
  }, []);
  
  // Handle cursor position change
  const handleCursorChange = useCallback(() => {
    const textarea = textareaRef.current;
    const editor = editorRef.current;
    if (!textarea || !editor) return;
    
    const selectionStart = textarea.selectionStart;
    const textBeforeCursor = content.substring(0, selectionStart);
    const lines = textBeforeCursor.split('\n');
    const line = lines.length;
    const column = lines[lines.length - 1].length + 1;
    
    onCursorChange({ line, column });
    
    // Calculate cursor pixel position relative to viewport
    const lineHeight = 20;
    const lineNumberWidth = 56; // Width of line number gutter
    const padding = 12;
    
    // Get editor's position on screen
    const editorRect = editor.getBoundingClientRect();
    const scrollTop = editor.scrollTop;
    const scrollLeft = editor.scrollLeft;
    
    // Calculate position relative to editor, accounting for scroll
    const relativeTop = (line - 1) * lineHeight + padding;
    const relativeLeft = (column - 1) * charWidth + lineNumberWidth + padding;
    
    // Convert to viewport coordinates
    const top = editorRect.top + relativeTop - scrollTop;
    const left = editorRect.left + relativeLeft - scrollLeft;
    
    setCursorPosition({ top, left });
  }, [content, onCursorChange, charWidth]);
  
  // Handle key events
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Tab to accept suggestion
    if (e.key === 'Tab' && showSuggestions && suggestions.length > 0) {
      e.preventDefault();
      onAcceptSuggestion(suggestions[selectedSuggestion]);
      return;
    }
    
    // Navigate suggestions
    if (showSuggestions && suggestions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedSuggestion((prev) => Math.min(prev + 1, suggestions.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedSuggestion((prev) => Math.max(prev - 1, 0));
        return;
      }
      if (e.key === 'Escape') {
        onDismissSuggestions();
        return;
      }
    }
    
    // Handle tab for indentation when no suggestions
    if (e.key === 'Tab' && !showSuggestions) {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (!textarea) return;
      
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newContent = content.substring(0, start) + '  ' + content.substring(end);
      onChange(newContent);
      
      // Move cursor after tab
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 2;
      }, 0);
    }
  };
  
  // Handle input change
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };
  
  // Update cursor position on various events
  useEffect(() => {
    const textarea = textareaRef.current;
    const editor = editorRef.current;
    if (!textarea || !editor) return;
    
    const updateCursor = () => handleCursorChange();
    
    textarea.addEventListener('click', updateCursor);
    textarea.addEventListener('keyup', updateCursor);
    editor.addEventListener('scroll', updateCursor);
    
    return () => {
      textarea.removeEventListener('click', updateCursor);
      textarea.removeEventListener('keyup', updateCursor);
      editor.removeEventListener('scroll', updateCursor);
    };
  }, [handleCursorChange]);
  
  // Reset selected suggestion when suggestions change
  useEffect(() => {
    setSelectedSuggestion(0);
  }, [suggestions]);
  
  const lines = content.split('\n');
  const lineCount = lines.length;
  
  // Calculate minimum height based on content
  const contentHeight = Math.max(lineCount * 20 + 24, 300); // 20px per line + padding
  
  return (
    <div ref={editorRef} className="relative h-full bg-[#1e1e1e] overflow-auto font-mono text-sm">
      {/* Hidden element to measure character width */}
      <span 
        ref={measureRef}
        className="absolute opacity-0 pointer-events-none"
        style={{
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
          fontSize: '14px',
        }}
      >
        0123456789
      </span>
      
      {/* Content wrapper - sets minimum size for scrolling */}
      <div 
        className="relative"
        style={{ minHeight: `${contentHeight}px`, minWidth: 'max-content' }}
      >
        {/* Line numbers */}
        <div 
          className="absolute left-0 top-0 w-14 bg-[#1e1e1e] border-r border-[#333] select-none z-10"
          style={{ paddingTop: '12px', minHeight: `${contentHeight}px` }}
        >
          {Array.from({ length: lineCount }, (_, i) => (
            <div
              key={i}
              className={`text-right pr-3 leading-5 ${
                lintErrors.some(e => e.line === i + 1) 
                  ? 'text-red-500' 
                  : 'text-[#858585]'
              }`}
            >
              {i + 1}
            </div>
          ))}
        </div>
        
        {/* Editor area */}
        <div className="ml-14" style={{ minHeight: `${contentHeight}px` }}>
          {/* Syntax highlighting layer (visual only) */}
          <div 
            className="absolute left-14 top-0 right-0 pointer-events-none"
            style={{ padding: '12px' }}
          >
            <SyntaxHighlighter
              language={highlightLanguage}
              style={vscDarkPlus}
              customStyle={{
                background: 'transparent',
                margin: 0,
                padding: 0,
                fontSize: '14px',
                lineHeight: '20px',
              }}
              codeTagProps={{
                style: {
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                },
              }}
            >
              {content || ' '}
            </SyntaxHighlighter>
          </div>
          
          {/* Textarea for editing */}
          <textarea
            ref={textareaRef}
            value={content}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            className="w-full bg-transparent text-transparent caret-white resize-none outline-none"
            style={{
              padding: '12px',
              fontSize: '14px',
              lineHeight: '20px',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
              tabSize: 2,
              minHeight: `${contentHeight}px`,
            }}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
          />
          
          {/* Lint error markers */}
          {lintErrors.map((error, i) => (
            <div
              key={i}
              className="absolute group"
              style={{
                top: (error.line - 1) * 20 + 12,
                left: (error.column - 1) * charWidth + 12,
              }}
            >
              <div 
                className={`w-2 h-2 rounded-full ${
                  error.severity === 'error' ? 'bg-red-500' :
                  error.severity === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                }`}
              />
              <div className="absolute left-4 top-0 hidden group-hover:block z-50 bg-[#333] text-white text-xs p-2 rounded shadow-lg whitespace-nowrap">
                {error.message}
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Suggestions overlay - positioned relative to viewport */}
      {showSuggestions && suggestions.length > 0 && (
        <div
          className="fixed z-50 bg-[#252526] border border-[#454545] rounded shadow-lg max-w-md overflow-hidden"
          style={{
            top: Math.min(cursorPosition.top + 20, window.innerHeight - 200), // +20 to appear below cursor
            left: Math.max(8, Math.min(cursorPosition.left, window.innerWidth - 400)),
          }}
        >
          <div className="px-2 py-1 text-xs text-[#858585] border-b border-[#454545]">
            Suggestions (Tab to accept)
          </div>
          {suggestions.slice(0, 5).map((suggestion, i) => (
            <div
              key={i}
              onClick={() => onAcceptSuggestion(suggestion)}
              className={`px-3 py-2 cursor-pointer text-sm font-mono ${
                i === selectedSuggestion
                  ? 'bg-[#094771] text-white'
                  : 'text-[#d4d4d4] hover:bg-[#2a2d2e]'
              }`}
            >
              <span className="text-[#858585]">→ </span>
              {suggestion.length > 60 ? suggestion.substring(0, 60) + '...' : suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
