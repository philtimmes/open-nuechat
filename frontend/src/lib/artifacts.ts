import type { Artifact } from '../types';

// Parse artifacts from message content
// Supports formats like:
// 
// Pattern 1: Filename on line before code fence (common LLM pattern)
// src/components/Button.tsx
// ```tsx
// content
// ```
//
// Pattern 2: artifact:title:type format
// ```artifact:title:type
// content
// ```
//
// Pattern 3: XML-like artifact tags
// <artifact title="Title" type="code" language="python">
// content
// </artifact>
//
// Pattern 4: XML-style filename tags (for naive LLM output)
// <Menu.cpp>content</Menu.cpp>
// <artifact=Menu.cpp>content</artifact>
// <file:src/utils.py>content</file>
// <code name="app.js">content</code>
//
// Pattern 5: Large code blocks (500+ chars) auto-converted to artifacts

let artifactCounter = 0;

/**
 * Parse code blocks with proper nesting support.
 * Handles cases where content contains nested code fences (e.g., markdown files with code examples).
 * 
 * Rules:
 * - ```lang opens a fence (depth++)
 * - ``` alone (no language) closes a fence (depth--)
 * - Only close the outer block when depth returns to 0
 */
function parseCodeBlocks(content: string): Array<{
  fullMatch: string;
  startIndex: number;
  endIndex: number;
  filepath: string | null;
  language: string;
  code: string;
}> {
  const results: Array<{
    fullMatch: string;
    startIndex: number;
    endIndex: number;
    filepath: string | null;
    language: string;
    code: string;
  }> = [];
  
  const lines = content.split('\n');
  let i = 0;
  let charIndex = 0;
  
  while (i < lines.length) {
    const line = lines[i];
    const lineStart = charIndex;
    
    // Check for file path on this line and code fence on next line
    const nextLine = lines[i + 1];
    const isFilePath = looksLikeFilePath(line);
    const fenceMatch = nextLine?.match(/^```(\w*)$/);
    
    // Also check if this line itself starts a code fence
    const directFenceMatch = line.match(/^```(\w+)$/);
    
    if (isFilePath && fenceMatch && !isXmlStyleFileTag(line)) {
      // Pattern: filepath\n```lang\ncode\n```
      // Skip if line looks like an XML tag (e.g., <Menu.cpp>) - those are handled by Pattern 4
      const filepath = cleanFilePath(line);
      const lang = fenceMatch[1] || '';
      const fenceLineStart = charIndex + line.length + 1; // +1 for newline
      const codeStartLine = i + 2;
      const codeStartIndex = fenceLineStart + nextLine.length + 1;
      
      // Find the true closing fence using depth tracking
      let depth = 1;
      let endLine = codeStartLine;
      let j = codeStartLine;
      
      while (j < lines.length && depth > 0) {
        const l = lines[j];
        if (l.match(/^```\w+$/)) {
          // Opening a nested fence (has language identifier)
          depth++;
        } else if (l.match(/^```\s*$/)) {
          // Closing fence (no language)
          depth--;
        }
        if (depth > 0) {
          j++;
        }
      }
      endLine = j;
      
      if (depth === 0 && endLine < lines.length) {
        const codeLines = lines.slice(codeStartLine, endLine);
        const code = codeLines.join('\n');
        
        // Calculate indices
        let endIndex = codeStartIndex;
        for (let k = codeStartLine; k <= endLine; k++) {
          endIndex += lines[k].length + 1;
        }
        
        const fullMatch = lines.slice(i, endLine + 1).join('\n');
        
        results.push({
          fullMatch,
          startIndex: lineStart,
          endIndex: endIndex - 1,
          filepath,
          language: lang,
          code: code.trim(),
        });
        
        // Skip to after the closing fence
        charIndex = endIndex;
        i = endLine + 1;
        continue;
      }
    } else if (directFenceMatch) {
      // Pattern: ```lang\ncode\n``` (no filepath)
      const lang = directFenceMatch[1];
      const codeStartLine = i + 1;
      const codeStartIndex = charIndex + line.length + 1;
      
      // Find the true closing fence using depth tracking
      let depth = 1;
      let j = codeStartLine;
      
      while (j < lines.length && depth > 0) {
        const l = lines[j];
        if (l.match(/^```\w+$/)) {
          depth++;
        } else if (l.match(/^```\s*$/)) {
          depth--;
        }
        if (depth > 0) {
          j++;
        }
      }
      const endLine = j;
      
      if (depth === 0 && endLine < lines.length) {
        const codeLines = lines.slice(codeStartLine, endLine);
        const code = codeLines.join('\n');
        
        const fullMatch = lines.slice(i, endLine + 1).join('\n');
        
        // Calculate end index
        let endIndex = codeStartIndex;
        for (let k = codeStartLine; k <= endLine; k++) {
          endIndex += lines[k].length + 1;
        }
        
        results.push({
          fullMatch,
          startIndex: lineStart,
          endIndex: endIndex - 1,
          filepath: null,
          language: lang,
          code: code.trim(),
        });
        
        charIndex = endIndex;
        i = endLine + 1;
        continue;
      }
    }
    
    // Move to next line
    charIndex += line.length + 1;
    i++;
  }
  
  return results;
}

/**
 * Clean up a filepath by removing angle brackets and trailing punctuation
 * Handles patterns like: <filename.py>, `filename.py`, filename.py:, etc.
 */
export function cleanFilePath(filepath: string): string {
  let cleaned = filepath.trim();
  
  // Remove surrounding angle brackets
  if (cleaned.startsWith('<') && cleaned.endsWith('>')) {
    cleaned = cleaned.slice(1, -1);
  }
  // Remove just leading or trailing angle brackets
  cleaned = cleaned.replace(/^[<>]+|[<>]+$/g, '');
  
  // Remove surrounding backticks
  if (cleaned.startsWith('`') && cleaned.endsWith('`')) {
    cleaned = cleaned.slice(1, -1);
  }
  cleaned = cleaned.replace(/^`+|`+$/g, '');
  
  // Remove trailing punctuation (colon, semicolon, comma, period if not part of extension)
  // But preserve the extension's dot
  cleaned = cleaned.replace(/[:;,]+$/, '');
  
  // Remove trailing period only if there's already an extension
  // e.g., "file.py." -> "file.py" but "file" stays "file"
  if (cleaned.match(/\.\w+\.$/)) {
    cleaned = cleaned.slice(0, -1);
  }
  
  // Remove any remaining angle brackets inside the path
  cleaned = cleaned.replace(/[<>]/g, '');
  
  return cleaned.trim();
}

function looksLikeFilePath(line: string): boolean {
  // Clean first, then check
  const trimmed = cleanFilePath(line);
  if (!trimmed) return false;
  
  // Must have extension or path separator
  if (!trimmed.includes('/') && !trimmed.includes('.')) return false;
  
  // Skip if it looks like a sentence (has spaces but no path separators)
  if (trimmed.includes(' ') && !trimmed.includes('/') && !trimmed.includes('\\')) return false;
  
  // Skip common false positives
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return false;
  if (trimmed.startsWith('#')) return false;
  if (trimmed.startsWith('*')) return false;
  if (trimmed.startsWith('-')) return false;
  
  return true;
}

// Check if a line is an XML-style opening tag for a file (e.g., <Menu.cpp>, <file:test.py>)
// These should be handled by Pattern 4, not Pattern 1
function isXmlStyleFileTag(line: string): boolean {
  const trimmed = line.trim();
  // Match: <filename.ext> or <tag:filename.ext> or <tag=filename.ext>
  return /^<(?:artifact|file|code)?[=:]?[A-Za-z_][\w\-./]*\.\w+>$/.test(trimmed);
}

export function extractArtifacts(content: string): { cleanContent: string; artifacts: Artifact[] } {
  const artifacts: Artifact[] = [];
  const lines = content.split('\n');
  const outputLines: string[] = [];
  
  // State machine - only one parse method active at a time
  // 0 = none, 1 = filepath+fence, 2 = artifact:title:type, 3 = <artifact>, 4 = <filename.ext>, 5 = large code block
  let activeMethod = 0;
  let buffer: string[] = [];
  let metadata: {
    filepath?: string;
    filename?: string;
    language?: string;
    title?: string;
    type?: string;
    tagName?: string;
    fenceDepth?: number;
    startLine?: number;
    prefixText?: string;  // Text before artifact tag on same line
  } = {};
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const nextLine = lines[i + 1];
    
    // If no active method, try to detect pattern starts
    if (activeMethod === 0) {
      // Pattern 4a: <filename.ext> (direct filename as tag)
      const xmlFileTagMatch = line.match(/^<([A-Za-z_][\w\-./]*\.\w+)>\s*$/);
      if (xmlFileTagMatch) {
        activeMethod = 4;
        metadata = { 
          tagName: xmlFileTagMatch[1],
          filepath: xmlFileTagMatch[1],
          filename: xmlFileTagMatch[1].split('/').pop() || xmlFileTagMatch[1],
          startLine: i
        };
        buffer = [];
        continue;
      }
      
      // Pattern 4b: <artifact=filename> or <file:filename> etc
      // Can have text before the tag (e.g., "File main.cpp // comment <artifact=src/main.cpp>")
      const xmlAttrTagMatch = line.match(/^(.*?)<(artifact|file|code)[=:]([A-Za-z_][\w\-./]*\.\w+)>\s*$/i);
      if (xmlAttrTagMatch) {
        const prefixText = xmlAttrTagMatch[1].trim();
        activeMethod = 4;
        metadata = {
          tagName: xmlAttrTagMatch[2].toLowerCase(),
          filepath: xmlAttrTagMatch[3],
          filename: xmlAttrTagMatch[3].split('/').pop() || xmlAttrTagMatch[3],
          startLine: i,
          prefixText: prefixText, // Store prefix to output later
        };
        buffer = [];
        continue;
      }
      
      // Pattern 4c: <artifact name="filename"> attribute style
      const xmlNameAttrMatch = line.match(/^<(artifact|file|code)\s+(?:name|filename|file|path)=["']([A-Za-z_][\w\-./]*\.\w+)["'][^>]*>\s*$/i);
      if (xmlNameAttrMatch) {
        activeMethod = 4;
        metadata = {
          tagName: xmlNameAttrMatch[1].toLowerCase(),
          filepath: xmlNameAttrMatch[2],
          filename: xmlNameAttrMatch[2].split('/').pop() || xmlNameAttrMatch[2],
          startLine: i
        };
        buffer = [];
        continue;
      }
      
      // Pattern 3: <artifact title="..." type="...">
      const xmlArtifactMatch = line.match(/^<artifact\s+(?:title="([^"]*)")?\s*(?:type="([^"]*)")?\s*(?:language="([^"]*)")?\s*(?:filename="([^"]*)")?[^>]*>\s*$/i);
      if (xmlArtifactMatch) {
        activeMethod = 3;
        metadata = {
          title: xmlArtifactMatch[1] || 'Untitled',
          type: xmlArtifactMatch[2] || 'code',
          language: xmlArtifactMatch[3],
          filename: xmlArtifactMatch[4],
          startLine: i
        };
        buffer = [];
        continue;
      }
      
      // Pattern 2: ```artifact:title:type
      const artifactFenceMatch = line.match(/^```artifact:([^:\n]+):(\w+)(?::(\w+))?$/);
      if (artifactFenceMatch) {
        activeMethod = 2;
        metadata = {
          title: artifactFenceMatch[1].trim(),
          type: artifactFenceMatch[2],
          language: artifactFenceMatch[3],
          fenceDepth: 1,
          startLine: i
        };
        buffer = [];
        continue;
      }
      
      // Pattern 1: filepath followed by code fence on next line
      // Skip if line looks like XML tag
      if (nextLine && looksLikeFilePath(line) && !isXmlStyleFileTag(line)) {
        const fenceMatch = nextLine.match(/^```(\w*)$/);
        if (fenceMatch) {
          activeMethod = 1;
          metadata = {
            filepath: cleanFilePath(line),
            filename: cleanFilePath(line).split('/').pop() || cleanFilePath(line),
            language: fenceMatch[1] || '',
            fenceDepth: 1,
            startLine: i
          };
          buffer = [];
          i++; // Skip the fence line
          continue;
        }
      }
      
      // Pattern 5: Standalone code fence (will check size at end)
      const standaloneFenceMatch = line.match(/^```(\w+)$/);
      if (standaloneFenceMatch) {
        activeMethod = 5;
        metadata = {
          language: standaloneFenceMatch[1],
          fenceDepth: 1,
          startLine: i
        };
        buffer = [];
        continue;
      }
      
      // No pattern matched, output line as-is
      outputLines.push(line);
      continue;
    }
    
    // Active method - look for closing
    
    // Pattern 4: XML-style tags - look for closing tag
    if (activeMethod === 4) {
      // Check for various closing tag formats
      // Escape special regex chars in filepath/filename
      const escapedPath = (metadata.filepath || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const escapedName = (metadata.filename || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const escapedTag = (metadata.tagName || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      
      const closePatterns = [
        new RegExp(`^<\\/${escapedPath}>\\s*$`),              // </path/to/filename.ext>
        new RegExp(`^<\\/${escapedName}>\\s*$`),              // </filename.ext>
        new RegExp(`^<\\/${escapedTag}>\\s*$`, 'i'),          // </artifact>, </file>, </code>
        new RegExp(`^<\\/${escapedTag}[=:]${escapedPath}>\\s*$`, 'i'),  // </file:filename>
        new RegExp(`^<\\/${escapedTag}[=:]${escapedName}>\\s*$`, 'i'),  // </file:filename> (just name)
      ];
      
      const isClosing = closePatterns.some(p => p.test(line));
      
      if (isClosing) {
        // Extract artifact
        let code = buffer.join('\n').trim();
        // Strip code fences if present inside
        const fenceMatch = code.match(/^```\w*\n([\s\S]*?)\n?```$/);
        if (fenceMatch) {
          code = fenceMatch[1].trim();
        }
        
        const ext = metadata.filename?.split('.').pop()?.toLowerCase() || '';
        const language = extToLang(ext);
        const type = langToType(language);
        
        if (code.length > 0) {
          artifacts.push({
            id: `artifact-${Date.now()}-${++artifactCounter}`,
            title: metadata.filename || 'Untitled',
            type,
            language,
            content: code,
            filename: metadata.filepath,
            created_at: new Date().toISOString(),
          });
          // Include prefix text (e.g., "File main.cpp // comment") before artifact placeholder
          const prefix = metadata.prefixText ? `${metadata.prefixText} ` : '';
          outputLines.push(`${prefix}[📦 Artifact: ${metadata.filename}]`);
        }
        
        activeMethod = 0;
        metadata = {};
        buffer = [];
        continue;
      }
      
      buffer.push(line);
      continue;
    }
    
    // Pattern 3: <artifact> tags
    if (activeMethod === 3) {
      if (line.match(/^<\/artifact>\s*$/i)) {
        const code = buffer.join('\n').trim();
        const artType = normalizeType(metadata.type || 'code');
        
        if (code.length > 0) {
          artifacts.push({
            id: `artifact-${Date.now()}-${++artifactCounter}`,
            title: metadata.title || 'Untitled',
            type: artType,
            language: metadata.language || inferLanguage(artType, code),
            content: code,
            filename: metadata.filename || generateFilename(metadata.title || 'Untitled', artType, metadata.language),
            created_at: new Date().toISOString(),
          });
          outputLines.push(`[📦 Artifact: ${metadata.title}]`);
        }
        
        activeMethod = 0;
        metadata = {};
        buffer = [];
        continue;
      }
      
      buffer.push(line);
      continue;
    }
    
    // Pattern 2: artifact:title:type fence
    if (activeMethod === 2) {
      // Track nested fences
      if (line.match(/^```\w+$/)) {
        metadata.fenceDepth = (metadata.fenceDepth || 1) + 1;
        buffer.push(line);
        continue;
      }
      if (line.match(/^```\s*$/)) {
        metadata.fenceDepth = (metadata.fenceDepth || 1) - 1;
        if (metadata.fenceDepth === 0) {
          const code = buffer.join('\n').trim();
          const artType = normalizeType(metadata.type || 'code');
          
          if (code.length > 0) {
            artifacts.push({
              id: `artifact-${Date.now()}-${++artifactCounter}`,
              title: metadata.title || 'Untitled',
              type: artType,
              language: metadata.language || inferLanguage(artType, code),
              content: code,
              filename: generateFilename(metadata.title || 'Untitled', artType, metadata.language),
              created_at: new Date().toISOString(),
            });
            outputLines.push(`[📦 Artifact: ${metadata.title}]`);
          }
          
          activeMethod = 0;
          metadata = {};
          buffer = [];
          continue;
        }
      }
      
      buffer.push(line);
      continue;
    }
    
    // Pattern 1: filepath + code fence
    if (activeMethod === 1) {
      // Track nested fences
      if (line.match(/^```\w+$/)) {
        metadata.fenceDepth = (metadata.fenceDepth || 1) + 1;
        buffer.push(line);
        continue;
      }
      if (line.match(/^```\s*$/)) {
        metadata.fenceDepth = (metadata.fenceDepth || 1) - 1;
        if (metadata.fenceDepth === 0) {
          const code = buffer.join('\n').trim();
          const ext = metadata.filename?.split('.').pop()?.toLowerCase() || '';
          const language = metadata.language || extToLang(ext);
          const type = langToType(language);
          
          if (code.length > 0) {
            artifacts.push({
              id: `artifact-${Date.now()}-${++artifactCounter}`,
              title: metadata.filename || 'Untitled',
              type,
              language,
              content: code,
              filename: metadata.filepath,
              created_at: new Date().toISOString(),
            });
            outputLines.push(`[📦 Artifact: ${metadata.filename}]`);
          }
          
          activeMethod = 0;
          metadata = {};
          buffer = [];
          continue;
        }
      }
      
      buffer.push(line);
      continue;
    }
    
    // Pattern 5: Standalone code fence (large blocks)
    if (activeMethod === 5) {
      // Track nested fences
      if (line.match(/^```\w+$/)) {
        metadata.fenceDepth = (metadata.fenceDepth || 1) + 1;
        buffer.push(line);
        continue;
      }
      if (line.match(/^```\s*$/)) {
        metadata.fenceDepth = (metadata.fenceDepth || 1) - 1;
        if (metadata.fenceDepth === 0) {
          const code = buffer.join('\n').trim();
          const language = metadata.language || 'text';
          const type = langToType(language);
          
          // Only convert to artifact if large enough (500+ chars)
          if (code.length >= 500) {
            const title = `${language.charAt(0).toUpperCase() + language.slice(1)} Code`;
            artifacts.push({
              id: `artifact-${Date.now()}-${++artifactCounter}`,
              title,
              type,
              language,
              content: code,
              filename: generateFilename(title, type, language),
              created_at: new Date().toISOString(),
            });
            outputLines.push(`[📦 Artifact: ${title}]`);
          } else {
            // Small code block - output as-is
            outputLines.push(`\`\`\`${metadata.language}`);
            outputLines.push(...buffer);
            outputLines.push('```');
          }
          
          activeMethod = 0;
          metadata = {};
          buffer = [];
          continue;
        }
      }
      
      buffer.push(line);
      continue;
    }
  }
  
  // Handle unclosed patterns at end of content
  if (activeMethod !== 0 && buffer.length > 0) {
    // For Pattern 4 (XML tags), try to extract artifact if we have a complete code block
    if (activeMethod === 4) {
      let code = buffer.join('\n').trim();
      // Strip any trailing extra backticks (common LLM mistake)
      code = code.replace(/\n```\s*$/g, '\n```').replace(/```\s*\n```\s*$/, '```');
      // Check if buffer contains a complete code fence (don't require $ anchor - LLM may add trailing content)
      const fenceMatch = code.match(/^```(\w*)\n([\s\S]*?)\n```/);
      if (fenceMatch && fenceMatch[2].trim().length > 0) {
        // We have a complete code block - extract as artifact
        code = fenceMatch[2].trim();
        const ext = metadata.filename?.split('.').pop()?.toLowerCase() || '';
        const language = extToLang(ext);
        const type = langToType(language);
        
        artifacts.push({
          id: `artifact-${Date.now()}-${++artifactCounter}`,
          title: metadata.filename || 'Untitled',
          type,
          language,
          content: code,
          filename: metadata.filepath,
          created_at: new Date().toISOString(),
        });
        // Include prefix text before artifact placeholder
        const prefix = metadata.prefixText ? `${metadata.prefixText} ` : '';
        outputLines.push(`${prefix}[📦 Artifact: ${metadata.filename}]`);
      } else {
        // Incomplete - output as-is with prefix
        const prefix = metadata.prefixText ? `${metadata.prefixText} ` : '';
        outputLines.push(`${prefix}<${metadata.tagName || metadata.filepath}>`);
        outputLines.push(...buffer);
      }
    } else if (activeMethod === 3) {
      outputLines.push(`<artifact>`);
      outputLines.push(...buffer);
    } else if (activeMethod === 2) {
      outputLines.push(`\`\`\`artifact:${metadata.title}:${metadata.type}`);
      outputLines.push(...buffer);
    } else if (activeMethod === 1) {
      outputLines.push(metadata.filepath || '');
      outputLines.push(`\`\`\`${metadata.language || ''}`);
      outputLines.push(...buffer);
    } else if (activeMethod === 5) {
      outputLines.push(`\`\`\`${metadata.language || ''}`);
      outputLines.push(...buffer);
    }
  }
  
  return { cleanContent: outputLines.join('\n'), artifacts };
}

function normalizeType(type: string): Artifact['type'] {
  const t = type.toLowerCase();
  switch (t) {
    case 'html':
    case 'webpage':
      return 'html';
    case 'react':
    case 'jsx':
    case 'tsx':
      return 'react';
    case 'svg':
    case 'image':
      return 'svg';
    case 'markdown':
    case 'md':
      return 'markdown';
    case 'mermaid':
    case 'diagram':
      return 'mermaid';
    case 'json':
      return 'json';
    case 'csv':
      return 'csv';
    case 'document':
    case 'doc':
      return 'document';
    default:
      return 'code';
  }
}

function langToType(lang: string): Artifact['type'] {
  switch (lang.toLowerCase()) {
    case 'html':
      return 'html';
    case 'jsx':
    case 'tsx':
      return 'react';
    case 'svg':
      return 'svg';
    case 'markdown':
    case 'md':
      return 'markdown';
    case 'mermaid':
      return 'mermaid';
    case 'json':
      return 'json';
    case 'csv':
      return 'csv';
    default:
      return 'code';
  }
}

function extToLang(ext: string): string {
  switch (ext.toLowerCase()) {
    case 'py': return 'python';
    case 'js': return 'javascript';
    case 'ts': return 'typescript';
    case 'tsx': return 'tsx';
    case 'jsx': return 'jsx';
    case 'html': return 'html';
    case 'htm': return 'html';
    case 'css': return 'css';
    case 'scss': return 'scss';
    case 'less': return 'less';
    case 'json': return 'json';
    case 'md': return 'markdown';
    case 'svg': return 'svg';
    case 'xml': return 'xml';
    case 'yaml': 
    case 'yml': return 'yaml';
    case 'toml': return 'toml';
    case 'sh': 
    case 'bash': return 'bash';
    case 'zsh': return 'zsh';
    case 'fish': return 'fish';
    case 'ps1': return 'powershell';
    case 'bat': 
    case 'cmd': return 'batch';
    case 'go': return 'go';
    case 'rs': return 'rust';
    case 'rb': return 'ruby';
    case 'php': return 'php';
    case 'java': return 'java';
    case 'kt': 
    case 'kts': return 'kotlin';
    case 'swift': return 'swift';
    case 'c': return 'c';
    case 'h': return 'c';
    case 'cpp': 
    case 'cc': 
    case 'cxx': return 'cpp';
    case 'hpp': 
    case 'hxx': return 'cpp';
    case 'cs': return 'csharp';
    case 'sql': return 'sql';
    case 'r': return 'r';
    case 'lua': return 'lua';
    case 'pl': return 'perl';
    case 'ex': 
    case 'exs': return 'elixir';
    case 'erl': return 'erlang';
    case 'hs': return 'haskell';
    case 'ml': return 'ocaml';
    case 'clj': return 'clojure';
    case 'scala': return 'scala';
    case 'vue': return 'vue';
    case 'svelte': return 'svelte';
    case 'dockerfile': return 'dockerfile';
    case 'makefile': return 'makefile';
    case 'env': return 'dotenv';
    default: return ext || 'text';
  }
}

function inferLanguage(type: string, content: string): string | undefined {
  if (type === 'code') {
    // Try to infer from content
    if (content.includes('def ') || content.includes('import ')) return 'python';
    if (content.includes('function ') || content.includes('const ')) return 'javascript';
    if (content.includes('interface ') || content.includes(': string')) return 'typescript';
    if (content.includes('package ') || content.includes('func ')) return 'go';
    if (content.includes('fn ') || content.includes('let mut')) return 'rust';
  }
  return undefined;
}

function generateFilename(title: string, type: string, language?: string): string {
  const base = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  
  const ext = (() => {
    switch (type) {
      case 'html': return 'html';
      case 'react': return 'jsx';
      case 'svg': return 'svg';
      case 'markdown': return 'md';
      case 'mermaid': return 'mmd';
      case 'json': return 'json';
      case 'csv': return 'csv';
      case 'code':
        switch (language) {
          case 'python': return 'py';
          case 'javascript': return 'js';
          case 'typescript': return 'ts';
          case 'go': return 'go';
          case 'rust': return 'rs';
          case 'java': return 'java';
          case 'cpp':
          case 'c++': return 'cpp';
          case 'c': return 'c';
          case 'ruby': return 'rb';
          case 'php': return 'php';
          case 'swift': return 'swift';
          case 'kotlin': return 'kt';
          case 'sql': return 'sql';
          case 'shell':
          case 'bash': return 'sh';
          default: return 'txt';
        }
      default: return 'txt';
    }
  })();
  
  return `${base}.${ext}`;
}

// Collect all artifacts from a chat for download
// Uses full filepath as key, so files with same name in different dirs are kept
export function collectChatArtifacts(
  messages: { content: string; artifacts?: Artifact[]; created_at: string }[]
): Map<string, Artifact> {
  const artifactMap = new Map<string, Artifact>();
  
  // Process messages in order, newer overwrites older (for same path)
  for (const msg of messages) {
    // Direct artifacts on the message
    if (msg.artifacts) {
      for (const art of msg.artifacts) {
        // Use filename (which may include path) as primary key
        const key = art.filename || art.title;
        artifactMap.set(key, art);
      }
    }
    
    // Extract from content
    const { artifacts } = extractArtifacts(msg.content);
    for (const art of artifacts) {
      const key = art.filename || art.title;
      artifactMap.set(key, art);
    }
  }
  
  return artifactMap;
}

// Group artifacts by filename, with versions sorted by timestamp (newest first)
// Deduplicates artifacts with identical content, keeping only the oldest version
export interface ArtifactGroup {
  filename: string;
  displayName: string;
  type: Artifact['type'];
  language?: string;
  versions: Artifact[];
  latestVersion: Artifact;
}

export function groupArtifactsByFilename(artifacts: Artifact[]): ArtifactGroup[] {
  const groups = new Map<string, Artifact[]>();
  
  // Group by filename
  for (const artifact of artifacts) {
    const key = artifact.filename || artifact.title;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(artifact);
  }
  
  // Convert to array and sort versions by timestamp (newest first)
  const result: ArtifactGroup[] = [];
  
  for (const [filename, versions] of groups) {
    // Sort versions by created_at ascending (oldest first) for deduplication
    const sortedByOldest = [...versions].sort((a, b) => {
      const dateA = new Date(a.created_at || 0).getTime();
      const dateB = new Date(b.created_at || 0).getTime();
      return dateA - dateB;
    });
    
    // Deduplicate by content - keep oldest version when content is identical
    const seenContent = new Set<string>();
    const uniqueVersions: Artifact[] = [];
    
    for (const version of sortedByOldest) {
      if (!seenContent.has(version.content)) {
        seenContent.add(version.content);
        uniqueVersions.push(version);
      }
    }
    
    // Sort unique versions by newest first for display
    const sortedVersions = uniqueVersions.sort((a, b) => {
      const dateA = new Date(a.created_at || 0).getTime();
      const dateB = new Date(b.created_at || 0).getTime();
      return dateB - dateA;
    });
    
    const latest = sortedVersions[0];
    const displayName = filename.includes('/') 
      ? filename.split('/').pop() || filename
      : filename;
    
    result.push({
      filename,
      displayName,
      type: latest.type,
      language: latest.language,
      versions: sortedVersions,
      latestVersion: latest,
    });
  }
  
  // Sort groups by most recent update
  return result.sort((a, b) => {
    const dateA = new Date(a.latestVersion.created_at || 0).getTime();
    const dateB = new Date(b.latestVersion.created_at || 0).getTime();
    return dateB - dateA;
  });
}

// Get only the latest version of each unique file (for downloads)
export function getLatestArtifacts(artifacts: Artifact[]): Artifact[] {
  const groups = groupArtifactsByFilename(artifacts);
  return groups.map(g => g.latestVersion);
}
