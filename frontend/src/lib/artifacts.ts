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
    
    if (isFilePath && fenceMatch) {
      // Pattern: filepath\n```lang\ncode\n```
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

export function extractArtifacts(content: string): { cleanContent: string; artifacts: Artifact[] } {
  const artifacts: Artifact[] = [];
  let cleanContent = content;
  
  // Parse code blocks with proper nesting support
  const codeBlocks = parseCodeBlocks(content);
  
  // Pattern 1: File path + code blocks (e.g., "src/file.ts" followed by ```ts)
  for (const block of codeBlocks) {
    if (block.filepath) {
      const filename = block.filepath.split('/').pop() || block.filepath;
      const ext = filename.split('.').pop()?.toLowerCase() || '';
      const language = block.language || extToLang(ext);
      const type = langToType(language);
      
      artifacts.push({
        id: `artifact-${Date.now()}-${++artifactCounter}`,
        title: filename,
        type,
        language,
        content: block.code,
        filename: block.filepath,
        created_at: new Date().toISOString(),
      });
      cleanContent = cleanContent.replace(block.fullMatch, `\n[📦 Artifact: ${filename}]`);
    }
  }
  
  // Pattern 2: ```artifact:title:type format
  const codeBlockPattern = /```artifact:([^:\n]+):(\w+)(?::(\w+))?\n([\s\S]*?)```/g;
  let match;
  
  while ((match = codeBlockPattern.exec(content)) !== null) {
    const [fullMatch, title, type, language, code] = match;
    // Skip if already processed
    if (!cleanContent.includes(fullMatch)) continue;
    
    artifacts.push({
      id: `artifact-${Date.now()}-${++artifactCounter}`,
      title: title.trim(),
      type: normalizeType(type),
      language: language || inferLanguage(type, code),
      content: code.trim(),
      filename: generateFilename(title, type, language),
      created_at: new Date().toISOString(),
    });
    cleanContent = cleanContent.replace(fullMatch, `[📦 Artifact: ${title.trim()}]`);
  }
  
  // Pattern 3: XML-like <artifact> tags
  const xmlPattern = /<artifact\s+(?:title="([^"]+)")?\s*(?:type="([^"]+)")?\s*(?:language="([^"]+)")?\s*(?:filename="([^"]+)")?[^>]*>([\s\S]*?)<\/artifact>/gi;
  
  while ((match = xmlPattern.exec(content)) !== null) {
    const [fullMatch, title, type, language, filename, code] = match;
    // Skip if already processed
    if (!cleanContent.includes(fullMatch)) continue;
    
    const artTitle = title || 'Untitled';
    const artType = normalizeType(type || 'code');
    
    artifacts.push({
      id: `artifact-${Date.now()}-${++artifactCounter}`,
      title: artTitle,
      type: artType,
      language: language || inferLanguage(artType, code),
      content: code.trim(),
      filename: filename || generateFilename(artTitle, artType, language),
      created_at: new Date().toISOString(),
    });
    cleanContent = cleanContent.replace(fullMatch, `[📦 Artifact: ${artTitle}]`);
  }
  
  // Pattern 4: XML-style filename tags (multiple variations for naive LLM output)
  // Supports:
  //   <Menu.cpp>...</Menu.cpp>           - filename as tag
  //   <artifact=Menu.cpp>...</artifact>  - artifact tag with = filename
  //   <artifact:Menu.cpp>...</artifact>  - artifact tag with : filename
  //   <file=Menu.cpp>...</file>          - file tag with = filename
  //   <code=Menu.cpp>...</code>          - code tag with = filename
  //   <artifact name="Menu.cpp">...</artifact> - attribute style
  
  // Pattern 4a: Direct filename as tag: <Menu.cpp>...</Menu.cpp>
  const xmlFilenamePattern = /<([A-Za-z_][\w\-./]*\.\w+)>\s*([\s\S]*?)\s*<\/\1>/g;
  
  while ((match = xmlFilenamePattern.exec(content)) !== null) {
    const [fullMatch, tagName, code] = match;
    if (!cleanContent.includes(fullMatch)) continue;
    
    const filepath = tagName;
    const filename = filepath.split('/').pop() || filepath;
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const language = extToLang(ext);
    const type = langToType(language);
    
    artifacts.push({
      id: `artifact-${Date.now()}-${++artifactCounter}`,
      title: filename,
      type,
      language,
      content: code.trim(),
      filename: filepath,
      created_at: new Date().toISOString(),
    });
    cleanContent = cleanContent.replace(fullMatch, `[📦 Artifact: ${filename}]`);
  }
  
  // Pattern 4b: <artifact=filename> or <file=filename> or <code=filename> style
  // Closing tag can be </artifact>, </file>, </code>, or </filename>
  const xmlAttrPattern = /<(artifact|file|code)[=:]([A-Za-z_][\w\-./]*\.\w+)>\s*([\s\S]*?)\s*<\/(?:\1|\2)>/gi;
  
  while ((match = xmlAttrPattern.exec(content)) !== null) {
    const [fullMatch, tagType, tagFilename, code] = match;
    if (!cleanContent.includes(fullMatch)) continue;
    
    const filepath = tagFilename;
    const filename = filepath.split('/').pop() || filepath;
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const language = extToLang(ext);
    const type = langToType(language);
    
    artifacts.push({
      id: `artifact-${Date.now()}-${++artifactCounter}`,
      title: filename,
      type,
      language,
      content: code.trim(),
      filename: filepath,
      created_at: new Date().toISOString(),
    });
    cleanContent = cleanContent.replace(fullMatch, `[📦 Artifact: ${filename}]`);
  }
  
  // Pattern 4c: <artifact name="filename"> or <file name="filename"> attribute style
  const xmlNameAttrPattern = /<(artifact|file|code)\s+(?:name|filename|file|path)=["']([A-Za-z_][\w\-./]*\.\w+)["'][^>]*>\s*([\s\S]*?)\s*<\/\1>/gi;
  
  while ((match = xmlNameAttrPattern.exec(content)) !== null) {
    const [fullMatch, tagType, tagFilename, code] = match;
    if (!cleanContent.includes(fullMatch)) continue;
    
    const filepath = tagFilename;
    const filename = filepath.split('/').pop() || filepath;
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const language = extToLang(ext);
    const type = langToType(language);
    
    artifacts.push({
      id: `artifact-${Date.now()}-${++artifactCounter}`,
      title: filename,
      type,
      language,
      content: code.trim(),
      filename: filepath,
      created_at: new Date().toISOString(),
    });
    cleanContent = cleanContent.replace(fullMatch, `[📦 Artifact: ${filename}]`);
  }
  
  // Pattern 5: Detect large code blocks and convert to artifacts
  // Use the parsed blocks to handle this correctly
  for (const block of codeBlocks) {
    if (!block.filepath && block.code.length >= 500) {
      const fullMatch = block.fullMatch;
      // Skip if already processed
      if (!cleanContent.includes(fullMatch)) continue;
      
      const language = block.language || 'text';
      const type = langToType(language);
      const title = `${language.charAt(0).toUpperCase() + language.slice(1)} Code`;
      
      artifacts.push({
        id: `artifact-${Date.now()}-${++artifactCounter}`,
        title,
        type,
        language,
        content: block.code,
        filename: generateFilename(title, type, language),
        created_at: new Date().toISOString(),
      });
      cleanContent = cleanContent.replace(fullMatch, `[📦 Artifact: ${title}]`);
    }
  }
  
  return { cleanContent, artifacts };
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
