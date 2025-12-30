import { createContext, useContext, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useAuthStore } from '../stores/authStore';
import { useChatStore } from '../stores/chatStore';
import { useBrandingStore } from '../stores/brandingStore';
import { chatApi } from '../lib/api';
import api from '../lib/api';
import { extractArtifacts, cleanFilePath } from '../lib/artifacts';
import { createFileChangeFromCode } from '../lib/signatures';
import { 
  parseServerEvent, 
  isStreamStart, isStreamChunk, isStreamEnd, isStreamError,
  isToolCall, isToolResult, isImageGeneration, isMessageSaved,
  extractErrorMessage
} from '../lib/wsTypes';
import type { Message, StreamChunk, WSMessage, ZipFileResponse, Artifact } from '../types';

// Regex patterns to detect file request tags in LLM responses
// Supports multiple formats LLMs might output:
// - <request_file path="file.cpp"/>
// - <request_file path='file.cpp'/>
// - <request_file path=file.cpp/>  (no quotes)
// - <request_file path=file.cpp>   (no self-close)
// - <request_file=path="file.cpp"/> (equals variant)
const FILE_REQUEST_PATTERNS = [
  /<request_file\s+path=["']([^"']+)["']\s*\/?>/gi,      // path="..." or path='...'
  /<request_file\s+path=([^\s>\/]+)\s*\/?>/gi,           // path=value (no quotes)
  /<request_file=path=["']([^"']+)["']\s*\/?>/gi,        // =path="..." variant
  /<request_file=path=([^\s>\/]+)\s*\/?>/gi,             // =path=value variant
];

// Regex patterns for search/replace operations
// Format 1: <replace_line path="file" find="search" replace="replacement">
// Format 2: <replace_line path="file" find="search" replace="replacement"/>
const REPLACE_LINE_PATTERNS = [
  /<replace_line\s+path=["']([^"']+)["']\s+find=["']([^"']+)["']\s+replace=["']([^"']*)["']\s*\/?>/gi,
  /<replace_line\s+path=([^\s>]+)\s+find=["']([^"']+)["']\s+replace=["']([^"']*)["']\s*\/?>/gi,
];

// Format: <replace_block path="file">\n===== SEARCH\nold code\n===== REPLACE\nnew code\n</replace_block>
const REPLACE_BLOCK_REGEX = /<replace_block\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*REPLACE\s*\n([\s\S]*?)\n?<\/replace_block>/gi;

// ============ NEW ARTIFACT OPERATIONS ============

// <find_line path="file" contains="text or regex"/>
const FIND_LINE_PATTERNS = [
  /<find_line\s+path=["']([^"']+)["']\s+contains=["']([^"']+)["']\s*\/?>/gi,
  /<find_line\s+path=([^\s>\/]+)\s+contains=["']([^"']+)["']\s*\/?>/gi,
  /<find_line\s+path=["']([^"']+)["']\s+contains=([^\s>\/]+)\s*\/?>/gi,
];

// <find path="optional/path" search="regex or text"/>
// <find search="regex or text" path="optional/path"/>
// <find search="regex or text"/>
const FIND_PATTERNS = [
  /<find\s+path=["']([^"']+)["']\s+search=["']([^"']+)["']\s*\/?>/gi,
  /<find\s+path=([^\s>\/]+)\s+search=["']([^"']+)["']\s*\/?>/gi,
  /<find\s+search=["']([^"']+)["']\s+path=["']([^"']+)["']\s*\/?>/gi,  // Reversed order
  /<find\s+search=["']([^"']+)["']\s+path=([^\s>\/]+)\s*\/?>/gi,  // Reversed, unquoted path
  /<find\s+search=["']([^"']+)["']\s*\/?>/gi,  // No path - search all
];

// <search_replace path="file">===== SEARCH\nold\n===== Replace\nnew\n</search_replace>
const SEARCH_REPLACE_REGEX = /<search_replace\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*[Rr]eplace\s*\n([\s\S]*?)\n?<\/search_replace>/gi;

// ============ STREAMING DETECTION PATTERNS ============
// These detect COMPLETE tool tags during streaming for immediate execution

// Single-line tool tags - match on closing > or /> (the /? handles self-closing tags)
const STREAM_FIND_LINE_PATTERN = /<find_line\s+path=["']([^"']+)["']\s+contains=["']([^"']+)["']\s*\/?>/i;
const STREAM_FIND_PATTERN_WITH_PATH = /<find\s+path=["']([^"']+)["']\s+search=["']([^"']+)["']\s*\/?>/i;
const STREAM_FIND_PATTERN_WITH_PATH_REV = /<find\s+search=["']([^"']+)["']\s+path=["']([^"']+)["']\s*\/?>/i;  // Reversed
const STREAM_FIND_PATTERN_NO_PATH = /<find\s+search=["']([^"']+)["']\s*\/?>/i;
const STREAM_REQUEST_FILE_PATTERN = /<request_file\s+path=["']([^"']+)["']\s*\/?>/i;
const STREAM_KB_SEARCH_PATTERN = /<kb_search\s+query=["']([^"']+)["']\s*\/?>/i;

// Multi-line tool tag - match on closing </search_replace>
const STREAM_SEARCH_REPLACE_PATTERN = /<search_replace\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*[Rr]eplace\s*\n([\s\S]*?)<\/search_replace>/i;

// Multi-line tool tag - match on closing </replace_block> (legacy format)
const STREAM_REPLACE_BLOCK_PATTERN = /<replace_block\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*REPLACE\s*\n([\s\S]*?)<\/replace_block>/i;

// ============ INCOMPLETE TOOL TAG PATTERNS ============
// These match tool tags that were started but never closed (LLM stopped mid-output)
// Used at stream_end to salvage operations even without closing tags

// Incomplete search_replace - has SEARCH and Replace sections but no closing tag
const INCOMPLETE_SEARCH_REPLACE_PATTERN = /<search_replace\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*[Rr]eplace\s*\n([\s\S]*)$/i;

// Incomplete replace_block - has SEARCH and REPLACE sections but no closing tag
const INCOMPLETE_REPLACE_BLOCK_PATTERN = /<replace_block\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*REPLACE\s*\n([\s\S]*)$/i;

// Artifact detection patterns for streaming notification
// Pattern: <artifact title="..." type="...">...</artifact>
const STREAM_ARTIFACT_XML_PATTERN = /<artifact\s+[^>]*title=["']([^"']+)["'][^>]*>([\s\S]*?)<\/artifact>/gi;
// Pattern: <artifact=filename>...</artifact> or <artifact= filename>...</artifact>
const STREAM_ARTIFACT_EQUALS_PATTERN = /<artifact=\s*([^>]+?)>([\s\S]*?)<\/artifact>/gi;
// Pattern: Filename tags like <Menu.cpp>...</Menu.cpp>
const STREAM_ARTIFACT_FILENAME_PATTERN = /<([A-Za-z_][A-Za-z0-9_]*\.[a-zA-Z]{1,10})>([\s\S]*?)<\/\1>/gi;
// Pattern: Code fence with filename on preceding line (most common LLM pattern)
// Matches: filename.ext\n```lang\ncode\n```
const STREAM_CODE_FENCE_PATTERN = /(?:^|\n)([^\n`]+\.(?:ts|tsx|js|jsx|py|cpp|h|hpp|c|cs|java|go|rs|rb|php|vue|svelte|html|css|scss|json|yaml|yml|md|sql|sh|bash))\s*\n```[a-z]*\n[\s\S]*?\n```/gi;

// Types for replace operations
interface ReplaceLineOp {
  type: 'line';
  path: string;
  find: string;
  replace: string;
}

interface ReplaceBlockOp {
  type: 'block';
  path: string;
  search: string;
  replace: string;
}

type ReplaceOp = ReplaceLineOp | ReplaceBlockOp;

// ============ NEW OPERATION TYPES ============

interface FindLineOp {
  path: string;
  contains: string;
}

interface FindOp {
  path: string | null;  // null means search all artifacts
  search: string;
}

interface SearchReplaceOp {
  path: string;
  search: string;
  replace: string;
}

// ============ CONTEXT WINDOW CHUNKING ============
// When tool results exceed 1/4 of context window, split into hidden agent files
// This prevents overflowing the LLM context while keeping data searchable

// ~32k chars ≈ 8k tokens, which is 1/4 of a 32k context window
// Adjust based on your typical model context size
const CONTEXT_CHUNK_THRESHOLD = 32000;
const CONTEXT_CHUNK_SIZE = 24000; // Size of each chunk (leave room for metadata)

// Agent file naming - matches backend convention: {AgentNNNN}.md
const AGENT_FILE_PREFIX = "{Agent";
const AGENT_FILE_SUFFIX = "}.md";

// Counter for unique agent file names within a session
let agentFileCounter = 0;

/**
 * Chunks large tool results into hidden agent files.
 * Returns the artifacts to add and a summary message for the LLM.
 */
function chunkLargeToolResult(
  toolName: string,
  results: string,
): { summary: string; chunkedFiles: string[]; newArtifacts: Artifact[] } {
  const chunkedFiles: string[] = [];
  const newArtifacts: Artifact[] = [];
  
  // Split into chunks, trying to break at newlines
  const chunks: string[] = [];
  let remaining = results;
  
  while (remaining.length > 0) {
    if (remaining.length <= CONTEXT_CHUNK_SIZE) {
      chunks.push(remaining);
      break;
    }
    
    // Try to find a good break point (double newline, then single newline)
    let breakPoint = remaining.lastIndexOf('\n\n', CONTEXT_CHUNK_SIZE);
    if (breakPoint < CONTEXT_CHUNK_SIZE / 2) {
      breakPoint = remaining.lastIndexOf('\n', CONTEXT_CHUNK_SIZE);
    }
    if (breakPoint < CONTEXT_CHUNK_SIZE / 2) {
      breakPoint = CONTEXT_CHUNK_SIZE;
    }
    
    chunks.push(remaining.substring(0, breakPoint));
    remaining = remaining.substring(breakPoint).trimStart();
  }
  
  // Create hidden artifacts for each chunk
  const timestamp = new Date().toISOString();
  for (let i = 0; i < chunks.length; i++) {
    const fileNum = String(++agentFileCounter).padStart(4, '0');
    const filename = `${AGENT_FILE_PREFIX}${fileNum}${AGENT_FILE_SUFFIX}`;
    
    const artifact: Artifact = {
      id: `agent-${Date.now()}-${fileNum}`,
      type: 'markdown',
      title: `Agent Context ${fileNum}`,
      language: 'markdown',
      content: `# Agent Context File ${fileNum}\n## Source: ${toolName} (Part ${i + 1} of ${chunks.length})\n\n${chunks[i]}`,
      filename,
      created_at: timestamp,
      source: 'generated',
      hidden: true,
    };
    
    newArtifacts.push(artifact);
    chunkedFiles.push(filename);
  }
  
  // Create summary for LLM
  const totalChars = results.length;
  const summary = `[LARGE RESULT - CHUNKED INTO ${chunks.length} FILES]
The ${toolName} result was ${totalChars.toLocaleString()} characters (too large for context).
Data has been split into the following searchable files:
${chunkedFiles.map((f, i) => `  - ${f} (Part ${i + 1})`).join('\n')}

Use <find search="..."/> or <request_file path="${chunkedFiles[0]}"/> to search/view these files.
First 500 chars preview:
${results.substring(0, 500)}${results.length > 500 ? '...' : ''}`;
  
  return { summary, chunkedFiles, newArtifacts };
}

// Extract all file request paths from content
function extractFileRequests(content: string): string[] {
  const paths: string[] = [];
  const seen = new Set<string>();
  
  for (const regex of FILE_REQUEST_PATTERNS) {
    let match;
    regex.lastIndex = 0; // Reset before each use
    while ((match = regex.exec(content)) !== null) {
      const path = match[1].trim();
      if (path && !seen.has(path)) {
        seen.add(path);
        paths.push(path);
      }
    }
  }
  
  return paths;
}

// Extract all replace operations from content
function extractReplaceOperations(content: string): ReplaceOp[] {
  const ops: ReplaceOp[] = [];
  
  // Extract replace_line operations
  for (const regex of REPLACE_LINE_PATTERNS) {
    let match;
    regex.lastIndex = 0;
    while ((match = regex.exec(content)) !== null) {
      ops.push({
        type: 'line',
        path: match[1].trim(),
        find: match[2],
        replace: match[3],
      });
    }
  }
  
  // Extract replace_block operations
  REPLACE_BLOCK_REGEX.lastIndex = 0;
  let blockMatch;
  while ((blockMatch = REPLACE_BLOCK_REGEX.exec(content)) !== null) {
    ops.push({
      type: 'block',
      path: blockMatch[1].trim(),
      search: blockMatch[2],
      replace: blockMatch[3],
    });
  }
  
  return ops;
}

// Apply replace operations to artifacts
function applyReplaceOperations(
  artifacts: Artifact[],
  ops: ReplaceOp[]
): { updatedArtifacts: Artifact[]; results: string[] } {
  const results: string[] = [];
  const artifactMap = new Map<string, Artifact>();
  
  // Build map for quick lookup
  for (const art of artifacts) {
    const key = (art.filename || art.title || '').replace(/^\.?\//, '');
    artifactMap.set(key, art);
    // Also add just the filename for easier matching
    const justName = key.split('/').pop() || key;
    if (justName !== key) {
      artifactMap.set(justName, art);
    }
  }
  
  // Track which artifacts were modified
  const modifiedArtifacts = new Set<string>();
  
  for (const op of ops) {
    const normalizedPath = op.path.replace(/^\.?\//, '');
    const artifact = artifactMap.get(normalizedPath) || 
                     artifactMap.get(normalizedPath.split('/').pop() || '');
    
    if (!artifact) {
      results.push(`[REPLACE_ERROR: File not found: ${op.path}]`);
      continue;
    }
    
    const artKey = artifact.filename || artifact.title || '';
    
    if (op.type === 'line') {
      // Simple string replacement (first occurrence)
      if (artifact.content.includes(op.find)) {
        artifact.content = artifact.content.replace(op.find, op.replace);
        modifiedArtifacts.add(artKey);
        results.push(`[REPLACED in ${artKey}: "${op.find.substring(0, 30)}${op.find.length > 30 ? '...' : ''}" → "${op.replace.substring(0, 30)}${op.replace.length > 30 ? '...' : ''}"]`);
      } else {
        results.push(`[REPLACE_ERROR: String not found in ${artKey}: "${op.find.substring(0, 50)}${op.find.length > 50 ? '...' : ''}"]`);
      }
    } else if (op.type === 'block') {
      // Block replacement - normalize whitespace for matching
      const searchNormalized = op.search.trim();
      const contentLines = artifact.content.split('\n');
      const searchLines = searchNormalized.split('\n');
      
      // Try to find the block
      let found = false;
      for (let i = 0; i <= contentLines.length - searchLines.length; i++) {
        const candidateBlock = contentLines.slice(i, i + searchLines.length).join('\n');
        if (candidateBlock.trim() === searchNormalized) {
          // Found it - replace
          const before = contentLines.slice(0, i);
          const after = contentLines.slice(i + searchLines.length);
          const replaceLines = op.replace.trim().split('\n');
          artifact.content = [...before, ...replaceLines, ...after].join('\n');
          modifiedArtifacts.add(artKey);
          found = true;
          results.push(`[REPLACED BLOCK in ${artKey}: ${searchLines.length} lines → ${replaceLines.length} lines]`);
          break;
        }
      }
      
      if (!found) {
        // Try exact match as fallback
        if (artifact.content.includes(op.search)) {
          artifact.content = artifact.content.replace(op.search, op.replace);
          modifiedArtifacts.add(artKey);
          results.push(`[REPLACED BLOCK in ${artKey} (exact match)]`);
        } else {
          results.push(`[REPLACE_ERROR: Block not found in ${artKey}]`);
        }
      }
    }
  }
  
  return { updatedArtifacts: artifacts, results };
}

// ============ NEW OPERATION EXTRACTORS ============

// Extract find_line operations from content
function extractFindLineOperations(content: string): FindLineOp[] {
  const ops: FindLineOp[] = [];
  const seen = new Set<string>();
  
  for (const regex of FIND_LINE_PATTERNS) {
    let match;
    regex.lastIndex = 0;
    while ((match = regex.exec(content)) !== null) {
      const key = `${match[1]}:${match[2]}`;
      if (!seen.has(key)) {
        seen.add(key);
        ops.push({
          path: match[1].trim(),
          contains: match[2],
        });
      }
    }
  }
  
  return ops;
}

// Extract find operations from content
function extractFindOperations(content: string): FindOp[] {
  const ops: FindOp[] = [];
  const seen = new Set<string>();
  
  for (const regex of FIND_PATTERNS) {
    let match;
    regex.lastIndex = 0;
    while ((match = regex.exec(content)) !== null) {
      // Check if this pattern has path (2 groups) or not (1 group)
      const hasPath = match.length > 2 && regex.source.includes('path=');
      
      let path: string | null = null;
      let search: string;
      
      if (hasPath) {
        // Check if pattern has search before path (reversed order)
        const isReversed = regex.source.indexOf('search=') < regex.source.indexOf('path=');
        if (isReversed) {
          search = match[1];  // search is first group
          path = match[2].trim();  // path is second group
        } else {
          path = match[1].trim();  // path is first group
          search = match[2];  // search is second group
        }
      } else {
        search = match[1];
      }
      
      const key = `${path || '*'}:${search}`;
      
      if (!seen.has(key)) {
        seen.add(key);
        ops.push({ path, search });
      }
    }
  }
  
  return ops;
}

// Extract search_replace operations from content
function extractSearchReplaceOperations(content: string): SearchReplaceOp[] {
  const ops: SearchReplaceOp[] = [];
  
  SEARCH_REPLACE_REGEX.lastIndex = 0;
  let match;
  while ((match = SEARCH_REPLACE_REGEX.exec(content)) !== null) {
    ops.push({
      path: match[1].trim(),
      search: match[2],
      replace: match[3],
    });
  }
  
  return ops;
}

// Extract INCOMPLETE search_replace/replace_block operations (missing closing tag)
// Used as fallback at stream_end when LLM stopped mid-output
function extractIncompleteSearchReplaceOperations(content: string): SearchReplaceOp[] {
  const ops: SearchReplaceOp[] = [];
  
  // Only try incomplete patterns if we have an opening tag but no closing tag
  const hasSearchReplaceOpen = content.includes('<search_replace');
  const hasSearchReplaceClose = content.includes('</search_replace>');
  const hasReplaceBlockOpen = content.includes('<replace_block');
  const hasReplaceBlockClose = content.includes('</replace_block>');
  
  // Try incomplete search_replace
  if (hasSearchReplaceOpen && !hasSearchReplaceClose) {
    const match = content.match(INCOMPLETE_SEARCH_REPLACE_PATTERN);
    if (match) {
      console.log('[INCOMPLETE_TOOL] Found incomplete search_replace, closing it');
      ops.push({
        path: match[1].trim(),
        search: match[2],
        replace: match[3].trimEnd(), // Trim trailing whitespace from incomplete content
      });
    }
  }
  
  // Try incomplete replace_block
  if (hasReplaceBlockOpen && !hasReplaceBlockClose) {
    const match = content.match(INCOMPLETE_REPLACE_BLOCK_PATTERN);
    if (match) {
      console.log('[INCOMPLETE_TOOL] Found incomplete replace_block, closing it');
      ops.push({
        path: match[1].trim(),
        search: match[2],
        replace: match[3].trimEnd(),
      });
    }
  }
  
  return ops;
}

// ============ NEW OPERATION EXECUTORS ============

// Helper to find artifact by path
function findArtifactByPath(artifacts: Artifact[], requestPath: string): Artifact | undefined {
  const normalizedRequest = requestPath.replace(/^\.?\//, '').toLowerCase();
  const reqBasename = requestPath.split('/').pop()?.toLowerCase() || '';
  
  return artifacts.find(art => {
    const artFilename = art.filename || art.title || '';
    const normalizedArt = artFilename.replace(/^\.?\//, '').toLowerCase();
    const artBasename = artFilename.split('/').pop()?.toLowerCase() || '';
    
    // Try various matching strategies
    return normalizedArt === normalizedRequest ||  // exact match
           normalizedArt.endsWith('/' + normalizedRequest) ||  // art has longer path
           normalizedRequest.endsWith('/' + normalizedArt) ||  // request has longer path
           artBasename === reqBasename ||  // just filename match (case insensitive)
           normalizedArt.includes(normalizedRequest) ||  // partial match
           normalizedRequest.includes(normalizedArt);  // reverse partial match
  });
}

// Helper to try regex or do literal search
// Auto-detects regex if pattern contains common metacharacters (.*  .+  \d  ^  $  etc)
function matchesPattern(line: string, pattern: string): boolean {
  // Explicit regex: wrapped in /pattern/ or /pattern/flags
  const explicitRegexMatch = pattern.match(/^\/(.+)\/([gi]*)$/);
  if (explicitRegexMatch) {
    try {
      const regex = new RegExp(explicitRegexMatch[1], explicitRegexMatch[2] || 'i');
      return regex.test(line);
    } catch {
      // Invalid regex, fall through to auto-detect
    }
  }
  
  // Auto-detect regex: if pattern contains common regex metacharacters
  const hasRegexChars = /\.\*|\.\+|\\\w|[\^\$\[\]\(\)\|\?]/.test(pattern);
  if (hasRegexChars) {
    try {
      const regex = new RegExp(pattern, 'i');
      return regex.test(line);
    } catch {
      // Invalid regex, fall through to literal
    }
  }
  
  // Default: literal case-insensitive search
  return line.toLowerCase().includes(pattern.toLowerCase());
}

// Execute find_line operations - returns line numbers with matches
function executeFindLineOperations(
  artifacts: Artifact[],
  ops: FindLineOp[]
): string[] {
  const results: string[] = [];
  
  // Check if we have any artifacts to search
  if (artifacts.length === 0) {
    results.push('[FIND_LINE_ERROR: No files available to search. Upload files or create artifacts first.]');
    return results;
  }
  
  for (const op of ops) {
    const artifact = findArtifactByPath(artifacts, op.path);
    
    if (!artifact) {
      // List available files to help the user
      const availableFiles = artifacts.map(a => a.filename || a.title || 'unnamed').slice(0, 10);
      results.push(`[FIND_LINE_ERROR: File not found: ${op.path}]\nAvailable files: ${availableFiles.join(', ')}${artifacts.length > 10 ? '...' : ''}`);
      continue;
    }
    
    const artName = artifact.filename || artifact.title || op.path;
    const lines = artifact.content.split('\n');
    const matchingLines: number[] = [];
    
    for (let i = 0; i < lines.length; i++) {
      if (matchesPattern(lines[i], op.contains)) {
        matchingLines.push(i + 1); // 1-indexed line numbers
      }
    }
    
    if (matchingLines.length === 0) {
      results.push(`[FIND_LINE: No matches for "${op.contains}" in ${artName}]`);
    } else if (matchingLines.length <= 10) {
      // Show line numbers and content for small result sets
      const lineDetails = matchingLines.map(lineNum => {
        const lineContent = lines[lineNum - 1].trim();
        const truncated = lineContent.length > 60 ? lineContent.substring(0, 60) + '...' : lineContent;
        return `  Line ${lineNum}: ${truncated}`;
      }).join('\n');
      results.push(`[FIND_LINE: ${matchingLines.length} match(es) in ${artName}]\n${lineDetails}`);
    } else {
      // Just show line numbers for large result sets
      results.push(`[FIND_LINE: ${matchingLines.length} matches in ${artName} at lines: ${matchingLines.slice(0, 20).join(', ')}${matchingLines.length > 20 ? '...' : ''}]`);
    }
  }
  
  return results;
}

// Execute find operations - returns matching files with search matches
function executeFindOperations(
  artifacts: Artifact[],
  ops: FindOp[]
): string[] {
  const results: string[] = [];
  
  // Check if we have any artifacts to search
  if (artifacts.length === 0) {
    results.push('[FIND_ERROR: No files available to search. Upload files or create artifacts first.]');
    return results;
  }
  
  for (const op of ops) {
    // Determine which artifacts to search
    let artifactsToSearch: Artifact[];
    
    if (op.path) {
      // Search within a specific path/directory
      const normalizedPath = op.path.replace(/^\.?\//, '').replace(/\/$/, '');
      artifactsToSearch = artifacts.filter(art => {
        const artPath = (art.filename || art.title || '').replace(/^\.?\//, '');
        return artPath.startsWith(normalizedPath + '/') || artPath === normalizedPath;
      });
      
      if (artifactsToSearch.length === 0) {
        // Try to find a single file match
        const singleMatch = findArtifactByPath(artifacts, op.path);
        if (singleMatch) {
          artifactsToSearch = [singleMatch];
        }
      }
    } else {
      // Search all artifacts
      artifactsToSearch = artifacts;
    }
    
    if (artifactsToSearch.length === 0) {
      const availableFiles = artifacts.map(a => a.filename || a.title || 'unnamed').slice(0, 10);
      results.push(`[FIND_ERROR: No files found${op.path ? ` matching path: ${op.path}` : ''}]\nAvailable files: ${availableFiles.join(', ')}${artifacts.length > 10 ? '...' : ''}`);
      continue;
    }
    
    const matches: { file: string; lineNumbers: number[]; preview: string }[] = [];
    
    for (const artifact of artifactsToSearch) {
      const artName = artifact.filename || artifact.title || 'unknown';
      const lines = artifact.content.split('\n');
      const matchingLines: number[] = [];
      
      for (let i = 0; i < lines.length; i++) {
        if (matchesPattern(lines[i], op.search)) {
          matchingLines.push(i + 1);
        }
      }
      
      if (matchingLines.length > 0) {
        // Get first match as preview
        const firstMatchLine = lines[matchingLines[0] - 1].trim();
        const preview = firstMatchLine.length > 50 ? firstMatchLine.substring(0, 50) + '...' : firstMatchLine;
        matches.push({ file: artName, lineNumbers: matchingLines, preview });
      }
    }
    
    if (matches.length === 0) {
      results.push(`[FIND: No matches for "${op.search}"${op.path ? ` in ${op.path}` : ' in any file'}]`);
    } else {
      const searchScope = op.path ? `in ${op.path}` : 'in all files';
      const matchDetails = matches.map(m => {
        const lineInfo = m.lineNumbers.length <= 5 
          ? `lines ${m.lineNumbers.join(', ')}`
          : `${m.lineNumbers.length} matches`;
        return `  ${m.file} (${lineInfo}): "${m.preview}"`;
      }).join('\n');
      results.push(`[FIND: ${matches.length} file(s) with matches for "${op.search}" ${searchScope}]\n${matchDetails}`);
    }
  }
  
  return results;
}

// Execute search_replace operations - modifies artifacts in place
function executeSearchReplaceOperations(
  artifacts: Artifact[],
  ops: SearchReplaceOp[]
): { updatedArtifacts: Artifact[]; results: string[]; modifiedFiles: string[] } {
  const results: string[] = [];
  const modifiedFiles: string[] = [];
  
  // Get list of available files for error messages
  const availableFiles = artifacts.map(a => a.filename || a.title || 'untitled').filter(Boolean);
  
  for (const op of ops) {
    const artifact = findArtifactByPath(artifacts, op.path);
    
    if (!artifact) {
      // File not found - show available files to help LLM correct the path
      const suggestion = availableFiles.length > 0 
        ? `\nAvailable files: ${availableFiles.slice(0, 10).join(', ')}${availableFiles.length > 10 ? ` (and ${availableFiles.length - 10} more)` : ''}`
        : '\nNo files currently available.';
      results.push(`[SEARCH_REPLACE_ERROR: File "${op.path}" not found.${suggestion}]`);
      continue;
    }
    
    const artName = artifact.filename || artifact.title || op.path;
    const searchText = op.search.trim();
    const replaceText = op.replace.trimEnd(); // Preserve leading whitespace in replacement
    const searchLineCount = searchText.split('\n').length;
    
    // Try exact match first
    if (artifact.content.includes(searchText)) {
      artifact.content = artifact.content.replace(searchText, replaceText);
      modifiedFiles.push(artName);
      
      const replaceLineCount = replaceText.split('\n').length;
      results.push(`[SEARCH_REPLACE: ✓ Replaced ${searchLineCount} lines with ${replaceLineCount} lines in "${artName}"]`);
      continue;
    }
    
    // Try line-by-line matching with trimmed comparison
    const contentLines = artifact.content.split('\n');
    const searchLines = searchText.split('\n');
    let found = false;
    
    for (let i = 0; i <= contentLines.length - searchLines.length; i++) {
      let matches = true;
      for (let j = 0; j < searchLines.length; j++) {
        if (contentLines[i + j].trim() !== searchLines[j].trim()) {
          matches = false;
          break;
        }
      }
      
      if (matches) {
        // Found it - replace preserving surrounding content
        const before = contentLines.slice(0, i);
        const after = contentLines.slice(i + searchLines.length);
        const replaceLines = replaceText.split('\n');
        artifact.content = [...before, ...replaceLines, ...after].join('\n');
        modifiedFiles.push(artName);
        found = true;
        results.push(`[SEARCH_REPLACE: ✓ Replaced ${searchLines.length} lines with ${replaceLines.length} lines in "${artName}" (whitespace-normalized match at line ${i + 1})]`);
        break;
      }
    }
    
    if (!found) {
      // Provide detailed error to help LLM understand what went wrong
      const searchFirstLine = searchText.split('\n')[0].trim();
      const searchLastLine = searchText.split('\n').slice(-1)[0].trim();
      
      // Try to find similar content to show what might be the issue
      let hint = '';
      
      // Check if first line exists anywhere
      const firstLineIdx = contentLines.findIndex(line => line.trim() === searchFirstLine);
      if (firstLineIdx >= 0) {
        // First line found - maybe the rest doesn't match
        const contextStart = Math.max(0, firstLineIdx - 1);
        const contextEnd = Math.min(contentLines.length, firstLineIdx + searchLineCount + 1);
        const actualContent = contentLines.slice(contextStart, contextEnd).map((l, i) => `  ${contextStart + i + 1}: ${l}`).join('\n');
        hint = `\nFirst line found at line ${firstLineIdx + 1}, but subsequent lines don't match.\nActual content around that area:\n${actualContent}`;
      } else if (searchFirstLine.length > 10) {
        // First line not found - try partial match
        const partialMatch = contentLines.findIndex(line => 
          line.includes(searchFirstLine.substring(0, 20)) || 
          searchFirstLine.includes(line.trim().substring(0, 20))
        );
        if (partialMatch >= 0) {
          hint = `\nNo exact match. Similar content found at line ${partialMatch + 1}:\n  "${contentLines[partialMatch].trim()}"`;
        } else {
          // Show file structure to help
          const totalLines = contentLines.length;
          hint = `\nFile "${artName}" has ${totalLines} lines. First line: "${contentLines[0]?.trim().substring(0, 50) || '(empty)'}..."`;
        }
      }
      
      results.push(`[SEARCH_REPLACE_ERROR: Content not found in "${artName}". Searched for ${searchLineCount} lines starting with:\n  "${searchFirstLine.substring(0, 60)}${searchFirstLine.length > 60 ? '...' : ''}"${hint}]`);
    }
  }
  
  return { updatedArtifacts: artifacts, results, modifiedFiles };
}

// Construct WebSocket URL from current location
function getWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}`;
}

// Streaming buffer for batching updates - aggressive throttling
class StreamingBuffer {
  private buffer: string = '';
  private timeoutId: ReturnType<typeof setTimeout> | null = null;
  private lastFlush: number = 0;
  private flushInterval: number = 100; // 100ms - match backend
  private maxBufferSize: number = 500; // characters - larger buffer
  private onFlush: (content: string) => void;
  private onToolDetected: ((toolTag: string, beforeContent: string) => void) | null = null;
  private paused: boolean = false; // Stop accepting content after tool interrupt
  
  // Tool tag patterns to watch for
  private static TOOL_STARTS = ['<tool_call', '<find', '<find_line', '<request_file', '<search_replace', '<replace_block', '<kb_search'];
  
  constructor(onFlush: (content: string) => void, onToolDetected?: (toolTag: string, beforeContent: string) => void) {
    this.onFlush = onFlush;
    this.onToolDetected = onToolDetected || null;
  }
  
  append(chunk: string) {
    // Don't accept new content if paused (tool was detected)
    if (this.paused) {
      console.log('[BUFFER] Paused - ignoring chunk:', chunk.substring(0, 50));
      return;
    }
    
    this.buffer += chunk;
    
    // Debug: Log buffer state when it contains tool tags
    if (this.buffer.includes('<search_replace') || this.buffer.includes('</search_replace')) {
      console.log('[BUFFER] Contains search_replace, buffer length:', this.buffer.length);
    }
    
    // Check for complete tool tags FIRST
    if (this.onToolDetected) {
      const toolResult = this.extractCompleteToolTag();
      if (toolResult) {
        console.log('[BUFFER] Found complete tool tag:', toolResult.tag.substring(0, 100));
        // Found complete tool tag - flush content before it, then trigger callback
        if (toolResult.before) {
          this.onFlush(toolResult.before);
        }
        this.buffer = toolResult.after;
        this.onToolDetected(toolResult.tag, toolResult.before);
        return; // Don't continue - tool handler will manage flow
      }
    }
    
    // Check if buffer contains a partial tool tag at the end
    const partialToolPos = this.findPartialToolTag();
    if (partialToolPos >= 0) {
      // Flush everything before the partial tag, keep the rest
      if (partialToolPos > 0) {
        const toFlush = this.buffer.substring(0, partialToolPos);
        this.buffer = this.buffer.substring(partialToolPos);
        this.onFlush(toFlush);
      }
      // Don't flush the partial tag yet - wait for more content
      return;
    }
    
    // No tool tags - normal flush behavior
    if (this.buffer.length >= this.maxBufferSize) {
      this.flushNow();
      return;
    }
    
    // Schedule flush if not already scheduled
    if (!this.timeoutId) {
      this.timeoutId = setTimeout(() => {
        this.timeoutId = null;
        this.flushNow();
      }, this.flushInterval);
    }
  }
  
  private findPartialToolTag(): number {
    // Look for any tool start that doesn't have its closing
    const bufferLower = this.buffer.toLowerCase();
    
    for (const toolStart of StreamingBuffer.TOOL_STARTS) {
      const pos = bufferLower.lastIndexOf(toolStart);
      if (pos >= 0) {
        const afterStart = bufferLower.substring(pos);
        
        // Tags that need closing tags (not self-closing)
        if (toolStart === '<tool_call') {
          if (!afterStart.includes('</tool_call>')) {
            return pos; // Partial - waiting for closing tag
          }
        } else if (toolStart === '<search_replace') {
          if (!afterStart.includes('</search_replace>')) {
            return pos; // Partial - waiting for closing tag
          }
        } else if (toolStart === '<replace_block') {
          if (!afterStart.includes('</replace_block>')) {
            return pos; // Partial - waiting for closing tag
          }
        } else {
          // Other tags just need closing >
          if (!afterStart.includes('>')) {
            return pos; // Partial tag found
          }
        }
      }
    }
    return -1; // No partial tag
  }
  
  private extractCompleteToolTag(): { tag: string; before: string; after: string } | null {
    const bufferLower = this.buffer.toLowerCase();
    
    // First check for <tool_call>...</tool_call> (wraps other tools)
    const toolCallStart = bufferLower.indexOf('<tool_call');
    if (toolCallStart >= 0) {
      const afterStart = bufferLower.substring(toolCallStart);
      const closePos = afterStart.indexOf('</tool_call>');
      if (closePos >= 0) {
        // Use original buffer for actual content, but positions from lowercase
        const tag = this.buffer.substring(toolCallStart, toolCallStart + closePos + '</tool_call>'.length);
        return {
          tag,
          before: this.buffer.substring(0, toolCallStart),
          after: this.buffer.substring(toolCallStart + closePos + '</tool_call>'.length),
        };
      }
      return null; // Incomplete tool_call
    }
    
    // Check for <search_replace>...</search_replace> (multi-line)
    const searchReplaceStart = bufferLower.indexOf('<search_replace');
    if (searchReplaceStart >= 0) {
      const afterStart = bufferLower.substring(searchReplaceStart);
      const closePos = afterStart.indexOf('</search_replace>');
      if (closePos >= 0) {
        const tag = this.buffer.substring(searchReplaceStart, searchReplaceStart + closePos + '</search_replace>'.length);
        return {
          tag,
          before: this.buffer.substring(0, searchReplaceStart),
          after: this.buffer.substring(searchReplaceStart + closePos + '</search_replace>'.length),
        };
      }
      return null; // Incomplete search_replace
    }
    
    // Check for <replace_block>...</replace_block> (multi-line, legacy)
    const replaceBlockStart = bufferLower.indexOf('<replace_block');
    if (replaceBlockStart >= 0) {
      const afterStart = bufferLower.substring(replaceBlockStart);
      const closePos = afterStart.indexOf('</replace_block>');
      if (closePos >= 0) {
        const tag = this.buffer.substring(replaceBlockStart, replaceBlockStart + closePos + '</replace_block>'.length);
        return {
          tag,
          before: this.buffer.substring(0, replaceBlockStart),
          after: this.buffer.substring(replaceBlockStart + closePos + '</replace_block>'.length),
        };
      }
      return null; // Incomplete replace_block
    }
    
    // Look for other complete tool tags (self-closing with >)
    for (const toolStart of StreamingBuffer.TOOL_STARTS) {
      // Skip tags already handled above
      if (toolStart === '<tool_call' || toolStart === '<search_replace' || toolStart === '<replace_block') continue;
      
      const startPos = bufferLower.indexOf(toolStart);
      if (startPos >= 0) {
        const afterStart = bufferLower.substring(startPos);
        const closePos = afterStart.indexOf('>');
        if (closePos >= 0) {
          // Use original buffer for actual content
          const tag = this.buffer.substring(startPos, startPos + closePos + 1);
          return {
            tag,
            before: this.buffer.substring(0, startPos),
            after: this.buffer.substring(startPos + closePos + 1),
          };
        }
      }
    }
    return null;
  }
  
  flushNow() {
    if (this.buffer.length > 0 && !this.paused) {
      // Before flushing, check one more time for tool tags
      if (this.onToolDetected) {
        const toolResult = this.extractCompleteToolTag();
        if (toolResult) {
          if (toolResult.before) {
            this.onFlush(toolResult.before);
          }
          this.buffer = toolResult.after;
          this.onToolDetected(toolResult.tag, toolResult.before);
          return;
        }
      }
      
      this.onFlush(this.buffer);
      this.buffer = '';
      this.lastFlush = performance.now();
    }
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
  
  // Pause buffer - stop accepting/flushing content (for tool interrupts)
  pause() {
    this.paused = true;
    this.buffer = '';
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
  
  // Resume buffer (called on new stream start)
  resume() {
    this.paused = false;
  }
  
  clear() {
    this.buffer = '';
    this.paused = false; // Also reset paused state
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
}

interface WebSocketContextValue {
  isConnected: boolean;
  connectionError: string | null;
  subscribe: (chatId: string) => void;
  unsubscribe: (chatId: string) => void;
  sendChatMessage: (chatId: string, content: string, attachments?: unknown[], parentId?: string | null) => void;
  sendClientMessage: (chatId: string, content: string) => void;
  regenerateMessage: (chatId: string, content: string, parentId: string) => void;
  stopGeneration: (chatId: string) => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  const { accessToken, isAuthenticated, logout } = useAuthStore();
  const {
    currentChat,
    addMessage,
    appendStreamingContent,
    setStreamingContent,
    setStreamingToolCall,
    clearStreaming,
    setIsSending,
    updateChatLocally,
    updateStreamingArtifacts,
  } = useChatStore();
  
  // Create streaming buffer that batches updates
  const streamingBufferRef = useRef<StreamingBuffer | null>(null);
  const lastArtifactCheckRef = useRef<number>(0);
  
  // Ref to hold the tool interrupt callback (set via useEffect to have access to wsRef)
  const toolInterruptCallbackRef = useRef<((content: string) => void) | null>(null);
  
  // Track which tool tags have been processed during current stream
  const processedToolTagsRef = useRef<Set<string>>(new Set());
  
  // Track tool calls across streams to detect loops (persistent across streams)
  const toolCallHistoryRef = useRef<{ key: string; timestamp: number }[]>([]);
  const MAX_SAME_TOOL_CALLS = 3; // Max times same tool can be called in sequence
  const TOOL_HISTORY_WINDOW_MS = 60000; // 1 minute window
  
  // Track artifacts that were saved during current stream (to notify LLM)
  const savedArtifactsDuringStreamRef = useRef<string[]>([]);
  
  // Track write count per file per chat to prevent infinite rewrites
  // Structure: { chatId: { filename: writeCount } }
  const artifactWriteCountRef = useRef<Record<string, Record<string, number>>>({});
  
  // Track the current streaming message ID for tool result continuations
  const currentStreamingMessageIdRef = useRef<string | null>(null);
  
  // Track which chat the current stream belongs to - prevents cross-chat contamination
  const currentStreamingChatIdRef = useRef<string | null>(null);
  
  // Clear streaming refs when chat changes to prevent cross-chat contamination
  useEffect(() => {
    // If user switched to a different chat while streaming, clear the refs
    if (currentChat?.id && currentStreamingChatIdRef.current && 
        currentStreamingChatIdRef.current !== currentChat.id) {
      console.log('[CHAT_CHANGE] Clearing streaming refs - switched from', 
        currentStreamingChatIdRef.current, 'to', currentChat.id);
      currentStreamingChatIdRef.current = null;
      currentStreamingMessageIdRef.current = null;
      streamingBufferRef.current?.clear();
    }
  }, [currentChat?.id]);
  
  // Tool detection callback ref - will be set up in useEffect
  const toolDetectedCallbackRef = useRef<((toolTag: string, beforeContent: string) => void) | null>(null);
  
  if (!streamingBufferRef.current) {
    streamingBufferRef.current = new StreamingBuffer(
      // onFlush callback
      (content) => {
        // Only append content if we're still streaming for the same chat
        const streamingChatId = currentStreamingChatIdRef.current;
        const { currentChat } = useChatStore.getState();
        
        // If user switched to a different chat, ignore this content
        if (streamingChatId && currentChat?.id && streamingChatId !== currentChat.id) {
          console.log('[BUFFER_FLUSH] Ignoring content for different chat:', streamingChatId, 'vs', currentChat.id);
          return;
        }
        
        const store = useChatStore.getState();
        store.appendStreamingContent(content);
        
        // Check for artifacts periodically (every 500ms)
        const now = performance.now();
        if (now - lastArtifactCheckRef.current > 500) {
          lastArtifactCheckRef.current = now;
          const { streamingContent } = useChatStore.getState();
          store.updateStreamingArtifacts(streamingContent);
        }
      },
      // onToolDetected callback
      (toolTag, beforeContent) => {
        console.log('[TOOL_DETECTED] Tag:', toolTag.substring(0, 100));
        if (toolDetectedCallbackRef.current) {
          toolDetectedCallbackRef.current(toolTag, beforeContent);
        }
      }
    );
  }
  
  // Handle file requests detected in LLM responses - fetch files and auto-continue
  // Checks artifacts first (files LLM created), then falls back to uploaded zip files
  const handleFileRequests = useCallback(async (chatId: string, paths: string[], parentMessageId: string) => {
    console.log(`[handleFileRequests] Fetching ${paths.length} files for chat ${chatId}, parent=${parentMessageId}`);
    
    // Log current state before making changes
    const { artifacts: currentArtifacts, messages: currentMessages } = useChatStore.getState();
    console.log(`[handleFileRequests] Current state: ${currentMessages.length} messages, ${currentArtifacts.length} artifacts`);
    
    // Clear previous streaming content since we're about to start a new response
    const { clearStreaming } = useChatStore.getState();
    clearStreaming();
    
    const fileContents: string[] = [];
    const fetchedPaths: string[] = [];
    
    // Helper to find artifact by path (checks filename and full path)
    const findArtifact = (requestPath: string) => {
      const normalizedRequest = requestPath.replace(/^\.?\//, ''); // Remove leading ./ or /
      return currentArtifacts.find(art => {
        const artFilename = art.filename || art.title || '';
        const normalizedArt = artFilename.replace(/^\.?\//, '');
        // Match full path or just filename
        return normalizedArt === normalizedRequest || 
               normalizedArt.endsWith('/' + normalizedRequest) ||
               normalizedRequest.endsWith('/' + normalizedArt) ||
               artFilename.split('/').pop() === requestPath.split('/').pop();
      });
    };
    
    for (const path of paths) {
      // First check if this file exists in artifacts (files LLM already created)
      const artifact = findArtifact(path);
      
      if (artifact) {
        console.log(`[handleFileRequests] Found in artifacts: ${path} -> ${artifact.filename || artifact.title}`);
        const formatted = `=== FILE: ${artifact.filename || artifact.title} ===\n${artifact.content}\n=== END FILE ===`;
        fileContents.push(formatted);
        fetchedPaths.push(path);
        continue;
      }
      
      // Not in artifacts, try to fetch from uploaded zip
      try {
        const response = await chatApi.getZipFile(chatId, path);
        const fileData = response.data as ZipFileResponse;
        
        console.log(`[handleFileRequests] Fetched from zip: ${path} (${fileData.content.length} chars)`);
        fileContents.push(fileData.formatted);
        fetchedPaths.push(path);
        
      } catch (error) {
        console.error(`[handleFileRequests] Failed to fetch file ${path}:`, error);
        fileContents.push(`[FILE_ERROR: Could not retrieve ${path} - not found in artifacts or uploaded files]`);
      }
    }
    
    // Combine all file contents and send as continuation
    if (fetchedPaths.length > 0 || fileContents.length > 0) {
      const combinedContent = fileContents.join('\n\n');
      
      // Send to LLM without creating a message (no chat branching)
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        console.log('[handleFileRequests] Auto-continuing conversation with file content');
        setIsSending(true);
        
        // Get zip context from store
        const { zipContext } = useChatStore.getState();
        
        // Create continuation prompt marked as system output
        const fileLabel = fetchedPaths.length === 1
          ? `file: ${fetchedPaths[0]}`
          : `${fetchedPaths.length} files`;
        const continuationPrompt = `[SYSTEM TOOL RESULT - request_file]\nThe following ${fileLabel} content was retrieved by the system, not typed by the user.\n\n${combinedContent}\n\n[END TOOL RESULT]`;
        
        console.log(`[handleFileRequests] Sending continuation with ${continuationPrompt.length} chars to LLM`);
        
        wsRef.current.send(JSON.stringify({
          type: 'chat_message',
          payload: {
            chat_id: chatId,
            content: continuationPrompt,
            parent_id: parentMessageId,
            zip_context: zipContext,
            save_user_message: false,
            is_tool_continuation: true,
          },
        }));
        
        console.log('[handleFileRequests] Continuation sent, waiting for LLM response...');
      } else {
        console.error('WebSocket not open, cannot send file content continuation');
        setIsSending(false);
      }
    } else {
      console.error('No files could be fetched');
      setIsSending(false);
    }
  }, [setIsSending]);
  
  // Handle find_line operations - search within a specific file and return line numbers
  const handleFindLineOperations = useCallback((
    chatId: string, 
    ops: FindLineOp[], 
    parentMessageId: string,
    allArtifacts: Artifact[]
  ) => {
    console.log(`[handleFindLine] Processing ${ops.length} find_line operations`);
    
    const results = executeFindLineOperations(allArtifacts, ops);
    const combinedResults = results.join('\n\n');
    
    // Send results back to LLM to continue (no message creation)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[handleFindLine] Sending results to LLM');
      setIsSending(true);
      
      const { zipContext } = useChatStore.getState();
      
      wsRef.current.send(JSON.stringify({
        type: 'chat_message',
        payload: {
          chat_id: chatId,
          content: `[SYSTEM TOOL RESULT - find_line]\nThe following results were generated by the system, not typed by the user.\n\n${combinedResults}\n\n[END TOOL RESULT]`,
          parent_id: parentMessageId,
          zip_context: zipContext,
          save_user_message: false,
          is_tool_continuation: true,
        },
      }));
    }
  }, [setIsSending]);
  
  // Handle find operations - search across multiple files
  const handleFindOperations = useCallback((
    chatId: string, 
    ops: FindOp[], 
    parentMessageId: string,
    allArtifacts: Artifact[]
  ) => {
    console.log(`[handleFind] Processing ${ops.length} find operations`);
    
    const results = executeFindOperations(allArtifacts, ops);
    const combinedResults = results.join('\n\n');
    
    // Send results back to LLM to continue (no message creation)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[handleFind] Sending results to LLM');
      setIsSending(true);
      
      const { zipContext } = useChatStore.getState();
      
      wsRef.current.send(JSON.stringify({
        type: 'chat_message',
        payload: {
          chat_id: chatId,
          content: `[SYSTEM TOOL RESULT - find]\nThe following results were generated by the system, not typed by the user.\n\n${combinedResults}\n\n[END TOOL RESULT]`,
          parent_id: parentMessageId,
          zip_context: zipContext,
          save_user_message: false,
          is_tool_continuation: true,
        },
      }));
    }
  }, [setIsSending]);
  
  // Handle search_replace operations - modify files and report results
  const handleSearchReplaceOperations = useCallback((
    chatId: string, 
    ops: SearchReplaceOp[], 
    parentMessageId: string,
    allArtifacts: Artifact[],
    messageId: string
  ) => {
    console.log(`[handleSearchReplace] Processing ${ops.length} search_replace operations`);
    
    const { updateArtifact } = useChatStore.getState();
    const { updatedArtifacts, results, modifiedFiles } = executeSearchReplaceOperations(allArtifacts, ops);
    
    // Update modified artifacts in store
    for (const art of updatedArtifacts) {
      const key = art.filename || art.title || '';
      if (key && modifiedFiles.includes(key)) {
        updateArtifact(key, art);
      }
    }
    
    // Save to backend
    if (modifiedFiles.length > 0) {
      const modifiedArtifacts = updatedArtifacts.filter(a => 
        modifiedFiles.includes(a.filename || a.title || '')
      );
      api.put(`/chats/${chatId}/messages/${messageId}/artifacts`, {
        artifacts: modifiedArtifacts
      }).catch(err => {
        console.error('[handleSearchReplace] Failed to save:', err);
      });
    }
    
    const combinedResults = results.join('\n\n');
    
    // Send results back to LLM to continue (no message creation)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[handleSearchReplace] Sending results to LLM');
      setIsSending(true);
      
      const { zipContext } = useChatStore.getState();
      
      wsRef.current.send(JSON.stringify({
        type: 'chat_message',
        payload: {
          chat_id: chatId,
          content: `[SYSTEM TOOL RESULT - search_replace]\nThe following results were generated by the system, not typed by the user.\n\n${combinedResults}\n\n[END TOOL RESULT]`,
          parent_id: parentMessageId,
          zip_context: zipContext,
          save_user_message: false,
          is_tool_continuation: true,
        },
      }));
    }
  }, [setIsSending]);
  
  const handleMessage = useCallback((message: WSMessage) => {
    console.log('WebSocket message received:', message.type, message.payload);
    
    switch (message.type) {
      case 'pong':
        // Heartbeat response
        break;
        
      case 'subscribed':
        console.log('Subscribed to chat:', message.payload);
        break;
        
      case 'message_saved': {
        // Server confirms user message was saved
        // Since we use client-generated UUIDs, the ID should already match
        const payload = message.payload as { message_id: string; parent_id?: string };
        console.log('Message saved confirmed:', payload.message_id);
        // If parent_id was modified by server (e.g., validation), update it
        if (payload.parent_id !== undefined) {
          const { messages } = useChatStore.getState();
          const msg = messages.find(m => m.id === payload.message_id);
          if (msg && msg.parent_id !== payload.parent_id) {
            useChatStore.getState().updateMessage(payload.message_id, { parent_id: payload.parent_id });
          }
        }
        break;
      }
        
      case 'chat_updated': {
        // Update chat in list (e.g., when title is generated)
        const payload = message.payload as { chat_id: string; title?: string };
        if (payload.chat_id && payload.title) {
          console.log('Chat title updated:', payload.chat_id, payload.title);
          updateChatLocally(payload.chat_id, { title: payload.title });
          
          // Also update document title immediately if this is the current chat
          const { currentChat } = useChatStore.getState();
          if (currentChat?.id === payload.chat_id && payload.title !== 'New Chat') {
            const appName = useBrandingStore.getState().config?.app_name || 'Open-NueChat';
            document.title = `${appName} - ${payload.title}`;
          }
        }
        break;
      }
        
      case 'stream_start': {
        const payload = message.payload as StreamChunk;
        const { currentChat } = useChatStore.getState();
        
        // Validate this stream is for the current chat
        if (payload.chat_id && currentChat?.id && payload.chat_id !== currentChat.id) {
          console.warn('[stream_start] Ignoring stream for different chat:', payload.chat_id, 'current:', currentChat.id);
          break;
        }
        
        console.log('[stream_start] Stream starting for chat:', payload.chat_id, 'message_id:', payload.message_id);
        // Log current state
        const { messages, artifacts } = useChatStore.getState();
        console.log(`[stream_start] State before clear: ${messages.length} messages, ${artifacts.length} artifacts`);
        
        // Track which chat this stream belongs to - prevents cross-chat contamination
        currentStreamingChatIdRef.current = payload.chat_id || null;
        
        // Track the current streaming message ID for tool continuations
        currentStreamingMessageIdRef.current = payload.message_id || null;
        
        streamingBufferRef.current?.clear();
        setStreamingContent('');
        setIsSending(true);
        
        // Clear processed tool tags and saved artifacts for new stream
        processedToolTagsRef.current.clear();
        savedArtifactsDuringStreamRef.current = [];
        
        // Verify state wasn't accidentally cleared
        const { messages: afterMsgs, artifacts: afterArts } = useChatStore.getState();
        console.log(`[stream_start] State after clear: ${afterMsgs.length} messages, ${afterArts.length} artifacts`);
        break;
      }
        
      case 'stream_chunk': {
        const chunk = message.payload as StreamChunk;
        const { currentChat } = useChatStore.getState();
        
        // Validate this chunk is for the current chat
        if (chunk.chat_id && currentChat?.id && chunk.chat_id !== currentChat.id) {
          // Silently ignore chunks for other chats (don't spam logs)
          break;
        }
        
        if (chunk.content) {
          // Debug: Log incoming chunks
          if (chunk.content.includes('<search_replace') || chunk.content.includes('</search_replace')) {
            console.log('[STREAM_CHUNK] Contains search_replace tag:', chunk.content.substring(0, 200));
          }
          // Use buffered append for performance
          streamingBufferRef.current?.append(chunk.content);
        }
        break;
      }
        
      case 'tool_call_start': {
        const chunk = message.payload as StreamChunk;
        if (chunk.tool_call) {
          setStreamingToolCall({
            name: chunk.tool_call.name,
            input: JSON.stringify(chunk.tool_call.input, null, 2),
          });
        }
        break;
      }
        
      case 'tool_call_end':
        setStreamingToolCall(null);
        break;
        
      case 'stream_end': {
        // Flush any remaining buffered content
        streamingBufferRef.current?.flushNow();
        
        const chunk = message.payload as StreamChunk & { parent_id?: string };
        console.warn('Stream ended:', chunk);
        
        // Clear the streaming chat ID ref - stream is over
        const streamingChatId = currentStreamingChatIdRef.current;
        currentStreamingChatIdRef.current = null;
        currentStreamingMessageIdRef.current = null;
        
        // Add final message
        if (chunk.message_id) {
          const { streamingContent, currentChat, generatedImages } = useChatStore.getState();
          console.warn('Final streaming content length:', streamingContent.length);
          
          // Debug: Check for tool tags in final content
          const hasToolTag = streamingContent.includes('<find') || 
                             streamingContent.includes('<request_file') || 
                             streamingContent.includes('<search_replace') ||
                             streamingContent.includes('<replace_block');
          if (hasToolTag) {
            console.warn('[STREAM_END] TOOL TAGS FOUND IN FINAL CONTENT!');
            console.warn('[STREAM_END] Content tail:', streamingContent.slice(-1000));
          }
          
          // CRITICAL: Validate stream belongs to current chat
          // Use BOTH chunk.chat_id AND our tracked streamingChatId for robustness
          const expectedChatId = chunk.chat_id || streamingChatId;
          if (expectedChatId && currentChat?.id && expectedChatId !== currentChat.id) {
            console.warn('[STREAM_END] Ignoring stream for different chat:', expectedChatId, 'current:', currentChat.id);
            clearStreaming();
            setIsSending(false);
            break;
          }
          
          // Extract artifacts from the content and attach to message
          // cleanContent has [📦 Artifact: ...] placeholders, artifacts has the extracted data
          const { cleanContent, artifacts: extractedArtifacts } = extractArtifacts(streamingContent);
          
          // Check if there's a generated image for this message
          let generatedImage = generatedImages[chunk.message_id];
          if (generatedImage) {
            console.log('%c[STREAM_END]', 'background: purple; color: white', 
              'Found generated image for message:', chunk.message_id, 'base64 len:', generatedImage.base64?.length);
          } else {
            console.log('%c[STREAM_END]', 'background: orange; color: black', 
              'No image found yet for message:', chunk.message_id);
          }
          
          const assistantMessage: Message = {
            id: chunk.message_id,
            chat_id: chunk.chat_id || '',
            role: 'assistant',
            // Use cleanContent which has artifact placeholders, preserving code blocks for markdown rendering
            // But actually we want the ORIGINAL content so code blocks render via markdown
            // The artifacts are stored separately for the panel
            content: streamingContent,
            parent_id: chunk.parent_id,
            input_tokens: chunk.usage?.input_tokens,
            output_tokens: chunk.usage?.output_tokens,
            created_at: new Date().toISOString(),
            // Attach extracted artifacts to the message
            artifacts: extractedArtifacts.length > 0 ? extractedArtifacts : undefined,
            // Attach generated image directly to message metadata
            metadata: generatedImage ? { generated_image: generatedImage } : undefined,
          };
          addMessage(assistantMessage);
          
          console.log('Added assistant message with', extractedArtifacts.length, 'artifacts');
          
          // Process any replace operations in the response
          const replaceOps = extractReplaceOperations(streamingContent);
          if (replaceOps.length > 0) {
            console.log('[REPLACE] Found', replaceOps.length, 'replace operations');
            
            // Get all artifacts from store (includes previously created ones)
            const { artifacts: allArtifacts, updateArtifact } = useChatStore.getState();
            
            // Combine with newly extracted artifacts for replacement
            const combinedArtifacts = [...allArtifacts];
            for (const newArt of extractedArtifacts) {
              const existingIdx = combinedArtifacts.findIndex(a => 
                (a.filename || a.title) === (newArt.filename || newArt.title)
              );
              if (existingIdx >= 0) {
                combinedArtifacts[existingIdx] = newArt;
              } else {
                combinedArtifacts.push(newArt);
              }
            }
            
            // Apply replacements
            const { updatedArtifacts, results } = applyReplaceOperations(combinedArtifacts, replaceOps);
            
            // Log results
            for (const result of results) {
              console.log('[REPLACE]', result);
            }
            
            // Update artifacts in store
            for (const art of updatedArtifacts) {
              const key = art.filename || art.title || '';
              if (key) {
                updateArtifact(key, art);
              }
            }
            
            // Save updated artifacts to backend
            if (chunk.chat_id && chunk.message_id) {
              api.put(`/chats/${chunk.chat_id}/messages/${chunk.message_id}/artifacts`, {
                artifacts: updatedArtifacts
              }).then(() => {
                console.log('[REPLACE] Saved updated artifacts to backend');
              }).catch(err => {
                console.error('[REPLACE] Failed to save updated artifacts:', err);
              });
            }
          }
          
          // ============ NEW ARTIFACT OPERATIONS ============
          
          // Helper to get all artifacts (stored + uploaded + newly extracted)
          const getAllArtifacts = () => {
            const { artifacts: storedArtifacts, uploadedArtifacts } = useChatStore.getState();
            const allArtifacts = [...storedArtifacts, ...uploadedArtifacts];
            // Add newly extracted artifacts
            for (const newArt of extractedArtifacts) {
              const existingIdx = allArtifacts.findIndex(a => 
                (a.filename || a.title) === (newArt.filename || newArt.title)
              );
              if (existingIdx >= 0) {
                allArtifacts[existingIdx] = newArt;
              } else {
                allArtifacts.push(newArt);
              }
            }
            return allArtifacts;
          };
          
          // Track if we need to auto-continue (any operation triggers continuation)
          let needsAutoContinue = false;
          const allArtifacts = getAllArtifacts();
          
          // Process find_line operations
          const findLineOps = extractFindLineOperations(streamingContent);
          if (findLineOps.length > 0 && currentChat?.id) {
            console.log('[FIND_LINE] Found', findLineOps.length, 'find_line operations');
            handleFindLineOperations(currentChat.id, findLineOps, chunk.message_id, allArtifacts);
            needsAutoContinue = true;
          }
          
          // Process find operations
          const findOps = extractFindOperations(streamingContent);
          if (findOps.length > 0 && currentChat?.id && !needsAutoContinue) {
            console.log('[FIND] Found', findOps.length, 'find operations');
            handleFindOperations(currentChat.id, findOps, chunk.message_id, allArtifacts);
            needsAutoContinue = true;
          }
          
          // Process search_replace operations (new format)
          let searchReplaceOps = extractSearchReplaceOperations(streamingContent);
          
          // Fallback: try to extract incomplete operations (LLM stopped without closing tag)
          if (searchReplaceOps.length === 0) {
            const incompleteOps = extractIncompleteSearchReplaceOperations(streamingContent);
            if (incompleteOps.length > 0) {
              console.log('[SEARCH_REPLACE] Using', incompleteOps.length, 'incomplete operations (missing closing tag)');
              searchReplaceOps = incompleteOps;
            }
          }
          
          if (searchReplaceOps.length > 0 && currentChat?.id && !needsAutoContinue) {
            console.log('[SEARCH_REPLACE] Found', searchReplaceOps.length, 'search_replace operations');
            handleSearchReplaceOperations(currentChat.id, searchReplaceOps, chunk.message_id, allArtifacts, chunk.message_id);
            needsAutoContinue = true;
          }
          
          // If any operation triggered auto-continuation, don't clear streaming yet
          if (needsAutoContinue) {
            break;
          }
          
          // Persist artifacts to backend for proper versioning on reload
          if (extractedArtifacts.length > 0 && chunk.chat_id && chunk.message_id) {
            api.put(`/chats/${chunk.chat_id}/messages/${chunk.message_id}/artifacts`, {
              artifacts: extractedArtifacts
            }).then(() => {
              console.log('[ARTIFACTS] Saved', extractedArtifacts.length, 'artifacts to backend');
            }).catch(err => {
              console.error('[ARTIFACTS] Failed to save artifacts:', err);
            });
          }
          
          // Update code summary with extracted artifacts
          if (extractedArtifacts.length > 0) {
            const { addFileToSummary, saveCodeSummary, codeSummary } = useChatStore.getState();
            
            console.log('[CODE_SUMMARY] Processing', extractedArtifacts.length, 'artifacts');
            console.log('[CODE_SUMMARY] Current summary files:', codeSummary?.files?.length || 0);
            
            // Track which files we're adding
            const existingPaths = new Set((codeSummary?.files || []).map(f => f.path));
            let filesAdded = 0;
            
            for (const artifact of extractedArtifacts) {
              // Use filename or title as the path, cleaned of any angle brackets/punctuation
              const rawPath = artifact.filename || artifact.title;
              const filepath = rawPath ? cleanFilePath(rawPath) : null;
              
              console.log(`[CODE_SUMMARY] Artifact: type=${artifact.type}, filename=${artifact.filename}, title=${artifact.title}, cleanPath=${filepath}, contentLen=${artifact.content?.length || 0}`);
              
              // Track all code-like artifacts
              const isCodeLike = ['code', 'react', 'html', 'json', 'csv'].includes(artifact.type);
              
              if (filepath && artifact.content && isCodeLike) {
                const action = existingPaths.has(filepath) ? 'modified' : 'created';
                const fileChange = createFileChangeFromCode(
                  filepath,
                  artifact.content,
                  action
                );
                addFileToSummary(fileChange);
                existingPaths.add(filepath); // Track for subsequent artifacts in same batch
                filesAdded++;
                console.log(`[CODE_SUMMARY] ✓ Added ${action} file: ${filepath} (${fileChange.signatures.length} signatures)`);
              } else {
                console.log(`[CODE_SUMMARY] ✗ Skipped: filepath=${!!filepath}, content=${!!artifact.content}, isCodeLike=${isCodeLike}`);
              }
            }
            
            console.log(`[CODE_SUMMARY] Total files added: ${filesAdded}`);
            
            // Save to backend if we added any files
            if (filesAdded > 0) {
              saveCodeSummary().then(() => {
                console.log('[CODE_SUMMARY] Saved to backend');
              }).catch(err => {
                console.error('[CODE_SUMMARY] Failed to save:', err);
              });
            }
          } else {
            console.log('[CODE_SUMMARY] No artifacts to process');
          }
          
          // Auto-continue if artifacts were created during streaming
          // This tells the LLM to keep generating more files if needed
          // Track writes per file to prevent rewriting same file repeatedly
          const MAX_WRITES_PER_FILE = 5;
          if (savedArtifactsDuringStreamRef.current.length > 0 && currentChat?.id && chunk.message_id) {
            // Initialize file write tracker for this chat if needed
            if (!artifactWriteCountRef.current[currentChat.id]) {
              artifactWriteCountRef.current[currentChat.id] = {};
            }
            
            const fileWriteCounts = artifactWriteCountRef.current[currentChat.id];
            const savedFiles: string[] = [];
            const blockedFiles: string[] = [];
            
            // Check each artifact for write limit
            for (const filename of savedArtifactsDuringStreamRef.current) {
              const count = (fileWriteCounts[filename] || 0) + 1;
              fileWriteCounts[filename] = count;
              
              if (count > MAX_WRITES_PER_FILE) {
                blockedFiles.push(`${filename} (${count} writes, max ${MAX_WRITES_PER_FILE})`);
              } else {
                savedFiles.push(filename);
              }
            }
            
            // Clear the saved artifacts list
            savedArtifactsDuringStreamRef.current = [];
            
            // If all files were blocked, don't continue
            if (savedFiles.length === 0 && blockedFiles.length > 0) {
              console.log(`[ARTIFACT_CONTINUE] All files blocked due to write limits: ${blockedFiles.join(', ')}`);
              // Fall through to normal stream end handling
            } else if (savedFiles.length > 0) {
              const savedFileList = savedFiles.join(', ');
              console.log(`[ARTIFACT_CONTINUE] ${savedFiles.length} files saved, sending continuation`);
              if (blockedFiles.length > 0) {
                console.log(`[ARTIFACT_CONTINUE] Blocked files: ${blockedFiles.join(', ')}`);
              }
              
              // Get zip context
              const { zipContext } = useChatStore.getState();
              
              // Build message with blocked file warning if needed
              let content = `[SYSTEM NOTIFICATION - FILES SAVED]\nThe following files were saved: ${savedFileList}`;
              if (blockedFiles.length > 0) {
                content += `\n\nWARNING: The following files have been written too many times and further changes are blocked: ${blockedFiles.join(', ')}. Please move on to other files.`;
              }
              content += `\n\nContinue generating any remaining files for this project. If you have finished creating all necessary files, provide a brief summary of what was created.\n[END NOTIFICATION]`;
              
              // Send continuation to let LLM know it can keep going
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                  type: 'chat_message',
                  payload: {
                    chat_id: currentChat.id,
                    content,
                    parent_id: chunk.message_id,
                    zip_context: zipContext,
                    save_user_message: false,
                    is_tool_continuation: true,
                  },
                }));
                // Don't clear streaming - LLM will continue
                break;
              }
            }
          }
          
          // If no image was found, schedule a check in case it arrives right after
          // (handles race condition where image_generated event is still processing)
          if (!generatedImage && chunk.message_id) {
            const messageId = chunk.message_id; // Capture for closure
            setTimeout(() => {
              const { generatedImages: imgs, messages: msgs, updateMessage } = useChatStore.getState();
              const img = imgs[messageId];
              if (img) {
                console.log('%c[DELAYED IMAGE CHECK]', 'background: green; color: white',
                  'Found image after delay, updating message');
                const msg = msgs.find(m => m.id === messageId);
                if (msg) {
                  updateMessage(messageId, {
                    metadata: { ...msg.metadata, generated_image: img }
                  });
                }
              }
            }, 200);
          }
          
          // Check for file request tags in the response
          const fileRequests = extractFileRequests(streamingContent);
          if (fileRequests.length > 0 && currentChat?.id) {
            console.log('Detected file requests:', fileRequests);
            // Fetch files and inject into conversation
            // Don't clear streaming state yet - handleFileRequests will manage it
            handleFileRequests(currentChat.id, fileRequests, chunk.message_id);
            // Don't call clearStreaming() or setIsSending(false) here
            // handleFileRequests will set isSending appropriately
            break;
          }
        }
        clearStreaming();
        setIsSending(false);
        break;
      }
        
      case 'stream_error': {
        const chunk = message.payload as StreamChunk;
        console.error('Stream error:', chunk.error);
        streamingBufferRef.current?.clear();
        currentStreamingChatIdRef.current = null;
        currentStreamingMessageIdRef.current = null;
        clearStreaming();
        setIsSending(false);
        break;
      }
      
      case 'stream_stopped': {
        // Flush any remaining buffered content
        streamingBufferRef.current?.flushNow();
        
        // Generation was stopped by user
        const chunk = message.payload as StreamChunk & { parent_id?: string };
        console.log('Stream stopped by user');
        
        // Clear the streaming chat ID ref
        const streamingChatId = currentStreamingChatIdRef.current;
        currentStreamingChatIdRef.current = null;
        currentStreamingMessageIdRef.current = null;
        
        // Add partial message if content exists AND it's for the current chat
        if (chunk.message_id) {
          const { streamingContent, currentChat } = useChatStore.getState();
          
          // Validate stream belongs to current chat
          const expectedChatId = chunk.chat_id || streamingChatId;
          if (expectedChatId && currentChat?.id && expectedChatId !== currentChat.id) {
            console.warn('[STREAM_STOPPED] Ignoring stream for different chat:', expectedChatId, 'current:', currentChat.id);
            clearStreaming();
            setIsSending(false);
            break;
          }
          
          // Don't add message here - the content is already being saved by backend
          // The "[Generation stopped]" marker will be in the database but not shown to user
          // This prevents duplicate messages when tools interrupt generation
        }
        clearStreaming();
        setIsSending(false);
        break;
      }
        
      case 'client_message': {
        // Message from another client in shared chat
        const msg = message.payload as Message;
        addMessage(msg);
        break;
      }
      
      case 'image_generation_started': {
        // Image generation has started - queue info available in payload
        break;
      }
      
      case 'image_generated': {
        const payload = message.payload as {
          chat_id: string;
          message_id: string;
          image: {
            base64?: string;
            url?: string;
            width: number;
            height: number;
            seed: number;
            prompt: string;
            generation_time?: number;
            job_id?: string;
          };
        };
        
        if (!payload.message_id || !payload.image) {
          break;
        }
        
        if (!payload.image.base64 && !payload.image.url) {
          break;
        }
        
        const { setGeneratedImage, messages, updateMessage } = useChatStore.getState();
        
        setGeneratedImage(payload.message_id, payload.image);
        
        const existingMessage = messages.find(m => m.id === payload.message_id);
        if (existingMessage) {
          updateMessage(payload.message_id, {
            content: "Here's the image I generated based on your request:",
            metadata: { ...existingMessage.metadata, generated_image: payload.image }
          });
        }
        break;
      }
      
      case 'image_generation_failed': {
        const failPayload = message.payload as {
          chat_id: string;
          message_id: string;
          error: string;
        };
        
        const { messages, updateMessage } = useChatStore.getState();
        const existingMsg = messages.find(m => m.id === failPayload.message_id);
        
        if (existingMsg) {
          updateMessage(failPayload.message_id, {
            content: `I wasn't able to generate the image: ${failPayload.error || 'Unknown error'}`,
            metadata: { 
              ...existingMsg.metadata, 
              image_generation: { 
                status: 'failed', 
                error: failPayload.error 
              }
            }
          });
        }
        break;
      }
        
      case 'error':
        console.error('WebSocket error:', message.payload);
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  }, [addMessage, appendStreamingContent, setStreamingContent, setStreamingToolCall, clearStreaming, setIsSending, updateChatLocally, updateStreamingArtifacts]);
  
  // Set up tool interrupt callback - checks streaming content for complete tool tags
  useEffect(() => {
    console.warn('[TOOL_SETUP] Setting up tool interrupt callback');
    toolInterruptCallbackRef.current = (streamingContent: string) => {
      const { currentChat, artifacts: storedArtifacts, uploadedArtifacts } = useChatStore.getState();
      if (!currentChat?.id || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.warn('[TOOL_INTERRUPT] Skipped - no chat or WS not connected');
        return;
      }
      
      // Capture parent ID at the START before any stop_generation clears it
      const capturedParentId = currentStreamingMessageIdRef.current;
      
      // ======== TRACK COMPLETED ARTIFACTS ========
      // Track artifacts as they complete during streaming (for notification in tool results)
      // We DON'T interrupt for artifacts - only track them to include in tool result messages
      const artifactPatterns = [
        { pattern: STREAM_ARTIFACT_XML_PATTERN, nameGroup: 1, contentGroup: 2 },
        { pattern: STREAM_ARTIFACT_EQUALS_PATTERN, nameGroup: 1, contentGroup: 2 },
        { pattern: STREAM_ARTIFACT_FILENAME_PATTERN, nameGroup: 1, contentGroup: 2 },
        { pattern: STREAM_CODE_FENCE_PATTERN, nameGroup: 1, contentGroup: 0 },
      ];
      
      for (const { pattern, nameGroup } of artifactPatterns) {
        // Reset lastIndex for global patterns
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(streamingContent)) !== null) {
          const artifactName = match[nameGroup];
          const artifactKey = `artifact:${artifactName}:${match[0].length}`;
          
          if (!processedToolTagsRef.current.has(artifactKey)) {
            processedToolTagsRef.current.add(artifactKey);
            // Track that this artifact was saved (for notification when we DO interrupt for a tool)
            if (!savedArtifactsDuringStreamRef.current.includes(artifactName)) {
              savedArtifactsDuringStreamRef.current.push(artifactName);
              console.log(`[ARTIFACT_TRACKED] Noted artifact during stream: ${artifactName}`);
            }
          }
        }
      }
      
      // Debug: Check if any tool-like patterns exist
      const hasToolTag = streamingContent.includes('<find') || 
                         streamingContent.includes('<request_file') || 
                         streamingContent.includes('<search_replace') ||
                         streamingContent.includes('<replace_block');
      if (hasToolTag) {
        console.warn('[TOOL_INTERRUPT] Tool tag detected in content, length:', streamingContent.length);
        console.warn('[TOOL_INTERRUPT] Last 500 chars:', streamingContent.slice(-500));
      }
      
      // Combine all artifacts for tool execution - always get fresh state
      const getAllArtifacts = () => {
        const { artifacts: freshArtifacts, uploadedArtifacts: freshUploaded } = useChatStore.getState();
        console.warn(`[TOOL_INTERRUPT] getAllArtifacts: ${freshArtifacts.length} artifacts, ${freshUploaded.length} uploaded`);
        return [...freshArtifacts, ...freshUploaded];
      };
      
      // Generate unique key for a tag to track if processed
      const getTagKey = (type: string, content: string) => `${type}:${content.length}:${content.slice(-50)}`;
      
      // Check for tool call loops (same tool called repeatedly)
      const isToolLoop = (tagKey: string): boolean => {
        const now = Date.now();
        // Clean old entries
        toolCallHistoryRef.current = toolCallHistoryRef.current.filter(
          entry => now - entry.timestamp < TOOL_HISTORY_WINDOW_MS
        );
        // Count how many times this exact call was made
        const sameCallCount = toolCallHistoryRef.current.filter(
          entry => entry.key === tagKey
        ).length;
        return sameCallCount >= MAX_SAME_TOOL_CALLS;
      };
      
      // Record tool call for loop detection
      const recordToolCall = (tagKey: string) => {
        toolCallHistoryRef.current.push({ key: tagKey, timestamp: Date.now() });
      };
      
      // Helper to send tool results without creating new messages
      // Uses capturedParentId to ensure proper conversation tree linkage
      // Includes notifications about artifacts that were saved during streaming
      // Large results are chunked into hidden agent files to prevent context overflow
      const sendToolResult = (toolName: string, results: string) => {
        const { zipContext, streamingContent, artifacts, setArtifacts } = useChatStore.getState();
        
        // Build notification about saved artifacts
        let artifactNotification = '';
        if (savedArtifactsDuringStreamRef.current.length > 0) {
          const savedFiles = savedArtifactsDuringStreamRef.current.join(', ');
          artifactNotification = `\n\n[FILES SAVED] The following files were saved during your response: ${savedFiles}\nYou can reference these files in your continued response.`;
          // Clear the saved artifacts list after notifying
          savedArtifactsDuringStreamRef.current = [];
        }
        
        // Ensure we always send something, even if results are empty
        let finalResults = results.trim() || `[${toolName.toUpperCase()}: No results returned]`;
        
        // Check if results exceed context threshold - chunk into hidden agent files
        if (finalResults.length > CONTEXT_CHUNK_THRESHOLD) {
          console.log(`[TOOL_RESULT] Large result (${finalResults.length} chars) - chunking into agent files`);
          const { summary, chunkedFiles, newArtifacts } = chunkLargeToolResult(toolName, finalResults);
          finalResults = summary;
          // Add new hidden artifacts to the store
          setArtifacts([...artifacts, ...newArtifacts]);
          console.log(`[TOOL_RESULT] Created ${chunkedFiles.length} agent files: ${chunkedFiles.join(', ')}`);
        }
        
        // Use captured streaming message ID as parent for linear conversation flow
        // This prevents tool continuations from creating branches
        console.log(`[TOOL_RESULT] Sending ${toolName} results (${finalResults.length} chars), parent_id: ${capturedParentId}, savedArtifacts: ${savedArtifactsDuringStreamRef.current.length}`);
        
        wsRef.current!.send(JSON.stringify({
          type: 'chat_message',
          payload: {
            chat_id: currentChat.id,
            content: `[SYSTEM TOOL RESULT - ${toolName}]\nThe following results were generated by the system, not typed by the user.\n\n${finalResults}${artifactNotification}\n\n[END TOOL RESULT]`,
            parent_id: capturedParentId,  // Link to current streaming message for linear conversation
            continue_message_id: capturedParentId,  // Append to this existing message instead of creating new
            zip_context: zipContext,
            save_user_message: false,
            is_tool_continuation: true,
          },
        }));
      };
      
      // Check for find_line tag
      const findLineMatch = streamingContent.match(STREAM_FIND_LINE_PATTERN);
      if (findLineMatch) {
        console.log('[TOOL_INTERRUPT] MATCHED find_line:', findLineMatch[0].substring(0, 100));
        const tagKey = getTagKey('find_line', findLineMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          console.log('[TOOL_INTERRUPT] Processing find_line (not yet processed)');
          
          // Check for tool loop
          if (isToolLoop(tagKey)) {
            console.warn('[TOOL_INTERRUPT] Loop detected for find_line, stopping');
            wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
            streamingBufferRef.current?.pause();
            sendToolResult('find_line', '[LOOP DETECTED] You have called this same tool multiple times in a row. Please try a different approach or proceed with the information you have.');
            return;
          }
          recordToolCall(tagKey);
          
          console.log('[TOOL_INTERRUPT] Detected find_line tag, executing immediately');
          
          // Stop generation and pause buffer to prevent spillover
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          // Execute tool and send results
          const ops: FindLineOp[] = [{ path: findLineMatch[1], contains: findLineMatch[2] }];
          const results = executeFindLineOperations(getAllArtifacts(), ops);
          sendToolResult('find_line', results.join('\n\n'));
          return;
        } else {
          console.log('[TOOL_INTERRUPT] find_line already processed, skipping');
        }
      }
      
      // Check for find tag (with path) - try both attribute orders
      let findWithPathMatch = streamingContent.match(STREAM_FIND_PATTERN_WITH_PATH);
      let pathGroup = 1, searchGroup = 2;  // Default: path first, search second
      
      if (!findWithPathMatch) {
        // Try reversed order: search first, path second
        findWithPathMatch = streamingContent.match(STREAM_FIND_PATTERN_WITH_PATH_REV);
        if (findWithPathMatch) {
          pathGroup = 2;  // Path is second group in reversed pattern
          searchGroup = 1;  // Search is first group
        }
      }
      
      if (findWithPathMatch) {
        console.log('[TOOL_INTERRUPT] MATCHED find (with path):', findWithPathMatch[0].substring(0, 100));
        const tagKey = getTagKey('find_path', findWithPathMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          
          // Check for tool loop
          if (isToolLoop(tagKey)) {
            console.warn('[TOOL_INTERRUPT] Loop detected for find, stopping');
            wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
            streamingBufferRef.current?.pause();
            sendToolResult('find', '[LOOP DETECTED] You have called this same tool multiple times in a row. Please try a different approach or proceed with the information you have.');
            return;
          }
          recordToolCall(tagKey);
          
          console.log('[TOOL_INTERRUPT] Detected find tag (with path), executing immediately');
          
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          const ops: FindOp[] = [{ path: findWithPathMatch[pathGroup], search: findWithPathMatch[searchGroup] }];
          const results = executeFindOperations(getAllArtifacts(), ops);
          sendToolResult('find', results.join('\n\n'));
          return;
        }
      }
      
      // Check for find tag (no path - search all)
      const findNoPathMatch = streamingContent.match(STREAM_FIND_PATTERN_NO_PATH);
      if (findNoPathMatch) {
        console.log('[TOOL_INTERRUPT] MATCHED find (no path):', findNoPathMatch[0].substring(0, 100));
        const tagKey = getTagKey('find_all', findNoPathMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          
          // Check for tool loop
          if (isToolLoop(tagKey)) {
            console.warn('[TOOL_INTERRUPT] Loop detected for find, stopping');
            wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
            streamingBufferRef.current?.pause();
            sendToolResult('find', '[LOOP DETECTED] You have called this same tool multiple times in a row. Please try a different approach or proceed with the information you have.');
            return;
          }
          recordToolCall(tagKey);
          
          console.log('[TOOL_INTERRUPT] Detected find tag (no path), executing immediately');
          
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          const ops: FindOp[] = [{ path: null, search: findNoPathMatch[1] }];
          const results = executeFindOperations(getAllArtifacts(), ops);
          sendToolResult('find', results.join('\n\n'));
          return;
        }
      }
      
      // Check for request_file tag
      const requestFileMatch = streamingContent.match(STREAM_REQUEST_FILE_PATTERN);
      if (requestFileMatch) {
        const tagKey = getTagKey('request_file', requestFileMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          console.log('[TOOL_INTERRUPT] Detected request_file tag, executing immediately');
          
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          const requestPath = requestFileMatch[1];
          const allArtifacts = getAllArtifacts();
          
          // Extensive debugging
          console.log(`[REQUEST_FILE] ==========================================`);
          console.log(`[REQUEST_FILE] Looking for: "${requestPath}"`);
          console.log(`[REQUEST_FILE] Total artifacts: ${allArtifacts.length}`);
          const { artifacts: debugArts, uploadedArtifacts: debugUploaded } = useChatStore.getState();
          console.log(`[REQUEST_FILE] Store state: artifacts=${debugArts.length}, uploadedArtifacts=${debugUploaded.length}`);
          allArtifacts.forEach((a, i) => {
            console.log(`[REQUEST_FILE] [${i}] filename="${a.filename}" title="${a.title}" source="${a.source || 'unknown'}"`);
          });
          console.log(`[REQUEST_FILE] ==========================================`);
          
          // Use shared helper for consistent path matching
          const artifact = findArtifactByPath(allArtifacts, requestPath);
          
          if (artifact) {
            console.log(`[REQUEST_FILE] ✓ Found: ${artifact.filename || artifact.title}`);
            const content = `=== FILE: ${artifact.filename || artifact.title} ===\n${artifact.content}\n=== END FILE ===`;
            sendToolResult('request_file', content);
          } else {
            // List available files to help LLM
            const availableFiles = allArtifacts.map(a => a.filename || a.title).filter(Boolean);
            console.log('[REQUEST_FILE] ✗ Not found. Available:', availableFiles.join(', '));
            const fileList = availableFiles.length > 0 
              ? `Available files:\n${availableFiles.map(f => `  - ${f}`).join('\n')}`
              : 'No files currently available. Upload files or create artifacts first.';
            sendToolResult('request_file', `[FILE_ERROR: Could not find "${requestPath}"]\n\n${fileList}`);
          }
          return;
        }
      }
      
      // Check for search_replace tag (multi-line)
      const searchReplaceMatch = streamingContent.match(STREAM_SEARCH_REPLACE_PATTERN);
      if (searchReplaceMatch) {
        console.log('[TOOL_INTERRUPT] MATCHED search_replace, path:', searchReplaceMatch[1]);
        const tagKey = getTagKey('search_replace', searchReplaceMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          console.log('[TOOL_INTERRUPT] Detected search_replace tag, executing immediately');
          
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          const ops: SearchReplaceOp[] = [{
            path: searchReplaceMatch[1],
            search: searchReplaceMatch[2],
            replace: searchReplaceMatch[3],
          }];
          
          const allArtifacts = getAllArtifacts();
          const { updateArtifact } = useChatStore.getState();
          const { updatedArtifacts, results, modifiedFiles } = executeSearchReplaceOperations(allArtifacts, ops);
          
          // Update modified artifacts in store
          for (const art of updatedArtifacts) {
            const key = art.filename || art.title || '';
            if (key && modifiedFiles.includes(key)) {
              updateArtifact(key, art);
            }
          }
          
          sendToolResult('search_replace', results.join('\n\n'));
          return;
        }
      }
      
      // Check for replace_block tag (legacy format, multi-line)
      const replaceBlockMatch = streamingContent.match(STREAM_REPLACE_BLOCK_PATTERN);
      if (replaceBlockMatch) {
        console.log('[TOOL_INTERRUPT] MATCHED replace_block, path:', replaceBlockMatch[1]);
        const tagKey = getTagKey('replace_block', replaceBlockMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          console.log('[TOOL_INTERRUPT] Detected replace_block tag, executing immediately');
          
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          // Convert to SearchReplaceOp format (same structure)
          const ops: SearchReplaceOp[] = [{
            path: replaceBlockMatch[1],
            search: replaceBlockMatch[2],
            replace: replaceBlockMatch[3],
          }];
          
          const allArtifacts = getAllArtifacts();
          const { updateArtifact } = useChatStore.getState();
          const { updatedArtifacts, results, modifiedFiles } = executeSearchReplaceOperations(allArtifacts, ops);
          
          // Update modified artifacts in store
          for (const art of updatedArtifacts) {
            const key = art.filename || art.title || '';
            if (key && modifiedFiles.includes(key)) {
              updateArtifact(key, art);
            }
          }
          
          sendToolResult('replace_block', results.join('\n\n'));
          return;
        }
      }
      
      // Check for kb_search tag - search knowledge bases
      const kbSearchMatch = streamingContent.match(STREAM_KB_SEARCH_PATTERN);
      if (kbSearchMatch) {
        console.log('[TOOL_INTERRUPT] MATCHED kb_search, query:', kbSearchMatch[1]);
        const tagKey = getTagKey('kb_search', kbSearchMatch[0]);
        if (!processedToolTagsRef.current.has(tagKey)) {
          processedToolTagsRef.current.add(tagKey);
          
          // Check for tool loop
          if (isToolLoop(tagKey)) {
            console.warn('[TOOL_INTERRUPT] Loop detected for kb_search, stopping');
            wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
            streamingBufferRef.current?.pause();
            sendToolResult('kb_search', '[LOOP DETECTED] You have searched for this same query multiple times. Please use the results you have or try a different query.');
            return;
          }
          recordToolCall(tagKey);
          
          console.log('[TOOL_INTERRUPT] Detected kb_search tag, executing API call');
          
          // Stop generation and pause buffer
          wsRef.current!.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
          streamingBufferRef.current?.pause();
          
          // Make API call to search knowledge bases
          const searchQuery = kbSearchMatch[1];
          api.post('/knowledge-stores/search', { query: searchQuery, top_k: 5 })
            .then((response) => {
              const { results, stores_searched } = response.data;
              if (results.length === 0) {
                sendToolResult('kb_search', `[KB_SEARCH: No results found for "${searchQuery}" across ${stores_searched} knowledge bases]`);
              } else {
                const formattedResults = results.map((r: { document_name: string; knowledge_store_name: string; score: number; content: string }, i: number) => 
                  `[Result ${i + 1}] From "${r.document_name}" (${r.knowledge_store_name}, score: ${r.score.toFixed(2)})\n${r.content}`
                ).join('\n\n---\n\n');
                sendToolResult('kb_search', `[KB_SEARCH RESULTS for "${searchQuery}"]\nSearched ${stores_searched} knowledge bases, found ${results.length} results:\n\n${formattedResults}`);
              }
            })
            .catch((err) => {
              console.error('[TOOL_INTERRUPT] kb_search API error:', err);
              sendToolResult('kb_search', `[KB_SEARCH ERROR: Failed to search knowledge bases - ${err.message || 'Unknown error'}]`);
            });
          return;
        }
      }
    };
    
    // Clear processed tags when streaming ends or on unmount
    return () => {
      toolInterruptCallbackRef.current = null;
    };
  }, []); // No dependencies - uses refs and getState()
  
  // Set up tool detected callback - handles complete tool tags found in buffer BEFORE they're displayed
  useEffect(() => {
    console.log('[TOOL_DETECTED_SETUP] Setting up tool detected callback');
    toolDetectedCallbackRef.current = (toolTag: string, _beforeContent: string) => {
      const { currentChat, artifacts, uploadedArtifacts } = useChatStore.getState();
      if (!currentChat?.id || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.warn('[TOOL_DETECTED] Skipped - no chat or WS not connected');
        return;
      }
      
      console.log('[TOOL_DETECTED] Processing tag:', toolTag.substring(0, 100));
      
      // If this is a <tool_call>...</tool_call> wrapper, extract inner content
      let tagToProcess = toolTag;
      const toolCallMatch = toolTag.match(/<tool_call[^>]*>([\s\S]*?)<\/tool_call>/i);
      if (toolCallMatch) {
        // Extract inner content - might contain <find> or other tags
        const innerContent = toolCallMatch[1].trim();
        console.log('[TOOL_DETECTED] Extracted from tool_call wrapper:', innerContent.substring(0, 100));
        
        // Look for inner tool tag
        const innerTagMatch = innerContent.match(/<(find|find_line|request_file|search_replace|kb_search)[^>]*>/i);
        if (innerTagMatch) {
          tagToProcess = innerTagMatch[0];
          console.log('[TOOL_DETECTED] Found inner tag:', tagToProcess);
        } else {
          // Maybe it's JSON format: {"tool": "find", "query": "..."}
          try {
            const jsonMatch = innerContent.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              const parsed = JSON.parse(jsonMatch[0]);
              const toolName = parsed.tool || parsed.name;
              if (toolName === 'find') {
                tagToProcess = `<find search="${parsed.query || parsed.search || ''}" ${parsed.path ? `path="${parsed.path}"` : ''}>`;
              }
            }
          } catch (e) {
            console.warn('[TOOL_DETECTED] Could not parse tool_call JSON:', innerContent.substring(0, 100));
          }
        }
      }
      
      // Capture parent ID BEFORE stopping - stream_end will clear it
      const capturedParentId = currentStreamingMessageIdRef.current;
      
      // Pause the buffer to stop further content
      streamingBufferRef.current?.pause();
      
      // Stop the LLM generation
      wsRef.current.send(JSON.stringify({ type: 'stop_generation', payload: { chat_id: currentChat.id } }));
      
      // Combine artifacts for searching
      const allArtifacts = [...artifacts, ...uploadedArtifacts];
      
      // Helper to send tool result - uses captured parentId
      const sendResult = (toolName: string, result: string) => {
        const { zipContext } = useChatStore.getState();
        
        console.log(`[TOOL_DETECTED] Sending ${toolName} result, parent_id: ${capturedParentId}, continue_message_id: ${capturedParentId}`);
        
        wsRef.current!.send(JSON.stringify({
          type: 'chat_message',
          payload: {
            chat_id: currentChat.id,
            content: `[SYSTEM TOOL RESULT - ${toolName}]\nThe following results were generated by the system, not typed by the user.\n\n${result}\n\n[END TOOL RESULT]`,
            parent_id: capturedParentId,
            continue_message_id: capturedParentId,  // Append to this existing message instead of creating new
            zip_context: zipContext,
            save_user_message: false,
            is_tool_continuation: true,
          },
        }));
      };
      
      // Parse and execute the tool tag
      // <find search="..." path="..."> or <find path="..." search="...">
      const findMatch = tagToProcess.match(/<find\s+(?:search=["']([^"']+)["']\s+path=["']([^"']+)["']|path=["']([^"']+)["']\s+search=["']([^"']+)["']|search=["']([^"']+)["'])\s*\/?>/i);
      if (findMatch) {
        // Extract search and path from whichever groups matched
        const search = findMatch[1] || findMatch[4] || findMatch[5];
        const path = findMatch[2] || findMatch[3] || null;
        
        console.log(`[TOOL_DETECTED] find: search="${search}", path="${path}"`);
        
        const ops: FindOp[] = [{ path, search }];
        const results = executeFindOperations(allArtifacts, ops);
        sendResult('find', results.join('\n\n'));
        return;
      }
      
      // <find_line path="..." contains="...">
      const findLineMatch = tagToProcess.match(/<find_line\s+path=["']([^"']+)["']\s+contains=["']([^"']+)["']\s*\/?>/i);
      if (findLineMatch) {
        const path = findLineMatch[1];
        const contains = findLineMatch[2];
        
        console.log(`[TOOL_DETECTED] find_line: path="${path}", contains="${contains}"`);
        
        const ops: FindLineOp[] = [{ path, contains }];
        const results = executeFindLineOperations(allArtifacts, ops);
        sendResult('find_line', results.join('\n\n'));
        return;
      }
      
      // <request_file path="...">
      const requestFileMatch = tagToProcess.match(/<request_file\s+path=["']([^"']+)["']\s*\/?>/i);
      if (requestFileMatch) {
        const path = requestFileMatch[1];
        console.log(`[TOOL_DETECTED] request_file: path="${path}"`);
        
        // Try to find the artifact
        const artifact = findArtifactByPath(allArtifacts, path);
        if (artifact) {
          sendResult('request_file', `[FILE: ${path}]\n\`\`\`\n${artifact.content}\n\`\`\``);
        } else {
          const availableFiles = allArtifacts.map(a => a.filename || a.title || 'unnamed').slice(0, 20);
          sendResult('request_file', `[FILE_ERROR: Could not find "${path}"]\nAvailable files: ${availableFiles.join(', ')}`);
        }
        return;
      }
      
      // <search_replace path="...">===== SEARCH...===== Replace...</search_replace>
      const searchReplaceMatch = tagToProcess.match(/<search_replace\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*[Rr]eplace\s*\n([\s\S]*?)<\/search_replace>/i);
      if (searchReplaceMatch) {
        const path = searchReplaceMatch[1];
        const searchText = searchReplaceMatch[2];
        const replaceText = searchReplaceMatch[3];
        
        console.log(`[TOOL_DETECTED] search_replace: path="${path}", search=${searchText.length} chars, replace=${replaceText.length} chars`);
        
        const ops: SearchReplaceOp[] = [{ path, search: searchText, replace: replaceText }];
        const { updateArtifact } = useChatStore.getState();
        const { updatedArtifacts, results, modifiedFiles } = executeSearchReplaceOperations(allArtifacts, ops);
        
        // Update modified artifacts in store
        for (const art of updatedArtifacts) {
          const key = art.filename || art.title || '';
          if (key && modifiedFiles.includes(key)) {
            updateArtifact(key, art);
          }
        }
        
        sendResult('search_replace', results.join('\n\n'));
        return;
      }
      
      // <replace_block path="...">===== SEARCH...===== Replace...</replace_block> (legacy)
      const replaceBlockMatch = tagToProcess.match(/<replace_block\s+path=["']?([^"'>\s]+)["']?\s*>\s*\n?=====\s*SEARCH\s*\n([\s\S]*?)\n=====\s*[Rr]eplace\s*\n([\s\S]*?)<\/replace_block>/i);
      if (replaceBlockMatch) {
        const path = replaceBlockMatch[1];
        const searchText = replaceBlockMatch[2];
        const replaceText = replaceBlockMatch[3];
        
        console.log(`[TOOL_DETECTED] replace_block: path="${path}", search=${searchText.length} chars, replace=${replaceText.length} chars`);
        
        const ops: SearchReplaceOp[] = [{ path, search: searchText, replace: replaceText }];
        const { updateArtifact } = useChatStore.getState();
        const { updatedArtifacts, results, modifiedFiles } = executeSearchReplaceOperations(allArtifacts, ops);
        
        // Update modified artifacts in store
        for (const art of updatedArtifacts) {
          const key = art.filename || art.title || '';
          if (key && modifiedFiles.includes(key)) {
            updateArtifact(key, art);
          }
        }
        
        sendResult('replace_block', results.join('\n\n'));
        return;
      }
      
      // <kb_search query="...">
      const kbSearchMatch = tagToProcess.match(/<kb_search\s+query=["']([^"']+)["']\s*\/?>/i);
      if (kbSearchMatch) {
        const query = kbSearchMatch[1];
        console.log(`[TOOL_DETECTED] kb_search: query="${query}"`);
        // KB search needs async API call - for now just acknowledge
        sendResult('kb_search', `[KB_SEARCH: Searching for "${query}"... (async operation)]`);
        return;
      }
      
      console.warn('[TOOL_DETECTED] Unrecognized tool tag:', tagToProcess);
    };
    
    return () => {
      toolDetectedCallbackRef.current = null;
    };
  }, []);
  
  const connect = useCallback(() => {
    if (!accessToken || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    try {
      const wsUrl = `${getWebSocketUrl()}/ws/ws?token=${accessToken}`;
      console.log('Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      let wasConnected = false;
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        wasConnected = true;
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttempts.current = 0;
        
        // Subscribe to current chat if any
        const { currentChat } = useChatStore.getState();
        if (currentChat) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            payload: { chat_id: currentChat.id },
          }));
        }
      };
      
      ws.onmessage = (event) => {
        // Use type-safe parser from wsTypes
        const data = parseServerEvent(event.data);
        if (!data) {
          console.warn('Received invalid WebSocket message');
          return;
        }
        
        // Debug: log ALL incoming messages except pong
        if (data.type !== 'pong') {
          console.log('%c[WS]', 'color: green; font-weight: bold', data.type, data.payload ? Object.keys(data.payload) : 'no payload');
        }
        if (data.type === 'image_generation') {
          console.log('%c[IMAGE]', 'color: blue; font-weight: bold; font-size: 16px', 'Got image!', data.payload?.message_id);
        }
        handleMessage(data as WSMessage);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Don't set error message yet - wait for onclose to determine if it's auth failure
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason, 'wasConnected:', wasConnected);
        setIsConnected(false);
        wsRef.current = null;
        
        // Handle auth errors - code 4001 or immediate close without ever connecting
        const isAuthError = event.code === 4001 || event.code === 1008 || 
          (event.code === 1006 && !wasConnected) ||
          event.reason?.toLowerCase().includes('token') ||
          event.reason?.toLowerCase().includes('auth');
        
        if (isAuthError) {
          console.log('WebSocket auth failed, logging out');
          logout();
          window.location.href = '/login';
          return;
        }
        
        // Set connection error for non-auth failures
        if (!wasConnected) {
          setConnectionError('Connection failed');
        }
        
        // Reconnect after delay if authenticated (with backoff)
        const { isAuthenticated } = useAuthStore.getState();
        if (isAuthenticated && event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          reconnectAttempts.current++;
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setConnectionError('Connection lost. Please refresh the page.');
        }
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setConnectionError('Failed to connect');
    }
  }, [accessToken, handleMessage, logout]);
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);
  
  const subscribe = useCallback((chatId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        payload: { chat_id: chatId },
      }));
    }
  }, []);
  
  const unsubscribe = useCallback((chatId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        payload: { chat_id: chatId },
      }));
    }
  }, []);
  
  const sendChatMessage = useCallback((chatId: string, content: string, attachments?: unknown[], parentId?: string | null) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Reset file write counter for this chat on new user message (new task)
      artifactWriteCountRef.current[chatId] = {};
      
      // Convert attachments to display format for the user message
      const displayAttachments = attachments?.map((att: any) => ({
        type: att.type as 'image' | 'file',
        name: att.filename || 'attachment',
        data: att.data,
        mime_type: att.mime_type,
        url: att.type === 'image' && att.data ? `data:${att.mime_type};base64,${att.data}` : undefined,
      }));
      
      console.log('[sendChatMessage] Attachments:', {
        rawCount: attachments?.length || 0,
        displayCount: displayAttachments?.length || 0,
        first: displayAttachments?.[0] ? { 
          type: displayAttachments[0].type, 
          name: displayAttachments[0].name, 
          hasData: !!displayAttachments[0].data,
          dataLen: displayAttachments[0].data?.length || 0,
          hasUrl: !!displayAttachments[0].url,
        } : null,
      });
      
      // Generate a proper UUID for the user message (used by both frontend and backend)
      const messageId = crypto.randomUUID();
      const userMessage: Message = {
        id: messageId,
        chat_id: chatId,
        role: 'user',
        content,
        attachments: displayAttachments,
        parent_id: parentId,
        created_at: new Date().toISOString(),
      };
      addMessage(userMessage);
      
      setIsSending(true);
      
      // Get zip context from store if available
      const { zipContext } = useChatStore.getState();
      
      console.log('Sending chat message:', { chatId, content: content.substring(0, 50), parentId, messageId, hasZipContext: !!zipContext });
      wsRef.current.send(JSON.stringify({
        type: 'chat_message',
        payload: {
          chat_id: chatId,
          content,
          attachments,
          parent_id: parentId,
          message_id: messageId,  // Send the ID so backend uses it
          zip_context: zipContext,  // Include zip manifest if available
        },
      }));
    } else {
      console.error('WebSocket not connected');
    }
  }, [addMessage, setIsSending]);
  
  // Regenerate without adding a new user message (for retry)
  const regenerateMessage = useCallback((chatId: string, content: string, parentId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setIsSending(true);
      
      console.log('Regenerating message:', { chatId, content: content.substring(0, 50), parentId });
      wsRef.current.send(JSON.stringify({
        type: 'regenerate_message',
        payload: {
          chat_id: chatId,
          content,
          parent_id: parentId,  // The user message whose response we're regenerating
        },
      }));
    } else {
      console.error('WebSocket not connected');
    }
  }, [setIsSending]);
  
  const sendClientMessage = useCallback((chatId: string, content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'client_message',
        payload: {
          chat_id: chatId,
          content,
        },
      }));
    }
  }, []);
  
  const stopGeneration = useCallback((chatId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('Stopping generation for chat:', chatId);
      
      // Immediately clear streaming state for responsive UI
      streamingBufferRef.current?.flushNow();
      clearStreaming();
      setIsSending(false);
      
      // Send stop request to server
      wsRef.current.send(JSON.stringify({
        type: 'stop_generation',
        payload: { chat_id: chatId },
      }));
    }
  }, [setIsSending, clearStreaming]);
  
  const ping = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);
  
  // Connect on mount if authenticated
  useEffect(() => {
    if (isAuthenticated && accessToken) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [isAuthenticated, accessToken, connect, disconnect]);
  
  // Track previous chat for proper unsubscribe
  const previousChatIdRef = useRef<string | null>(null);
  
  // Subscribe to chat when it changes - with proper cleanup
  useEffect(() => {
    if (!isConnected) return;
    
    const newChatId = currentChat?.id || null;
    const oldChatId = previousChatIdRef.current;
    
    // Unsubscribe from old chat first
    if (oldChatId && oldChatId !== newChatId) {
      console.log('[WS] Unsubscribing from old chat:', oldChatId);
      unsubscribe(oldChatId);
      // Clear any lingering streaming state when switching chats
      clearStreaming();
    }
    
    // Subscribe to new chat
    if (newChatId) {
      console.log('[WS] Subscribing to chat:', newChatId);
      subscribe(newChatId);
    }
    
    previousChatIdRef.current = newChatId;
    
    // Cleanup on unmount
    return () => {
      if (newChatId) {
        unsubscribe(newChatId);
      }
    };
  }, [isConnected, currentChat?.id, subscribe, unsubscribe, clearStreaming]);
  
  // Heartbeat
  useEffect(() => {
    if (!isConnected) return;
    
    const interval = setInterval(() => {
      ping();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [isConnected, ping]);
  
  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        connectionError,
        subscribe,
        unsubscribe,
        sendChatMessage,
        sendClientMessage,
        regenerateMessage,
        stopGeneration,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}
