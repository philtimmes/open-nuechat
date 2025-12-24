/**
 * File processor for converting uploaded files to artifacts
 * Extracts code signatures and prepares files for LLM context
 */

import type { Artifact, CodeSignature } from '../types';

// Binary document types that need backend extraction
const BINARY_EXTENSIONS = new Set(['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.rtf']);

// Gzip extensions
const GZIP_EXTENSIONS = new Set(['.gz', '.gzip']);

// Size limits for safety
const MAX_COMPRESSED_SIZE = 10 * 1024 * 1024;  // 10MB max compressed
const MAX_DECOMPRESSED_SIZE = 50 * 1024 * 1024;  // 50MB max decompressed
const DECOMPRESSION_RATIO_LIMIT = 100;  // Max 100x expansion (zip bomb protection)

// Extension to language and artifact type mapping
const EXT_MAP: Record<string, { language: string; type: Artifact['type'] }> = {
  '.py': { language: 'python', type: 'code' },
  '.js': { language: 'javascript', type: 'code' },
  '.jsx': { language: 'javascript', type: 'react' },
  '.ts': { language: 'typescript', type: 'code' },
  '.tsx': { language: 'typescript', type: 'react' },
  '.java': { language: 'java', type: 'code' },
  '.go': { language: 'go', type: 'code' },
  '.rs': { language: 'rust', type: 'code' },
  '.rb': { language: 'ruby', type: 'code' },
  '.php': { language: 'php', type: 'code' },
  '.c': { language: 'c', type: 'code' },
  '.cpp': { language: 'cpp', type: 'code' },
  '.cc': { language: 'cpp', type: 'code' },
  '.h': { language: 'c', type: 'code' },
  '.hpp': { language: 'cpp', type: 'code' },
  '.cs': { language: 'csharp', type: 'code' },
  '.swift': { language: 'swift', type: 'code' },
  '.kt': { language: 'kotlin', type: 'code' },
  '.scala': { language: 'scala', type: 'code' },
  '.lua': { language: 'lua', type: 'code' },
  '.r': { language: 'r', type: 'code' },
  '.sh': { language: 'bash', type: 'code' },
  '.bash': { language: 'bash', type: 'code' },
  '.yaml': { language: 'yaml', type: 'code' },
  '.yml': { language: 'yaml', type: 'code' },
  '.toml': { language: 'toml', type: 'code' },
  '.xml': { language: 'xml', type: 'code' },
  '.html': { language: 'html', type: 'html' },
  '.css': { language: 'css', type: 'code' },
  '.scss': { language: 'scss', type: 'code' },
  '.sass': { language: 'sass', type: 'code' },
  '.sql': { language: 'sql', type: 'code' },
  '.graphql': { language: 'graphql', type: 'code' },
  '.proto': { language: 'protobuf', type: 'code' },
  '.asm': { language: 'asm', type: 'code' },
  '.s': { language: 'asm', type: 'code' },
  '.md': { language: 'markdown', type: 'markdown' },
  '.json': { language: 'json', type: 'json' },
  '.csv': { language: 'csv', type: 'csv' },
  '.txt': { language: 'text', type: 'document' },
  '.log': { language: 'log', type: 'document' },
  '.pdf': { language: 'pdf', type: 'document' },
  '.docx': { language: 'docx', type: 'document' },
  '.doc': { language: 'doc', type: 'document' },
  '.xlsx': { language: 'xlsx', type: 'document' },
  '.xls': { language: 'xls', type: 'document' },
  '.rtf': { language: 'rtf', type: 'document' },
};

// Signature extraction patterns by language
const SIGNATURE_PATTERNS: Record<string, Array<{ kind: string; pattern: RegExp }>> = {
  python: [
    { kind: 'class', pattern: /^class\s+(\w+)(?:\s*\(([^)]*)\))?:/gm },
    { kind: 'function', pattern: /^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:/gm },
    { kind: 'variable', pattern: /^([A-Z][A-Z0-9_]*)\s*=/gm },
  ],
  javascript: [
    { kind: 'class', pattern: /^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?/gm },
    { kind: 'function', pattern: /^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(/gm },
    { kind: 'function', pattern: /^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>/gm },
    { kind: 'variable', pattern: /^(?:export\s+)?(?:const|let|var)\s+([A-Z][A-Z0-9_]*)\s*=/gm },
  ],
  typescript: [
    { kind: 'interface', pattern: /^(?:export\s+)?interface\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+[^{]+)?/gm },
    { kind: 'type', pattern: /^(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*=/gm },
    { kind: 'class', pattern: /^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+(\w+))?/gm },
    { kind: 'function', pattern: /^(?:export\s+)?(?:async\s+)?function\s+(\w+)(?:<[^>]+>)?\s*\(/gm },
    { kind: 'function', pattern: /^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)(?:\s*:[^=]+)?\s*=>/gm },
  ],
  java: [
    { kind: 'class', pattern: /^(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)/gm },
    { kind: 'interface', pattern: /^(?:public\s+)?interface\s+(\w+)/gm },
    { kind: 'method', pattern: /^\s+(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(/gm },
  ],
  go: [
    { kind: 'function', pattern: /^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(/gm },
    { kind: 'type', pattern: /^type\s+(\w+)\s+(?:struct|interface)/gm },
  ],
  rust: [
    { kind: 'function', pattern: /^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\(/gm },
    { kind: 'struct', pattern: /^(?:pub\s+)?struct\s+(\w+)/gm },
    { kind: 'enum', pattern: /^(?:pub\s+)?enum\s+(\w+)/gm },
    { kind: 'trait', pattern: /^(?:pub\s+)?trait\s+(\w+)/gm },
    { kind: 'impl', pattern: /^impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)/gm },
  ],
  c: [
    { kind: 'function', pattern: /^(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:;|\{)/gm },
    { kind: 'struct', pattern: /^(?:typedef\s+)?struct\s+(\w+)/gm },
    { kind: 'define', pattern: /^#define\s+(\w+)/gm },
  ],
  cpp: [
    { kind: 'class', pattern: /^class\s+(\w+)/gm },
    { kind: 'function', pattern: /^(?:[\w:]+\s+)+(\w+)\s*\([^)]*\)(?:\s*const)?(?:\s*override)?(?:\s*=\s*0)?(?:\s*;|\s*\{)/gm },
    { kind: 'struct', pattern: /^struct\s+(\w+)/gm },
    { kind: 'namespace', pattern: /^namespace\s+(\w+)/gm },
  ],
  asm: [
    { kind: 'label', pattern: /^(\w+):/gm },
    { kind: 'function', pattern: /^\.(?:global|globl)\s+(\w+)/gm },
  ],
};

/**
 * Extract code signatures from file content
 */
export function extractSignatures(content: string, language: string): CodeSignature[] {
  const signatures: CodeSignature[] = [];
  const patterns = SIGNATURE_PATTERNS[language] || [];
  const lines = content.split('\n');
  
  for (const { kind, pattern } of patterns) {
    // Reset regex state
    pattern.lastIndex = 0;
    let match;
    
    while ((match = pattern.exec(content)) !== null) {
      const name = match[1];
      const lineNumber = content.substring(0, match.index).split('\n').length;
      const signatureLine = lines[lineNumber - 1]?.trim() || match[0];
      
      // Get docstring (line before if it looks like a comment)
      let docstring: string | undefined;
      if (lineNumber > 1) {
        const prevLine = lines[lineNumber - 2]?.trim();
        if (prevLine?.startsWith('//') || prevLine?.startsWith('#') || 
            prevLine?.startsWith('*') || prevLine?.startsWith('"""') ||
            prevLine?.startsWith("'''")) {
          docstring = prevLine.replace(/^[#/*'"]+\s*/, '');
        }
      }
      
      signatures.push({
        name,
        kind,
        line: lineNumber,
        signature: signatureLine.length > 100 ? signatureLine.substring(0, 100) + '...' : signatureLine,
        docstring,
      });
    }
  }
  
  // Sort by line number
  return signatures.sort((a, b) => a.line - b.line);
}

/**
 * Read file as text
 */
export async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

/**
 * Read file as base64 (for images)
 */
export async function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix to get pure base64
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

/**
 * Read file as ArrayBuffer
 */
async function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Check if a file is gzipped by magic bytes (1f 8b)
 */
function isGzipByMagicBytes(buffer: ArrayBuffer): boolean {
  const view = new Uint8Array(buffer);
  return view.length >= 2 && view[0] === 0x1f && view[1] === 0x8b;
}

/**
 * Decompress gzipped data using native DecompressionStream API
 * Falls back to manual implementation if not available
 */
async function decompressGzip(buffer: ArrayBuffer): Promise<string> {
  // Check file size before decompression
  if (buffer.byteLength > MAX_COMPRESSED_SIZE) {
    throw new Error(`Compressed file too large: ${(buffer.byteLength / 1024 / 1024).toFixed(2)}MB (max ${MAX_COMPRESSED_SIZE / 1024 / 1024}MB)`);
  }
  
  // Use native DecompressionStream if available (modern browsers)
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const stream = new Response(buffer).body;
      if (!stream) throw new Error('Failed to create stream');
      
      const decompressedStream = stream.pipeThrough(new DecompressionStream('gzip'));
      const reader = decompressedStream.getReader();
      const chunks: Uint8Array[] = [];
      let totalSize = 0;
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        totalSize += value.length;
        
        // Safety check: prevent zip bombs
        if (totalSize > MAX_DECOMPRESSED_SIZE) {
          reader.cancel();
          throw new Error(`Decompressed size exceeds limit: ${(totalSize / 1024 / 1024).toFixed(2)}MB (max ${MAX_DECOMPRESSED_SIZE / 1024 / 1024}MB)`);
        }
        
        // Check decompression ratio
        const ratio = totalSize / buffer.byteLength;
        if (ratio > DECOMPRESSION_RATIO_LIMIT) {
          reader.cancel();
          throw new Error(`Suspicious decompression ratio (${ratio.toFixed(0)}x) - possible zip bomb`);
        }
        
        chunks.push(value);
      }
      
      // Combine chunks and decode as UTF-8
      const combined = new Uint8Array(totalSize);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      
      const decoder = new TextDecoder('utf-8', { fatal: false });
      return decoder.decode(combined);
    } catch (error) {
      console.error('[decompressGzip] DecompressionStream failed:', error);
      throw error;
    }
  } else {
    // DecompressionStream not available
    throw new Error('Gzip decompression not supported in this browser. Please use a modern browser or upload the uncompressed file.');
  }
}

/**
 * Get the inner filename from a .gz file (strip .gz extension)
 */
function getInnerFilename(filename: string): string {
  const lowerName = filename.toLowerCase();
  if (lowerName.endsWith('.gz')) {
    return filename.slice(0, -3);
  } else if (lowerName.endsWith('.gzip')) {
    return filename.slice(0, -5);
  }
  return filename;
}

/**
 * Error line patterns for log files
 */
const ERROR_PATTERNS = [
  /\berror\b/i,
  /\bfailed\b/i,
  /\bfailure\b/i,
  /\bexception\b/i,
  /\btraceback\b/i,
  /\bcritical\b/i,
  /\bfatal\b/i,
  /\bpanic\b/i,
  /\bsegfault\b/i,
  /\bsegmentation fault\b/i,
  /\bstack trace\b/i,
  /\bcore dump\b/i,
  /\bcrash\b/i,
  /\babort\b/i,
  /\bwarning\b/i,
  /\bwarn\b/i,
  /\b4\d{2}\b/,  // HTTP 4xx errors
  /\b5\d{2}\b/,  // HTTP 5xx errors
  /errno/i,
  /ENOENT|EACCES|EPERM|ECONNREFUSED|ETIMEDOUT/i,
];

/**
 * Check if a line contains an error pattern
 */
function isErrorLine(line: string): boolean {
  return ERROR_PATTERNS.some(pattern => pattern.test(line));
}

/**
 * Extract error lines with context from content
 * Returns lines matching error patterns with their preceding and following non-error lines
 */
export function extractErrorLinesWithContext(content: string, maxErrors: number = 50): {
  errorSummary: string;
  errorCount: number;
  totalLines: number;
} {
  const lines = content.split('\n');
  const totalLines = lines.length;
  const errorIndices: number[] = [];
  
  // Find all error line indices
  for (let i = 0; i < lines.length; i++) {
    if (isErrorLine(lines[i])) {
      errorIndices.push(i);
    }
  }
  
  if (errorIndices.length === 0) {
    return {
      errorSummary: '',
      errorCount: 0,
      totalLines,
    };
  }
  
  // Limit number of errors to process
  const indicesToProcess = errorIndices.slice(0, maxErrors);
  const summaryLines: string[] = [];
  const processedRanges = new Set<string>();
  
  summaryLines.push(`=== ERROR SUMMARY (${errorIndices.length} errors found in ${totalLines} lines) ===`);
  if (errorIndices.length > maxErrors) {
    summaryLines.push(`(Showing first ${maxErrors} errors)`);
  }
  summaryLines.push('');
  
  for (const errorIdx of indicesToProcess) {
    // Find preceding non-error line
    let prevIdx = errorIdx - 1;
    while (prevIdx >= 0 && isErrorLine(lines[prevIdx])) {
      prevIdx--;
    }
    
    // Find following non-error line
    let nextIdx = errorIdx + 1;
    while (nextIdx < lines.length && isErrorLine(lines[nextIdx])) {
      nextIdx++;
    }
    
    // Create range key to avoid duplicates
    const rangeKey = `${Math.max(0, prevIdx)}-${Math.min(lines.length - 1, nextIdx)}`;
    if (processedRanges.has(rangeKey)) continue;
    processedRanges.add(rangeKey);
    
    // Add context lines
    summaryLines.push(`--- Line ${errorIdx + 1} ---`);
    
    // Previous non-error line (if exists and different from error)
    if (prevIdx >= 0 && prevIdx !== errorIdx) {
      summaryLines.push(`${prevIdx + 1}: ${lines[prevIdx].substring(0, 200)}`);
    }
    
    // Error line(s)
    for (let i = Math.max(0, prevIdx + 1); i <= Math.min(lines.length - 1, nextIdx - 1); i++) {
      summaryLines.push(`${i + 1}> ${lines[i].substring(0, 200)}`);
    }
    
    // Following non-error line (if exists)
    if (nextIdx < lines.length) {
      summaryLines.push(`${nextIdx + 1}: ${lines[nextIdx].substring(0, 200)}`);
    }
    
    summaryLines.push('');
  }
  
  summaryLines.push('=== END ERROR SUMMARY ===');
  
  return {
    errorSummary: summaryLines.join('\n'),
    errorCount: errorIndices.length,
    totalLines,
  };
}

/**
 * Extract text from a binary document via backend API
 */
async function extractTextFromDocument(file: File): Promise<string> {
  console.log(`[extractTextFromDocument v4] Starting extraction for ${file.name}, size=${file.size}, type=${file.type}`);
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    // Get auth token
    const authData = localStorage.getItem('nexus-auth');
    let authHeader = '';
    if (authData) {
      try {
        const { state } = JSON.parse(authData);
        if (state?.accessToken) {
          authHeader = `Bearer ${state.accessToken}`;
          console.log(`[extractTextFromDocument v4] Using auth token: ${authHeader.substring(0, 20)}...`);
        }
      } catch {
        console.warn('[extractTextFromDocument v4] Failed to parse auth data');
      }
    }
    
    if (!authHeader) {
      console.error('[extractTextFromDocument v4] No auth token available!');
      alert('Not authenticated - please log in again');
      return '';
    }
    
    console.log(`[extractTextFromDocument v4] Sending POST to /api/documents/extract-text`);
    
    // Use fetch directly to avoid axios Content-Type issues with FormData
    const response = await fetch('/api/documents/extract-text', {
      method: 'POST',
      headers: {
        'Authorization': authHeader,
      },
      body: formData,
    });
    
    console.log(`[extractTextFromDocument v4] Response status: ${response.status}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[extractTextFromDocument v4] Error response:`, errorText);
      let errorDetail = errorText;
      try {
        const errorJson = JSON.parse(errorText);
        errorDetail = errorJson.detail || JSON.stringify(errorJson);
      } catch {
        // Keep raw text
      }
      throw new Error(`HTTP ${response.status}: ${errorDetail}`);
    }
    
    const data = await response.json();
    console.log(`[extractTextFromDocument v4] Success:`, response.status, data?.chars, 'chars');
    return data.text || '';
  } catch (error: unknown) {
    const err = error as { message?: string };
    console.error(`[extractTextFromDocument v4] FAILED for ${file.name}:`, err.message);
    alert(`Failed to extract text from ${file.name}: ${err.message || 'Unknown error'}`);
    return '';
  }
}

/**
 * Check if a file is a log file (by name or extension)
 */
function isLogFile(filename: string): boolean {
  const lowerName = filename.toLowerCase();
  // Check for .log extension
  if (lowerName.endsWith('.log')) return true;
  // Check for "log" in filename (e.g., syslog, access_log, error.log.1)
  if (lowerName.includes('log')) return true;
  // Common log file patterns
  if (lowerName.includes('syslog') || lowerName.includes('messages') || 
      lowerName.includes('dmesg') || lowerName.includes('journal')) return true;
  return false;
}

/**
 * Process a file into an artifact, with gzip support and log error extraction
 */
export async function processFileToArtifact(file: File): Promise<{
  artifact: Artifact | null;
  errorSummary?: string;
  errorCount?: number;
}> {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  const isGzip = GZIP_EXTENSIONS.has(ext);
  
  console.log(`[processFileToArtifact v3] Processing ${file.name}, ext=${ext}, isGzip=${isGzip}`);
  
  // Handle images separately (they're attachments, not artifacts for now)
  if (file.type.startsWith('image/')) {
    console.log(`[processFileToArtifact v3] Skipping image: ${file.name}`);
    return { artifact: null };
  }
  
  try {
    let content: string;
    let actualFilename = file.name;
    let actualExt = ext;
    
    // Handle gzip files
    if (isGzip) {
      console.log(`[processFileToArtifact v3] Decompressing gzip file: ${file.name}`);
      const buffer = await readFileAsArrayBuffer(file);
      
      // Verify it's actually gzip by magic bytes
      if (!isGzipByMagicBytes(buffer)) {
        console.warn(`[processFileToArtifact v3] File ${file.name} has .gz extension but no gzip magic bytes`);
        // Try reading as text anyway
        content = await readFileAsText(file);
      } else {
        try {
          content = await decompressGzip(buffer);
          // Get inner filename (strip .gz)
          actualFilename = getInnerFilename(file.name);
          actualExt = '.' + actualFilename.split('.').pop()?.toLowerCase();
          console.log(`[processFileToArtifact v3] Decompressed ${file.name} -> ${actualFilename}, ${content.length} chars`);
        } catch (error) {
          console.error(`[processFileToArtifact v3] Gzip decompression failed:`, error);
          throw error;
        }
      }
    } else if (BINARY_EXTENSIONS.has(ext)) {
      // Binary documents need backend extraction
      console.log(`[processFileToArtifact v3] Binary file detected, calling extract API for ${file.name}`);
      content = await extractTextFromDocument(file);
      if (!content) {
        console.warn(`[processFileToArtifact v3] No text extracted from ${file.name}`);
        return { artifact: null };
      }
      console.log(`[processFileToArtifact v3] Extracted ${content.length} chars from ${file.name}`);
    } else {
      // Text files can be read directly
      console.log(`[processFileToArtifact v3] Reading text file ${file.name}`);
      content = await readFileAsText(file);
    }
    
    const extInfo = EXT_MAP[actualExt];
    const language = extInfo?.language || 'text';
    const type = extInfo?.type || 'document';
    
    // Extract signatures for code files
    const signatures = type === 'code' || type === 'react' 
      ? extractSignatures(content, language)
      : [];
    
    // Extract error summary for log files
    let errorSummary: string | undefined;
    let errorCount: number | undefined;
    
    if (isLogFile(actualFilename)) {
      console.log(`[processFileToArtifact v3] Detected log file, extracting errors: ${actualFilename}`);
      const errorInfo = extractErrorLinesWithContext(content);
      if (errorInfo.errorCount > 0) {
        errorSummary = errorInfo.errorSummary;
        errorCount = errorInfo.errorCount;
        console.log(`[processFileToArtifact v3] Found ${errorCount} errors in ${actualFilename}`);
      }
    }
    
    const artifact: Artifact = {
      id: `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      title: actualFilename,
      language,
      content,
      filename: actualFilename,
      created_at: new Date().toISOString(),
      size: content.length,  // Use decompressed size
      signatures,
      source: 'upload',
    };
    
    console.log(`[processFileToArtifact v3] Created artifact for ${actualFilename}: ${content.length} chars, ${signatures.length} signatures`);
    return { artifact, errorSummary, errorCount };
  } catch (error) {
    console.error(`[processFileToArtifact v3] Failed to process file ${file.name}:`, error);
    return { artifact: null };
  }
}

/**
 * Process multiple files into artifacts, collecting error summaries
 */
export async function processFilesToArtifacts(files: File[]): Promise<{
  artifacts: Artifact[];
  logErrors: Array<{ filename: string; errorSummary: string; errorCount: number }>;
}> {
  const artifacts: Artifact[] = [];
  const logErrors: Array<{ filename: string; errorSummary: string; errorCount: number }> = [];
  
  for (const file of files) {
    const result = await processFileToArtifact(file);
    if (result.artifact) {
      artifacts.push(result.artifact);
    }
    if (result.errorSummary && result.errorCount) {
      logErrors.push({
        filename: result.artifact?.filename || file.name,
        errorSummary: result.errorSummary,
        errorCount: result.errorCount,
      });
    }
  }
  
  return { artifacts, logErrors };
}

/**
 * Generate LLM context manifest for uploaded files
 */
export function generateFileManifest(artifacts: Artifact[]): string {
  if (artifacts.length === 0) return '';
  
  const lines: string[] = [
    '=== UPLOADED FILES ===',
    `Total files: ${artifacts.length}`,
    '',
  ];
  
  for (const artifact of artifacts) {
    const lineCount = artifact.content.split('\n').length;
    lines.push(`📄 ${artifact.filename} (${artifact.language}, ${lineCount} lines)`);
    
    if (artifact.signatures && artifact.signatures.length > 0) {
      lines.push('   Signatures:');
      for (const sig of artifact.signatures.slice(0, 10)) {
        lines.push(`   - ${sig.kind}: ${sig.name} (line ${sig.line})`);
      }
      if (artifact.signatures.length > 10) {
        lines.push(`   ... and ${artifact.signatures.length - 10} more`);
      }
    }
    lines.push('');
  }
  
  lines.push('Use view_file_lines or search_in_file tools to examine file contents.');
  lines.push('=== END FILES ===');
  
  return lines.join('\n');
}

/**
 * Get lines from content
 */
export function getFileLines(content: string, startLine: number, endLine?: number): string {
  const lines = content.split('\n');
  const start = Math.max(0, startLine - 1);
  const end = endLine ? Math.min(lines.length, endLine) : lines.length;
  
  return lines.slice(start, end)
    .map((line, i) => `${start + i + 1}: ${line}`)
    .join('\n');
}

/**
 * Search in file content
 */
export function searchInFile(content: string, pattern: string, contextLines: number = 2): string[] {
  const lines = content.split('\n');
  const results: string[] = [];
  const regex = new RegExp(pattern, 'gi');
  
  for (let i = 0; i < lines.length; i++) {
    if (regex.test(lines[i])) {
      const start = Math.max(0, i - contextLines);
      const end = Math.min(lines.length, i + contextLines + 1);
      const context = lines.slice(start, end)
        .map((line, j) => `${start + j + 1}${j + start === i ? '>' : ':'} ${line}`)
        .join('\n');
      results.push(context);
    }
  }
  
  return results;
}
