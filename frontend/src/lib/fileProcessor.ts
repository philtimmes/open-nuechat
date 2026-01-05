/**
 * File processor for converting uploaded files to artifacts
 * Extracts code signatures and prepares files for LLM context
 */

import type { Artifact, CodeSignature } from '../types';

// Binary document types that need backend extraction
const BINARY_EXTENSIONS = new Set(['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.rtf']);

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
 * Process a file into an artifact
 */
export async function processFileToArtifact(file: File): Promise<Artifact | null> {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  const extInfo = EXT_MAP[ext];
  
  console.log(`[processFileToArtifact v2] Processing ${file.name}, ext=${ext}, type=${file.type}, isBinary=${BINARY_EXTENSIONS.has(ext)}`);
  
  // Handle images separately (they're attachments, not artifacts for now)
  if (file.type.startsWith('image/')) {
    console.log(`[processFileToArtifact v2] Skipping image: ${file.name}`);
    return null; // Images handled separately as attachments
  }
  
  try {
    let content: string;
    
    // Binary documents need backend extraction
    if (BINARY_EXTENSIONS.has(ext)) {
      console.log(`[processFileToArtifact v2] Binary file detected, calling extract API for ${file.name}`);
      content = await extractTextFromDocument(file);
      if (!content) {
        console.warn(`[processFileToArtifact v2] No text extracted from ${file.name}`);
        return null;
      }
      console.log(`[processFileToArtifact v2] Extracted ${content.length} chars from ${file.name}`);
    } else {
      // Text files can be read directly
      console.log(`[processFileToArtifact v2] Reading text file ${file.name}`);
      content = await readFileAsText(file);
    }
    
    const language = extInfo?.language || 'text';
    const type = extInfo?.type || 'document';
    
    // Extract signatures for code files
    const signatures = type === 'code' || type === 'react' 
      ? extractSignatures(content, language)
      : [];
    
    const artifact: Artifact = {
      id: `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      title: file.name,
      language,
      content,
      filename: file.name,
      created_at: new Date().toISOString(),
      size: file.size,
      signatures,
      source: 'upload',
    };
    
    console.log(`[processFileToArtifact v2] Created artifact for ${file.name}: ${content.length} chars, ${signatures.length} signatures`);
    return artifact;
  } catch (error) {
    console.error(`[processFileToArtifact v2] Failed to process file ${file.name}:`, error);
    return null;
  }
}

/**
 * Process multiple files into artifacts
 */
export async function processFilesToArtifacts(files: File[]): Promise<Artifact[]> {
  const artifacts: Artifact[] = [];
  
  for (const file of files) {
    const artifact = await processFileToArtifact(file);
    if (artifact) {
      artifacts.push(artifact);
    }
  }
  
  return artifacts;
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
