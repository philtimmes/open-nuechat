/**
 * Code signature extraction utilities
 * Extracts function/class/type signatures from code for tracking
 */

import type { CodeSignatureEntry, FileChange, SignatureWarning } from '../types';

// Language detection based on file extension
export function detectLanguage(filepath: string): string | undefined {
  const ext = filepath.split('.').pop()?.toLowerCase();
  const langMap: Record<string, string> = {
    'ts': 'typescript',
    'tsx': 'typescript',
    'js': 'javascript',
    'jsx': 'javascript',
    'py': 'python',
    'rs': 'rust',
    'go': 'go',
    'java': 'java',
    'rb': 'ruby',
    'php': 'php',
    'cs': 'csharp',
    'cpp': 'cpp',
    'c': 'c',
    'h': 'c',
    'hpp': 'cpp',
    'swift': 'swift',
    'kt': 'kotlin',
    'scala': 'scala',
    'sql': 'sql',
    'sh': 'bash',
    'bash': 'bash',
    'zsh': 'bash',
    'yaml': 'yaml',
    'yml': 'yaml',
    'json': 'json',
    'md': 'markdown',
    'html': 'html',
    'css': 'css',
    'scss': 'scss',
    'less': 'less',
  };
  return ext ? langMap[ext] : undefined;
}

// Extract signatures from TypeScript/JavaScript code
function extractTSSignatures(code: string, filepath: string): CodeSignatureEntry[] {
  const signatures: CodeSignatureEntry[] = [];
  const lines = code.split('\n');
  
  lines.forEach((line, idx) => {
    const lineNum = idx + 1;
    const trimmed = line.trim();
    
    // Functions: function name(...) or async function name(...)
    const funcMatch = trimmed.match(/^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(\([^)]*\))/);
    if (funcMatch) {
      signatures.push({
        name: funcMatch[1],
        type: 'function',
        signature: `function ${funcMatch[1]}${funcMatch[2]}`,
        file: filepath,
        line: lineNum,
      });
    }
    
    // Arrow functions: const name = (...) => or const name: Type = (...) =>
    const arrowMatch = trimmed.match(/^(?:export\s+)?(?:const|let|var)\s+(\w+)(?:\s*:\s*[^=]+)?\s*=\s*(?:async\s*)?\(([^)]*)\)\s*(?::\s*[^=]+)?\s*=>/);
    if (arrowMatch) {
      signatures.push({
        name: arrowMatch[1],
        type: 'function',
        signature: `const ${arrowMatch[1]} = (${arrowMatch[2]}) =>`,
        file: filepath,
        line: lineNum,
      });
    }
    
    // Classes
    const classMatch = trimmed.match(/^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?/);
    if (classMatch) {
      signatures.push({
        name: classMatch[1],
        type: 'class',
        signature: trimmed.replace(/\s*\{.*$/, ''),
        file: filepath,
        line: lineNum,
      });
    }
    
    // Interfaces
    const interfaceMatch = trimmed.match(/^(?:export\s+)?interface\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+[\w,\s<>]+)?/);
    if (interfaceMatch) {
      signatures.push({
        name: interfaceMatch[1],
        type: 'interface',
        signature: trimmed.replace(/\s*\{.*$/, ''),
        file: filepath,
        line: lineNum,
      });
    }
    
    // Type aliases
    const typeMatch = trimmed.match(/^(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*=/);
    if (typeMatch) {
      signatures.push({
        name: typeMatch[1],
        type: 'type',
        signature: trimmed.length > 80 ? trimmed.slice(0, 77) + '...' : trimmed,
        file: filepath,
        line: lineNum,
      });
    }
    
    // Methods in classes (simplified)
    const methodMatch = trimmed.match(/^(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:async\s+)?(\w+)\s*(\([^)]*\))(?:\s*:\s*[^{]+)?\s*\{/);
    if (methodMatch && !['if', 'for', 'while', 'switch', 'catch', 'function', 'class'].includes(methodMatch[1])) {
      // Only add if we're likely inside a class (heuristic)
      const prevLines = lines.slice(Math.max(0, idx - 20), idx).join('\n');
      if (prevLines.includes('class ')) {
        signatures.push({
          name: methodMatch[1],
          type: 'method',
          signature: `${methodMatch[1]}${methodMatch[2]}`,
          file: filepath,
          line: lineNum,
        });
      }
    }
  });
  
  return signatures;
}

// Extract signatures from Python code
function extractPythonSignatures(code: string, filepath: string): CodeSignatureEntry[] {
  const signatures: CodeSignatureEntry[] = [];
  const lines = code.split('\n');
  
  lines.forEach((line, idx) => {
    const lineNum = idx + 1;
    const trimmed = line.trim();
    
    // Functions and methods: def name(...):
    const funcMatch = trimmed.match(/^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)/);
    if (funcMatch) {
      const isMethod = funcMatch[2].startsWith('self') || funcMatch[2].startsWith('cls');
      signatures.push({
        name: funcMatch[1],
        type: isMethod ? 'method' : 'function',
        signature: `def ${funcMatch[1]}(${funcMatch[2]})`,
        file: filepath,
        line: lineNum,
      });
    }
    
    // Classes
    const classMatch = trimmed.match(/^class\s+(\w+)(?:\([^)]*\))?:/);
    if (classMatch) {
      signatures.push({
        name: classMatch[1],
        type: 'class',
        signature: trimmed.replace(/:$/, ''),
        file: filepath,
        line: lineNum,
      });
    }
    
    // FastAPI endpoints
    const endpointMatch = trimmed.match(/^@(?:router|app)\.(get|post|put|patch|delete)\s*\(\s*["']([^"']+)["']/);
    if (endpointMatch) {
      // Look at next line for the function name
      const nextLine = lines[idx + 1]?.trim() || '';
      const funcNameMatch = nextLine.match(/^(?:async\s+)?def\s+(\w+)/);
      if (funcNameMatch) {
        signatures.push({
          name: funcNameMatch[1],
          type: 'endpoint',
          signature: `${endpointMatch[1].toUpperCase()} ${endpointMatch[2]}`,
          file: filepath,
          line: lineNum + 1,
        });
      }
    }
  });
  
  return signatures;
}

// Main extraction function
export function extractSignatures(code: string, filepath: string): CodeSignatureEntry[] {
  const lang = detectLanguage(filepath);
  
  switch (lang) {
    case 'typescript':
    case 'javascript':
      return extractTSSignatures(code, filepath);
    case 'python':
      return extractPythonSignatures(code, filepath);
    // Add more languages as needed
    default:
      return [];
  }
}

// Create a FileChange from an artifact
export function createFileChangeFromCode(
  filepath: string,
  code: string,
  action: 'created' | 'modified' | 'deleted' = 'created'
): FileChange {
  const language = detectLanguage(filepath);
  const signatures = extractSignatures(code, filepath);
  
  return {
    path: filepath,
    action,
    language,
    signatures,
    timestamp: new Date().toISOString(),
  };
}

// Check for common library imports and verify they're documented
export function detectLibraryImports(code: string, filepath: string): string[] {
  const imports: string[] = [];
  const lang = detectLanguage(filepath);
  
  if (lang === 'typescript' || lang === 'javascript') {
    // Match import statements
    const importMatches = code.matchAll(/import\s+(?:{[^}]+}|[\w*]+)\s+from\s+['"]([^'"]+)['"]/g);
    for (const match of importMatches) {
      const pkg = match[1];
      // Only track external packages (not relative imports)
      if (!pkg.startsWith('.') && !pkg.startsWith('@/')) {
        imports.push(pkg);
      }
    }
    
    // Match require statements
    const requireMatches = code.matchAll(/require\s*\(\s*['"]([^'"]+)['"]\s*\)/g);
    for (const match of requireMatches) {
      const pkg = match[1];
      if (!pkg.startsWith('.')) {
        imports.push(pkg);
      }
    }
  } else if (lang === 'python') {
    // Match import statements
    const importMatches = code.matchAll(/^(?:from\s+(\S+)\s+)?import\s+(\S+)/gm);
    for (const match of importMatches) {
      const pkg = match[1] || match[2];
      // Only track external packages (not relative imports)
      if (!pkg.startsWith('.')) {
        imports.push(pkg.split('.')[0]);
      }
    }
  }
  
  return [...new Set(imports)]; // Deduplicate
}

// Generate warnings for signature inconsistencies
export function generateSignatureWarnings(
  currentFiles: FileChange[],
  newFile: FileChange,
  expectedSignatures?: CodeSignatureEntry[]
): SignatureWarning[] {
  const warnings: SignatureWarning[] = [];
  
  // Check if signatures in new file match expected signatures (if provided)
  if (expectedSignatures) {
    for (const expected of expectedSignatures) {
      const found = newFile.signatures.find(s => s.name === expected.name);
      if (!found) {
        warnings.push({
          type: 'missing',
          message: `Expected signature "${expected.name}" not found in ${newFile.path}`,
          file: newFile.path,
          signature: expected.signature,
          suggestion: `Add ${expected.type} ${expected.name} to match the documented interface`,
        });
      } else if (found.type !== expected.type) {
        warnings.push({
          type: 'mismatch',
          message: `Type mismatch: "${found.name}" is a ${found.type} but expected ${expected.type}`,
          file: newFile.path,
          signature: found.signature,
          suggestion: `Verify the implementation matches the expected type`,
        });
      }
    }
  }
  
  // Check for orphaned references (functions called but not defined)
  // This is a simplified check - in practice you'd want more sophisticated analysis
  
  return warnings;
}
