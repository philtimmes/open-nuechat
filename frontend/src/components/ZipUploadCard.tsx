import React, { useState, useMemo } from 'react';
import type { ZipUploadResult, CodeSignature } from '../types';

interface ZipUploadCardProps {
  result: Partial<ZipUploadResult>;
  onFileClick?: (path: string) => void;
}

type TreeNode = {
  _type?: 'file' | 'directory';
  _path?: string;
  _size?: number;
  _language?: string;
  [key: string]: TreeNode | string | number | undefined;
};

export default function ZipUploadCard({ result, onFileClick }: ZipUploadCardProps) {
  const [activeTab, setActiveTab] = useState<'files' | 'signatures' | 'summary'>('files');
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());

  // Destructure with safe defaults for partial data (e.g., when restoring from backend)
  const {
    filename = 'Uploaded Files',
    total_files = result.artifacts?.length || 0,
    total_size = 0,
    languages = {},
    file_tree: providedFileTree = {},
    signature_index = {},
    summary: providedSummary = '',
    artifacts = [],
  } = result;

  // Rebuild file_tree from artifacts if not provided
  const file_tree = Object.keys(providedFileTree).length > 0 
    ? providedFileTree 
    : buildFileTreeFromArtifacts(artifacts);

  // Generate summary from artifacts if not provided (backwards compatibility)
  const summary = providedSummary || generateSummaryFromArtifacts(artifacts, signature_index, languages, total_files, total_size);

  // Generate a summary similar to backend's format_signature_summary
  function generateSummaryFromArtifacts(
    arts: typeof artifacts,
    sigIndex: Record<string, CodeSignature[]>,
    langs: Record<string, number>,
    fileCount: number,
    totalBytes: number
  ): string {
    if (arts.length === 0) return '';
    
    const lines: string[] = [
      '# Zip Archive Summary',
      '',
      `**Files:** ${fileCount || arts.length}`,
      `**Total Size:** ${totalBytes.toLocaleString()} bytes`,
    ];
    
    // Languages
    const langEntries = Object.entries(langs).sort((a, b) => b[1] - a[1]);
    if (langEntries.length > 0) {
      lines.push(`**Languages:** ${langEntries.map(([l, c]) => `${l} (${c})`).join(', ')}`);
    } else {
      // Infer languages from artifacts
      const inferredLangs: Record<string, number> = {};
      for (const art of arts) {
        if (art.language) {
          inferredLangs[art.language] = (inferredLangs[art.language] || 0) + 1;
        }
      }
      const inferredEntries = Object.entries(inferredLangs).sort((a, b) => b[1] - a[1]);
      if (inferredEntries.length > 0) {
        lines.push(`**Languages:** ${inferredEntries.map(([l, c]) => `${l} (${c})`).join(', ')}`);
      }
    }
    
    lines.push('', '## File Structure', '```');
    
    // Format file tree
    function formatTree(node: TreeNode, prefix: string = ''): string[] {
      const result: string[] = [];
      const entries = Object.entries(node).filter(([k]) => !k.startsWith('_'));
      const dirs = entries.filter(([, v]) => (v as TreeNode)?._type === 'directory');
      const files = entries.filter(([, v]) => (v as TreeNode)?._type === 'file');
      
      for (const [name, child] of [...dirs.sort(), ...files.sort()]) {
        const childNode = child as TreeNode;
        if (childNode._type === 'file') {
          const size = childNode._size || 0;
          const lang = childNode._language || '';
          result.push(`${prefix}${name} (${size.toLocaleString()} bytes${lang ? `, ${lang}` : ''})`);
        } else {
          result.push(`${prefix}${name}/`);
          result.push(...formatTree(childNode, prefix + '  '));
        }
      }
      return result;
    }
    
    lines.push(...formatTree(file_tree as TreeNode));
    lines.push('```', '');
    
    // Code signatures
    const sigEntries = Object.entries(sigIndex).filter(([, sigs]) => sigs && sigs.length > 0);
    if (sigEntries.length > 0) {
      lines.push('## Code Signatures', '');
      
      for (const [filepath, sigs] of sigEntries.sort()) {
        lines.push(`### ${filepath}`);
        for (const sig of sigs) {
          const signature = sig.signature || sig.name;
          lines.push(`- **${sig.kind}** \`${signature}\` (line ${sig.line})`);
        }
        lines.push('');
      }
    }
    
    return lines.join('\n');
  }

  // Build a file tree structure from a flat list of artifacts
  function buildFileTreeFromArtifacts(arts: typeof artifacts): TreeNode {
    const tree: TreeNode = {};
    
    for (const artifact of arts) {
      const filepath = artifact.filename || artifact.title;
      if (!filepath) continue;
      
      const parts = filepath.split('/');
      let current = tree;
      
      // Compute size from content if not provided
      const artifactSize = artifact.size || (artifact.content?.length || 0);
      
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        const isLast = i === parts.length - 1;
        
        if (isLast) {
          // File node
          current[part] = {
            _type: 'file',
            _path: filepath,
            _size: artifactSize,
            _language: artifact.language,
          };
        } else {
          // Directory node
          if (!current[part]) {
            current[part] = { _type: 'directory' };
          }
          current = current[part] as TreeNode;
        }
      }
    }
    
    return tree;
  }

  const toggleDir = (path: string) => {
    setExpandedDirs(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const renderTree = (node: TreeNode, path: string = '', depth: number = 0): React.ReactNode[] => {
    const items: React.ReactNode[] = [];
    
    // Get entries, filtering out metadata keys
    const entries = Object.entries(node).filter(([key]) => !key.startsWith('_'));
    
    // Sort: directories first, then files
    const dirs = entries.filter(([, v]) => (v as TreeNode)?._type === 'directory');
    const files = entries.filter(([, v]) => (v as TreeNode)?._type === 'file');
    
    for (const [name, value] of [...dirs.sort(), ...files.sort()]) {
      const childNode = value as TreeNode;
      const fullPath = path ? `${path}/${name}` : name;
      const isDir = childNode._type === 'directory';
      const isExpanded = expandedDirs.has(fullPath);
      
      if (isDir) {
        items.push(
          <div key={fullPath}>
            <div
              className="flex items-center gap-2 py-1 px-2 hover:bg-[var(--color-surface-hover)] rounded cursor-pointer"
              style={{ paddingLeft: `${depth * 16 + 8}px` }}
              onClick={() => toggleDir(fullPath)}
            >
              <svg 
                className={`w-4 h-4 text-[var(--color-text-secondary)] transition-transform ${isExpanded ? 'rotate-90' : ''}`} 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <svg className="w-4 h-4 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
              </svg>
              <span className="text-sm text-[var(--color-text)]">{name}</span>
            </div>
            {isExpanded && renderTree(childNode, fullPath, depth + 1)}
          </div>
        );
      } else {
        const lang = childNode._language || '';
        const size = childNode._size || 0;
        items.push(
          <div
            key={fullPath}
            className="flex items-center gap-2 py-1 px-2 hover:bg-[var(--color-surface-hover)] rounded cursor-pointer"
            style={{ paddingLeft: `${depth * 16 + 24}px` }}
            onClick={() => onFileClick?.(childNode._path || fullPath)}
          >
            <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="text-sm text-[var(--color-text)] flex-1">{name}</span>
            {lang && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--color-button)]/80 text-[var(--color-button-text)]">
                {lang}
              </span>
            )}
            <span className="text-xs text-[var(--color-text-secondary)]">{formatSize(size)}</span>
          </div>
        );
      }
    }
    
    return items;
  };

  const renderSignatures = () => {
    const entries = Object.entries(signature_index);
    if (entries.length === 0) {
      return (
        <div className="text-center text-[var(--color-text-secondary)] py-4">
          No code signatures found
        </div>
      );
    }

    return entries.map(([filepath, signatures]) => (
      <div key={filepath} className="mb-4">
        <div 
          className="flex items-center gap-2 text-sm font-medium text-[var(--color-text)] mb-2 cursor-pointer hover:text-[var(--color-primary)]"
          onClick={() => onFileClick?.(filepath)}
        >
          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {filepath}
        </div>
        <div className="space-y-1 ml-4">
          {(signatures as CodeSignature[]).map((sig, idx) => (
            <div key={idx} className="flex items-center gap-2 text-sm">
              <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                sig.kind === 'function' || sig.kind === 'method' ? 'bg-blue-500/20 text-blue-400' :
                sig.kind === 'class' || sig.kind === 'struct' ? 'bg-purple-500/20 text-purple-400' :
                sig.kind === 'interface' || sig.kind === 'type' ? 'bg-green-500/20 text-green-400' :
                'bg-gray-500/20 text-gray-400'
              }`}>
                {sig.kind}
              </span>
              <span className="font-mono text-[var(--color-text)]">{sig.name}</span>
              <span className="text-xs text-[var(--color-text-secondary)]">line {sig.line}</span>
            </div>
          ))}
        </div>
      </div>
    ));
  };

  // Count total signatures
  const totalSignatures = Object.values(signature_index).reduce(
    (sum, sigs) => sum + (sigs as CodeSignature[]).length, 0
  );

  return (
    <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl overflow-hidden max-w-2xl">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-background)]">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[var(--color-button)]/80">
            <svg className="w-5 h-5 text-[var(--color-button-text)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
            </svg>
          </div>
          <div className="flex-1">
            <div className="font-medium text-[var(--color-text)]">{filename}</div>
            <div className="text-sm text-[var(--color-text-secondary)]">
              {total_files} files • {formatSize(total_size)} • {totalSignatures} signatures
            </div>
          </div>
        </div>
        
        {/* Language breakdown */}
        {Object.keys(languages).length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {Object.entries(languages)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 6)
              .map(([lang, count]) => (
                <span 
                  key={lang}
                  className="text-xs px-2 py-1 rounded-full bg-[var(--color-surface)] text-[var(--color-text-secondary)]"
                >
                  {lang}: {count}
                </span>
              ))}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[var(--color-border)]">
        <button
          onClick={() => setActiveTab('files')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'files'
              ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
          }`}
        >
          Files ({total_files})
        </button>
        <button
          onClick={() => setActiveTab('signatures')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'signatures'
              ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
          }`}
        >
          Signatures ({totalSignatures})
        </button>
        <button
          onClick={() => setActiveTab('summary')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'summary'
              ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
          }`}
        >
          Summary
        </button>
      </div>

      {/* Content */}
      <div className="max-h-80 overflow-auto p-2">
        {activeTab === 'files' && (
          <div className="space-y-0.5">
            {Object.keys(file_tree).length > 0 ? (
              renderTree(file_tree as TreeNode)
            ) : (
              <div className="text-center text-[var(--color-text-secondary)] py-4">
                No files available
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'signatures' && (
          <div className="p-2">
            {renderSignatures()}
          </div>
        )}
        
        {activeTab === 'summary' && (
          <div className="p-2">
            {summary ? (
              <pre className="text-xs text-[var(--color-text)] whitespace-pre-wrap font-mono leading-relaxed">
                {summary}
              </pre>
            ) : (
              <div className="text-center text-[var(--color-text-secondary)] py-4">
                {artifacts.length > 0 
                  ? `${artifacts.length} files loaded from previous session`
                  : 'No summary available'
                }
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
