import { useState } from 'react';
import type { CodeSummary, FileChange, SignatureWarning, ZipUploadResult } from '../types';
import ZipUploadCard from './ZipUploadCard';

interface SummaryPanelProps {
  summary: CodeSummary | null;
  zipUploadResult?: Partial<ZipUploadResult> | null;
  onClose: () => void;
  onClearWarnings?: () => void;
  onFileClick?: (path: string) => void;
}

export default function SummaryPanel({ summary, zipUploadResult, onClose, onClearWarnings, onFileClick }: SummaryPanelProps) {
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<'uploads' | 'files' | 'warnings'>(zipUploadResult ? 'uploads' : 'files');
  
  const toggleFile = (path: string) => {
    const newExpanded = new Set(expandedFiles);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedFiles(newExpanded);
  };
  
  const getActionColor = (action: FileChange['action']) => {
    switch (action) {
      case 'created': return 'text-green-400 bg-green-500/10';
      case 'modified': return 'text-yellow-400 bg-yellow-500/10';
      case 'deleted': return 'text-red-400 bg-red-500/10';
      default: return 'text-[var(--color-text-secondary)] bg-[var(--color-surface)]';
    }
  };
  
  const getWarningColor = (type: SignatureWarning['type']) => {
    switch (type) {
      case 'missing': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
      case 'mismatch': return 'text-red-400 bg-red-500/10 border-red-500/30';
      case 'orphan': return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
      case 'library_not_found': return 'text-purple-400 bg-purple-500/10 border-purple-500/30';
      default: return 'text-[var(--color-text-secondary)] bg-[var(--color-surface)] border-[var(--color-border)]';
    }
  };
  
  const getWarningIcon = (type: SignatureWarning['type']) => {
    switch (type) {
      case 'missing': return '⚠️';
      case 'mismatch': return '❌';
      case 'orphan': return '🔍';
      case 'library_not_found': return '📦';
      default: return '❓';
    }
  };
  
  const warningCount = summary?.warnings?.length || 0;
  const fileCount = summary?.files?.length || 0;
  const uploadCount = zipUploadResult?.total_files || zipUploadResult?.artifacts?.length || 0;
  
  return (
    <div className="w-96 bg-[var(--color-surface)] border-l border-[var(--color-border)] flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-[var(--color-border)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">📋</span>
          <h2 className="font-semibold text-[var(--color-text)]">Project Summary</h2>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      {/* Tabs */}
      <div className="flex border-b border-[var(--color-border)]">
        {uploadCount > 0 && (
          <button
            onClick={() => setActiveTab('uploads')}
            className={`flex-1 px-3 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-1 ${
              activeTab === 'uploads'
                ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            Uploads
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-[var(--color-button)]/80 text-[var(--color-button-text)]">
              {uploadCount}
            </span>
          </button>
        )}
        <button
          onClick={() => setActiveTab('files')}
          className={`flex-1 px-3 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-1 ${
            activeTab === 'files'
              ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
          }`}
        >
          Changes
          {fileCount > 0 && (
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-[var(--color-button)]/80 text-[var(--color-button-text)]">
              {fileCount}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('warnings')}
          className={`flex-1 px-3 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-1 ${
            activeTab === 'warnings'
              ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
          }`}
        >
          Warnings
          {warningCount > 0 && (
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-red-500/20 text-red-400">
              {warningCount}
            </span>
          )}
        </button>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'uploads' && zipUploadResult ? (
          <ZipUploadCard result={zipUploadResult} onFileClick={onFileClick} />
        ) : !summary ? (
          <div className="text-center py-8 text-[var(--color-text-secondary)]">
            <svg className="w-8 h-8 mb-2 mx-auto text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-sm">No code changes tracked yet</p>
            <p className="text-xs mt-1">Code summaries will appear here as files are created or modified</p>
          </div>
        ) : activeTab === 'files' ? (
          <div className="space-y-2">
            {summary.files.length === 0 ? (
              <div className="text-center py-8 text-[var(--color-text-secondary)]">
                <p className="text-sm">No files tracked</p>
              </div>
            ) : (
              summary.files.map((file) => (
                <div
                  key={file.path}
                  className="rounded-lg border border-[var(--color-border)] overflow-hidden"
                >
                  <button
                    onClick={() => toggleFile(file.path)}
                    className="w-full px-3 py-2 flex items-center justify-between hover:bg-[var(--color-background)] transition-colors"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${getActionColor(file.action)}`}>
                        {file.action}
                      </span>
                      <span className="text-sm text-[var(--color-text)] truncate">
                        {file.path}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {file.language && (
                        <span className="text-xs text-[var(--color-text-secondary)]">
                          {file.language}
                        </span>
                      )}
                      <svg
                        className={`w-4 h-4 text-[var(--color-text-secondary)] transition-transform ${
                          expandedFiles.has(file.path) ? 'rotate-180' : ''
                        }`}
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </button>
                  
                  {expandedFiles.has(file.path) && file.signatures.length > 0 && (
                    <div className="px-3 pb-3 pt-1 border-t border-[var(--color-border)] bg-[var(--color-background)]">
                      <p className="text-xs text-[var(--color-text-secondary)] mb-2">
                        Signatures ({file.signatures.length})
                      </p>
                      <div className="space-y-1">
                        {file.signatures.map((sig, idx) => (
                          <div
                            key={idx}
                            className="text-xs font-mono bg-[var(--color-surface)] rounded px-2 py-1 overflow-x-auto"
                          >
                            <span className="text-[var(--color-primary)]">{sig.type}</span>
                            <span className="text-[var(--color-text)]"> {sig.name}</span>
                            {sig.line && (
                              <span className="text-[var(--color-text-secondary)]"> :L{sig.line}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {summary.warnings.length === 0 ? (
              <div className="text-center py-8 text-[var(--color-text-secondary)]">
                <span className="text-3xl mb-2 block">✅</span>
                <p className="text-sm">No warnings</p>
                <p className="text-xs mt-1">All code appears consistent with documented signatures</p>
              </div>
            ) : (
              <>
                <div className="flex justify-end mb-2">
                  {onClearWarnings && (
                    <button
                      onClick={onClearWarnings}
                      className="text-xs px-2 py-1 text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-background)] rounded transition-colors"
                    >
                      Clear All
                    </button>
                  )}
                </div>
                {summary.warnings.map((warning, idx) => (
                  <div
                    key={idx}
                    className={`rounded-lg border p-3 ${getWarningColor(warning.type)}`}
                  >
                    <div className="flex items-start gap-2">
                      <span className="text-base flex-shrink-0">{getWarningIcon(warning.type)}</span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium">{warning.message}</p>
                        {warning.file && (
                          <p className="text-xs mt-1 opacity-80">
                            File: {warning.file}
                          </p>
                        )}
                        {warning.signature && (
                          <p className="text-xs mt-1 font-mono opacity-80 truncate">
                            {warning.signature}
                          </p>
                        )}
                        {warning.suggestion && (
                          <p className="text-xs mt-2 italic opacity-90 flex items-start gap-1">
                            <svg className="w-3 h-3 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                            {warning.suggestion}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </div>
      
      {/* Footer */}
      {summary && (
        <div className="p-3 border-t border-[var(--color-border)] text-xs text-[var(--color-text-secondary)]">
          Last updated: {new Date(summary.last_updated).toLocaleString()}
        </div>
      )}
    </div>
  );
}
