import { useState, useEffect, useCallback } from 'react';
import api from '../lib/api';
import type { Document, SearchResult, KnowledgeStore } from '../types';

type Tab = 'documents' | 'knowledge-bases';

// Allowed file extensions for upload
const ALLOWED_FILE_EXTENSIONS = [
  // Documents
  '.pdf', '.txt', '.md', '.json', '.csv',
  // Office documents
  '.docx', '.doc', '.xlsx', '.xls', '.rtf',
  // Python
  '.py', '.pyi',
  // JavaScript/TypeScript
  '.js', '.jsx', '.ts', '.tsx',
  // C/C++
  '.c', '.h', '.cc', '.cpp', '.hpp',
  // Java
  '.java',
  // Rust
  '.rs',
  // Go
  '.go',
  // Ruby
  '.rb',
  // Web
  '.html', '.css', '.scss',
  // Config
  '.yaml', '.yml', '.xml', '.toml', '.ini',
  // Shell
  '.sh',
  // Other
  '.sql', '.swift', '.kt', '.scala', '.php',
].join(',');

export default function Documents() {
  const [activeTab, setActiveTab] = useState<Tab>('documents');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [knowledgeStores, setKnowledgeStores] = useState<KnowledgeStore[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Knowledge store form
  const [showKSForm, setShowKSForm] = useState(false);
  const [ksName, setKsName] = useState('');
  const [ksDescription, setKsDescription] = useState('');
  const [ksIcon, setKsIcon] = useState('KB');
  const [ksColor, setKsColor] = useState('#6366f1');
  const [ksIsPublic, setKsIsPublic] = useState(false);
  const [editingKS, setEditingKS] = useState<KnowledgeStore | null>(null);
  
  // NC-0.8.0.1.1: Document keyword editing
  const [editingDocKeywords, setEditingDocKeywords] = useState<Document | null>(null);
  const [docKeywordsEnabled, setDocKeywordsEnabled] = useState(false);
  const [docKeywordsText, setDocKeywordsText] = useState('');
  const [docKeywordsMode, setDocKeywordsMode] = useState<'any' | 'all' | 'mixed'>('any');
  
  // Expanded knowledge base in documents view
  const [expandedKS, setExpandedKS] = useState<string | null>(null);
  
  useEffect(() => {
    fetchData();
  }, []);
  
  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [docsRes, ksRes] = await Promise.all([
        api.get('/documents'),
        api.get('/knowledge-stores').catch(() => ({ data: [] })),
      ]);
      setDocuments(docsRes.data || []);
      setKnowledgeStores(ksRes.data || []);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setError('Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleUpload = async (files: FileList | null, knowledgeStoreId?: string) => {
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    setError(null);
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('file', file);
        
        // Use KB-specific endpoint when uploading to a knowledge base
        const endpoint = knowledgeStoreId 
          ? `/knowledge-stores/${knowledgeStoreId}/documents`
          : '/documents';
        
        await api.post(endpoint, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const progress = progressEvent.total
              ? Math.round((progressEvent.loaded / progressEvent.total) * 100)
              : 0;
            setUploadProgress(((i / files.length) * 100) + (progress / files.length));
          },
        });
      }
      
      await fetchData();
    } catch (err: unknown) {
      console.error('Upload failed:', err);
      const error = err as { response?: { data?: { detail?: string | Array<{ msg?: string; loc?: string[] }> } } };
      const detail = error.response?.data?.detail;
      
      if (Array.isArray(detail)) {
        const messages = detail.map(e => e.msg || 'Error').join(', ');
        setError(messages || 'Upload failed');
      } else if (typeof detail === 'string') {
        setError(detail);
      } else {
        setError('Upload failed');
      }
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };
  
  const handleDelete = async (docId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
      await api.delete(`/documents/${docId}`);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
    } catch (err) {
      console.error('Delete failed:', err);
      setError('Failed to delete document');
    }
  };
  
  const handleReprocess = async (docId: string) => {
    try {
      await api.post(`/documents/${docId}/reprocess`);
      await fetchData();
    } catch (err) {
      console.error('Reprocess failed:', err);
      setError('Failed to reprocess document');
    }
  };
  
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    
    setIsSearching(true);
    try {
      const response = await api.post('/documents/search', {
        query: searchQuery,
        top_k: 5,
      });
      setSearchResults(response.data || []);
    } catch (err) {
      console.error('Search failed:', err);
      setError('Search failed');
    } finally {
      setIsSearching(false);
    }
  };
  
  // Knowledge Store CRUD
  const handleCreateKS = async () => {
    if (!ksName.trim()) return;
    
    try {
      const data = {
        name: ksName,
        description: ksDescription || undefined,
        icon: ksIcon,
        color: ksColor,
        is_public: ksIsPublic,
      };
      
      if (editingKS) {
        await api.patch(`/knowledge-stores/${editingKS.id}`, data);
      } else {
        await api.post('/knowledge-stores', data);
      }
      
      await fetchData();
      resetKSForm();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string | Array<{ msg?: string; loc?: string[] }> } } };
      const detail = error.response?.data?.detail;
      
      if (Array.isArray(detail)) {
        const messages = detail.map(e => {
          const field = e.loc?.slice(-1)[0] || 'field';
          return `${field}: ${e.msg}`;
        });
        setError(messages.join(', ') || 'Validation failed');
      } else if (typeof detail === 'string') {
        setError(detail);
      } else {
        setError('Failed to save knowledge base');
      }
    }
  };
  
  const handleDeleteKS = async (ksId: string) => {
    if (!confirm('Delete this knowledge base and all its documents?')) return;
    
    try {
      await api.delete(`/knowledge-stores/${ksId}`);
      await fetchData();
    } catch (err) {
      setError('Failed to delete knowledge base');
    }
  };
  
  const editKS = (ks: KnowledgeStore) => {
    setEditingKS(ks);
    setKsName(ks.name);
    setKsDescription(ks.description || '');
    setKsIcon(ks.icon);
    setKsColor(ks.color);
    setKsIsPublic(ks.is_public);
    setShowKSForm(true);
  };
  
  const resetKSForm = () => {
    setShowKSForm(false);
    setEditingKS(null);
    setKsName('');
    setKsDescription('');
    setKsIcon('KB');
    setKsColor('#6366f1');
    setKsIsPublic(false);
  };
  
  // NC-0.8.0.1.1: Document keyword editing functions
  const openDocKeywordsEditor = (doc: Document) => {
    setEditingDocKeywords(doc);
    setDocKeywordsEnabled(doc.require_keywords_enabled || false);
    setDocKeywordsText(doc.required_keywords || '');
    setDocKeywordsMode(doc.keyword_match_mode || 'any');
  };
  
  const saveDocKeywords = async () => {
    if (!editingDocKeywords) return;
    
    try {
      await api.patch(
        `/knowledge-stores/${editingDocKeywords.knowledge_store_id}/documents/${editingDocKeywords.id}/keywords`,
        {
          require_keywords_enabled: docKeywordsEnabled,
          required_keywords: docKeywordsText,
          keyword_match_mode: docKeywordsMode,
        }
      );
      await fetchData();
      setEditingDocKeywords(null);
    } catch (err) {
      setError('Failed to save document keywords');
    }
  };
  
  // Drag and drop handlers
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleUpload(e.dataTransfer.files);
  }, []);
  
  const getFileIcon = (fileType: string | undefined | null) => {
    const type = fileType || '';
    if (type.includes('pdf')) {
      return (
        <svg className="w-8 h-8 text-red-500" fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 2l5 5h-5V4z" />
        </svg>
      );
    }
    if (type.includes('wordprocessingml') || type.includes('msword')) {
      // Word document (.docx, .doc)
      return (
        <svg className="w-8 h-8 text-blue-600" fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 2l5 5h-5V4zM9 13h6v1H9v-1zm0 2h6v1H9v-1zm0 2h4v1H9v-1z" />
        </svg>
      );
    }
    if (type.includes('spreadsheetml') || type.includes('ms-excel')) {
      // Excel spreadsheet (.xlsx, .xls)
      return (
        <svg className="w-8 h-8 text-green-600" fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 2l5 5h-5V4zM8 12h3v2H8v-2zm4 0h4v2h-4v-2zm-4 3h3v2H8v-2zm4 0h4v2h-4v-2z" />
        </svg>
      );
    }
    if (type.includes('rtf')) {
      // RTF document
      return (
        <svg className="w-8 h-8 text-purple-500" fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 2l5 5h-5V4zM9 13h6v1H9v-1zm0 2h6v1H9v-1z" />
        </svg>
      );
    }
    if (type.includes('json')) {
      return (
        <svg className="w-8 h-8 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a2 2 0 01-2-2V6a2 2 0 012-2h8z" />
        </svg>
      );
    }
    if (type.includes('csv') || type.includes('spreadsheet')) {
      return (
        <svg className="w-8 h-8 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      );
    }
    return (
      <svg className="w-8 h-8 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  };
  
  const formatFileSize = (bytes: number | undefined | null) => {
    if (!bytes) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };
  
  // Icon options for knowledge bases - using simple shapes/letters instead of emojis
  const iconOptions = ['KB', 'DB', 'LIB', 'DOC', 'REF', 'SRC', 'LAB', 'BIZ', 'EDU', 'SYS'];
  
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-[var(--color-text)]">Documents & Knowledge</h1>
          <div className="flex gap-2">
            <label className="flex items-center gap-2 px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 cursor-pointer transition-opacity">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Upload
              <input
                type="file"
                multiple
                onChange={(e) => handleUpload(e.target.files)}
                className="hidden"
                accept={ALLOWED_FILE_EXTENSIONS}
              />
            </label>
          </div>
        </div>
        
        {/* Tabs */}
        <div className="flex gap-1 mb-6 border-b border-[var(--color-border)]">
          <button
            onClick={() => setActiveTab('documents')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === 'documents'
                ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Documents ({documents.length})
          </button>
          <button
            onClick={() => setActiveTab('knowledge-bases')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === 'knowledge-bases'
                ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Knowledge Bases ({knowledgeStores.length})
          </button>
        </div>
        
        {/* Error */}
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-[var(--color-error)]/10 border border-[var(--color-error)]/30 text-[var(--color-error)] text-sm flex justify-between items-center">
            {error}
            <button onClick={() => setError(null)} className="text-[var(--color-error)] hover:opacity-70">✕</button>
          </div>
        )}
        
        {/* Upload progress */}
        {isUploading && (
          <div className="mb-4 p-4 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
            <div className="flex justify-between mb-2">
              <span className="text-sm text-[var(--color-text)]">Uploading...</span>
              <span className="text-sm text-[var(--color-text-secondary)]">{Math.round(uploadProgress)}%</span>
            </div>
            <div className="h-2 bg-[var(--color-background)] rounded-full overflow-hidden">
              <div className="h-full bg-[var(--color-primary)] transition-all" style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        )}
        
        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <>
            {/* Drop zone */}
            <div
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className={`mb-6 border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                isDragging
                  ? 'border-[var(--color-button)] bg-[var(--color-button)]/10'
                  : 'border-[var(--color-border)] hover:border-[var(--color-text-secondary)]'
              }`}
            >
              <svg className="w-12 h-12 mx-auto text-[var(--color-text-secondary)] mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-[var(--color-text)] font-medium">
                {isDragging ? 'Drop files here' : 'Drag and drop files here'}
              </p>
              <p className="text-[var(--color-text-secondary)] text-sm mt-1">or click Upload above</p>
              <p className="text-zinc-500/60 text-xs mt-3">PDF, TXT, MD, JSON, CSV (max 50MB)</p>
            </div>
            
            {/* Search */}
            <div className="mb-6">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Search across all documents..."
                    className="w-full pl-10 pr-4 py-3 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                  />
                </div>
                <button
                  onClick={handleSearch}
                  disabled={isSearching || !searchQuery.trim()}
                  className="px-6 py-3 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                  {isSearching ? 'Searching...' : 'Search'}
                </button>
              </div>
            </div>
            
            {/* Search results */}
            {searchResults.length > 0 && (
              <div className="mb-8">
                <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Search Results</h2>
                <div className="space-y-3">
                  {searchResults.map((result, index) => (
                    <div key={index} className="bg-[var(--color-surface)] rounded-xl p-4 border border-[var(--color-border)]">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-[var(--color-text)]">{result.document_name}</span>
                        <span className="text-xs px-2 py-1 rounded-full bg-[var(--color-button)]/80 text-[var(--color-button-text)]">
                          {(result.similarity * 100).toFixed(1)}% match
                        </span>
                      </div>
                      <p className="text-sm text-[var(--color-text-secondary)] line-clamp-3">{result.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Documents list - grouped by knowledge base */}
            <div>
              <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Your Documents ({documents.length})</h2>
              
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <svg className="animate-spin h-8 w-8 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-12 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
                  <svg className="w-16 h-16 mx-auto text-zinc-500/40 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-[var(--color-text-secondary)]">No documents yet</p>
                  <p className="text-zinc-500/60 text-sm mt-1">Upload files to enable AI-powered search</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Documents in Knowledge Bases */}
                  {knowledgeStores.filter(ks => documents.some(d => d.knowledge_store_id === ks.id)).map((ks) => {
                    const ksDocs = documents.filter(d => d.knowledge_store_id === ks.id);
                    const isExpanded = expandedKS === ks.id;
                    
                    return (
                      <div key={ks.id} className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] overflow-hidden">
                        {/* Knowledge Base Header */}
                        <button
                          onClick={() => setExpandedKS(isExpanded ? null : ks.id)}
                          className="w-full flex items-center gap-3 p-4 hover:bg-[var(--color-background)]/50 transition-colors"
                        >
                          <div
                            className="w-10 h-10 rounded-lg flex items-center justify-center text-xl flex-shrink-0"
                            style={{ backgroundColor: `${ks.color}20` }}
                          >
                            {ks.icon}
                          </div>
                          <div className="flex-1 text-left">
                            <div className="flex items-center gap-2">
                              <h3 className="font-medium text-[var(--color-text)]">{ks.name}</h3>
                              {ks.is_global && (
                                <span className="px-1.5 py-0.5 text-xs rounded bg-purple-500/10 text-purple-400" title="Auto-searched on every query">Global</span>
                              )}
                              {ks.is_public ? (
                                <span className="px-1.5 py-0.5 text-xs rounded bg-[var(--color-success)]/10 text-[var(--color-success)]">Public</span>
                              ) : (
                                <span className="px-1.5 py-0.5 text-xs rounded bg-[var(--color-text-secondary)]/10 text-[var(--color-text-secondary)]">Private</span>
                              )}
                            </div>
                            <p className="text-sm text-[var(--color-text-secondary)]">
                              {ksDocs.length} documents • {formatFileSize(ks.total_size_bytes)}
                            </p>
                          </div>
                          <svg className={`w-5 h-5 text-[var(--color-text-secondary)] transition-transform ${isExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </button>
                        
                        {/* Documents in this KB */}
                        {isExpanded && (
                          <div className="border-t border-[var(--color-border)] p-4 space-y-2">
                            {ksDocs.map((doc) => (
                              <div key={doc.id} className="flex items-center gap-3 p-3 rounded-lg bg-[var(--color-background)] hover:bg-[var(--color-border)]/30 transition-colors">
                                {getFileIcon(doc.file_type)}
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2">
                                    <h4 className="text-sm font-medium text-[var(--color-text)] truncate">{doc.name}</h4>
                                    {doc.require_keywords_enabled && (
                                      <span className="px-1.5 py-0.5 text-[10px] bg-[var(--color-primary)]/20 text-[var(--color-primary)] rounded" title="Has keyword filter">
                                        🔑
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                                    <span>{formatFileSize(doc.file_size)}</span>
                                    <span>•</span>
                                    <span>{doc.chunk_count || 0} chunks</span>
                                    <span>•</span>
                                    <div className="flex items-center gap-1">
                                      <div className={`w-1.5 h-1.5 rounded-full ${doc.is_processed ? 'bg-[var(--color-success)]' : 'bg-[var(--color-warning)]'}`} />
                                      <span>{doc.is_processed ? 'Ready' : 'Processing'}</span>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-1">
                                  {/* Keyword filter button */}
                                  {ks.is_global && (
                                    <button
                                      onClick={() => openDocKeywordsEditor(doc)}
                                      className={`p-1.5 rounded hover:bg-[var(--color-primary)]/10 ${doc.require_keywords_enabled ? 'text-[var(--color-primary)]' : 'text-[var(--color-text-secondary)]'}`}
                                      title="Keyword Filter (Global KB)"
                                    >
                                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                                      </svg>
                                    </button>
                                  )}
                                  {!doc.is_processed && (
                                    <button
                                      onClick={() => handleReprocess(doc.id)}
                                      className="p-1.5 rounded text-[var(--color-warning)] hover:bg-[var(--color-warning)]/10"
                                      title="Reprocess"
                                    >
                                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                      </svg>
                                    </button>
                                  )}
                                  <button
                                    onClick={() => handleDelete(doc.id)}
                                    className="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-error)] hover:bg-red-500/10"
                                    title="Delete"
                                  >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                  </button>
                                </div>
                              </div>
                            ))}
                            {/* Add more docs button */}
                            <label className="flex items-center justify-center gap-2 p-3 rounded-lg border-2 border-dashed border-[var(--color-border)] hover:border-[var(--color-primary)] cursor-pointer transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-primary)]">
                              <input
                                type="file"
                                multiple
                                onChange={(e) => handleUpload(e.target.files, ks.id)}
                                className="hidden"
                                accept={ALLOWED_FILE_EXTENSIONS}
                              />
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                              </svg>
                              <span className="text-sm">Add more documents</span>
                            </label>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  
                  {/* Unassigned Documents */}
                  {documents.filter(d => !d.knowledge_store_id).length > 0 && (
                    <div className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] overflow-hidden">
                      <button
                        onClick={() => setExpandedKS(expandedKS === 'unassigned' ? null : 'unassigned')}
                        className="w-full flex items-center gap-3 p-4 hover:bg-[var(--color-background)]/50 transition-colors"
                      >
                        <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 bg-zinc-500/10">
                          <svg className="w-5 h-5 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div className="flex-1 text-left">
                          <h3 className="font-medium text-[var(--color-text)]">Unassigned Documents</h3>
                          <p className="text-sm text-[var(--color-text-secondary)]">
                            {documents.filter(d => !d.knowledge_store_id).length} documents not in any knowledge base
                          </p>
                        </div>
                        <svg className={`w-5 h-5 text-[var(--color-text-secondary)] transition-transform ${expandedKS === 'unassigned' ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                      
                      {expandedKS === 'unassigned' && (
                        <div className="border-t border-[var(--color-border)] p-4 space-y-2">
                          {documents.filter(d => !d.knowledge_store_id).map((doc) => (
                            <div key={doc.id} className="flex items-center gap-3 p-3 rounded-lg bg-[var(--color-background)] hover:bg-[var(--color-border)]/30 transition-colors">
                              {getFileIcon(doc.file_type)}
                              <div className="flex-1 min-w-0">
                                <h4 className="text-sm font-medium text-[var(--color-text)] truncate">{doc.name}</h4>
                                <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                                  <span>{formatFileSize(doc.file_size)}</span>
                                  <span>•</span>
                                  <span>{doc.chunk_count || 0} chunks</span>
                                  <span>•</span>
                                  <div className="flex items-center gap-1">
                                    <div className={`w-1.5 h-1.5 rounded-full ${doc.is_processed ? 'bg-[var(--color-success)]' : 'bg-[var(--color-warning)]'}`} />
                                    <span>{doc.is_processed ? 'Ready' : 'Processing'}</span>
                                  </div>
                                </div>
                              </div>
                              <div className="flex items-center gap-1">
                                {!doc.is_processed && (
                                  <button
                                    onClick={() => handleReprocess(doc.id)}
                                    className="p-1.5 rounded text-[var(--color-warning)] hover:bg-[var(--color-warning)]/10"
                                    title="Reprocess"
                                  >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                  </button>
                                )}
                                <button
                                  onClick={() => handleDelete(doc.id)}
                                  className="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-error)] hover:bg-red-500/10"
                                  title="Delete"
                                >
                                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </>
        )}
        
        {/* Knowledge Bases Tab */}
        {activeTab === 'knowledge-bases' && (
          <>
            <div className="mb-6 flex justify-between items-center">
              <p className="text-[var(--color-text-secondary)] text-sm">
                Organize documents into knowledge bases for use with Custom GPTs.
              </p>
              <button
                onClick={() => setShowKSForm(true)}
                className="flex items-center gap-2 px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Knowledge Base
              </button>
            </div>
            
            {/* Knowledge Base Form Modal */}
            {showKSForm && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={resetKSForm}>
                <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-md border border-[var(--color-border)]" onClick={(e) => e.stopPropagation()}>
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
                    {editingKS ? 'Edit Knowledge Base' : 'Create Knowledge Base'}
                  </h3>
                  
                  {/* Show size stats when editing */}
                  {editingKS && (
                    <div className="mb-4 p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <div className="text-lg font-semibold text-[var(--color-text)]">{editingKS.document_count}</div>
                          <div className="text-xs text-[var(--color-text-secondary)]">Documents</div>
                        </div>
                        <div>
                          <div className="text-lg font-semibold text-[var(--color-text)]">{editingKS.total_chunks}</div>
                          <div className="text-xs text-[var(--color-text-secondary)]">Chunks</div>
                        </div>
                        <div>
                          <div className="text-lg font-semibold text-[var(--color-text)]">{formatFileSize(editingKS.total_size_bytes)}</div>
                          <div className="text-xs text-[var(--color-text-secondary)]">Size</div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Name</label>
                      <input
                        type="text"
                        value={ksName}
                        onChange={(e) => setKsName(e.target.value)}
                        placeholder="e.g., Company Docs"
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Description</label>
                      <textarea
                        value={ksDescription}
                        onChange={(e) => setKsDescription(e.target.value)}
                        placeholder="What is this knowledge base about?"
                        rows={2}
                        className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                      />
                    </div>
                    
                    {/* Visibility Toggle */}
                    <div>
                      <label className="block text-sm text-[var(--color-text-secondary)] mb-2">Visibility</label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setKsIsPublic(false)}
                          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border transition-all ${
                            !ksIsPublic
                              ? 'border-[var(--color-button)] bg-[var(--color-button)] text-[var(--color-button-text)]'
                              : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-text-secondary)]'
                          }`}
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                          </svg>
                          Private
                        </button>
                        <button
                          type="button"
                          onClick={() => setKsIsPublic(true)}
                          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border transition-all ${
                            ksIsPublic
                              ? 'border-[var(--color-success)] bg-[var(--color-success)]/10 text-[var(--color-success)]'
                              : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-text-secondary)]'
                          }`}
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Public
                        </button>
                      </div>
                      <p className="text-xs text-[var(--color-text-secondary)] mt-2">
                        {ksIsPublic 
                          ? 'Anyone can discover and use this knowledge base'
                          : 'Only you can access this knowledge base directly. GPTs using it will still work for others.'}
                      </p>
                    </div>
                    
                    <div className="flex gap-4">
                      <div className="flex-1">
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Icon</label>
                        <div className="flex flex-wrap gap-2">
                          {iconOptions.map((icon) => (
                            <button
                              key={icon}
                              onClick={() => setKsIcon(icon)}
                              className={`w-10 h-10 rounded-lg text-xl flex items-center justify-center transition-all ${
                                ksIcon === icon
                                  ? 'bg-[var(--color-button)] ring-2 ring-[var(--color-border)] text-[var(--color-button-text)]'
                                  : 'bg-[var(--color-background)] hover:bg-[var(--color-background)]/80'
                              }`}
                            >
                              {icon}
                            </button>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Color</label>
                        <input
                          type="color"
                          value={ksColor}
                          onChange={(e) => setKsColor(e.target.value)}
                          className="w-10 h-10 rounded-lg cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex justify-end gap-3 mt-6">
                    <button
                      onClick={resetKSForm}
                      className="px-4 py-2 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-background)]"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleCreateKS}
                      disabled={!ksName.trim()}
                      className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
                    >
                      {editingKS ? 'Save' : 'Create'}
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {/* NC-0.8.0.1.1: Document Keyword Filter Modal */}
            {editingDocKeywords && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setEditingDocKeywords(null)}>
                <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-md border border-[var(--color-border)]" onClick={(e) => e.stopPropagation()}>
                  <h3 className="text-lg font-semibold text-[var(--color-text)] mb-2">
                    Document Keyword Filter
                  </h3>
                  <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                    {editingDocKeywords.name}
                  </p>
                  
                  <div className="space-y-4">
                    {/* Enable toggle */}
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={docKeywordsEnabled}
                        onChange={(e) => setDocKeywordsEnabled(e.target.checked)}
                        className="w-5 h-5 rounded border-[var(--color-border)] bg-[var(--color-background)]"
                      />
                      <div>
                        <div className="text-sm text-[var(--color-text)]">Enable Keyword Filter</div>
                        <div className="text-xs text-[var(--color-text-secondary)]">
                          Only include this document when keywords match
                        </div>
                      </div>
                    </label>
                    
                    {docKeywordsEnabled && (
                      <>
                        {/* Keywords input */}
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                            Required Keywords
                          </label>
                          <textarea
                            value={docKeywordsText}
                            onChange={(e) => setDocKeywordsText(e.target.value)}
                            placeholder={`"exact phrase", keyword1, keyword2`}
                            rows={3}
                            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                          />
                          <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
                            Comma-separated. Use "quotes" for exact phrases.
                          </p>
                        </div>
                        
                        {/* Match mode */}
                        <div>
                          <label className="block text-sm text-[var(--color-text-secondary)] mb-2">
                            Match Mode
                          </label>
                          <div className="grid grid-cols-3 gap-2">
                            <button
                              type="button"
                              onClick={() => setDocKeywordsMode('any')}
                              className={`px-3 py-2 rounded-lg text-sm border transition-all ${
                                docKeywordsMode === 'any'
                                  ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                                  : 'border-[var(--color-border)] text-[var(--color-text-secondary)]'
                              }`}
                            >
                              Any
                            </button>
                            <button
                              type="button"
                              onClick={() => setDocKeywordsMode('all')}
                              className={`px-3 py-2 rounded-lg text-sm border transition-all ${
                                docKeywordsMode === 'all'
                                  ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                                  : 'border-[var(--color-border)] text-[var(--color-text-secondary)]'
                              }`}
                            >
                              All
                            </button>
                            <button
                              type="button"
                              onClick={() => setDocKeywordsMode('mixed')}
                              className={`px-3 py-2 rounded-lg text-sm border transition-all ${
                                docKeywordsMode === 'mixed'
                                  ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                                  : 'border-[var(--color-border)] text-[var(--color-text-secondary)]'
                              }`}
                            >
                              Mixed
                            </button>
                          </div>
                          <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                            {docKeywordsMode === 'any' && 'Any phrase OR any keyword matches (most permissive)'}
                            {docKeywordsMode === 'all' && 'ALL phrases AND ALL keywords must match (most restrictive)'}
                            {docKeywordsMode === 'mixed' && 'All phrases must match, but only any keyword needed'}
                          </p>
                        </div>
                      </>
                    )}
                  </div>
                  
                  <div className="flex justify-end gap-3 mt-6">
                    <button
                      onClick={() => setEditingDocKeywords(null)}
                      className="px-4 py-2 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-background)]"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={saveDocKeywords}
                      className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
                    >
                      Save
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Knowledge Bases Grid */}
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <svg className="animate-spin h-8 w-8 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
            ) : knowledgeStores.length === 0 ? (
              <div className="text-center py-12 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
                <svg className="w-12 h-12 mx-auto mb-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <p className="text-[var(--color-text-secondary)]">No knowledge bases yet</p>
                <p className="text-zinc-500/60 text-sm mt-1">Create one to organize your documents</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {knowledgeStores.map((ks) => (
                  <div
                    key={ks.id}
                    className="bg-[var(--color-surface)] rounded-xl p-5 border border-[var(--color-border)] hover:border-[var(--color-text-secondary)]/30 transition-all group"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl"
                        style={{ backgroundColor: `${ks.color}20` }}
                      >
                        {ks.icon}
                      </div>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => editKS(ks)}
                          className="p-1.5 rounded text-[var(--color-text-secondary)] hover:bg-[var(--color-background)]"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                          </svg>
                        </button>
                        <button
                          onClick={() => handleDeleteKS(ks.id)}
                          className="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-error)] hover:bg-red-500/10"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                    
                    <h3 className="font-semibold text-[var(--color-text)] mb-1">{ks.name}</h3>
                    {ks.description && (
                      <p className="text-sm text-[var(--color-text-secondary)] line-clamp-2 mb-3">{ks.description}</p>
                    )}
                    
                    <div className="flex items-center gap-4 text-xs text-[var(--color-text-secondary)]">
                      <span>{ks.document_count} docs</span>
                      <span>{ks.total_chunks} chunks</span>
                      <span>{formatFileSize(ks.total_size_bytes)}</span>
                    </div>
                    
                    <div className="flex items-center gap-2 mt-4 pt-4 border-t border-[var(--color-border)]">
                      <label className="flex-1">
                        <input
                          type="file"
                          multiple
                          onChange={(e) => handleUpload(e.target.files, ks.id)}
                          className="hidden"
                          accept={ALLOWED_FILE_EXTENSIONS}
                        />
                        <span className="flex items-center justify-center gap-1 px-3 py-1.5 text-sm rounded-lg bg-[var(--color-background)] hover:bg-[var(--color-border)] cursor-pointer transition-colors">
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                          Add Docs
                        </span>
                      </label>
                      {ks.is_global && (
                        <span className="px-2 py-1 text-xs rounded-full bg-purple-500/10 text-purple-400" title="Auto-searched on every query">
                          Global
                        </span>
                      )}
                      {ks.is_public ? (
                        <span className="px-2 py-1 text-xs rounded-full bg-[var(--color-success)]/10 text-[var(--color-success)]">
                          Public
                        </span>
                      ) : (
                        <span className="px-2 py-1 text-xs rounded-full bg-[var(--color-text-secondary)]/10 text-[var(--color-text-secondary)]">
                          Private
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
        
        {/* Info section */}
        <div className="mt-8 p-6 bg-zinc-900/50 rounded-xl border border-[var(--color-border)]">
          <h3 className="font-medium text-[var(--color-text)] mb-2">About Documents & Knowledge Bases</h3>
          <p className="text-sm text-[var(--color-text-secondary)]">
            Documents are processed and indexed for semantic search. Organize them into Knowledge Bases 
            to use with Custom GPTs. When chatting with AI, it can search your documents 
            to provide more accurate, contextual responses.
          </p>
        </div>
      </div>
    </div>
  );
}
