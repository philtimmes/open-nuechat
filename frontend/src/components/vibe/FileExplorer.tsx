/**
 * FileExplorer - File tree with menu, folder support, and multi-select
 */

import { useState, useRef } from 'react';
import type { VibeFile } from '../../pages/VibeCode';

// Clipboard type
interface ClipboardData {
  files: VibeFile[];
  mode: 'cut' | 'copy';
}

interface FileExplorerProps {
  files: VibeFile[];
  activeFile: VibeFile | null;
  onFileSelect: (file: VibeFile) => void;
  onCreateFile: (name: string, language?: string) => void;
  onCreateFolder: (name: string) => void;
  onDeleteFile: (fileId: string) => void;
  onDeleteFiles: (fileIds: string[]) => void;
  onMoveFiles: (fileIds: string[], targetFolder: string) => void;
  onCopyFiles: (fileIds: string[], targetFolder: string) => void;
  onZipUpload: (file: File) => void;
  onDownload: () => void;
  isUploadingZip: boolean;
  isDownloading?: boolean;
}

// File icons by extension
const fileIcons: Record<string, string> = {
  'ts': '📘',
  'tsx': '⚛️',
  'js': '📒',
  'jsx': '⚛️',
  'py': '🐍',
  'rs': '🦀',
  'go': '🐹',
  'java': '☕',
  'cpp': '⚙️',
  'c': '⚙️',
  'h': '📄',
  'css': '🎨',
  'scss': '🎨',
  'html': '🌐',
  'json': '📋',
  'yaml': '📋',
  'yml': '📋',
  'md': '📝',
  'sql': '🗃️',
  'sh': '🖥️',
  'txt': '📄',
  'default': '📄',
};

export default function FileExplorer({
  files,
  activeFile,
  onFileSelect,
  onCreateFile,
  onCreateFolder,
  onDeleteFile,
  onDeleteFiles,
  onMoveFiles,
  onCopyFiles,
  onZipUpload,
  onDownload,
  isUploadingZip,
  isDownloading = false,
}: FileExplorerProps) {
  // Menu state
  const [showMenu, setShowMenu] = useState(false);
  const [showNewSubmenu, setShowNewSubmenu] = useState(false);
  const [showFileActionsSubmenu, setShowFileActionsSubmenu] = useState(false);
  
  // Creation state
  const [showNewFile, setShowNewFile] = useState(false);
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [newItemName, setNewItemName] = useState('');
  const [currentFolder, setCurrentFolder] = useState<string>(''); // Track current folder for new files
  
  // Selection state
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  
  // Clipboard state
  const [clipboard, setClipboard] = useState<ClipboardData | null>(null);
  
  // Delete confirmation
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  
  // Context menu
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; file: VibeFile } | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  
  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    return fileIcons[ext] || fileIcons['default'];
  };
  
  const handleCreateFile = () => {
    if (newItemName.trim()) {
      // If a folder is open, create file in that folder
      const fileName = currentFolder 
        ? `${currentFolder}/${newItemName.trim()}`
        : newItemName.trim();
      onCreateFile(fileName);
      setNewItemName('');
      setShowNewFile(false);
    }
  };
  
  const handleCreateFolder = () => {
    if (newItemName.trim()) {
      // If a folder is open, create subfolder
      const folderName = currentFolder
        ? `${currentFolder}/${newItemName.trim()}`
        : newItemName.trim();
      onCreateFolder(folderName);
      setNewItemName('');
      setShowNewFolder(false);
    }
  };
  
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.zip')) {
      onZipUpload(file);
    }
    e.target.value = '';
  };
  
  const handleContextMenu = (e: React.MouseEvent, file: VibeFile) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, file });
  };
  
  // Toggle item selection
  const toggleSelection = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const newSelected = new Set(selectedItems);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedItems(newSelected);
  };
  
  // Toggle folder expansion
  const toggleFolder = (folder: string) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folder)) {
      newExpanded.delete(folder);
      // If closing a folder, clear currentFolder if it was this one
      if (currentFolder === folder) {
        setCurrentFolder('');
      }
    } else {
      newExpanded.add(folder);
      // Set this as the current folder for new file creation
      setCurrentFolder(folder);
    }
    setExpandedFolders(newExpanded);
  };
  
  // Select all
  const selectAll = () => {
    setSelectedItems(new Set(files.map(f => f.id)));
  };
  
  // Clear selection
  const clearSelection = () => {
    setSelectedItems(new Set());
  };
  
  // Clipboard operations
  const handleCut = () => {
    const selectedFiles = files.filter(f => selectedItems.has(f.id));
    if (selectedFiles.length > 0) {
      setClipboard({ files: selectedFiles, mode: 'cut' });
    }
  };
  
  const handleCopy = () => {
    const selectedFiles = files.filter(f => selectedItems.has(f.id));
    if (selectedFiles.length > 0) {
      setClipboard({ files: selectedFiles, mode: 'copy' });
    }
  };
  
  const handlePaste = () => {
    if (!clipboard) return;
    
    const fileIds = clipboard.files.map(f => f.id);
    const targetFolder = currentFolder;
    
    if (clipboard.mode === 'cut') {
      onMoveFiles(fileIds, targetFolder);
      setClipboard(null); // Clear clipboard after cut-paste
    } else {
      onCopyFiles(fileIds, targetFolder);
      // Keep clipboard for copy (can paste multiple times)
    }
    
    // Clear selection after paste
    setSelectedItems(new Set());
  };
  
  // Delete selected items
  const handleDeleteSelected = () => {
    if (selectedItems.size > 0) {
      setShowDeleteConfirm(true);
    }
  };
  
  const confirmDelete = () => {
    onDeleteFiles(Array.from(selectedItems));
    setSelectedItems(new Set());
    setShowDeleteConfirm(false);
  };
  
  // Close menu when clicking outside
  const closeMenu = () => {
    setShowMenu(false);
    setShowNewSubmenu(false);
    setShowFileActionsSubmenu(false);
  };
  
  // Group files by directory
  const groupedFiles = files.reduce((acc, file) => {
    const parts = file.path.split('/');
    if (parts.length > 1) {
      const dir = parts.slice(0, -1).join('/');
      if (!acc[dir]) acc[dir] = [];
      acc[dir].push(file);
    } else {
      if (!acc['']) acc[''] = [];
      acc[''].push(file);
    }
    return acc;
  }, {} as Record<string, VibeFile[]>);
  
  // Get unique folders
  const folders = Object.keys(groupedFiles).filter(d => d !== '').sort();
  const rootFiles = groupedFiles[''] || [];
  
  return (
    <div className="h-full flex flex-col bg-[var(--color-surface)]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--color-border)]">
        <span className="text-xs font-medium text-[var(--color-text-secondary)] uppercase">
          Explorer
        </span>
        <div className="flex items-center gap-1">
          {/* Download Button */}
          <button
            onClick={onDownload}
            disabled={isDownloading || files.length === 0}
            className="p-1 rounded hover:bg-[var(--color-background)] text-[var(--color-text-secondary)] disabled:opacity-50"
            title="Download as Zip"
          >
            {isDownloading ? (
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
              </svg>
            )}
          </button>
          {/* Upload Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploadingZip}
            className="p-1 rounded hover:bg-[var(--color-background)] text-[var(--color-text-secondary)] disabled:opacity-50"
            title="Upload Zip"
          >
            {isUploadingZip ? (
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            )}
          </button>
        </div>
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        onChange={handleFileUpload}
        className="hidden"
      />
      
      {/* Menu Button */}
      <div className="px-3 py-2 border-b border-[var(--color-border)] relative" ref={menuRef}>
        <button
          onClick={() => setShowMenu(!showMenu)}
          className="w-full flex items-center justify-between px-3 py-2 text-sm rounded bg-[var(--color-primary)] text-white hover:opacity-90 transition-opacity"
        >
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
            Menu
          </div>
          <svg className={`w-4 h-4 transition-transform ${showMenu ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        {/* Dropdown Menu */}
        {showMenu && (
          <>
            <div className="fixed inset-0 z-40" onClick={closeMenu} />
            <div className="absolute left-3 right-3 top-full mt-1 z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-xl overflow-hidden">
              {/* New submenu - click to expand */}
              <div>
                <button
                  onClick={() => setShowNewSubmenu(!showNewSubmenu)}
                  className="w-full flex items-center justify-between px-4 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
                >
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    New
                    {currentFolder && (
                      <span className="text-xs text-[var(--color-text-secondary)]">
                        (in {currentFolder})
                      </span>
                    )}
                  </div>
                  <svg className={`w-4 h-4 transition-transform ${showNewSubmenu ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                
                {/* New submenu items - drops down inline */}
                {showNewSubmenu && (
                  <div className="bg-[var(--color-background)] border-t border-[var(--color-border)]">
                    <button
                      onClick={() => {
                        setShowNewFolder(true);
                        closeMenu();
                      }}
                      className="w-full flex items-center gap-2 px-6 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)]"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                      </svg>
                      Folder
                    </button>
                    <button
                      onClick={() => {
                        setShowNewFile(true);
                        closeMenu();
                      }}
                      className="w-full flex items-center gap-2 px-6 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)]"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      File
                    </button>
                  </div>
                )}
              </div>
              
              {/* File Actions submenu - only show when files are selected */}
              {selectedItems.size > 0 && (
                <div>
                  <button
                    onClick={() => setShowFileActionsSubmenu(!showFileActionsSubmenu)}
                    className="w-full flex items-center justify-between px-4 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
                  >
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      File Actions ({selectedItems.size})
                    </div>
                    <svg className={`w-4 h-4 transition-transform ${showFileActionsSubmenu ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  
                  {/* File Actions submenu items */}
                  {showFileActionsSubmenu && (
                    <div className="bg-[var(--color-background)] border-t border-[var(--color-border)]">
                      <button
                        onClick={() => {
                          handleCut();
                          closeMenu();
                        }}
                        className="w-full flex items-center gap-2 px-6 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)]"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12L9.121 9.121m0 5.758a3 3 0 10-4.243 4.243 3 3 0 004.243-4.243zm0-5.758a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z" />
                        </svg>
                        Cut
                      </button>
                      <button
                        onClick={() => {
                          handleCopy();
                          closeMenu();
                        }}
                        className="w-full flex items-center gap-2 px-6 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)]"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        Copy
                      </button>
                      <button
                        onClick={() => {
                          handlePaste();
                          closeMenu();
                        }}
                        disabled={!clipboard}
                        className="w-full flex items-center gap-2 px-6 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)] disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        Paste {clipboard && `(${clipboard.files.length} ${clipboard.mode === 'cut' ? 'cut' : 'copied'})`}
                      </button>
                    </div>
                  )}
                </div>
              )}
              
              {/* Paste option when nothing selected but clipboard has content */}
              {selectedItems.size === 0 && clipboard && (
                <button
                  onClick={() => {
                    handlePaste();
                    closeMenu();
                  }}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Paste ({clipboard.files.length} {clipboard.mode === 'cut' ? 'cut' : 'copied'})
                  {currentFolder && <span className="text-xs text-[var(--color-text-secondary)]">to {currentFolder}</span>}
                </button>
              )}
              
              <div className="border-t border-[var(--color-border)]" />
              
              {/* Select All / Clear Selection */}
              <button
                onClick={() => {
                  selectAll();
                  closeMenu();
                }}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                Select All
              </button>
              
              {selectedItems.size > 0 && (
                <button
                  onClick={() => {
                    clearSelection();
                    closeMenu();
                  }}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Clear Selection ({selectedItems.size})
                </button>
              )}
              
              <div className="border-t border-[var(--color-border)]" />
              
              {/* Delete Selected */}
              <button
                onClick={() => {
                  handleDeleteSelected();
                  closeMenu();
                }}
                disabled={selectedItems.size === 0}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-[var(--color-background)] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete Selected {selectedItems.size > 0 && `(${selectedItems.size})`}
              </button>
            </div>
          </>
        )}
      </div>
      
      {/* New folder input */}
      {showNewFolder && (
        <div className="px-3 py-2 border-b border-[var(--color-border)] bg-[var(--color-background)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">New Folder</div>
          <input
            type="text"
            value={newItemName}
            onChange={(e) => setNewItemName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleCreateFolder();
              if (e.key === 'Escape') {
                setShowNewFolder(false);
                setNewItemName('');
              }
            }}
            placeholder="folder-name"
            className="w-full px-2 py-1 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[var(--color-text)]"
            autoFocus
          />
        </div>
      )}
      
      {/* New file input */}
      {showNewFile && (
        <div className="px-3 py-2 border-b border-[var(--color-border)] bg-[var(--color-background)]">
          <div className="text-xs text-[var(--color-text-secondary)] mb-1">New File</div>
          <input
            type="text"
            value={newItemName}
            onChange={(e) => setNewItemName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleCreateFile();
              if (e.key === 'Escape') {
                setShowNewFile(false);
                setNewItemName('');
              }
            }}
            placeholder="filename.ts"
            className="w-full px-2 py-1 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[var(--color-text)]"
            autoFocus
          />
        </div>
      )}
      
      {/* Selection indicator */}
      {selectedItems.size > 0 && (
        <div className="px-3 py-1 bg-[var(--color-primary)]/10 text-xs text-[var(--color-primary)] border-b border-[var(--color-border)]">
          {selectedItems.size} item{selectedItems.size !== 1 ? 's' : ''} selected
        </div>
      )}
      
      {/* File tree */}
      <div className="flex-1 overflow-y-auto py-1">
        {/* Root files first */}
        {rootFiles.map(file => (
          <div
            key={file.id}
            onClick={() => onFileSelect(file)}
            onContextMenu={(e) => handleContextMenu(e, file)}
            className={`flex items-center gap-2 px-3 py-1 cursor-pointer text-sm ${
              activeFile?.id === file.id
                ? 'bg-[var(--color-primary)]/20 text-[var(--color-primary)]'
                : 'text-[var(--color-text)] hover:bg-[var(--color-background)]'
            }`}
          >
            {/* Checkbox */}
            <input
              type="checkbox"
              checked={selectedItems.has(file.id)}
              onChange={() => {}}
              onClick={(e) => toggleSelection(file.id, e)}
              className="w-4 h-4 rounded border-[var(--color-border)] text-[var(--color-primary)] focus:ring-[var(--color-primary)] cursor-pointer"
            />
            <span className="text-base">{getFileIcon(file.name)}</span>
            <span className={`truncate flex-1 ${file.isModified ? 'italic' : ''}`}>
              {file.name}
              {file.isModified && ' •'}
            </span>
          </div>
        ))}
        
        {/* Folders */}
        {folders.map(dir => {
          const isExpanded = expandedFolders.has(dir);
          const folderFiles = groupedFiles[dir];
          const allSelected = folderFiles.every(f => selectedItems.has(f.id));
          const someSelected = folderFiles.some(f => selectedItems.has(f.id));
          
          return (
            <div key={dir}>
              {/* Folder header */}
              <div
                onClick={() => toggleFolder(dir)}
                className="flex items-center gap-2 px-3 py-1 cursor-pointer text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
              >
                {/* Folder checkbox */}
                <input
                  type="checkbox"
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelected && !allSelected;
                  }}
                  onChange={() => {}}
                  onClick={(e) => {
                    e.stopPropagation();
                    const newSelected = new Set(selectedItems);
                    if (allSelected) {
                      folderFiles.forEach(f => newSelected.delete(f.id));
                    } else {
                      folderFiles.forEach(f => newSelected.add(f.id));
                    }
                    setSelectedItems(newSelected);
                  }}
                  className="w-4 h-4 rounded border-[var(--color-border)] text-[var(--color-primary)] focus:ring-[var(--color-primary)] cursor-pointer"
                />
                <svg 
                  className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`} 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <svg className="w-4 h-4 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <span className="truncate">{dir}</span>
                <span className="text-xs text-[var(--color-text-secondary)]">({folderFiles.length})</span>
              </div>
              
              {/* Folder contents */}
              {isExpanded && folderFiles.map(file => (
                <div
                  key={file.id}
                  onClick={() => onFileSelect(file)}
                  onContextMenu={(e) => handleContextMenu(e, file)}
                  className={`flex items-center gap-2 px-3 py-1 pl-10 cursor-pointer text-sm ${
                    activeFile?.id === file.id
                      ? 'bg-[var(--color-primary)]/20 text-[var(--color-primary)]'
                      : 'text-[var(--color-text)] hover:bg-[var(--color-background)]'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedItems.has(file.id)}
                    onChange={() => {}}
                    onClick={(e) => toggleSelection(file.id, e)}
                    className="w-4 h-4 rounded border-[var(--color-border)] text-[var(--color-primary)] focus:ring-[var(--color-primary)] cursor-pointer"
                  />
                  <span className="text-base">{getFileIcon(file.name)}</span>
                  <span className={`truncate flex-1 ${file.isModified ? 'italic' : ''}`}>
                    {file.name}
                    {file.isModified && ' •'}
                  </span>
                </div>
              ))}
            </div>
          );
        })}
        
        {files.length === 0 && (
          <div className="px-3 py-4 text-center text-sm text-[var(--color-text-secondary)]">
            <div className="text-2xl mb-2">📁</div>
            <div>No files yet</div>
            <div className="text-xs mt-1">Use Menu → New to create files</div>
          </div>
        )}
      </div>
      
      {/* Context menu */}
      {contextMenu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setContextMenu(null)}
          />
          <div
            className="fixed z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded shadow-lg py-1"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <button
              onClick={() => {
                onFileSelect(contextMenu.file);
                setContextMenu(null);
              }}
              className="w-full px-4 py-1 text-left text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
            >
              Open
            </button>
            <button
              onClick={() => {
                const newSelected = new Set(selectedItems);
                if (newSelected.has(contextMenu.file.id)) {
                  newSelected.delete(contextMenu.file.id);
                } else {
                  newSelected.add(contextMenu.file.id);
                }
                setSelectedItems(newSelected);
                setContextMenu(null);
              }}
              className="w-full px-4 py-1 text-left text-sm text-[var(--color-text)] hover:bg-[var(--color-background)]"
            >
              {selectedItems.has(contextMenu.file.id) ? 'Deselect' : 'Select'}
            </button>
            <div className="border-t border-[var(--color-border)] my-1" />
            <button
              onClick={() => {
                onDeleteFile(contextMenu.file.id);
                setContextMenu(null);
              }}
              className="w-full px-4 py-1 text-left text-sm text-red-400 hover:bg-[var(--color-background)]"
            >
              Delete
            </button>
          </div>
        </>
      )}
      
      {/* Delete confirmation dialog */}
      {showDeleteConfirm && (
        <>
          <div className="fixed inset-0 bg-black/50 z-50" onClick={() => setShowDeleteConfirm(false)} />
          <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-xl p-4 w-80">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
                <svg className="w-5 h-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-[var(--color-text)]">Delete Files</h3>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  {selectedItems.size} item{selectedItems.size !== 1 ? 's' : ''} will be deleted
                </p>
              </div>
            </div>
            
            <p className="text-sm text-[var(--color-text-secondary)] mb-4">
              Are you sure you want to delete the selected files? This action cannot be undone.
            </p>
            
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-sm rounded bg-[var(--color-background)] text-[var(--color-text)] hover:bg-[var(--color-border)]"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 text-sm rounded bg-red-500 text-white hover:bg-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
