/**
 * VibeCode - AI-Powered Code Editor
 * 
 * Features:
 * - Monaco-based code editor with AI completion
 * - Zip file upload for project context
 * - Named projects with localStorage persistence
 * - Multiple projects support
 * - Agentic coding from prompts
 * - Linting and formatting
 * - Agents.md for tracking solutions
 * - State machine for complex tasks
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import JSZip from 'jszip';
import { useAuthStore } from '../stores/authStore';
import { useModelsStore } from '../stores/modelsStore';
import api from '../lib/api';
import CodeEditor from '../components/vibe/CodeEditor';
import FileExplorer from '../components/vibe/FileExplorer';
import VibeChat from '../components/vibe/VibeChat';
import AgentStatus from '../components/vibe/AgentStatus';

// Types for Custom GPTs in model selector
interface CustomAssistant {
  id: string;
  name: string;
  type: string;
  assistant_id: string;
  icon?: string;
  color?: string;
}

export interface VibeFile {
  id: string;
  name: string;
  path: string;
  content: string;
  language: string;
  isModified: boolean;
  isNew?: boolean;
}

export interface VibeProject {
  id: string;
  name: string;
  files: VibeFile[];
  agentsMd: string;  // Tracks user feedback and solutions
  createdAt: string;
  updatedAt: string;
}

export interface AgentTask {
  id: string;
  type: 'create' | 'edit' | 'fix' | 'refactor' | 'explain' | 'test';
  status: 'pending' | 'running' | 'completed' | 'failed';
  description: string;
  affectedFiles: string[];
  result?: string;
  error?: string;
}

// Agent state machine states
type AgentState = 
  | 'idle'
  | 'analyzing'      // Analyzing user request
  | 'planning'       // Creating execution plan
  | 'generating'     // Generating code
  | 'reviewing'      // Self-reviewing generated code
  | 'applying'       // Applying changes
  | 'testing'        // Running tests/linting
  | 'complete'
  | 'error';

// Storage keys
const STORAGE_KEY_PROJECTS = 'vibecode-projects';
const STORAGE_KEY_ACTIVE_PROJECT = 'vibecode-active-project';

export default function VibeCode() {
  const navigate = useNavigate();
  const { user, isAuthenticated } = useAuthStore();
  const { models, defaultModel, subscribedAssistants, fetchModels } = useModelsStore();
  
  // Project management state
  const [projects, setProjects] = useState<VibeProject[]>([]);
  const [showProjectSelector, setShowProjectSelector] = useState(false);
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [showRenameDialog, setShowRenameDialog] = useState(false);
  const [renameProjectName, setRenameProjectName] = useState('');
  
  // Project state
  const [project, setProject] = useState<VibeProject | null>(null);
  const [files, setFiles] = useState<VibeFile[]>([]);
  const [activeFile, setActiveFile] = useState<VibeFile | null>(null);
  const [openTabs, setOpenTabs] = useState<VibeFile[]>([]);
  
  // Editor state
  const [editorContent, setEditorContent] = useState('');
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  
  // Agent state
  const [agentState, setAgentState] = useState<AgentState>('idle');
  const [currentTask, setCurrentTask] = useState<AgentTask | null>(null);
  const [taskHistory, setTaskHistory] = useState<AgentTask[]>([]);
  const [agentsMd, setAgentsMd] = useState('# Agents.md\n\nTracking user feedback, errors, and solutions.\n\n');
  
  // Chat state
  const [chatMessages, setChatMessages] = useState<Array<{
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: string;
  }>>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // UI state
  const [selectedModel, setSelectedModel] = useState(defaultModel);
  const [selectedAssistantId, setSelectedAssistantId] = useState<string | null>(null);
  const [myAssistants, setMyAssistants] = useState<CustomAssistant[]>([]);
  const [isUploadingZip, setIsUploadingZip] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [lintErrors, setLintErrors] = useState<Array<{
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning' | 'info';
  }>>([]);
  
  // WebSocket for real-time updates
  const wsRef = useRef<WebSocket | null>(null);
  
  // Load projects from localStorage
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }
    
    // Load saved projects
    const savedProjects = localStorage.getItem(STORAGE_KEY_PROJECTS);
    const savedActiveId = localStorage.getItem(STORAGE_KEY_ACTIVE_PROJECT);
    
    if (savedProjects) {
      try {
        const parsed = JSON.parse(savedProjects) as VibeProject[];
        setProjects(parsed);
        
        // Load active project
        if (savedActiveId) {
          const active = parsed.find(p => p.id === savedActiveId);
          if (active) {
            loadProject(active);
          } else if (parsed.length > 0) {
            loadProject(parsed[0]);
          } else {
            createNewProject('New Project');
          }
        } else if (parsed.length > 0) {
          loadProject(parsed[0]);
        } else {
          createNewProject('New Project');
        }
      } catch (e) {
        console.error('Failed to parse saved projects:', e);
        createNewProject('New Project');
      }
    } else {
      createNewProject('New Project');
    }
    
    // Load models and assistants
    fetchModels();
    loadMyAssistants();
  }, [isAuthenticated, navigate]);
  
  // Save projects to localStorage when they change
  useEffect(() => {
    if (projects.length > 0) {
      localStorage.setItem(STORAGE_KEY_PROJECTS, JSON.stringify(projects));
    }
  }, [projects]);
  
  // Save active project ID
  useEffect(() => {
    if (project) {
      localStorage.setItem(STORAGE_KEY_ACTIVE_PROJECT, project.id);
    }
  }, [project?.id]);
  
  // Auto-save current project when files change
  useEffect(() => {
    if (project && files.length > 0) {
      const updatedProject = {
        ...project,
        files,
        agentsMd,
        updatedAt: new Date().toISOString(),
      };
      setProjects(prev => prev.map(p => p.id === project.id ? updatedProject : p));
    }
  }, [files, agentsMd]);
  
  // Load owned assistants
  const loadMyAssistants = async () => {
    try {
      const res = await api.get('/assistants');
      const assistants = res.data.map((a: any) => ({
        id: `gpt:${a.id}`,
        name: a.name,
        type: 'assistant',
        assistant_id: a.id,
        icon: a.icon,
        color: a.color,
      }));
      setMyAssistants(assistants);
    } catch (err) {
      console.error('Failed to load assistants:', err);
    }
  };
  
  // Handle model/assistant selection
  const handleModelSelect = (modelId: string, assistantId?: string) => {
    setSelectedModel(modelId);
    setSelectedAssistantId(assistantId || null);
  };
  
  // Get display name for current selection
  const getSelectedDisplayName = () => {
    if (selectedAssistantId) {
      // Check owned assistants
      const owned = myAssistants.find(a => a.assistant_id === selectedAssistantId);
      if (owned) return owned.name;
      // Check subscribed assistants
      const subscribed = subscribedAssistants.find(a => a.assistant_id === selectedAssistantId);
      if (subscribed) return subscribed.name;
    }
    // Find model name
    const model = models.find(m => m.id === selectedModel);
    return model?.name || selectedModel || 'Select Model';
  };
  
  // Create a new project
  const createNewProject = (name: string, initialFiles?: VibeFile[]) => {
    const newProject: VibeProject = {
      id: `project-${Date.now()}`,
      name,
      files: initialFiles || [],
      agentsMd: '# Agents.md\n\nTracking user feedback, errors, and solutions.\n\n',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    
    // Create welcome file if no initial files
    if (!initialFiles || initialFiles.length === 0) {
      const welcomeFile: VibeFile = {
        id: `file-${Date.now()}`,
        name: 'welcome.md',
        path: 'welcome.md',
        content: `# Welcome to ${name}! 🚀

An AI-powered code editor that helps you write better code faster.

## Features

- **AI Code Completion**: Get intelligent suggestions as you type
- **Agentic Coding**: Describe what you want, AI writes it
- **Smart Linting**: Catch errors before they happen
- **Code Formatting**: Keep your code beautiful
- **Project Context**: Upload zip files for full project understanding
- **Multiple Projects**: Create and switch between projects

## Getting Started

1. Upload a zip file or create new files
2. Use the chat to ask questions or request code
3. Press Tab to accept AI suggestions
4. Let the AI agent handle complex tasks

Happy coding! 🎉
`,
        language: 'markdown',
        isModified: false,
      };
      newProject.files = [welcomeFile];
    }
    
    setProjects(prev => [...prev, newProject]);
    loadProject(newProject);
    setShowNewProjectDialog(false);
    setNewProjectName('');
  };
  
  // Load a project
  const loadProject = (proj: VibeProject) => {
    setProject(proj);
    setFiles(proj.files);
    setAgentsMd(proj.agentsMd);
    setChatMessages([]);
    setOpenTabs([]);
    
    // Open first file
    if (proj.files.length > 0) {
      handleFileSelect(proj.files[0]);
    } else {
      setActiveFile(null);
      setEditorContent('');
    }
    
    setShowProjectSelector(false);
  };
  
  // Delete a project
  const deleteProject = (projectId: string) => {
    const remaining = projects.filter(p => p.id !== projectId);
    setProjects(remaining);
    
    // If deleting current project, switch to another or create new
    if (project?.id === projectId) {
      if (remaining.length > 0) {
        loadProject(remaining[0]);
      } else {
        createNewProject('New Project');
      }
    }
  };
  
  // Rename current project
  const renameProject = () => {
    if (!project || !renameProjectName.trim()) return;
    
    const updated = { ...project, name: renameProjectName.trim(), updatedAt: new Date().toISOString() };
    setProject(updated);
    setProjects(prev => prev.map(p => p.id === project.id ? updated : p));
    setShowRenameDialog(false);
    setRenameProjectName('');
  };
  
  const initializeProject = () => {
    // This is now handled by createNewProject
    createNewProject('New Project');
  };
  
  // Handle file selection
  const handleFileSelect = (file: VibeFile) => {
    // Save current file if modified
    if (activeFile && activeFile.isModified) {
      saveFile(activeFile);
    }
    
    setActiveFile(file);
    setEditorContent(file.content);
    
    // Add to open tabs if not already there
    if (!openTabs.find(t => t.id === file.id)) {
      setOpenTabs([...openTabs, file]);
    }
    
    // Clear suggestions when switching files
    setSuggestions([]);
    setShowSuggestions(false);
  };
  
  // Handle content change
  const handleContentChange = (newContent: string) => {
    setEditorContent(newContent);
    
    if (activeFile) {
      const updatedFile = { ...activeFile, content: newContent, isModified: true };
      setActiveFile(updatedFile);
      
      // Update in files list
      setFiles(files.map(f => f.id === activeFile.id ? updatedFile : f));
      
      // Update in open tabs
      setOpenTabs(openTabs.map(t => t.id === activeFile.id ? updatedFile : t));
    }
    
    // Trigger completion after typing
    requestCompletion(newContent);
  };
  
  // Request AI completion
  const requestCompletion = useCallback(async (content: string) => {
    if (!activeFile || agentState !== 'idle') return;
    
    // Debounce completion requests
    // Only suggest after user pauses typing
    try {
      const response = await api.post('/vibe/complete', {
        file_path: activeFile.path,
        content,
        cursor_line: cursorPosition.line,
        cursor_column: cursorPosition.column,
        language: activeFile.language,
        context_files: files.slice(0, 5).map(f => ({ path: f.path, content: f.content })),
        model: selectedModel,
        assistant_id: selectedAssistantId,
      });
      
      if (response.data.suggestions && response.data.suggestions.length > 0) {
        setSuggestions(response.data.suggestions);
        setShowSuggestions(true);
      }
    } catch (error) {
      console.error('Completion error:', error);
    }
  }, [activeFile, cursorPosition, files, selectedModel, selectedAssistantId, agentState]);
  
  // Accept suggestion
  const handleAcceptSuggestion = (suggestion: string) => {
    if (!activeFile) return;
    
    // Insert suggestion at cursor position
    const lines = editorContent.split('\n');
    const line = lines[cursorPosition.line - 1] || '';
    const before = line.slice(0, cursorPosition.column - 1);
    const after = line.slice(cursorPosition.column - 1);
    
    lines[cursorPosition.line - 1] = before + suggestion + after;
    const newContent = lines.join('\n');
    
    handleContentChange(newContent);
    setShowSuggestions(false);
    setSuggestions([]);
  };
  
  // Save file
  const saveFile = async (file: VibeFile) => {
    const updatedFile = { ...file, isModified: false };
    setFiles(files.map(f => f.id === file.id ? updatedFile : f));
    setOpenTabs(openTabs.map(t => t.id === file.id ? updatedFile : t));
    if (activeFile?.id === file.id) {
      setActiveFile(updatedFile);
    }
  };
  
  // Create new file
  const handleCreateFile = (path: string, language?: string) => {
    // Extract just the filename from the path
    const name = path.split('/').pop() || path;
    // Always determine language from path if not explicitly provided
    const detectedLanguage = language || getLanguageFromPath(path);
    const newFile: VibeFile = {
      id: `file-${Date.now()}`,
      name,
      path,
      content: '',
      language: detectedLanguage,
      isModified: false,
      isNew: true,
    };
    
    setFiles([...files, newFile]);
    handleFileSelect(newFile);
  };
  
  // Create file with content (from chat code snippets)
  const handleCreateFileWithContent = (path: string, content: string, language?: string) => {
    // Extract just the filename from the path
    const name = path.split('/').pop() || path;
    // Always determine language from path if not explicitly provided
    const detectedLanguage = language || getLanguageFromPath(path);
    const newFile: VibeFile = {
      id: `file-${Date.now()}`,
      name,
      path,
      content,
      language: detectedLanguage,
      isModified: true,
      isNew: true,
    };
    
    setFiles([...files, newFile]);
    handleFileSelect(newFile);
    setEditorContent(content);
  };
  
  // Insert code at cursor position
  const handleInsertCode = (code: string) => {
    if (!activeFile) {
      // No active file - create a new one
      const newFile: VibeFile = {
        id: `file-${Date.now()}`,
        name: 'untitled.txt',
        path: 'untitled.txt',
        content: code,
        language: 'text',
        isModified: true,
        isNew: true,
      };
      setFiles([...files, newFile]);
      handleFileSelect(newFile);
      setEditorContent(code);
      return;
    }
    
    // Insert at current position
    const lines = editorContent.split('\n');
    const lineIndex = Math.max(0, cursorPosition.line - 1);
    const colIndex = Math.max(0, cursorPosition.column - 1);
    
    if (lineIndex < lines.length) {
      const line = lines[lineIndex];
      const newLine = line.slice(0, colIndex) + code + line.slice(colIndex);
      lines[lineIndex] = newLine;
    } else {
      lines.push(code);
    }
    
    const newContent = lines.join('\n');
    handleContentChange(newContent);
  };
  
  // Delete file
  const handleDeleteFile = (fileId: string) => {
    setFiles(files.filter(f => f.id !== fileId));
    setOpenTabs(openTabs.filter(t => t.id !== fileId));
    
    if (activeFile?.id === fileId) {
      const remainingFiles = files.filter(f => f.id !== fileId);
      if (remainingFiles.length > 0) {
        handleFileSelect(remainingFiles[0]);
      } else {
        setActiveFile(null);
        setEditorContent('');
      }
    }
  };
  
  // Delete multiple files
  const handleDeleteFiles = (fileIds: string[]) => {
    const idsSet = new Set(fileIds);
    setFiles(files.filter(f => !idsSet.has(f.id)));
    setOpenTabs(openTabs.filter(t => !idsSet.has(t.id)));
    
    if (activeFile && idsSet.has(activeFile.id)) {
      const remainingFiles = files.filter(f => !idsSet.has(f.id));
      if (remainingFiles.length > 0) {
        handleFileSelect(remainingFiles[0]);
      } else {
        setActiveFile(null);
        setEditorContent('');
      }
    }
  };
  
  // Create folder (creates a placeholder file in the folder)
  const handleCreateFolder = (name: string) => {
    // Create a .gitkeep placeholder file in the folder
    const placeholder: VibeFile = {
      id: `file-${Date.now()}`,
      name: '.gitkeep',
      path: `${name}/.gitkeep`,
      content: '',
      language: 'plaintext',
      isModified: false,
      isNew: true,
    };
    setFiles([...files, placeholder]);
  };
  
  // Move files to a target folder (cut + paste)
  const handleMoveFiles = (fileIds: string[], targetFolder: string) => {
    setFiles(prevFiles => prevFiles.map(file => {
      if (fileIds.includes(file.id)) {
        const newPath = targetFolder 
          ? `${targetFolder}/${file.name}`
          : file.name;
        return {
          ...file,
          path: newPath,
          isModified: true,
        };
      }
      return file;
    }));
    
    // Update open tabs if any moved files are open
    setOpenTabs(prevTabs => prevTabs.map(tab => {
      if (fileIds.includes(tab.id)) {
        const newPath = targetFolder 
          ? `${targetFolder}/${tab.name}`
          : tab.name;
        return {
          ...tab,
          path: newPath,
          isModified: true,
        };
      }
      return tab;
    }));
    
    // Update active file if it was moved
    if (activeFile && fileIds.includes(activeFile.id)) {
      const newPath = targetFolder 
        ? `${targetFolder}/${activeFile.name}`
        : activeFile.name;
      setActiveFile({
        ...activeFile,
        path: newPath,
        isModified: true,
      });
    }
  };
  
  // Copy files to a target folder (copy + paste)
  const handleCopyFiles = (fileIds: string[], targetFolder: string) => {
    const filesToCopy = files.filter(f => fileIds.includes(f.id));
    const newFiles = filesToCopy.map(file => {
      const newPath = targetFolder 
        ? `${targetFolder}/${file.name}`
        : file.name;
      
      // Check if file already exists at target, add suffix if needed
      let finalPath = newPath;
      let counter = 1;
      while (files.some(f => f.path === finalPath)) {
        const ext = file.name.includes('.') 
          ? `.${file.name.split('.').pop()}`
          : '';
        const baseName = file.name.replace(ext, '');
        const newName = `${baseName}_copy${counter > 1 ? counter : ''}${ext}`;
        finalPath = targetFolder ? `${targetFolder}/${newName}` : newName;
        counter++;
      }
      
      return {
        ...file,
        id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        path: finalPath,
        name: finalPath.split('/').pop() || file.name,
        isModified: true,
        isNew: true,
      };
    });
    
    setFiles([...files, ...newFiles]);
  };
  
  // Close tab
  const handleCloseTab = (fileId: string) => {
    const newTabs = openTabs.filter(t => t.id !== fileId);
    setOpenTabs(newTabs);
    
    if (activeFile?.id === fileId && newTabs.length > 0) {
      handleFileSelect(newTabs[newTabs.length - 1]);
    } else if (newTabs.length === 0) {
      setActiveFile(null);
      setEditorContent('');
    }
  };
  
  // Handle zip upload - can create new project or add to existing
  const handleZipUpload = async (file: File, createNewProjectFromZip: boolean = false) => {
    setIsUploadingZip(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post('/vibe/upload-zip', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      const uploadedFiles: VibeFile[] = response.data.files.map((f: any) => ({
        id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: f.name,
        path: f.path,
        content: f.content,
        language: getLanguageFromPath(f.path),
        isModified: false,
      }));
      
      if (createNewProjectFromZip) {
        // Create a new project with the zip contents
        const projectName = file.name.replace(/\.zip$/i, '');
        createNewProject(projectName, uploadedFiles);
      } else {
        // Add to current project
        setFiles([...files, ...uploadedFiles]);
        
        // Open first file
        if (uploadedFiles.length > 0) {
          handleFileSelect(uploadedFiles[0]);
        }
      }
      
      // Log to Agents.md
      updateAgentsMd(`## Zip Upload: ${file.name}\n- Files: ${uploadedFiles.length}\n- Time: ${new Date().toISOString()}\n\n`);
      
    } catch (error) {
      console.error('Zip upload error:', error);
    } finally {
      setIsUploadingZip(false);
    }
  };
  
  // Handle zip upload that creates a new project
  const handleZipUploadAsNewProject = async (file: File) => {
    await handleZipUpload(file, true);
  };
  
  // Download project as zip
  const handleDownload = async () => {
    if (files.length === 0) return;
    
    setIsDownloading(true);
    
    try {
      const zip = new JSZip();
      
      // Add all files to zip
      for (const file of files) {
        // Skip .gitkeep files (folder placeholders)
        if (file.name === '.gitkeep') continue;
        zip.file(file.path, file.content);
      }
      
      // Generate zip
      const blob = await zip.generateAsync({ type: 'blob' });
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project?.name || 'vibecode-project'}-${Date.now()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Download error:', error);
    } finally {
      setIsDownloading(false);
    }
  };
  
  // Get language from file path
  const getLanguageFromPath = (path: string): string => {
    const ext = path.split('.').pop()?.toLowerCase() || '';
    const langMap: Record<string, string> = {
      // TypeScript/JavaScript
      'ts': 'typescript',
      'tsx': 'typescript',
      'js': 'javascript',
      'jsx': 'javascript',
      'mjs': 'javascript',
      'cjs': 'javascript',
      // Python
      'py': 'python',
      'pyw': 'python',
      'pyx': 'python',
      // Rust
      'rs': 'rust',
      // Go
      'go': 'go',
      // Java/Kotlin
      'java': 'java',
      'kt': 'kotlin',
      'kts': 'kotlin',
      // C/C++
      'cpp': 'cpp',
      'cc': 'cpp',
      'cxx': 'cpp',
      'c': 'c',
      'h': 'c',
      'hpp': 'cpp',
      'hxx': 'cpp',
      // C#
      'cs': 'csharp',
      // Ruby
      'rb': 'ruby',
      'erb': 'ruby',
      // PHP
      'php': 'php',
      // Swift
      'swift': 'swift',
      // Web
      'css': 'css',
      'scss': 'scss',
      'sass': 'scss',
      'less': 'less',
      'html': 'html',
      'htm': 'html',
      'vue': 'vue',
      'svelte': 'svelte',
      // Data formats
      'json': 'json',
      'yaml': 'yaml',
      'yml': 'yaml',
      'toml': 'toml',
      'xml': 'xml',
      'csv': 'csv',
      // Docs
      'md': 'markdown',
      'markdown': 'markdown',
      'rst': 'restructuredtext',
      'txt': 'plaintext',
      // Database
      'sql': 'sql',
      'psql': 'sql',
      // Shell
      'sh': 'shell',
      'bash': 'shell',
      'zsh': 'shell',
      'fish': 'shell',
      'ps1': 'powershell',
      // Config
      'dockerfile': 'dockerfile',
      'makefile': 'makefile',
      'cmake': 'cmake',
      'env': 'dotenv',
      'gitignore': 'gitignore',
      // Other
      'r': 'r',
      'lua': 'lua',
      'pl': 'perl',
      'scala': 'scala',
      'hs': 'haskell',
      'ex': 'elixir',
      'exs': 'elixir',
      'erl': 'erlang',
      'clj': 'clojure',
      'ml': 'ocaml',
      'fs': 'fsharp',
      'dart': 'dart',
      'zig': 'zig',
      'nim': 'nim',
      'v': 'v',
      'sol': 'solidity',
      'tf': 'terraform',
      'proto': 'protobuf',
      'graphql': 'graphql',
      'gql': 'graphql',
    };
    return langMap[ext] || 'plaintext';
  };
  
  // Format code
  const handleFormat = async () => {
    if (!activeFile) return;
    
    try {
      const response = await api.post('/vibe/format', {
        content: editorContent,
        language: activeFile.language,
      });
      
      if (response.data.formatted) {
        handleContentChange(response.data.formatted);
      }
    } catch (error) {
      console.error('Format error:', error);
    }
  };
  
  // Lint code
  const handleLint = async () => {
    if (!activeFile) return;
    
    try {
      const response = await api.post('/vibe/lint', {
        content: editorContent,
        language: activeFile.language,
        file_path: activeFile.path,
      });
      
      setLintErrors(response.data.errors || []);
    } catch (error) {
      console.error('Lint error:', error);
    }
  };
  
  // Update Agents.md
  const updateAgentsMd = (entry: string) => {
    const newContent = agentsMd + entry;
    setAgentsMd(newContent);
    
    // Update Agents.md file if it exists
    const agentsFile = files.find(f => f.name === 'Agents.md');
    if (agentsFile) {
      const updatedFile = { ...agentsFile, content: newContent, isModified: true };
      setFiles(files.map(f => f.id === agentsFile.id ? updatedFile : f));
    }
  };
  
  // Send chat message
  const handleSendMessage = async (message: string) => {
    if (!message.trim() || isStreaming) return;
    
    const userMessage = {
      id: `msg-${Date.now()}`,
      role: 'user' as const,
      content: message,
      timestamp: new Date().toISOString(),
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);
    
    // Determine if this is an agentic task
    const isAgenticTask = detectAgenticTask(message);
    
    if (isAgenticTask) {
      await handleAgenticTask(message);
    } else {
      await handleChatResponse(message);
    }
    
    setIsStreaming(false);
  };
  
  // Detect if message requires agentic task
  const detectAgenticTask = (message: string): boolean => {
    const agenticPatterns = [
      /create\s+(a\s+)?(\w+\s+)?(file|function|class|component|module|app|application|program)/i,
      /write\s+(a\s+)?(\w+\s+)?(file|function|class|component|module|code)/i,
      /generate\s+(a\s+)?(\w+\s+)?(file|function|class|component|module|code)/i,
      /build\s+(a\s+)?(\w+\s+)?(app|application|program|feature)/i,
      /implement\s+/i,
      /refactor\s+/i,
      /fix\s+(the\s+)?(bug|error|issue)/i,
      /add\s+(a\s+)?(\w+\s+)?(feature|function|method)/i,
    ];
    
    return agenticPatterns.some(pattern => pattern.test(message));
  };
  
  // Handle agentic task
  const handleAgenticTask = async (message: string) => {
    const task: AgentTask = {
      id: `task-${Date.now()}`,
      type: 'create',
      status: 'pending',
      description: message,
      affectedFiles: [],
    };
    
    setCurrentTask(task);
    setAgentState('analyzing');
    
    try {
      // Phase 1: Analyze
      const analysisResponse = await api.post('/vibe/agent/analyze', {
        message,
        files: files.map(f => ({ path: f.path, content: f.content })),
        model: selectedModel,
        assistant_id: selectedAssistantId,
      });
      
      setAgentState('planning');
      
      // Phase 2: Plan
      const planResponse = await api.post('/vibe/agent/plan', {
        analysis: analysisResponse.data,
        message,
        model: selectedModel,
        assistant_id: selectedAssistantId,
      });
      
      setAgentState('generating');
      
      // Phase 3: Generate code
      const generateResponse = await api.post('/vibe/agent/generate', {
        plan: planResponse.data,
        files: files.map(f => ({ path: f.path, content: f.content })),
        model: selectedModel,
        assistant_id: selectedAssistantId,
      });
      
      setAgentState('reviewing');
      
      // Phase 4: Review
      const reviewResponse = await api.post('/vibe/agent/review', {
        generated: generateResponse.data,
        model: selectedModel,
        assistant_id: selectedAssistantId,
      });
      
      setAgentState('applying');
      
      // Phase 5: Apply changes
      const changes = reviewResponse.data.approved_changes || generateResponse.data.changes;
      
      for (const change of changes) {
        if (change.type === 'create') {
          handleCreateFile(change.path, getLanguageFromPath(change.path));
          // Set content after creation
          setTimeout(() => {
            const newFile = files.find(f => f.path === change.path);
            if (newFile) {
              const updatedFile = { ...newFile, content: change.content };
              setFiles(prev => prev.map(f => f.path === change.path ? updatedFile : f));
            }
          }, 100);
        } else if (change.type === 'modify') {
          const file = files.find(f => f.path === change.path);
          if (file) {
            const updatedFile = { ...file, content: change.content, isModified: true };
            setFiles(prev => prev.map(f => f.id === file.id ? updatedFile : f));
          }
        }
        
        task.affectedFiles.push(change.path);
      }
      
      setAgentState('testing');
      
      // Phase 6: Test/Lint
      await handleLint();
      
      setAgentState('complete');
      task.status = 'completed';
      task.result = `Successfully applied ${changes.length} changes`;
      
      // Add assistant message
      const assistantMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant' as const,
        content: `✅ Task completed!\n\n${task.result}\n\nAffected files:\n${task.affectedFiles.map(f => `- ${f}`).join('\n')}`,
        timestamp: new Date().toISOString(),
      };
      setChatMessages(prev => [...prev, assistantMessage]);
      
      // Log to Agents.md
      updateAgentsMd(`## Task: ${message}\n- Status: Completed\n- Files: ${task.affectedFiles.join(', ')}\n- Time: ${new Date().toISOString()}\n\n`);
      
    } catch (error: any) {
      setAgentState('error');
      task.status = 'failed';
      task.error = error.message;
      
      const errorMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant' as const,
        content: `❌ Task failed: ${error.message}`,
        timestamp: new Date().toISOString(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
      
      // Log error to Agents.md
      updateAgentsMd(`## Error: ${message}\n- Error: ${error.message}\n- Time: ${new Date().toISOString()}\n\n`);
    } finally {
      setTaskHistory(prev => [...prev, task]);
      setCurrentTask(null);
      setTimeout(() => setAgentState('idle'), 2000);
    }
  };
  
  // Handle regular chat response
  const handleChatResponse = async (message: string) => {
    try {
      const response = await api.post('/vibe/chat', {
        message,
        files: files.map(f => ({ path: f.path, content: f.content })),
        active_file: activeFile ? { path: activeFile.path, content: activeFile.content } : null,
        cursor_position: cursorPosition,
        model: selectedModel,
        assistant_id: selectedAssistantId,
      });
      
      const assistantMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant' as const,
        content: response.data.response,
        timestamp: new Date().toISOString(),
      };
      setChatMessages(prev => [...prev, assistantMessage]);
      
    } catch (error: any) {
      const errorMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant' as const,
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
    }
  };
  
  return (
    <div className="flex h-full bg-[var(--color-background)]">
      {/* Left Sidebar - File Explorer */}
      <div className="w-64 flex flex-col border-r border-[var(--color-border)]">
        {/* Upper - File Explorer */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <FileExplorer
            files={files}
            activeFile={activeFile}
            onFileSelect={handleFileSelect}
            onCreateFile={handleCreateFile}
            onCreateFolder={handleCreateFolder}
            onDeleteFile={handleDeleteFile}
            onDeleteFiles={handleDeleteFiles}
            onMoveFiles={handleMoveFiles}
            onCopyFiles={handleCopyFiles}
            onZipUpload={handleZipUpload}
            onZipUploadAsNewProject={handleZipUploadAsNewProject}
            onDownload={handleDownload}
            isUploadingZip={isUploadingZip}
            isDownloading={isDownloading}
          />
        </div>
        
        {/* Lower - Project & Settings */}
        <div className="h-72 border-t border-[var(--color-border)] p-3 overflow-y-auto">
          {/* Project Section */}
          <div className="text-xs text-[var(--color-text-secondary)] uppercase mb-2 flex justify-between items-center">
            <span>Project</span>
            <span className="text-[var(--color-text-secondary)]">({projects.length})</span>
          </div>
          
          {/* Project selector button */}
          <button
            onClick={() => setShowProjectSelector(!showProjectSelector)}
            className="w-full flex items-center justify-between px-2 py-1.5 mb-2 text-sm rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]"
          >
            <span className="truncate flex-1 text-left">{project?.name || 'Select Project'}</span>
            <svg className={`w-4 h-4 ml-2 transition-transform ${showProjectSelector ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {/* Project dropdown */}
          {showProjectSelector && (
            <div className="mb-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded shadow-lg max-h-48 overflow-y-auto">
              {/* New project button */}
              <button
                onClick={() => {
                  setShowProjectSelector(false);
                  setShowNewProjectDialog(true);
                }}
                className="w-full px-3 py-2 text-left text-sm text-[var(--color-primary)] hover:bg-[var(--color-background)] flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Project
              </button>
              <div className="border-t border-[var(--color-border)]" />
              
              {/* Project list */}
              {projects.map(p => (
                <div
                  key={p.id}
                  className={`flex items-center justify-between px-3 py-2 text-sm cursor-pointer ${
                    p.id === project?.id 
                      ? 'bg-[var(--color-primary)]/20 text-[var(--color-primary)]' 
                      : 'text-[var(--color-text)] hover:bg-[var(--color-background)]'
                  }`}
                >
                  <span 
                    className="truncate flex-1"
                    onClick={() => loadProject(p)}
                  >
                    {p.name}
                  </span>
                  <span className="text-xs text-[var(--color-text-secondary)] mr-2">
                    {p.files.length} files
                  </span>
                  {projects.length > 1 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Delete project "${p.name}"?`)) {
                          deleteProject(p.id);
                        }
                      }}
                      className="w-5 h-5 flex items-center justify-center text-[var(--color-text-secondary)] hover:text-red-500 rounded hover:bg-[var(--color-background)]"
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          
          {/* Project actions */}
          <div className="flex gap-1 mb-3">
            <button
              onClick={() => {
                setRenameProjectName(project?.name || '');
                setShowRenameDialog(true);
              }}
              className="flex-1 px-2 py-1 text-xs rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]"
              title="Rename project"
            >
              Rename
            </button>
            <button
              onClick={() => setShowNewProjectDialog(true)}
              className="flex-1 px-2 py-1 text-xs rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]"
              title="Create new project"
            >
              + New
            </button>
          </div>
          
          <div className="text-xs text-[var(--color-text-secondary)] uppercase mb-2">Model / GPT</div>
          <select
            value={selectedAssistantId ? `gpt:${selectedAssistantId}` : selectedModel}
            onChange={(e) => {
              const value = e.target.value;
              if (value.startsWith('gpt:')) {
                const assistantId = value.replace('gpt:', '');
                // Find the assistant to get its underlying model
                const owned = myAssistants.find(a => a.assistant_id === assistantId);
                const subscribed = subscribedAssistants.find(a => a.assistant_id === assistantId);
                // For now, use default model with the assistant context
                handleModelSelect(defaultModel, assistantId);
              } else {
                handleModelSelect(value);
              }
            }}
            className="w-full px-2 py-1 text-xs rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)]"
          >
            {/* LLM Models */}
            <optgroup label="🤖 Models">
              {models.map(model => (
                <option key={model.id} value={model.id}>{model.name || model.id}</option>
              ))}
            </optgroup>
            
            {/* My Custom GPTs */}
            {myAssistants.length > 0 && (
              <optgroup label="⭐ My GPTs">
                {myAssistants.map(assistant => (
                  <option key={assistant.id} value={`gpt:${assistant.assistant_id}`}>
                    {assistant.icon || '🤖'} {assistant.name}
                  </option>
                ))}
              </optgroup>
            )}
            
            {/* Subscribed Custom GPTs */}
            {subscribedAssistants.length > 0 && (
              <optgroup label="📚 Subscribed GPTs">
                {subscribedAssistants.map(assistant => (
                  <option key={assistant.id} value={`gpt:${assistant.assistant_id}`}>
                    {assistant.icon || '🤖'} {assistant.name}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          
          {/* Show selected GPT info */}
          {selectedAssistantId && (
            <div className="mt-2 text-xs text-[var(--color-primary)] truncate">
              Using: {getSelectedDisplayName()}
            </div>
          )}
          
          <div className="mt-3 flex gap-2">
            <button
              onClick={handleFormat}
              className="flex-1 px-2 py-1 text-xs rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]"
            >
              Format
            </button>
            <button
              onClick={handleLint}
              className="flex-1 px-2 py-1 text-xs rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]"
            >
              Lint
            </button>
          </div>
          
          <button
            onClick={() => navigate('/chat')}
            className="mt-2 w-full px-2 py-1 text-xs rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-background)]"
          >
            ← Back to Chat
          </button>
        </div>
      </div>
      
      {/* Middle - Code Editor */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {/* Tabs */}
        <div className="flex items-center border-b border-[var(--color-border)] bg-[var(--color-surface)] overflow-x-auto">
          {openTabs.map(tab => (
            <div
              key={tab.id}
              onClick={() => handleFileSelect(tab)}
              className={`flex items-center gap-2 px-3 py-2 text-sm cursor-pointer border-r border-[var(--color-border)] ${
                activeFile?.id === tab.id
                  ? 'bg-[var(--color-background)] text-[var(--color-text)]'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-background)]'
              }`}
            >
              <span className={tab.isModified ? 'italic' : ''}>
                {tab.name}
                {tab.isModified && ' •'}
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); handleCloseTab(tab.id); }}
                className="w-4 h-4 rounded hover:bg-[var(--color-border)] flex items-center justify-center"
              >
                ×
              </button>
            </div>
          ))}
        </div>
        
        {/* Agent Status Bar */}
        {agentState !== 'idle' && (
          <AgentStatus state={agentState} task={currentTask} />
        )}
        
        {/* Editor */}
        <div className="flex-1 relative overflow-hidden min-h-0">
          {activeFile ? (
            <CodeEditor
              content={editorContent}
              language={activeFile.language}
              onChange={handleContentChange}
              onCursorChange={setCursorPosition}
              suggestions={suggestions}
              showSuggestions={showSuggestions}
              onAcceptSuggestion={handleAcceptSuggestion}
              onDismissSuggestions={() => setShowSuggestions(false)}
              lintErrors={lintErrors}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-[var(--color-text-secondary)]">
              <div className="text-center">
                <div className="text-4xl mb-4">🚀</div>
                <div className="text-lg font-medium mb-2">Welcome to VibeCode</div>
                <div className="text-sm">Create a new file or upload a zip to get started</div>
              </div>
            </div>
          )}
        </div>
        
        {/* Status Bar */}
        <div className="flex items-center justify-between px-3 py-1 text-xs bg-[var(--color-primary)] text-white">
          <div className="flex items-center gap-4">
            <span>VibeCode</span>
            {activeFile && (
              <>
                <span>{activeFile.language}</span>
                <span>Ln {cursorPosition.line}, Col {cursorPosition.column}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-4">
            {lintErrors.length > 0 && (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                {lintErrors.filter(e => e.severity === 'error').length} errors
              </span>
            )}
            <span>{files.length} files</span>
          </div>
        </div>
      </div>
      
      {/* Right - Chat Panel */}
      <div className="w-96 border-l border-[var(--color-border)]">
        <VibeChat
          messages={chatMessages}
          onSendMessage={handleSendMessage}
          onCreateFile={handleCreateFileWithContent}
          onInsertCode={handleInsertCode}
          isStreaming={isStreaming}
          agentState={agentState}
        />
      </div>
      
      {/* New Project Dialog */}
      {showNewProjectDialog && (
        <>
          <div 
            className="fixed inset-0 bg-black/50 z-50" 
            onClick={() => setShowNewProjectDialog(false)} 
          />
          <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-xl p-4 w-96">
            <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">New Project</h3>
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && newProjectName.trim()) {
                  createNewProject(newProjectName.trim());
                }
              }}
              placeholder="Project name"
              className="w-full px-3 py-2 mb-4 bg-[var(--color-background)] border border-[var(--color-border)] rounded text-[var(--color-text)]"
              autoFocus
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowNewProjectDialog(false);
                  setNewProjectName('');
                }}
                className="px-4 py-2 text-sm rounded bg-[var(--color-background)] text-[var(--color-text)] hover:bg-[var(--color-border)]"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (newProjectName.trim()) {
                    createNewProject(newProjectName.trim());
                  }
                }}
                disabled={!newProjectName.trim()}
                className="px-4 py-2 text-sm rounded bg-[var(--color-primary)] text-white hover:opacity-90 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </>
      )}
      
      {/* Rename Project Dialog */}
      {showRenameDialog && (
        <>
          <div 
            className="fixed inset-0 bg-black/50 z-50" 
            onClick={() => setShowRenameDialog(false)} 
          />
          <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-xl p-4 w-96">
            <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">Rename Project</h3>
            <input
              type="text"
              value={renameProjectName}
              onChange={(e) => setRenameProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && renameProjectName.trim()) {
                  renameProject();
                }
              }}
              placeholder="New name"
              className="w-full px-3 py-2 mb-4 bg-[var(--color-background)] border border-[var(--color-border)] rounded text-[var(--color-text)]"
              autoFocus
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowRenameDialog(false);
                  setRenameProjectName('');
                }}
                className="px-4 py-2 text-sm rounded bg-[var(--color-background)] text-[var(--color-text)] hover:bg-[var(--color-border)]"
              >
                Cancel
              </button>
              <button
                onClick={renameProject}
                disabled={!renameProjectName.trim()}
                className="px-4 py-2 text-sm rounded bg-[var(--color-primary)] text-white hover:opacity-90 disabled:opacity-50"
              >
                Rename
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
