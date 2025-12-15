import { useState, useRef, useCallback, useEffect, type RefObject } from 'react';
import type { Artifact } from '../types';
import { processFilesToArtifacts, readFileAsBase64 } from '../lib/fileProcessor';

interface ProcessedAttachment {
  type: 'image' | 'file';
  filename: string;
  mime_type: string;
  data?: string;  // base64 for images
  content?: string;  // text content for files
}

interface ChatInputProps {
  onSend: (content: string, attachments?: ProcessedAttachment[]) => void;
  onStop?: () => void;
  onZipUpload?: (file: File) => Promise<void>;
  onFilesUploaded?: (artifacts: Artifact[]) => void;
  onVoiceModeToggle?: () => void;
  disabled?: boolean;
  isStreaming?: boolean;
  isVoiceMode?: boolean;
  isListening?: boolean;
  placeholder?: string;
  isUploadingZip?: boolean;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
}

export default function ChatInput({ 
  onSend, 
  onStop, 
  onZipUpload, 
  onFilesUploaded,
  onVoiceModeToggle,
  disabled, 
  isStreaming, 
  isVoiceMode,
  isListening,
  placeholder, 
  isUploadingZip,
  inputRef 
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [processedAttachments, setProcessedAttachments] = useState<ProcessedAttachment[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessingFiles, setIsProcessingFiles] = useState(false);
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const textareaRef = inputRef || internalRef;
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleSubmit = () => {
    if (!message.trim() && processedAttachments.length === 0) return;
    if (disabled) return;
    
    onSend(message.trim(), processedAttachments.length > 0 ? processedAttachments : undefined);
    setMessage('');
    setAttachments([]);
    setProcessedAttachments([]);
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  };
  
  const handleFileSelect = async (files: FileList | null) => {
    if (!files) return;
    
    const imageFiles: File[] = [];
    const textFiles: File[] = [];
    
    for (const file of Array.from(files)) {
      // Limit file size to 10MB (50MB for zip)
      const maxSize = file.name.toLowerCase().endsWith('.zip') ? 50 * 1024 * 1024 : 10 * 1024 * 1024;
      if (file.size > maxSize) {
        alert(`File ${file.name} is too large. Maximum size is ${file.name.toLowerCase().endsWith('.zip') ? '50MB' : '10MB'}.`);
        continue;
      }
      
      // Handle zip files specially
      if (file.name.toLowerCase().endsWith('.zip') && onZipUpload) {
        onZipUpload(file);
        continue;
      }
      
      // Separate images from other files
      if (file.type.startsWith('image/')) {
        imageFiles.push(file);
      } else {
        textFiles.push(file);
      }
    }
    
    // Process files
    setIsProcessingFiles(true);
    try {
      const newProcessed: ProcessedAttachment[] = [];
      
      // Process images as base64 attachments
      for (const img of imageFiles) {
        const base64 = await readFileAsBase64(img);
        newProcessed.push({
          type: 'image',
          filename: img.name,
          mime_type: img.type,
          data: base64,
        });
      }
      
      // Process text files into artifacts
      if (textFiles.length > 0) {
        console.log(`[ChatInput v2] Processing ${textFiles.length} text files:`, textFiles.map(f => `${f.name} (${f.type})`));
        const artifacts = await processFilesToArtifacts(textFiles);
        console.log(`[ChatInput v2] Created ${artifacts.length} artifacts`);
        
        // Notify parent of new artifacts
        if (onFilesUploaded && artifacts.length > 0) {
          console.log(`[ChatInput v2] Notifying parent of ${artifacts.length} artifacts`);
          onFilesUploaded(artifacts);
        }
        
        // Add file content as attachments for LLM context
        for (const artifact of artifacts) {
          console.log(`[ChatInput v2] Adding artifact to processed attachments: ${artifact.filename}, ${artifact.content.length} chars`);
          newProcessed.push({
            type: 'file',
            filename: artifact.filename || artifact.title,
            mime_type: `text/${artifact.language || 'plain'}`,
            content: artifact.content,
          });
        }
      }
      
      // Update state
      console.log(`[ChatInput v2] Setting ${newProcessed.length} processed attachments`);
      setAttachments((prev) => [...prev, ...imageFiles, ...textFiles]);
      setProcessedAttachments((prev) => [...prev, ...newProcessed]);
    } catch (error) {
      console.error('Error processing files:', error);
      alert('Failed to process some files');
    } finally {
      setIsProcessingFiles(false);
    }
  };
  
  const removeAttachment = (index: number) => {
    const removedFile = attachments[index];
    setAttachments((prev) => prev.filter((_, i) => i !== index));
    // Also remove from processed attachments by matching filename
    setProcessedAttachments((prev) => prev.filter(p => p.filename !== removedFile?.name));
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
    handleFileSelect(e.dataTransfer.files);
  }, []);
  
  // Paste handler for files
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    const files: File[] = [];
    
    for (const item of Array.from(items)) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          files.push(file);
        }
      }
    }
    
    if (files.length > 0) {
      e.preventDefault();
      // Create a FileList-like object
      const dt = new DataTransfer();
      files.forEach(f => dt.items.add(f));
      handleFileSelect(dt.files);
    }
  }, []);
  
  return (
    <div
      className={`relative ${isDragging ? 'ring-2 ring-[var(--color-primary)] rounded-xl' : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-[var(--color-primary)]/10 border-2 border-dashed border-[var(--color-primary)] rounded-xl flex items-center justify-center z-10">
          <div className="text-[var(--color-primary)] font-medium text-base md:text-sm">
            Drop files here
          </div>
        </div>
      )}
      
      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2 px-1">
          {attachments.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-3 py-2 md:py-1.5 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)]"
            >
              {file.type.startsWith('image/') ? (
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="w-10 h-10 md:w-8 md:h-8 rounded object-cover"
                />
              ) : (
                <svg className="w-6 h-6 md:w-5 md:h-5 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              )}
              <span className="text-base md:text-sm text-[var(--color-text)] truncate max-w-[120px]">
                {file.name}
              </span>
              <button
                onClick={() => removeAttachment(index)}
                className="p-1 text-[var(--color-text-secondary)] hover:text-[var(--color-error)] transition-colors"
              >
                <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
      
      {/* Input container */}
      <div className="flex items-end gap-2 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] p-2 md:p-2">
        {/* File upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isUploadingZip}
          className="p-3 md:p-2 rounded-lg text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {isUploadingZip ? (
            <svg className="w-6 h-6 md:w-5 md:h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : (
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
            </svg>
          )}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={(e) => handleFileSelect(e.target.files)}
          className="hidden"
          accept="image/*,.pdf,.docx,.doc,.xlsx,.xls,.rtf,.txt,.md,.json,.csv,.zip,.py,.js,.jsx,.ts,.tsx,.java,.go,.rs,.rb,.php,.c,.cpp,.h,.hpp,.cs,.swift,.kt,.scala,.lua,.r,.sh,.bash,.yaml,.yml,.toml,.xml,.html,.css,.scss,.sass,.sql,.graphql,.proto,.asm,.s"
        />
        
        {/* Talk to Me (Voice Mode) button */}
        {onVoiceModeToggle && (
          <button
            onClick={onVoiceModeToggle}
            disabled={disabled}
            className={`p-3 md:p-2 rounded-lg transition-all ${
              isVoiceMode
                ? 'bg-[var(--color-primary)] text-white'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-zinc-700/30'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
            title={isVoiceMode ? "Stop voice mode (or say STOP)" : "Talk to Me - Voice conversation mode"}
          >
            {isListening ? (
              <svg className="w-6 h-6 md:w-5 md:h-5 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
            ) : isVoiceMode ? (
              <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
            ) : (
              <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            )}
          </button>
        )}
        
        {/* Text input */}
        <textarea
          ref={textareaRef}
          data-chat-input
          value={message}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled}
          placeholder={placeholder || 'Type your message...'}
          rows={1}
          className="flex-1 resize-none bg-transparent text-base md:text-sm text-[var(--color-text)] placeholder-zinc-500/50 focus:outline-none py-3 md:py-2 px-2 max-h-[200px]"
        />
        
        {/* Send or Stop button */}
        {isStreaming ? (
          <button
            onClick={onStop}
            className="p-3 md:p-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-all"
            title="Stop generating"
          >
            <svg className="w-6 h-6 md:w-5 md:h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="1" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={disabled || isProcessingFiles || (!message.trim() && attachments.length === 0)}
            className="p-3 md:p-2 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isProcessingFiles ? (
              <svg className="w-6 h-6 md:w-5 md:h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-6 h-6 md:w-5 md:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        )}
      </div>
      
      {/* Helper text - hide on mobile */}
      <p className="hidden md:block text-xs text-zinc-500/60 mt-2 px-1">
        Press Enter to send, Shift+Enter for new line. Drop files or click the attachment button. Zip files extract to artifacts.
      </p>
    </div>
  );
}
