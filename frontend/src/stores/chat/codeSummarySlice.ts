/**
 * Code summary slice - Code tracking for LLM context
 */
import type { CodeSummary, FileChange, SignatureWarning } from '../../types';
import type { CodeSummarySlice, SliceCreator } from './types';
import api from '../../lib/api';

export const createCodeSummarySlice: SliceCreator<CodeSummarySlice> = (set, get) => ({
  codeSummary: null,
  showSummary: false,

  setShowSummary: (show: boolean) => {
    set({ showSummary: show });
  },

  updateCodeSummary: (files: FileChange[], warnings: SignatureWarning[] = []) => {
    const { currentChat, codeSummary } = get();
    if (!currentChat) return;
    
    const newSummary: CodeSummary = {
      id: codeSummary?.id || `summary_${Date.now()}`,
      chat_id: currentChat.id,
      files,
      warnings,
      last_updated: new Date().toISOString(),
      auto_generated: true,
    };
    set({ codeSummary: newSummary });
  },

  addFileToSummary: (file: FileChange) => {
    const { currentChat, codeSummary } = get();
    if (!currentChat) return;
    
    const existingFiles = codeSummary?.files || [];
    // Update or add file
    const fileIndex = existingFiles.findIndex((f) => f.path === file.path);
    const updatedFiles = fileIndex >= 0
      ? existingFiles.map((f, i) => i === fileIndex ? file : f)
      : [...existingFiles, file];
    
    const newSummary: CodeSummary = {
      id: codeSummary?.id || `summary_${Date.now()}`,
      chat_id: currentChat.id,
      files: updatedFiles,
      warnings: codeSummary?.warnings || [],
      last_updated: new Date().toISOString(),
      auto_generated: true,
    };
    set({ codeSummary: newSummary });
  },

  addWarning: (warning: SignatureWarning) => {
    const { currentChat, codeSummary } = get();
    if (!currentChat) return;
    
    const existingWarnings = codeSummary?.warnings || [];
    // Avoid duplicate warnings
    const exists = existingWarnings.some(
      (w) => w.type === warning.type && w.file === warning.file && w.signature === warning.signature
    );
    if (exists) return;
    
    const newSummary: CodeSummary = {
      id: codeSummary?.id || `summary_${Date.now()}`,
      chat_id: currentChat.id,
      files: codeSummary?.files || [],
      warnings: [...existingWarnings, warning],
      last_updated: new Date().toISOString(),
      auto_generated: true,
    };
    set({ codeSummary: newSummary });
  },

  clearSummary: () => {
    set({ codeSummary: null });
  },

  fetchCodeSummary: async (chatId: string) => {
    try {
      const response = await api.get(`/chats/${chatId}/summary`);
      if (response.data) {
        set({ codeSummary: response.data });
      }
    } catch (error) {
      // Code summary may not exist - that's OK
      console.debug('No code summary found for chat:', chatId);
    }
  },

  saveCodeSummary: async () => {
    const { currentChat, codeSummary } = get();
    if (!currentChat || !codeSummary) return;
    
    try {
      await api.put(`/chats/${currentChat.id}/summary`, {
        files: codeSummary.files,
        warnings: codeSummary.warnings,
      });
      console.log('[CODE_SUMMARY] Saved to backend');
    } catch (error) {
      console.error('Failed to save code summary:', error);
    }
  },
});
