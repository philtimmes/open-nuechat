/**
 * Artifact slice - Artifact and file upload management
 */
import type { Artifact, GeneratedImage, ZipUploadResult, CodeSignature } from '../../types';
import type { ArtifactSlice, SliceCreator } from './types';
import api from '../../lib/api';
import { collectChatArtifacts, extractArtifacts } from '../../lib/artifacts';

export const createArtifactSlice: SliceCreator<ArtifactSlice> = (set, get) => ({
  artifacts: [],
  selectedArtifact: null,
  showArtifacts: false,
  uploadedArtifacts: [],
  zipUploadResult: null,
  zipContext: null,
  generatedImages: {},
  pendingImageContext: null,

  setSelectedArtifact: (artifact: Artifact | null) => {
    set({ selectedArtifact: artifact, showArtifacts: artifact !== null });
  },

  setShowArtifacts: (show: boolean) => {
    set({ showArtifacts: show, selectedArtifact: show ? get().selectedArtifact : null });
  },

  collectAllArtifacts: () => {
    const { messages } = get();
    const artifactMap = collectChatArtifacts(messages);
    return Array.from(artifactMap.values());
  },

  addUploadedArtifacts: (artifacts: Artifact[]) => {
    console.log('[artifactSlice] addUploadedArtifacts called with', artifacts.length, 'artifacts');
    console.log('[artifactSlice] Current state before:', {
      uploadedArtifacts: get().uploadedArtifacts.length,
      showArtifacts: get().showArtifacts,
    });
    set((state) => ({
      uploadedArtifacts: [...state.uploadedArtifacts, ...artifacts],
      // Auto-show artifacts panel when files are uploaded
      showArtifacts: artifacts.length > 0 || state.showArtifacts,
      // Auto-select the first new artifact if nothing is selected
      selectedArtifact: state.selectedArtifact || (artifacts.length > 0 ? artifacts[0] : null),
    }));
    console.log('[artifactSlice] After update:', {
      uploadedArtifacts: get().uploadedArtifacts.length,
      showArtifacts: get().showArtifacts,
      selectedArtifact: get().selectedArtifact?.filename,
    });
  },

  setZipUploadResult: (result) => {
    set({ zipUploadResult: result });
    if (result?.artifacts) {
      set({ uploadedArtifacts: result.artifacts });
    }
  },

  fetchUploadedData: async (chatId: string) => {
    try {
      const response = await api.get(`/chats/${chatId}/uploaded-files`);
      const { artifacts, archive } = response.data;
      
      if (artifacts && artifacts.length > 0) {
        set({ uploadedArtifacts: artifacts });
        
        // Reconstruct zipUploadResult from archive info
        if (archive) {
          const zipResult: Partial<ZipUploadResult> = {
            filename: archive.filename,
            total_files: archive.total_files,
            total_size: archive.total_size,
            languages: archive.languages || {},
            artifacts: artifacts,
            llm_manifest: archive.llm_manifest,
            summary: archive.summary || '',
            // Reconstruct signature_index from artifacts
            signature_index: artifacts.reduce((acc: Record<string, CodeSignature[]>, art: Artifact) => {
              if (art.signatures && art.signatures.length > 0 && art.filename) {
                acc[art.filename] = art.signatures;
              }
              return acc;
            }, {}),
          };
          set({ 
            zipUploadResult: zipResult,
            zipContext: archive.llm_manifest || null,
          });
        }
        
        // Merge uploaded artifacts with message-extracted artifacts
        set((state) => ({
          artifacts: [...state.artifacts.filter(a => a.source !== 'upload'), ...artifacts],
        }));
      }
    } catch (err) {
      // No uploaded files yet, that's OK
      console.log('No uploaded files for chat');
    }
  },

  setZipContext: (context: string | null) => {
    set({ zipContext: context });
  },

  setGeneratedImage: (messageId: string, image: GeneratedImage) => {
    set((state) => ({
      generatedImages: { ...state.generatedImages, [messageId]: image },
    }));
  },

  setPendingImageContext: (image: GeneratedImage | null) => {
    set({ pendingImageContext: image });
  },

  setArtifacts: (artifacts: Artifact[]) => {
    set({ artifacts });
  },

  updateArtifact: (key: string, artifact: Artifact) => {
    set((state) => {
      const normalizedKey = key.replace(/^\.?\//, '');
      
      // Helper to check if an artifact matches the key
      const matches = (a: Artifact) => {
        const artKey = (a.filename || a.title || '').replace(/^\.?\//, '');
        return artKey === normalizedKey || artKey.split('/').pop() === normalizedKey.split('/').pop();
      };
      
      // NC-0.8.0.12: Version tracking - find highest existing version for this file
      const allMatching = [
        ...state.artifacts.filter(matches),
        ...state.uploadedArtifacts.filter(matches),
      ];
      const maxVersion = allMatching.reduce((max, a) => Math.max(max, a.version || 1), 0);
      const newVersion = maxVersion + 1;
      
      // Create versioned artifact
      const versionedArtifact: Artifact = {
        ...artifact,
        version: newVersion,
        id: `${artifact.id || key}_v${newVersion}`,
        title: artifact.title || normalizedKey.split('/').pop() || normalizedKey,
      };
      
      // Mark old versions in uploadedArtifacts (don't remove - keep history)
      // Add the new version to the list
      const newUploadedArtifacts = [...state.uploadedArtifacts, versionedArtifact];
      
      // For artifacts (message-extracted), replace in-place to avoid clutter
      let foundInArtifacts = state.artifacts.some(matches);
      const newArtifacts = state.artifacts.map((a) => {
        if (matches(a)) {
          return { ...a, ...versionedArtifact };
        }
        return a;
      });
      
      // If not found in artifacts at all, add it there too
      if (!foundInArtifacts && !allMatching.length) {
        newArtifacts.push(versionedArtifact);
      }
      
      return { 
        artifacts: newArtifacts,
        uploadedArtifacts: newUploadedArtifacts,
        // Auto-select the updated file
        selectedArtifact: versionedArtifact,
        showArtifacts: true,
      };
    });
  },

  // NC-0.8.0.12: Revert to an older version by "touching" it (updating created_at to now)
  revertArtifact: (artifactId: string) => {
    const now = new Date().toISOString();
    set((state) => {
      const newUploaded = state.uploadedArtifacts.map((a) =>
        a.id === artifactId ? { ...a, created_at: now } : a
      );
      const newArtifacts = state.artifacts.map((a) =>
        a.id === artifactId ? { ...a, created_at: now } : a
      );
      
      // Find the touched artifact to auto-select it
      const reverted = newUploaded.find(a => a.id === artifactId)
        || newArtifacts.find(a => a.id === artifactId);
      
      return {
        uploadedArtifacts: newUploaded,
        artifacts: newArtifacts,
        selectedArtifact: reverted || state.selectedArtifact,
      };
    });
    
    // Also update backend session storage with the reverted content
    const state = get();
    const reverted = [...state.uploadedArtifacts, ...state.artifacts].find(a => a.id === artifactId);
    if (reverted?.filename && state.currentChat?.id) {
      api.put(`/chats/${state.currentChat.id}/revert-file`, {
        filename: reverted.filename,
        content: reverted.content,
      }).catch(err => console.warn('[revertArtifact] Backend sync failed:', err));
    }
  },
});
