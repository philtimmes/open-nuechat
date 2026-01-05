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
      
      // First check if it's in artifacts
      let foundInArtifacts = state.artifacts.some(matches);
      let foundInUploaded = state.uploadedArtifacts.some(matches);
      
      // Update artifacts array
      const newArtifacts = state.artifacts.map((a) => {
        if (matches(a)) {
          return { ...a, ...artifact };
        }
        return a;
      });
      
      // Update uploadedArtifacts array
      const newUploadedArtifacts = state.uploadedArtifacts.map((a) => {
        if (matches(a)) {
          return { ...a, ...artifact };
        }
        return a;
      });
      
      // If not found in either, add to artifacts
      if (!foundInArtifacts && !foundInUploaded) {
        newArtifacts.push(artifact);
      }
      
      return { 
        artifacts: newArtifacts,
        uploadedArtifacts: newUploadedArtifacts,
      };
    });
  },
});
