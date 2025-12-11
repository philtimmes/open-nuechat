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
});
