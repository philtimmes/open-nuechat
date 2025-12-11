import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../lib/api';

export interface Voice {
  id: string;
  name: string;
  lang: string;
  gender: string;
  category?: 'natural' | 'local'; // natural = Kokoro, local = OS/Browser
}

export interface LocalVoice {
  id: string;
  name: string;
  lang: string;
  voiceURI: string;
  localService: boolean;
}

export type TTSMethod = 'natural' | 'local';

export interface VoiceSettings {
  // TTS Settings
  ttsEnabled: boolean;
  ttsMethod: TTSMethod; // 'natural' (Kokoro server) or 'local' (Web Speech API)
  selectedVoice: string;
  selectedLocalVoice: string;
  autoReadResponses: boolean;
  
  // STT Settings
  sttEnabled: boolean;
  selectedLanguage: string;
  
  // Talk to Me Mode
  talkToMeMode: boolean;
  
  // Available options (fetched from API)
  availableVoices: Voice[];
  localVoices: LocalVoice[];
  availableLanguages: { code: string; name: string }[];
  
  // Service status
  ttsAvailable: boolean;
  sttAvailable: boolean;
  localTtsAvailable: boolean;
  
  // Loading states
  isLoadingVoices: boolean;
  isLoadingLanguages: boolean;
  
  // Actions
  setTtsEnabled: (enabled: boolean) => void;
  setTtsMethod: (method: TTSMethod) => void;
  setSelectedVoice: (voiceId: string) => void;
  setSelectedLocalVoice: (voiceId: string) => void;
  setAutoReadResponses: (enabled: boolean) => void;
  setSttEnabled: (enabled: boolean) => void;
  setSelectedLanguage: (langCode: string) => void;
  setTalkToMeMode: (enabled: boolean) => void;
  fetchVoices: () => Promise<void>;
  fetchLocalVoices: () => void;
  fetchLanguages: () => Promise<void>;
  checkServiceStatus: () => Promise<void>;
}

// Detect platform/browser for grouping
function detectPlatform(): string {
  const ua = navigator.userAgent;
  if (/iPad|iPhone|iPod/.test(ua)) return 'iOS';
  if (/Mac/.test(ua) && 'ontouchend' in document) return 'iOS'; // iPad with desktop UA
  if (/Mac/.test(ua)) return 'macOS';
  if (/Android/.test(ua)) return 'Android';
  if (/Windows/.test(ua)) return 'Windows';
  if (/Linux/.test(ua)) return 'Linux';
  return 'Browser';
}

export const useVoiceStore = create<VoiceSettings>()(
  persist(
    (set, get) => ({
      // Defaults
      ttsEnabled: true,
      ttsMethod: 'natural',
      selectedVoice: 'af_heart',
      selectedLocalVoice: '',
      autoReadResponses: false,
      sttEnabled: true,
      selectedLanguage: 'en',
      talkToMeMode: false,
      availableVoices: [],
      localVoices: [],
      availableLanguages: [],
      ttsAvailable: false,
      sttAvailable: false,
      localTtsAvailable: false,
      isLoadingVoices: false,
      isLoadingLanguages: false,
      
      setTtsEnabled: (enabled) => set({ ttsEnabled: enabled }),
      setTtsMethod: (method) => set({ ttsMethod: method }),
      setSelectedVoice: (voiceId) => set({ selectedVoice: voiceId }),
      setSelectedLocalVoice: (voiceId) => set({ selectedLocalVoice: voiceId }),
      setAutoReadResponses: (enabled) => set({ autoReadResponses: enabled }),
      setSttEnabled: (enabled) => set({ sttEnabled: enabled }),
      setSelectedLanguage: (langCode) => set({ selectedLanguage: langCode }),
      setTalkToMeMode: (enabled) => set({ talkToMeMode: enabled }),
      
      fetchVoices: async () => {
        set({ isLoadingVoices: true });
        try {
          const res = await api.get('/tts/voices');
          // Mark Kokoro voices as 'natural'
          const voices = (res.data || []).map((v: Voice) => ({
            ...v,
            category: 'natural' as const
          }));
          set({ availableVoices: voices });
        } catch (err) {
          console.error('Failed to fetch voices:', err);
        } finally {
          set({ isLoadingVoices: false });
        }
      },
      
      fetchLocalVoices: () => {
        // Web Speech API
        if (!('speechSynthesis' in window)) {
          set({ localTtsAvailable: false, localVoices: [] });
          return;
        }
        
        const loadVoices = () => {
          const voices = window.speechSynthesis.getVoices();
          if (voices.length === 0) return;
          
          const platform = detectPlatform();
          
          const localVoices: LocalVoice[] = voices.map(v => ({
            id: `local:${v.voiceURI}`,
            name: `${v.name}${v.localService ? '' : ' (Network)'}`,
            lang: v.lang,
            voiceURI: v.voiceURI,
            localService: v.localService,
          }));
          
          // Sort: prioritize local services, then by name
          localVoices.sort((a, b) => {
            if (a.localService !== b.localService) {
              return a.localService ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
          });
          
          set({ 
            localVoices, 
            localTtsAvailable: localVoices.length > 0,
            // Auto-select first local voice if none selected
            selectedLocalVoice: get().selectedLocalVoice || (localVoices[0]?.id || '')
          });
          
          console.log(`Loaded ${localVoices.length} local voices for ${platform}`);
        };
        
        // Voices may load async (especially on Chrome)
        loadVoices();
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
          window.speechSynthesis.onvoiceschanged = loadVoices;
        }
      },
      
      fetchLanguages: async () => {
        set({ isLoadingLanguages: true });
        try {
          const res = await api.get('/stt/languages');
          set({ availableLanguages: res.data || [] });
        } catch (err) {
          console.error('Failed to fetch languages:', err);
        } finally {
          set({ isLoadingLanguages: false });
        }
      },
      
      checkServiceStatus: async () => {
        try {
          const [ttsRes, sttRes] = await Promise.allSettled([
            api.get('/tts/status'),
            api.get('/stt/status')
          ]);
          
          set({
            ttsAvailable: ttsRes.status === 'fulfilled' && ttsRes.value.data?.available,
            sttAvailable: sttRes.status === 'fulfilled' && sttRes.value.data?.available
          });
          
          // Also check local TTS
          get().fetchLocalVoices();
        } catch (err) {
          console.error('Failed to check service status:', err);
        }
      },
    }),
    {
      name: 'voice-settings',
      partialize: (state) => ({
        ttsEnabled: state.ttsEnabled,
        ttsMethod: state.ttsMethod,
        selectedVoice: state.selectedVoice,
        selectedLocalVoice: state.selectedLocalVoice,
        autoReadResponses: state.autoReadResponses,
        sttEnabled: state.sttEnabled,
        selectedLanguage: state.selectedLanguage,
      }),
    }
  )
);
