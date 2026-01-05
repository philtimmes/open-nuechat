/**
 * Branding Store
 * 
 * Manages application branding and configuration loaded from the backend.
 * This allows the app name, favicon, theme, and other settings to be
 * customized via .env without rebuilding the frontend.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface BrandingConfig {
  app_name: string;
  app_tagline: string;
  app_version: string;
  app_description: string;
  favicon_url: string;
  logo_url: string | null;
  logo_text: string;
  default_theme: string;
  brand_colors: {
    primary: string | null;
    secondary: string | null;
    accent: string | null;
  };
  footer_text: string | null;
  privacy_url: string | null;
  terms_url: string | null;
  support_email: string | null;
  features: {
    registration: boolean;
    oauth_google: boolean;
    oauth_github: boolean;
    billing: boolean;
    public_assistants: boolean;
    public_knowledge_stores: boolean;
  };
  welcome: {
    title: string;
    message: string;
  };
}

interface BrandingState {
  config: BrandingConfig | null;
  isLoading: boolean;
  error: string | null;
  isLoaded: boolean;
  
  // Actions
  loadConfig: (forceReload?: boolean) => Promise<void>;
  setConfig: (config: BrandingConfig) => void;
  
  // Helpers
  getAppName: () => string;
  getFaviconUrl: () => string;
  getDefaultTheme: () => string;
}

const defaultConfig: BrandingConfig = {
  app_name: 'Open-NueChat',
  app_tagline: 'AI-Powered Chat Platform',
  app_version: '1.0.0',
  app_description: 'A full-featured LLM chat platform',
  favicon_url: '/favicon.ico',
  logo_url: null,
  logo_text: 'Open-NueChat',
  default_theme: 'dark',
  brand_colors: {
    primary: null,
    secondary: null,
    accent: null,
  },
  footer_text: null,
  privacy_url: null,
  terms_url: null,
  support_email: null,
  features: {
    registration: true,
    oauth_google: true,
    oauth_github: true,
    billing: true,
    public_assistants: true,
    public_knowledge_stores: true,
  },
  welcome: {
    title: 'Welcome to Open-NueChat!',
    message: 'Start a conversation with AI.',
  },
};

export const useBrandingStore = create<BrandingState>()(
  persist(
    (set, get) => ({
      config: null,
      isLoading: false,
      error: null,
      isLoaded: false,
      
      loadConfig: async (forceReload = false) => {
        // Don't reload if already loaded (unless forced)
        if (!forceReload && get().isLoaded && get().config) {
          return;
        }
        
        set({ isLoading: true, error: null });
        
        try {
          const response = await fetch('/api/branding/config');
          
          if (!response.ok) {
            throw new Error('Failed to load branding config');
          }
          
          const config = await response.json();
          
          set({ 
            config, 
            isLoading: false, 
            isLoaded: true,
            error: null 
          });
          
          // Apply branding
          applyBranding(config);
          
        } catch (error) {
          console.warn('Using default branding config:', error);
          set({ 
            config: defaultConfig, 
            isLoading: false, 
            isLoaded: true,
            error: null 
          });
          applyBranding(defaultConfig);
        }
      },
      
      setConfig: (config) => {
        set({ config, isLoaded: true });
        applyBranding(config);
      },
      
      getAppName: () => {
        return get().config?.app_name || defaultConfig.app_name;
      },
      
      getFaviconUrl: () => {
        return get().config?.favicon_url || defaultConfig.favicon_url;
      },
      
      getDefaultTheme: () => {
        return get().config?.default_theme || defaultConfig.default_theme;
      },
    }),
    {
      name: 'nexus-branding',
      partialize: (state) => ({ 
        config: state.config,
        isLoaded: state.isLoaded,
      }),
    }
  )
);

/**
 * Apply branding settings to the document
 */
function applyBranding(config: BrandingConfig) {
  // Update page title
  document.title = config.app_name;
  
  // Update favicon
  updateFavicon(config.favicon_url);
  
  // Apply brand colors as CSS variables (override theme defaults)
  if (config.brand_colors.primary) {
    document.documentElement.style.setProperty('--color-primary', config.brand_colors.primary);
  }
  if (config.brand_colors.secondary) {
    document.documentElement.style.setProperty('--color-secondary', config.brand_colors.secondary);
  }
  if (config.brand_colors.accent) {
    document.documentElement.style.setProperty('--color-accent', config.brand_colors.accent);
  }
  
  // Update meta tags
  updateMetaTag('description', config.app_description);
  updateMetaTag('application-name', config.app_name);
}

function updateFavicon(url: string) {
  let link = document.querySelector<HTMLLinkElement>("link[rel~='icon']");
  
  if (!link) {
    link = document.createElement('link');
    link.rel = 'icon';
    document.head.appendChild(link);
  }
  
  link.href = url;
}

function updateMetaTag(name: string, content: string) {
  let meta = document.querySelector<HTMLMetaElement>(`meta[name="${name}"]`);
  
  if (!meta) {
    meta = document.createElement('meta');
    meta.name = name;
    document.head.appendChild(meta);
  }
  
  meta.content = content;
}

// Export helper hook for easy access
export function useAppName() {
  return useBrandingStore((state) => state.config?.app_name || 'Open-NueChat');
}

export function useBrandingConfig() {
  return useBrandingStore((state) => state.config);
}

export function useFeatureFlags() {
  return useBrandingStore((state) => state.config?.features || defaultConfig.features);
}
