/**
 * Branding Store
 * 
 * Manages application branding and configuration loaded from the backend.
 * This allows the app name, favicon, theme, and other settings to be
 * customized via .env without rebuilding the frontend.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface CustomTheme {
  id: string;
  name: string;
  [key: string]: string;  // CSS variables
}

export interface BrandingConfig {
  app_name: string;
  app_tagline: string;
  app_version: string;
  app_description: string;
  favicon_url: string;
  logo_url: string | null;
  logo_text: string;
  default_theme: string;
  custom_css: string;
  custom_themes: CustomTheme[];
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
  getCustomThemes: () => CustomTheme[];
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
  custom_css: '',
  custom_themes: [],
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
          // Try the admin public branding endpoint first
          const response = await fetch('/api/admin/public/branding');
          
          if (!response.ok) {
            throw new Error('Failed to load branding config');
          }
          
          const data = await response.json();
          
          // Parse custom themes JSON
          let customThemes: CustomTheme[] = [];
          try {
            if (data.custom_themes) {
              customThemes = JSON.parse(data.custom_themes);
            }
          } catch (e) {
            console.warn('Failed to parse custom themes:', e);
          }
          
          const config: BrandingConfig = {
            ...defaultConfig,
            app_name: data.app_name || defaultConfig.app_name,
            app_tagline: data.app_tagline || defaultConfig.app_tagline,
            favicon_url: data.favicon_url || defaultConfig.favicon_url,
            logo_url: data.logo_url || defaultConfig.logo_url,
            logo_text: data.app_name || defaultConfig.logo_text,
            custom_css: data.custom_css || '',
            custom_themes: customThemes,
            welcome: {
              title: `Welcome to ${data.app_name || 'Open-NueChat'}!`,
              message: data.app_tagline || 'Start a conversation with AI.',
            },
          };
          
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
      
      getCustomThemes: () => {
        return get().config?.custom_themes || [];
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
  if (config.favicon_url) {
    updateFavicon(config.favicon_url);
  }
  
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
  
  // Apply custom CSS
  applyCustomCSS(config.custom_css);
  
  // Update meta tags
  updateMetaTag('description', config.app_description);
  updateMetaTag('application-name', config.app_name);
}

function updateFavicon(url: string) {
  if (!url) return;
  
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

function applyCustomCSS(css: string) {
  // Remove existing custom CSS
  const existingStyle = document.getElementById('custom-branding-css');
  if (existingStyle) {
    existingStyle.remove();
  }
  
  // Add new custom CSS
  if (css && css.trim()) {
    const style = document.createElement('style');
    style.id = 'custom-branding-css';
    style.textContent = css;
    document.head.appendChild(style);
  }
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

export function useCustomThemes() {
  return useBrandingStore((state) => state.config?.custom_themes || []);
}
