import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Theme, ThemeColors } from '../types';
import api from '../lib/api';

interface ThemeStore {
  themes: Theme[];
  currentTheme: Theme | null;
  isLoading: boolean;
  
  // Actions
  fetchThemes: () => Promise<void>;
  setCurrentTheme: (theme: Theme) => void;
  setThemeById: (themeId: string) => void;
  applyTheme: (theme: Theme) => void;
  applyThemeToAccount: (themeId: string) => Promise<void>;
}

// Apply theme colors to CSS variables
const applyThemeToCss = (colors: ThemeColors, fonts?: Theme['fonts']) => {
  const root = document.documentElement;
  
  root.style.setProperty('--color-primary', colors.primary);
  root.style.setProperty('--color-secondary', colors.secondary);
  root.style.setProperty('--color-background', colors.background);
  root.style.setProperty('--color-surface', colors.surface);
  root.style.setProperty('--color-text', colors.text);
  root.style.setProperty('--color-text-secondary', colors.text_secondary);
  root.style.setProperty('--color-accent', colors.accent);
  root.style.setProperty('--color-error', colors.error);
  root.style.setProperty('--color-success', colors.success);
  root.style.setProperty('--color-warning', colors.warning);
  root.style.setProperty('--color-border', colors.border);
  
  // Button colors - use theme-specific or fallback to primary
  const colorsAny = colors as unknown as Record<string, string>;
  const buttonColor = colorsAny.button || colors.primary;
  const buttonTextColor = colorsAny.button_text || '#FFFFFF';
  root.style.setProperty('--color-button', buttonColor);
  root.style.setProperty('--color-button-text', buttonTextColor);
  
  if (fonts) {
    root.style.setProperty('--font-heading', fonts.heading);
    root.style.setProperty('--font-body', fonts.body);
    root.style.setProperty('--font-code', fonts.code);
  }
  
  // Update body background immediately
  document.body.style.backgroundColor = colors.background;
  document.body.style.color = colors.text;
};

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set, get) => ({
      themes: [],
      currentTheme: null,
      isLoading: false,
      
      fetchThemes: async () => {
        set({ isLoading: true });
        try {
          const response = await api.get('/themes');
          // Backend returns array directly
          const themes = Array.isArray(response.data) ? response.data : [];
          set({ themes, isLoading: false });
          
          // Apply current theme if set
          const { currentTheme } = get();
          if (currentTheme) {
            applyThemeToCss(currentTheme.colors, currentTheme.fonts);
          }
        } catch (err) {
          console.error('Failed to fetch themes:', err);
          set({ themes: [], isLoading: false });
        }
      },
      
      setCurrentTheme: (theme) => {
        set({ currentTheme: theme });
        applyThemeToCss(theme.colors, theme.fonts);
      },
      
      setThemeById: (themeId) => {
        const themes = get().themes;
        const safeThemes = Array.isArray(themes) ? themes : [];
        const theme = safeThemes.find((t) => t.id === themeId || t.name?.toLowerCase() === themeId?.toLowerCase());
        if (theme) {
          set({ currentTheme: theme });
          applyThemeToCss(theme.colors, theme.fonts);
        }
      },
      
      applyTheme: (theme) => {
        applyThemeToCss(theme.colors, theme.fonts);
        set({ currentTheme: theme });
      },
      
      applyThemeToAccount: async (themeId) => {
        try {
          await api.post(`/themes/apply/${themeId}`);
          const theme = get().themes.find((t) => t.id === themeId);
          if (theme) {
            get().applyTheme(theme);
          }
        } catch (err) {
          console.error('Failed to apply theme:', err);
          throw err;
        }
      },
    }),
    {
      name: 'nexus-theme',
      partialize: (state) => ({
        currentTheme: state.currentTheme,
      }),
      onRehydrateStorage: () => (state) => {
        // Apply theme on rehydration
        if (state?.currentTheme) {
          applyThemeToCss(state.currentTheme.colors, state.currentTheme.fonts);
        }
      },
    }
  )
);
