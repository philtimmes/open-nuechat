import React, { useEffect, Component, ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './stores/authStore';
import { useThemeStore } from './stores/themeStore';
import { useBrandingStore } from './stores/brandingStore';
import { useModelsStore } from './stores/modelsStore';
import { WebSocketProvider, useWebSocket } from './contexts/WebSocketContext';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import ChatPage from './pages/ChatPage';
import Settings from './pages/Settings';
import Billing from './pages/Billing';
import Documents from './pages/Documents';
import Admin from './pages/Admin';
import CustomGPTs from './pages/CustomGPTs';
import SharedChat from './pages/SharedChat';
import OAuthCallback from './pages/OAuthCallback';
import AgentFlows from './pages/AgentFlows';
import VibeCode from './pages/VibeCode';

// Components
import Layout from './components/Layout';

// Error Boundary
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '40px', 
          backgroundColor: '#1E293B', 
          color: '#F8FAFC',
          minHeight: '100vh',
          fontFamily: 'system-ui, sans-serif'
        }}>
          <h1 style={{ color: '#EF4444', marginBottom: '20px' }}>Application Error</h1>
          <pre style={{ 
            backgroundColor: '#0F172A', 
            padding: '20px', 
            borderRadius: '8px',
            overflow: 'auto',
            whiteSpace: 'pre-wrap'
          }}>
            {this.state.error?.message}
            {'\n\n'}
            {this.state.error?.stack}
          </pre>
          <button 
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: '#0EA5E9',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Protected route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, accessToken, _hasHydrated } = useAuthStore();
  const [isValidating, setIsValidating] = React.useState(true);
  const hasValidated = React.useRef(false);
  
  // Validate token once after hydration
  useEffect(() => {
    if (!_hasHydrated || hasValidated.current) return;
    
    const validateToken = async () => {
      hasValidated.current = true;
      
      const { isAuthenticated, accessToken, logout } = useAuthStore.getState();
      
      if (isAuthenticated && accessToken) {
        try {
          // Try to fetch user to validate token
          const response = await fetch('/api/auth/me', {
            headers: { 'Authorization': `Bearer ${accessToken}` }
          });
          if (!response.ok) throw new Error('Invalid token');
          setIsValidating(false);
        } catch {
          // Token is invalid, logout
          logout();
          setIsValidating(false);
        }
      } else {
        setIsValidating(false);
      }
    };
    
    validateToken();
  }, [_hasHydrated]);
  
  // Wait for hydration
  if (!_hasHydrated || isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)]">
        <div className="flex flex-col items-center gap-4">
          <svg className="animate-spin h-8 w-8 text-[var(--color-primary)]" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span className="text-[var(--color-text-secondary)] text-sm">Loading...</span>
        </div>
      </div>
    );
  }
  
  if (!isAuthenticated || !accessToken) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
}

// Connection error banner (uses the shared WebSocket context)
function ConnectionErrorBanner() {
  const { connectionError } = useWebSocket();
  
  if (!connectionError) return null;
  
  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-[var(--color-error)] text-white px-4 py-2 text-center text-sm">
      Connection error: {connectionError}
    </div>
  );
}

function App() {
  const { isAuthenticated, accessToken, fetchUser } = useAuthStore();
  const { fetchThemes, currentTheme, setThemeById } = useThemeStore();
  const { loadConfig, config, isLoaded: brandingLoaded } = useBrandingStore();
  const { fetchModels } = useModelsStore();
  
  // Load branding config first (this sets favicon, title, etc.)
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);
  
  // Fetch user, themes, and models after branding is loaded
  useEffect(() => {
    if (brandingLoaded) {
      if (isAuthenticated && accessToken) {
        fetchUser();
        // Force refetch models when authenticated to get subscribed assistants
        fetchModels(true);
      } else {
        // Fetch models without auth (gets base models only)
        fetchModels();
      }
      fetchThemes();
    }
  }, [isAuthenticated, accessToken, fetchUser, fetchThemes, fetchModels, brandingLoaded]);
  
  // Apply default theme from branding config if none selected
  useEffect(() => {
    if (brandingLoaded && !currentTheme && config?.default_theme) {
      // Try to set the theme from config
      setThemeById(config.default_theme);
    }
    
    // Fallback if still no theme
    if (!currentTheme) {
      // Apply a default dark theme (matches index.css :root)
      document.documentElement.style.setProperty('--color-primary', '#0EA5E9');
      document.documentElement.style.setProperty('--color-secondary', '#38BDF8');
      document.documentElement.style.setProperty('--color-background', '#0F172A');
      document.documentElement.style.setProperty('--color-surface', '#1E293B');
      document.documentElement.style.setProperty('--color-text', '#F8FAFC');
      document.documentElement.style.setProperty('--color-text-secondary', '#94A3B8');
      document.documentElement.style.setProperty('--color-accent', '#06B6D4');
      document.documentElement.style.setProperty('--color-error', '#EF4444');
      document.documentElement.style.setProperty('--color-success', '#22C55E');
      document.documentElement.style.setProperty('--color-warning', '#F59E0B');
      document.documentElement.style.setProperty('--color-border', '#334155');
      document.body.style.backgroundColor = '#0F172A';
      document.body.style.color = '#F8FAFC';
    }
  }, [currentTheme, brandingLoaded, config, setThemeById]);
  
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/shared/:shareId" element={<SharedChat />} />
        <Route path="/oauth/callback" element={<OAuthCallback />} />
        
        {/* Protected routes */}
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <WebSocketProvider>
                <ConnectionErrorBanner />
                <Layout>
                  <Routes>
                    <Route path="/" element={<ChatPage />} />
                    <Route path="/chat/:chatId" element={<ChatPage />} />
                    <Route path="/settings" element={<Settings />} />
                    <Route path="/billing" element={<Billing />} />
                    <Route path="/documents" element={<Documents />} />
                    <Route path="/admin" element={<Admin />} />
                    <Route path="/gpts" element={<CustomGPTs />} />
                    <Route path="/agents" element={<AgentFlows />} />
                    <Route path="/vibe" element={<VibeCode />} />
                  </Routes>
                </Layout>
              </WebSocketProvider>
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

// Export App wrapped with ErrorBoundary
export default function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}
