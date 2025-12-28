import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useBrandingStore, useFeatureFlags } from '../stores/brandingStore';
import api from '../lib/api';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();
  const { config, loadConfig } = useBrandingStore();
  const features = useFeatureFlags();
  
  // Force reload branding config on login page to get latest OAuth settings
  useEffect(() => {
    loadConfig(true);
  }, [loadConfig]);
  
  const appName = config?.app_name || 'Open-NueChat';
  const appTagline = config?.app_tagline || 'Your AI-powered conversation platform';
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    console.log('Login attempt:', { email, password: '***' });
    console.log('API URL:', '/api/auth/login');
    
    try {
      const response = await api.post('/auth/login', { email, password });
      console.log('Login response:', response.data);
      const { user, access_token, refresh_token } = response.data;
      setAuth(user, access_token, refresh_token);
      
      // Check if we need to continue a shared chat
      const continueShareId = sessionStorage.getItem('continueSharedChat');
      if (continueShareId) {
        sessionStorage.removeItem('continueSharedChat');
        // Clone the chat and redirect
        try {
          const cloneResponse = await api.post(`/shared/${continueShareId}/clone`);
          navigate(`/?chat=${cloneResponse.data.chat_id}`);
          return;
        } catch (cloneErr) {
          console.error('Failed to clone shared chat:', cloneErr);
          // Fall through to normal redirect
        }
      }
      
      navigate('/');
    } catch (err: any) {
      console.error('Login error:', err);
      console.error('Response:', err.response);
      setError(err.response?.data?.detail || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleOAuth = (provider: 'google' | 'github') => {
    window.location.href = `${import.meta.env.VITE_API_URL || '/api'}/auth/oauth/${provider}`;
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)] p-4">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-[var(--color-primary)] opacity-10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-[var(--color-secondary)] opacity-10 rounded-full blur-3xl" />
      </div>
      
      <div className="relative w-full max-w-md">
        {/* Logo and Branding */}
        <div className="text-center mb-8">
          {config?.logo_url && (
            <img 
              src={config.logo_url} 
              alt={appName} 
              className="h-12 mx-auto mb-4"
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
          )}
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] bg-clip-text text-transparent">
            {appName}
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-2">
            {appTagline}
          </p>
        </div>
        
        {/* Card */}
        <div className="bg-[var(--color-surface)] rounded-2xl p-8 shadow-xl border border-[var(--color-border)]">
          <h2 className="text-2xl font-semibold text-[var(--color-text)] mb-6">
            Welcome back
          </h2>
          
          {error && (
            <div className="bg-[var(--color-error)]/10 border border-[var(--color-error)]/30 text-[var(--color-error)] px-4 py-3 rounded-lg mb-4 text-sm">
              {error}
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1.5">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent transition-all"
                placeholder="you@example.com"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1.5">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] placeholder-zinc-500/50 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent transition-all"
                placeholder="••••••••"
                required
              />
            </div>
            
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 px-4 bg-[var(--color-button)] text-[var(--color-button-text)] font-medium rounded-lg hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Signing in...
                </span>
              ) : (
                'Sign in'
              )}
            </button>
          </form>
          
          {/* OAuth buttons - conditionally rendered based on feature flags */}
          {(features.oauth_google || features.oauth_github) && (
            <>
              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-[var(--color-border)]" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-[var(--color-surface)] text-[var(--color-text-secondary)]">
                    or continue with
                  </span>
                </div>
              </div>
              
              <div className={`grid gap-3 ${features.oauth_google && features.oauth_github ? 'grid-cols-2' : 'grid-cols-1'}`}>
                {features.oauth_google && (
                  <button
                    onClick={() => handleOAuth('google')}
                    className="flex items-center justify-center gap-2 px-4 py-3 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] hover:bg-zinc-700/30 transition-all"
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                      <path fill="#EA4335" d="M5.27 9.76A7.08 7.08 0 0 1 12 4.91c1.71 0 3.26.61 4.48 1.62l3.35-3.35C17.76 1.19 15.05 0 12 0 7.27 0 3.2 2.7 1.24 6.65l4.03 3.11Z" />
                      <path fill="#34A853" d="M16.04 18.01A6.94 6.94 0 0 1 12 19.09a7.08 7.08 0 0 1-6.73-4.85l-4.03 3.11C3.2 21.3 7.27 24 12 24c2.93 0 5.73-1.07 7.83-3.05l-3.79-2.94Z" />
                      <path fill="#4285F4" d="m23.49 12.27-.01-.54H12v4.55h6.47a5.97 5.97 0 0 1-2.43 3.73l3.79 2.94C22.21 20.77 24 16.95 24 12.55c0-.09 0-.19-.01-.28Z" />
                      <path fill="#FBBC05" d="M5.27 14.24A7.13 7.13 0 0 1 4.91 12c0-.78.13-1.54.36-2.24L1.24 6.65A11.88 11.88 0 0 0 0 12c0 1.93.46 3.75 1.24 5.35l4.03-3.11Z" />
                    </svg>
                    Google
                  </button>
                )}
                
                {features.oauth_github && (
                  <button
                    onClick={() => handleOAuth('github')}
                    className="flex items-center justify-center gap-2 px-4 py-3 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] hover:bg-zinc-700/30 transition-all"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.6.11.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12Z" />
                    </svg>
                    GitHub
                  </button>
                )}
              </div>
            </>
          )}
          
          {features.registration && (
            <p className="mt-6 text-center text-sm text-[var(--color-text-secondary)]">
              Don't have an account?{' '}
              <Link
                to="/register"
                className="text-[var(--color-primary)] hover:text-[var(--color-accent)] font-medium transition-colors"
              >
                Sign up
              </Link>
            </p>
          )}
        </div>
        
        {/* Footer with configurable links */}
        <div className="mt-8 text-center text-xs text-zinc-500/60 space-y-2">
          <p>
            By signing in, you agree to our{' '}
            {config?.terms_url ? (
              <a href={config.terms_url} className="underline hover:text-[var(--color-text-secondary)]">Terms of Service</a>
            ) : (
              'Terms of Service'
            )}
            {' '}and{' '}
            {config?.privacy_url ? (
              <a href={config.privacy_url} className="underline hover:text-[var(--color-text-secondary)]">Privacy Policy</a>
            ) : (
              'Privacy Policy'
            )}
          </p>
          {config?.footer_text && (
            <p>{config.footer_text}</p>
          )}
        </div>
      </div>
    </div>
  );
}
