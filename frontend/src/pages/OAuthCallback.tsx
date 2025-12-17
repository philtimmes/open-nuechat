import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import axios from 'axios';

export default function OAuthCallback() {
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      // Extract tokens from URL fragment (after #)
      const hash = window.location.hash.substring(1);
      const params = new URLSearchParams(hash);
      
      const accessToken = params.get('access_token');
      const refreshToken = params.get('refresh_token');
      
      if (!accessToken || !refreshToken) {
        setError('OAuth callback missing tokens');
        setTimeout(() => navigate('/login'), 3000);
        return;
      }
      
      try {
        // Fetch user info using the token directly (don't rely on localStorage yet)
        const response = await axios.get('/api/auth/me', {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        });
        const user = response.data;
        
        // Set auth state (this stores tokens in Zustand/localStorage)
        setAuth(user, accessToken, refreshToken);
        
        // Clear the hash from URL for cleanliness
        window.history.replaceState(null, '', '/oauth/callback');
        
        // Check if we need to continue a shared chat
        const continueShareId = sessionStorage.getItem('continueSharedChat');
        if (continueShareId) {
          sessionStorage.removeItem('continueSharedChat');
          // Clone the chat and redirect
          try {
            const cloneResponse = await axios.post(`/api/shared/${continueShareId}/clone`, {}, {
              headers: {
                'Authorization': `Bearer ${accessToken}`
              }
            });
            navigate(`/?chat=${cloneResponse.data.chat_id}`);
            return;
          } catch (cloneErr) {
            console.error('Failed to clone shared chat:', cloneErr);
            // Fall through to normal redirect
          }
        }
        
        // Redirect to home
        navigate('/');
      } catch (err: any) {
        console.error('OAuth callback error:', err);
        setError(err.response?.data?.detail || 'Failed to complete sign in');
        setTimeout(() => navigate('/login'), 3000);
      }
    };
    
    handleCallback();
  }, [navigate, setAuth]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)]">
      <div className="text-center">
        {error ? (
          <div className="space-y-4">
            <div className="text-red-400 text-lg">{error}</div>
            <div className="text-[var(--color-text-secondary)]">Redirecting to login...</div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="animate-spin h-8 w-8 border-4 border-[var(--color-primary)] border-t-transparent rounded-full mx-auto"></div>
            <div className="text-[var(--color-text)]">Completing sign in...</div>
          </div>
        )}
      </div>
    </div>
  );
}
