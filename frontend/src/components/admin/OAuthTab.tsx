import { OAuthSettings } from './types';

interface Props {
  settings: OAuthSettings;
  setSettings: (settings: OAuthSettings) => void;
  onSave: () => void;
  isSaving: boolean;
}

export default function OAuthTab({ settings, setSettings, onSave, isSaving }: Props) {
  return (
    <div className="space-y-6">
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">🔷 Google OAuth</h2>
        
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.google_oauth_enabled}
                onChange={(e) => setSettings({ ...settings, google_oauth_enabled: e.target.checked })}
                className="w-4 h-4 rounded border-[var(--color-border)]"
              />
              <span className="text-sm text-[var(--color-text)]">Enable Google OAuth</span>
            </label>
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client ID</label>
            <input
              type="text"
              value={settings.google_client_id}
              onChange={(e) => setSettings({ ...settings, google_client_id: e.target.value })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client Secret</label>
            <input
              type="password"
              value={settings.google_client_secret}
              onChange={(e) => setSettings({ ...settings, google_client_secret: e.target.value })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Timeout (seconds)</label>
            <input
              type="number"
              min="5"
              max="300"
              value={settings.google_oauth_timeout}
              onChange={(e) => setSettings({ ...settings, google_oauth_timeout: parseInt(e.target.value) || 30 })}
              className="w-32 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
        </div>
      </div>
      
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">⬛ GitHub OAuth</h2>
        
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.github_oauth_enabled}
                onChange={(e) => setSettings({ ...settings, github_oauth_enabled: e.target.checked })}
                className="w-4 h-4 rounded border-[var(--color-border)]"
              />
              <span className="text-sm text-[var(--color-text)]">Enable GitHub OAuth</span>
            </label>
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client ID</label>
            <input
              type="text"
              value={settings.github_client_id}
              onChange={(e) => setSettings({ ...settings, github_client_id: e.target.value })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Client Secret</label>
            <input
              type="password"
              value={settings.github_client_secret}
              onChange={(e) => setSettings({ ...settings, github_client_secret: e.target.value })}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
          
          <div>
            <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Timeout (seconds)</label>
            <input
              type="number"
              min="5"
              max="300"
              value={settings.github_oauth_timeout}
              onChange={(e) => setSettings({ ...settings, github_oauth_timeout: parseInt(e.target.value) || 30 })}
              className="w-32 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>
        </div>
      </div>
      
      <button
        onClick={onSave}
        disabled={isSaving}
        className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
      >
        {isSaving ? 'Saving...' : 'Save OAuth Settings'}
      </button>
    </div>
  );
}
