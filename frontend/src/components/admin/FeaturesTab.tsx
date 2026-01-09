import { FeatureFlags } from './types';

interface Props {
  flags: FeatureFlags;
  setFlags: (flags: FeatureFlags) => void;
  onSave: () => void;
  isSaving: boolean;
}

export default function FeaturesTab({ flags, setFlags, onSave, isSaving }: Props) {
  return (
    <div className="space-y-6">
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Feature Flags</h2>
        
        <div className="space-y-4">
          <label className="flex items-center gap-3 cursor-pointer p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
            <input
              type="checkbox"
              checked={flags.enable_registration}
              onChange={(e) => setFlags({ ...flags, enable_registration: e.target.checked })}
              className="w-5 h-5 rounded border-[var(--color-border)]"
            />
            <div>
              <div className="text-sm font-medium text-[var(--color-text)]">Enable Registration</div>
              <div className="text-xs text-[var(--color-text-secondary)]">Allow new users to create accounts</div>
            </div>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
            <input
              type="checkbox"
              checked={flags.enable_billing}
              onChange={(e) => setFlags({ ...flags, enable_billing: e.target.checked })}
              className="w-5 h-5 rounded border-[var(--color-border)]"
            />
            <div>
              <div className="text-sm font-medium text-[var(--color-text)]">Enable Billing</div>
              <div className="text-xs text-[var(--color-text-secondary)]">Show billing page and enforce token limits</div>
            </div>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
            <input
              type="checkbox"
              checked={flags.freeforall}
              onChange={(e) => setFlags({ ...flags, freeforall: e.target.checked })}
              className="w-5 h-5 rounded border-[var(--color-border)]"
            />
            <div>
              <div className="text-sm font-medium text-[var(--color-text)]">Free For All Mode</div>
              <div className="text-xs text-[var(--color-text-secondary)]">Disable all token limits and restrictions</div>
            </div>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
            <input
              type="checkbox"
              checked={flags.enable_safety_filters}
              onChange={(e) => setFlags({ ...flags, enable_safety_filters: e.target.checked })}
              className="w-5 h-5 rounded border-[var(--color-border)]"
            />
            <div>
              <div className="text-sm font-medium text-[var(--color-text)]">Enable Safety Filters</div>
              <div className="text-xs text-[var(--color-text-secondary)]">Apply content filters to messages</div>
            </div>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
            <input
              type="checkbox"
              checked={flags.enable_mermaid_rendering}
              onChange={(e) => setFlags({ ...flags, enable_mermaid_rendering: e.target.checked })}
              className="w-5 h-5 rounded border-[var(--color-border)]"
            />
            <div>
              <div className="text-sm font-medium text-[var(--color-text)]">Enable Mermaid Rendering</div>
              <div className="text-xs text-[var(--color-text-secondary)]">Render Mermaid diagrams as interactive graphics (disable to show raw code)</div>
            </div>
          </label>
        </div>
      </div>
      
      <button
        onClick={onSave}
        disabled={isSaving}
        className="px-6 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
      >
        {isSaving ? 'Saving...' : 'Save Feature Flags'}
      </button>
    </div>
  );
}
