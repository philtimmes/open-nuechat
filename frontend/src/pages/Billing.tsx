import { useState, useEffect } from 'react';
import { useAuthStore } from '../stores/authStore';
import api from '../lib/api';
import type { UsageSummary, DailyUsage } from '../types';

interface TierConfig {
  id: string;
  name: string;
  price: number;
  tokens: number;
  features: string[];
  popular: boolean;
}

export default function Billing() {
  const { user } = useAuthStore();
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [history, setHistory] = useState<DailyUsage[]>([]);
  const [tiers, setTiers] = useState<TierConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    fetchBillingData();
  }, []);
  
  const fetchBillingData = async () => {
    setIsLoading(true);
    try {
      const [usageRes, historyRes, tiersRes] = await Promise.all([
        api.get('/billing/usage'),
        api.get('/billing/usage/history'),
        api.get('/admin/public/tiers'),
      ]);
      setUsage(usageRes.data);
      setHistory(historyRes.data);
      setTiers(tiersRes.data.tiers);
    } catch (err) {
      console.error('Failed to fetch billing data:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const formatTokens = (tokens: number | undefined | null) => {
    if (tokens == null) return '0';
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
    return tokens.toString();
  };
  
  const formatCurrency = (amount: number | undefined | null) => {
    if (amount == null) amount = 0;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(amount);
  };
  
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <svg className="animate-spin h-8 w-8 text-[var(--color-text-secondary)]" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      </div>
    );
  }
  
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6">
        <h1 className="text-2xl font-bold text-[var(--color-text)] mb-6">Billing & Usage</h1>
        
        {/* Current usage */}
        {usage && (
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Current Period Usage</h2>
            <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {/* Usage bar */}
                <div className="md:col-span-2">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[var(--color-text-secondary)] text-sm">Token Usage</span>
                    <span className="text-[var(--color-text)] font-medium">
                      {formatTokens(usage.total_tokens)} / {formatTokens(usage.tier_limit)}
                    </span>
                  </div>
                  <div className="h-4 bg-[var(--color-background)] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        (usage.usage_percentage ?? 0) > 90
                          ? 'bg-[var(--color-error)]'
                          : (usage.usage_percentage ?? 0) > 70
                          ? 'bg-[var(--color-warning)]'
                          : 'bg-[var(--color-primary)]'
                      }`}
                      style={{ width: `${Math.min(usage.usage_percentage ?? 0, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-2">
                    {(usage.usage_percentage ?? 0).toFixed(1)}% of monthly limit used
                  </p>
                </div>
                
                {/* Input tokens */}
                <div>
                  <span className="text-[var(--color-text-secondary)] text-sm">Input Tokens</span>
                  <p className="text-2xl font-bold text-[var(--color-text)]">
                    {formatTokens(usage.input_tokens)}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">
                    {formatCurrency(usage.input_cost)}
                  </p>
                </div>
                
                {/* Output tokens */}
                <div>
                  <span className="text-[var(--color-text-secondary)] text-sm">Output Tokens</span>
                  <p className="text-2xl font-bold text-[var(--color-text)]">
                    {formatTokens(usage.output_tokens)}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">
                    {formatCurrency(usage.output_cost)}
                  </p>
                </div>
              </div>
              
              {/* Total cost */}
              <div className="mt-6 pt-6 border-t border-[var(--color-border)] flex items-center justify-between">
                <span className="text-[var(--color-text-secondary)]">Estimated Cost This Period</span>
                <span className="text-2xl font-bold text-[var(--color-primary)]">
                  {formatCurrency(usage.total_cost)}
                </span>
              </div>
            </div>
          </section>
        )}
        
        {/* Usage history chart */}
        {history.length > 0 && (
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Daily Usage (Last 14 Days)</h2>
            <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <div className="flex items-end gap-1 h-40">
                {history.slice(-14).map((day, index) => {
                  const maxTokens = Math.max(...history.slice(-14).map(d => (d.input_tokens ?? 0) + (d.output_tokens ?? 0)));
                  const dayTokens = (day.input_tokens ?? 0) + (day.output_tokens ?? 0);
                  const height = maxTokens > 0 ? (dayTokens / maxTokens) * 100 : 0;
                  
                  return (
                    <div
                      key={index}
                      className="flex-1 group relative"
                    >
                      <div
                        className="bg-[var(--color-primary)] rounded-t hover:bg-[var(--color-accent)] transition-colors cursor-pointer"
                        style={{ height: `${Math.max(height, 2)}%` }}
                      />
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[var(--color-background)] border border-[var(--color-border)] rounded text-xs text-[var(--color-text)] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                        <p>{day.date ? new Date(day.date).toLocaleDateString() : 'Unknown'}</p>
                        <p>{formatTokens(dayTokens)} tokens</p>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between mt-2 text-xs text-[var(--color-text-secondary)]">
                <span>{history.length > 0 ? new Date(history[Math.max(0, history.length - 14)].date).toLocaleDateString() : ''}</span>
                <span>{history.length > 0 ? new Date(history[history.length - 1].date).toLocaleDateString() : ''}</span>
              </div>
            </div>
          </section>
        )}
        
        {/* Pricing tiers */}
        <section>
          <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Plans</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {tiers.map((tier) => (
              <div
                key={tier.id}
                className={`relative bg-[var(--color-surface)] rounded-xl p-6 border-2 transition-all ${
                  tier.id.toLowerCase() === user?.tier?.toLowerCase()
                    ? 'border-[var(--color-primary)] ring-2 ring-[var(--color-primary)]'
                    : tier.popular
                    ? 'border-[var(--color-secondary)]'
                    : 'border-[var(--color-border)]'
                }`}
              >
                {tier.popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-[var(--color-button)] text-[var(--color-button-text)] text-xs font-medium rounded-full">
                    Most Popular
                  </div>
                )}
                
                <div className="text-center mb-6">
                  <h3 className="text-xl font-bold text-[var(--color-text)]">{tier.name}</h3>
                  <div className="mt-2">
                    <span className="text-3xl font-bold text-[var(--color-text)]">
                      ${tier.price}
                    </span>
                    <span className="text-[var(--color-text-secondary)]">/month</span>
                  </div>
                </div>
                
                <ul className="space-y-3 mb-6">
                  {tier.features.map((feature, index) => (
                    <li key={index} className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
                      <svg className="w-5 h-5 text-[var(--color-success)] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
                
                <button
                  disabled={tier.id.toLowerCase() === user?.tier?.toLowerCase()}
                  className={`w-full py-2.5 rounded-lg font-medium transition-all ${
                    tier.id.toLowerCase() === user?.tier?.toLowerCase()
                      ? 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
                      : 'bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90'
                  }`}
                >
                  {tier.id.toLowerCase() === user?.tier?.toLowerCase() ? 'Current Plan' : 'Upgrade'}
                </button>
              </div>
            ))}
          </div>
        </section>
        
        {/* Payment info */}
        <section className="mt-8">
          <div className="bg-zinc-900/50 rounded-xl p-6 border border-[var(--color-border)]">
            <div className="flex items-center gap-3 mb-4">
              <svg className="w-6 h-6 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
              </svg>
              <h2 className="text-lg font-semibold text-[var(--color-text)]">Payment Method</h2>
            </div>
            <p className="text-[var(--color-text-secondary)] text-sm mb-4">
              {user?.tier === 'free'
                ? 'Add a payment method to upgrade your plan.'
                : 'Manage your payment methods and billing information.'}
            </p>
            <button className="px-4 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-zinc-700/30 transition-colors">
              {user?.tier === 'free' ? 'Add Payment Method' : 'Manage Payment Methods'}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
