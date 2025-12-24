import { useState, useEffect } from 'react';
import { useAuthStore } from '../stores/authStore';
import api from '../lib/api';
import type { UsageSummary, DailyUsage } from '../types';

interface TierConfig {
  id: string;
  name: string;
  price: number;
  tokens_limit: number;
  features: string[];
  popular: boolean;
}

interface PaymentMethod {
  id: string;
  provider: string;
  type: string;
  last_four?: string;
  brand?: string;
  exp_month?: number;
  exp_year?: number;
  email?: string;
  is_default: boolean;
}

interface Transaction {
  id: string;
  type: string;
  amount: number;
  currency: string;
  status: string;
  provider: string;
  description: string;
  created_at: string;
}

interface Subscription {
  id: string;
  tier: string;
  status: string;
  provider?: string;
  current_period_start?: string;
  current_period_end?: string;
  cancel_at_period_end: boolean;
}

interface PaymentProviders {
  available_providers: string[];
  currency: string;
  stripe_publishable_key?: string;
  google_pay?: {
    environment: string;
    merchantInfo: {
      merchantId: string;
      merchantName: string;
    };
  };
}

export default function Billing() {
  const { user, fetchUser } = useAuthStore();
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [history, setHistory] = useState<DailyUsage[]>([]);
  const [tiers, setTiers] = useState<TierConfig[]>([]);
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [providers, setProviders] = useState<PaymentProviders | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>('stripe');
  const [showTransactions, setShowTransactions] = useState(false);
  
  useEffect(() => {
    fetchBillingData();
    
    // Check for payment status in URL
    const params = new URLSearchParams(window.location.search);
    const status = params.get('status');
    if (status === 'success') {
      // Refresh data after successful payment
      setTimeout(() => {
        fetchBillingData();
        fetchUser();
      }, 2000);
    }
  }, []);
  
  const fetchBillingData = async () => {
    setIsLoading(true);
    try {
      const [usageRes, historyRes, tiersRes, subRes, providersRes, methodsRes, txRes] = await Promise.all([
        api.get('/billing/usage'),
        api.get('/billing/usage/history'),
        api.get('/billing/tiers'),
        api.get('/billing/subscription'),
        api.get('/billing/providers'),
        api.get('/billing/payment-methods'),
        api.get('/billing/transactions?limit=5'),
      ]);
      setUsage(usageRes.data);
      setHistory(historyRes.data);
      setTiers(tiersRes.data.tiers);
      setSubscription(subRes.data.subscription);
      setProviders(providersRes.data);
      setPaymentMethods(methodsRes.data.payment_methods);
      setTransactions(txRes.data.transactions);
      
      // Set default provider
      if (providersRes.data.available_providers?.length > 0) {
        setSelectedProvider(providersRes.data.available_providers[0]);
      }
    } catch (err) {
      console.error('Failed to fetch billing data:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSubscribe = async (tier: string) => {
    if (tier === 'free') return;
    
    setIsProcessing(true);
    try {
      const res = await api.post(`/billing/subscribe/${tier}?provider=${selectedProvider}`);
      
      // Redirect to payment provider
      if (res.data.url) {
        window.location.href = res.data.url;
      }
    } catch (err: any) {
      console.error('Subscription failed:', err);
      alert(err.response?.data?.detail || 'Failed to create subscription');
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleCancelSubscription = async () => {
    if (!confirm('Are you sure you want to cancel your subscription? You will retain access until the end of your billing period.')) {
      return;
    }
    
    setIsProcessing(true);
    try {
      await api.post('/billing/cancel-subscription');
      await fetchBillingData();
      fetchUser();
    } catch (err: any) {
      console.error('Cancel failed:', err);
      alert(err.response?.data?.detail || 'Failed to cancel subscription');
    } finally {
      setIsProcessing(false);
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
  
  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case 'stripe':
        return (
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
            <path d="M13.976 9.15c-2.172-.806-3.356-1.426-3.356-2.409 0-.831.683-1.305 1.901-1.305 2.227 0 4.515.858 6.09 1.631l.89-5.494C18.252.975 15.697 0 12.165 0 9.667 0 7.589.654 6.104 1.872 4.56 3.147 3.757 4.992 3.757 7.218c0 4.039 2.467 5.76 6.476 7.219 2.585.92 3.445 1.574 3.445 2.583 0 .98-.84 1.545-2.354 1.545-1.875 0-4.965-.921-6.99-2.109l-.9 5.555C5.175 22.99 8.385 24 11.714 24c2.641 0 4.843-.624 6.328-1.813 1.664-1.305 2.525-3.236 2.525-5.732 0-4.128-2.524-5.851-6.591-7.305z"/>
          </svg>
        );
      case 'paypal':
        return (
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7.076 21.337H2.47a.641.641 0 0 1-.633-.74L4.944.901C5.026.382 5.474 0 5.998 0h7.46c2.57 0 4.578.543 5.69 1.81 1.01 1.15 1.304 2.42 1.012 4.287-.023.143-.047.288-.077.437-.983 5.05-4.349 6.797-8.647 6.797h-2.19c-.524 0-.968.382-1.05.9l-1.12 7.106zm14.146-14.42a3.35 3.35 0 0 0-.607-.541c-.013.076-.026.175-.041.254-.93 4.778-4.005 7.201-9.138 7.201h-2.19a.563.563 0 0 0-.556.479l-1.187 7.527h-.506l-.24 1.516a.56.56 0 0 0 .554.647h3.882c.46 0 .85-.334.922-.788.06-.26.76-4.852.816-5.09a.932.932 0 0 1 .923-.788h.58c3.76 0 6.705-1.528 7.565-5.946.36-1.847.174-3.388-.777-4.471z"/>
          </svg>
        );
      case 'google_pay':
        return (
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z"/>
          </svg>
        );
      default:
        return null;
    }
  };
  
  const getStatusBadge = (status: string) => {
    const colors: Record<string, string> = {
      active: 'bg-green-500/20 text-green-400',
      past_due: 'bg-yellow-500/20 text-yellow-400',
      cancelled: 'bg-red-500/20 text-red-400',
      completed: 'bg-green-500/20 text-green-400',
      pending: 'bg-yellow-500/20 text-yellow-400',
      failed: 'bg-red-500/20 text-red-400',
    };
    return colors[status] || 'bg-gray-500/20 text-gray-400';
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
                </div>
                
                {/* Output tokens */}
                <div>
                  <span className="text-[var(--color-text-secondary)] text-sm">Output Tokens</span>
                  <p className="text-2xl font-bold text-[var(--color-text)]">
                    {formatTokens(usage.output_tokens)}
                  </p>
                </div>
              </div>
              
              {/* Subscription status */}
              {subscription && (
                <div className="mt-6 pt-6 border-t border-[var(--color-border)]">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-[var(--color-text-secondary)] text-sm">Current Plan</span>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-lg font-semibold text-[var(--color-text)] capitalize">{subscription.tier}</span>
                        <span className={`px-2 py-0.5 rounded-full text-xs ${getStatusBadge(subscription.status)}`}>
                          {subscription.status.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    {subscription.status === 'active' && subscription.tier !== 'free' && (
                      <button
                        onClick={handleCancelSubscription}
                        disabled={isProcessing}
                        className="px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                      >
                        Cancel Subscription
                      </button>
                    )}
                  </div>
                  {subscription.cancel_at_period_end && subscription.current_period_end && (
                    <p className="text-sm text-yellow-400 mt-2">
                      ⚠️ Subscription will end on {new Date(subscription.current_period_end).toLocaleDateString()}
                    </p>
                  )}
                </div>
              )}
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
        
        {/* Payment Provider Selection */}
        {providers && providers.available_providers.length > 0 && (
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Payment Method</h2>
            <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
              <p className="text-sm text-[var(--color-text-secondary)] mb-4">Select your preferred payment method:</p>
              <div className="flex flex-wrap gap-3">
                {providers.available_providers.map((provider) => (
                  <button
                    key={provider}
                    onClick={() => setSelectedProvider(provider)}
                    className={`flex items-center gap-2 px-4 py-3 rounded-lg border-2 transition-all ${
                      selectedProvider === provider
                        ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10'
                        : 'border-[var(--color-border)] hover:border-[var(--color-text-secondary)]'
                    }`}
                  >
                    <span className="text-[var(--color-text)]">{getProviderIcon(provider)}</span>
                    <span className="text-[var(--color-text)] font-medium capitalize">
                      {provider === 'google_pay' ? 'Google Pay' : provider}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}
        
        {/* Pricing tiers */}
        <section className="mb-8">
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
                  onClick={() => handleSubscribe(tier.id)}
                  disabled={tier.id.toLowerCase() === user?.tier?.toLowerCase() || isProcessing || tier.price === 0}
                  className={`w-full py-2.5 rounded-lg font-medium transition-all ${
                    tier.id.toLowerCase() === user?.tier?.toLowerCase()
                      ? 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
                      : tier.price === 0
                      ? 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
                      : 'bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90'
                  }`}
                >
                  {isProcessing ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Processing...
                    </span>
                  ) : tier.id.toLowerCase() === user?.tier?.toLowerCase() ? (
                    'Current Plan'
                  ) : tier.price === 0 ? (
                    'Free Forever'
                  ) : (
                    'Upgrade'
                  )}
                </button>
              </div>
            ))}
          </div>
        </section>
        
        {/* Stored Payment Methods */}
        {paymentMethods.length > 0 && (
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-[var(--color-text)] mb-4">Saved Payment Methods</h2>
            <div className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)] divide-y divide-[var(--color-border)]">
              {paymentMethods.map((method) => (
                <div key={method.id} className="p-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-[var(--color-text)]">{getProviderIcon(method.provider)}</span>
                    <div>
                      {method.type === 'card' ? (
                        <p className="text-[var(--color-text)]">
                          {method.brand?.toUpperCase()} •••• {method.last_four}
                        </p>
                      ) : (
                        <p className="text-[var(--color-text)]">{method.email}</p>
                      )}
                      {method.exp_month && method.exp_year && (
                        <p className="text-xs text-[var(--color-text-secondary)]">
                          Expires {method.exp_month}/{method.exp_year}
                        </p>
                      )}
                    </div>
                  </div>
                  {method.is_default && (
                    <span className="px-2 py-1 bg-[var(--color-primary)]/20 text-[var(--color-primary)] text-xs rounded">
                      Default
                    </span>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}
        
        {/* Transaction History */}
        <section className="mb-8">
          <button
            onClick={() => setShowTransactions(!showTransactions)}
            className="flex items-center gap-2 text-lg font-semibold text-[var(--color-text)] mb-4 hover:text-[var(--color-primary)] transition-colors"
          >
            <span>Transaction History</span>
            <svg className={`w-5 h-5 transition-transform ${showTransactions ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {showTransactions && (
            <div className="bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
              {transactions.length === 0 ? (
                <p className="p-6 text-center text-[var(--color-text-secondary)]">No transactions yet</p>
              ) : (
                <div className="divide-y divide-[var(--color-border)]">
                  {transactions.map((tx) => (
                    <div key={tx.id} className="p-4 flex items-center justify-between">
                      <div>
                        <p className="text-[var(--color-text)]">{tx.description}</p>
                        <p className="text-xs text-[var(--color-text-secondary)]">
                          {new Date(tx.created_at).toLocaleDateString()} • {tx.provider}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-[var(--color-text)] font-medium">
                          {formatCurrency(tx.amount)}
                        </p>
                        <span className={`px-2 py-0.5 rounded-full text-xs ${getStatusBadge(tx.status)}`}>
                          {tx.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
        
        {/* No Payment Providers Warning */}
        {(!providers || providers.available_providers.length === 0) && (
          <section className="mb-8">
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <svg className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <h3 className="text-lg font-semibold text-yellow-400">Payment Not Configured</h3>
                  <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                    Payment providers have not been configured. To enable payments, add your Stripe or PayPal API keys in the environment configuration.
                  </p>
                </div>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
