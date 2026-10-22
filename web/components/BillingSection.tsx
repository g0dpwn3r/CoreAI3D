'use client';

import { CreditCard, Receipt, ArrowUpRight, Check, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';

const PLANS = [
  {
    id: 'free',
    name: 'Free',
    price: 0,
    period: 'month',
    description: 'Perfect for trying out CoreAI3D',
    features: [
      '100 messages per day',
      'Access to CoreAI3D Fast model',
      'Basic chat history (7 days)',
      'Community support',
    ],
    cta: 'Current Plan',
    popular: false,
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 20,
    period: 'month',
    description: 'For power users and developers',
    features: [
      'Unlimited messages',
      'All models including Reasoning',
      'Full chat history',
      'Priority support',
      'API access (10k requests/month)',
      'File uploads (100MB)',
    ],
    cta: 'Upgrade to Pro',
    popular: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 99,
    period: 'month',
    description: 'For teams and organizations',
    features: [
      'Everything in Pro',
      'Team collaboration',
      'Custom model fine-tuning',
      'Dedicated support',
      'SLA guarantee',
      'API access (unlimited)',
      'File uploads (1GB)',
      'SSO & advanced admin',
    ],
    cta: 'Contact Sales',
    popular: false,
  },
];

export function BillingSection() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-text-primary mb-2">Billing & Plans</h2>
        <p className="text-text-muted">Manage your subscription and billing details</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {PLANS.map((plan) => (
          <div
            key={plan.id}
            className={cn(
              'relative p-6 bg-core-card border rounded-xl transition-all',
              plan.popular
                ? 'border-electric-blue/50 shadow-blue-glow'
                : 'border-core-border hover:border-electric-blue/30'
            )}
          >
            {plan.popular && (
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-electric-blue text-white text-xs font-medium rounded-full">
                Most Popular
              </div>
            )}
            <div className="mb-6">
              <h3 className="text-xl font-bold text-text-primary">{plan.name}</h3>
              <p className="text-text-muted text-sm mt-1">{plan.description}</p>
            </div>
            <div className="mb-6">
              <span className="text-4xl font-bold text-text-primary">${plan.price}</span>
              <span className="text-text-muted">/{plan.period}</span>
            </div>
            <ul className="space-y-3 mb-6">
              {plan.features.map((feature, index) => (
                <li key={index} className="flex items-start gap-3 text-sm text-text-secondary">
                  <Check className="w-5 h-5 text-electric-blue flex-shrink-0 mt-0.5" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
            <button
              className={cn(
                'w-full py-3 px-4 rounded-lg font-medium transition-colors',
                plan.popular
                  ? 'bg-electric-blue text-white hover:bg-electric-blue/90'
                  : plan.id === 'free'
                  ? 'bg-core-cardHover text-text-muted cursor-not-allowed'
                  : 'bg-core-cardHover text-text-primary hover:bg-electric-blue/10 hover:text-electric-blue border border-core-border'
              )}
              disabled={plan.id === 'free'}
            >
              {plan.cta}
              {plan.id !== 'free' && !plan.popular && (
                <ArrowUpRight className="w-4 h-4 ml-2" />
              )}
            </button>
          </div>
        ))}
      </div>

      <div className="bg-core-card border border-core-border rounded-xl p-6">
        <h3 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
          <CreditCard className="w-5 h-5 text-electric-blue" />
          Payment Methods
        </h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-core-bgSecondary border border-core-border rounded-lg">
            <div className="flex items-center gap-4">
              <div className="w-12 h-8 bg-gradient-to-r from-blue-600 to-blue-400 rounded flex items-center justify-center">
                <span className="text-white font-bold text-xs">VISA</span>
              </div>
              <div>
                <p className="font-medium text-text-primary">•••• •••• •••• 4242</p>
                <p className="text-sm text-text-muted">Expires 12/2026</p>
              </div>
            </div>
            <span className="px-2 py-1 text-xs bg-green-500/20 text-green-400 rounded-full">Default</span>
          </div>
          <button className="text-sm text-electric-blue hover:text-electric-blue/80 flex items-center gap-1">
            <CreditCard className="w-4 h-4" />
            Add Payment Method
          </button>
        </div>
      </div>

      <div className="bg-core-card border border-core-border rounded-xl p-6">
        <h3 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
          <Receipt className="w-5 h-5 text-electric-blue" />
          Invoice History
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-core-border text-left text-sm font-medium text-text-muted">
                <th className="pb-3 px-4">Date</th>
                <th className="pb-3 px-4">Description</th>
                <th className="pb-3 px-4">Amount</th>
                <th className="pb-3 px-4">Status</th>
                <th className="pb-3 px-4">Action</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-core-border/50">
                <td className="py-3 px-4 text-sm text-text-secondary">—</td>
                <td className="py-3 px-4 text-sm text-text-primary">No invoices yet</td>
                <td className="py-3 px-4 text-sm text-text-muted">—</td>
                <td className="py-3 px-4">—</td>
                <td className="py-3 px-4">—</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-sm text-text-muted mt-4">
          Invoices will appear here after your first paid subscription period.
        </p>
      </div>
    </div>
  );
}