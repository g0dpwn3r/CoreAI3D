'use client';

import { useState } from 'react';
import { Key, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BetaKeyFormProps {
  onSubmit: (betaKey: string) => Promise<void>;
  error?: string;
}

export function BetaKeyForm({ onSubmit, error }: BetaKeyFormProps) {
  const [betaKey, setBetaKey] = useState('');
  const [validating, setValidating] = useState(false);
  const [keyValid, setKeyValid] = useState<'idle' | 'valid' | 'invalid'>('idle');

  const validateKey = async () => {
    if (!betaKey.trim() || betaKey.length < 10) {
      setKeyValid('invalid');
      return;
    }
    
    setValidating(true);
    try {
      const res = await fetch('/api/auth/validate-beta-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ beta_key: betaKey }),
      });
      const data = await res.json();
      setKeyValid(data.valid ? 'valid' : 'invalid');
    } catch {
      setKeyValid('invalid');
    } finally {
      setValidating(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (keyValid !== 'valid') return;
    await onSubmit(betaKey);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label htmlFor="beta-key" className="block text-sm font-medium text-text-secondary mb-2">
          Beta Key <span className="text-electric-blue">*</span>
        </label>
        <div className="relative">
          <input
            id="beta-key"
            type="text"
            value={betaKey}
            onChange={(e) => {
              setBetaKey(e.target.value.toUpperCase());
              setKeyValid('idle');
            }}
            onBlur={validateKey}
            placeholder="CA3D-XXXXXXXXXXXXXXXXXXXX"
            className={cn(
              'w-full px-4 py-3 bg-core-card border rounded-lg',
              'text-text-primary placeholder-text-muted',
              'focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue',
              keyValid === 'valid' && 'border-green-500/50',
              keyValid === 'invalid' && 'border-red-500/50',
              (keyValid === 'idle' || validating) && 'border-core-border'
            )}
            disabled={validating}
            autoComplete="off"
            autoFocus
          />
          {validating && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <Loader2 className="w-5 h-5 text-electric-blue animate-spin" />
            </div>
          )}
          {keyValid === 'valid' && !validating && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
          )}
          {keyValid === 'invalid' && !validating && betaKey.length > 0 && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <AlertCircle className="w-5 h-5 text-red-400" />
            </div>
          )}
        </div>
        {keyValid === 'valid' && (
          <p className="mt-2 text-sm text-green-400 flex items-center gap-1">
            <CheckCircle className="w-4 h-4" />
            Valid beta key - you can proceed with registration
          </p>
        )}
        {keyValid === 'invalid' && betaKey.length > 0 && (
          <p className="mt-2 text-sm text-red-400 flex items-center gap-1">
            <AlertCircle className="w-4 h-4" />
            Invalid or already used beta key
          </p>
        )}
        {error && (
          <p className="mt-2 text-sm text-red-400 flex items-center gap-1">
            <AlertCircle className="w-4 h-4" />
            {error}
          </p>
        )}
      </div>

      <div className="bg-core-card/50 border border-core-border rounded-lg p-4">
        <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
          <Key className="w-5 h-5 text-electric-blue" />
          About Beta Keys
        </h4>
        <ul className="space-y-1 text-sm text-text-muted">
          <li>• Beta keys are required for registration during the beta period</li>
          <li>• Each key can only be used once</li>
          <li>• Keys are distributed by CoreAI3D administrators</li>
          <li>• Request a key from the CoreAI3D team or Linux GUI admin</li>
        </ul>
      </div>

      <button
        type="submit"
        disabled={keyValid !== 'valid'}
        className={cn(
          'w-full py-3 px-4 rounded-lg font-medium transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-electric-blue focus:ring-offset-2 focus:ring-offset-core-bg',
          keyValid === 'valid'
            ? 'bg-electric-blue text-white hover:bg-electric-blue/90'
            : 'bg-core-cardHover text-text-muted cursor-not-allowed'
        )}
      >
        Continue to Registration
      </button>
    </form>
  );
}