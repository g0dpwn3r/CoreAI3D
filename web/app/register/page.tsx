'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Mail, Lock, Eye, EyeOff, AlertCircle, Loader2, Key, CheckCircle, ArrowLeft } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuth } from '@/lib/auth-context';

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();
  const [step, setStep] = useState<'beta-key' | 'register'>('beta-key');
  const [betaKey, setBetaKey] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [keyValid, setKeyValid] = useState<'idle' | 'valid' | 'invalid'>('idle');
  const [validatingKey, setValidatingKey] = useState(false);

  const validateBetaKey = async () => {
    if (!betaKey.trim() || betaKey.length < 10) {
      setKeyValid('invalid');
      return;
    }
    
    setValidatingKey(true);
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
      setValidatingKey(false);
    }
  };

  const handleBetaKeySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (keyValid !== 'valid') return;
    setStep('register');
    setError('');
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setLoading(true);
    try {
      await register(email, password, betaKey);
      router.push('/chat');
      router.refresh();
    } catch (err: any) {
      setError(err.message || 'Registration failed');
      setStep('beta-key');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-core-bg px-4 py-12">
      <div className="w-full max-w-md animate-slide-up">
        <div className="flex items-center gap-2 mb-8">
          <Link 
            href="/login" 
            onClick={(e) => { if (step === 'register') { e.preventDefault(); setStep('beta-key'); } }}
            className="p-2 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <Link href="/" className="flex items-center gap-2 ml-auto">
            <div className="w-10 h-10 bg-electric-blue rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-xl">CA</span>
            </div>
            <span className="text-xl font-bold text-text-primary">CoreAI3D</span>
          </Link>
        </div>

        <div className="text-center mb-8">
          {step === 'beta-key' && (
            <>
              <div className="w-14 h-14 bg-electric-blue/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Key className="w-7 h-7 text-electric-blue" />
              </div>
              <h1 className="text-3xl font-bold text-text-primary">Enter Beta Key</h1>
              <p className="text-text-muted mt-2">Beta keys are required for registration during the beta period</p>
            </>
          )}
          {step === 'register' && (
            <>
              <div className="w-14 h-14 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-7 h-7 text-green-400" />
              </div>
              <h1 className="text-3xl font-bold text-text-primary">Create Account</h1>
              <p className="text-text-muted mt-2">Your beta key is valid. Complete your registration.</p>
            </>
          )}
        </div>

        <div className="bg-core-card border border-core-border rounded-xl p-8 space-y-6">
          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {step === 'beta-key' && (
            <form onSubmit={handleBetaKeySubmit} className="space-y-4">
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
                    onBlur={validateBetaKey}
                    placeholder="CA3D-XXXXXXXXXXXXXXXXXXXX"
                    className={cn(
                      'input-base',
                      keyValid === 'valid' && 'border-green-500/50',
                      keyValid === 'invalid' && 'border-red-500/50'
                    )}
                    disabled={validatingKey || loading}
                    autoComplete="off"
                    autoFocus
                    required
                  />
                  {validatingKey && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                      <Loader2 className="w-5 h-5 text-electric-blue animate-spin" />
                    </div>
                  )}
                  {keyValid === 'valid' && !validatingKey && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    </div>
                  )}
                  {keyValid === 'invalid' && !validatingKey && betaKey.length > 0 && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                    </div>
                  )}
                </div>
                {keyValid === 'valid' && (
                  <p className="mt-2 text-sm text-green-400 flex items-center gap-1 justify-center">
                    <CheckCircle className="w-4 h-4" />
                    Valid beta key
                  </p>
                )}
                {keyValid === 'invalid' && betaKey.length > 0 && (
                  <p className="mt-2 text-sm text-red-400 flex items-center gap-1 justify-center">
                    <AlertCircle className="w-4 h-4" />
                    Invalid or already used beta key
                  </p>
                )}
              </div>

              <button
                type="submit"
                disabled={keyValid !== 'valid' || loading}
                className={cn(
                  'w-full py-3 px-4 rounded-lg font-medium transition-colors',
                  'focus:outline-none focus:ring-2 focus:ring-electric-blue focus:ring-offset-2 focus:ring-offset-core-bg',
                  keyValid === 'valid'
                    ? 'bg-electric-blue text-white hover:bg-electric-blue/90'
                    : 'bg-core-cardHover text-text-muted cursor-not-allowed'
                )}
              >
                Continue
              </button>
            </form>
          )}

          {step === 'register' && (
            <form onSubmit={handleRegister} className="space-y-4">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-text-secondary mb-2">
                  Email
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
                  <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="input-base pl-10 bg-core-bgSecondary"
                    placeholder="you@example.com"
                    required
                    autoComplete="email"
                    disabled={loading}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-text-secondary mb-2">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
                  <input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="input-base pl-10 pr-12 bg-core-bgSecondary"
                    placeholder="••••••••"
                    required
                    autoComplete="new-password"
                    disabled={loading}
                    minLength={8}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <div>
                <label htmlFor="confirm-password" className="block text-sm font-medium text-text-secondary mb-2">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
                  <input
                    id="confirm-password"
                    type={showPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="input-base pl-10 bg-core-bgSecondary"
                    placeholder="••••••••"
                    required
                    autoComplete="new-password"
                    disabled={loading}
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className={cn(
                  'w-full py-3 px-4 rounded-lg font-medium transition-colors',
                  'focus:outline-none focus:ring-2 focus:ring-electric-blue focus:ring-offset-2 focus:ring-offset-core-bg',
                  loading
                    ? 'bg-core-cardHover text-text-muted cursor-not-allowed'
                    : 'bg-electric-blue text-white hover:bg-electric-blue/90'
                )}
              >
                {loading ? <Loader2 className="w-5 h-5 mx-auto animate-spin" /> : 'Create Account'}
              </button>
            </form>
          )}

          <div className="bg-core-card/50 border border-core-border rounded-lg p-4 text-sm text-text-muted">
            <p className="flex items-center gap-2">
              <Key className="w-4 h-4 text-electric-blue" />
              <strong>Beta Key:</strong> {betaKey || 'Not entered'}
            </p>
            <p className="mt-2">Each beta key can only be used once. Keys are distributed by CoreAI3D administrators.</p>
          </div>
        </div>

        <p className="text-center text-xs text-text-muted mt-6">
          By continuing, you agree to our{' '}
          <Link href="/terms" className="link-base">Terms of Service</Link>{' '}
          and{' '}
          <Link href="/privacy" className="link-base">Privacy Policy</Link>
        </p>
      </div>
    </div>
  );
}