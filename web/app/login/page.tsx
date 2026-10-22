'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Mail, Lock, Eye, EyeOff, AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuth } from '@/lib/auth-context';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(email, password);
      router.push('/chat');
      router.refresh();
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleAdminLogin = async () => {
    setError('');
    setLoading(true);
    setIsAdmin(true);

    try {
      await login('admin@coreai3d.local', 'admin123');
      router.push('/dashboard');
      router.refresh();
    } catch (err: any) {
      setError(err.message || 'Admin login failed');
      setIsAdmin(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-core-bg px-4">
      <div className="w-full max-w-md animate-slide-up">
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2 mb-6">
            <div className="w-12 h-12 bg-electric-blue rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-2xl">CA</span>
            </div>
            <span className="text-2xl font-bold text-text-primary">CoreAI3D</span>
          </Link>
          <h1 className="text-3xl font-bold text-text-primary">Welcome back</h1>
          <p className="text-text-muted mt-2">Sign in to continue to CoreAI3D</p>
        </div>

        <div className="bg-core-card border border-core-border rounded-xl p-8 space-y-6">
          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
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
                  className={cn(
                    'input-base pl-10',
                    'bg-core-bgSecondary'
                  )}
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
                  className={cn(
                    'input-base pl-10 pr-12',
                    'bg-core-bgSecondary'
                  )}
                  placeholder="••••••••"
                  required
                  autoComplete="current-password"
                  disabled={loading}
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
              {loading ? <Loader2 className="w-5 h-5 mx-auto animate-spin" /> : 'Sign In'}
            </button>
          </form>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-core-border" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-core-card text-text-muted">Or</span>
            </div>
          </div>

          <button
            onClick={handleAdminLogin}
            disabled={loading}
            className={cn(
              'w-full py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
              'focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-core-bg',
              loading
                ? 'bg-core-cardHover text-text-muted cursor-not-allowed'
                : 'bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 border border-purple-500/30'
            )}
          >
            <Loader2 className="w-4 h-4" />
            Login as Admin (Demo)
          </button>

          <p className="text-center text-sm text-text-muted">
            Don't have an account?{' '}
            <Link href="/register" className="link-base">Register with Beta Key</Link>
          </p>
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