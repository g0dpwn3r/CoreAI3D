'use client';

import { useState } from 'react';
import { 
  User, 
  Mail, 
  Key, 
  Copy, 
  Check, 
  Loader2,
  Shield,
  Settings
} from 'lucide-react';
import { cn, formatDateTime } from '@/lib/utils';
import { api, userApi } from '@/lib/api-client';

interface ProfileCardProps {
  user: {
    id: string;
    email: string;
    role: string;
    name: string | null;
    avatar_url: string | null;
    api_key: string | null;
    created_at: string;
  } | null;
  stats: {
    conversations_count: number;
    messages_count: number;
  } | null;
  onUpdate: (user: any) => void;
}

export function ProfileCard({ user, stats, onUpdate }: ProfileCardProps) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(user?.name || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  const handleSave = async () => {
    try {
      const data = await userApi.updateProfile({ name });
      onUpdate(data.user);
      setEditing(false);
    } catch (error) {
      console.error('Failed to update profile:', error);
      alert('Failed to update profile');
    }
  };

  const handleCopyApiKey = async () => {
    if (user?.api_key) {
      await navigator.clipboard.writeText(user.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRegenerateApiKey = async () => {
    if (!confirm('Regenerate API key? This will invalidate your current key.')) return;
    setRegenerating(true);
    try {
      const res = await fetch('/api/user/regenerate-api-key', {
        method: 'POST',
        credentials: 'include',
      });
      const data = await res.json();
      if (data.api_key) {
        onUpdate({ ...user, api_key: data.api_key });
      }
    } catch (error) {
      console.error('Failed to regenerate API key:', error);
      alert('Failed to regenerate API key');
    } finally {
      setRegenerating(false);
    }
  };

  const getInitials = (name: string | null, email: string) => {
    if (name) return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
    return email.slice(0, 2).toUpperCase();
  };

  return (
    <div className="space-y-6">
      <div className="bg-core-card border border-core-border rounded-xl p-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-electric-blue/20 rounded-xl flex items-center justify-center">
            {user?.avatar_url ? (
              <img src={user.avatar_url} alt={user.name || user.email} className="w-16 h-16 rounded-xl" />
            ) : (
              <span className="text-2xl font-bold text-electric-blue">
                {getInitials(user?.name || null, user?.email || '')}
              </span>
            )}
          </div>
          <div className="flex-1">
            {editing ? (
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="px-3 py-2 bg-core-bgSecondary border border-core-border rounded-lg text-text-primary focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue"
                autoFocus
              />
            ) : (
              <h2 className="text-xl font-bold text-text-primary">{user?.name || 'Unnamed User'}</h2>
            )}
            <p className="text-text-muted flex items-center gap-2 mt-1">
              <Mail className="w-4 h-4" />
              {user?.email}
            </p>
            <div className="flex items-center gap-4 mt-2">
              <span className={cn(
                'px-2 py-1 text-xs rounded-full',
                user?.role === 'admin'
                  ? 'bg-purple-500/20 text-purple-400'
                  : 'bg-electric-blue/20 text-electric-blue'
              )}>
                {user?.role}
              </span>
              <span className="text-sm text-text-muted">
                Member since {formatDateTime(user?.created_at || '')}
              </span>
            </div>
          </div>
          {editing ? (
            <div className="flex gap-2">
              <button onClick={handleSave} className="px-4 py-2 bg-electric-blue text-white rounded-lg hover:bg-electric-blue/90 transition-colors">
                Save
              </button>
              <button onClick={() => { setName(user?.name || ''); setEditing(false); }} className="px-4 py-2 bg-core-cardHover text-text-primary rounded-lg hover:bg-core-border transition-colors">
                Cancel
              </button>
            </div>
          ) : (
            <button onClick={() => setEditing(true)} className="px-4 py-2 bg-core-cardHover text-text-primary rounded-lg hover:bg-core-border transition-colors flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Edit
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-core-card border border-core-border rounded-xl p-4">
          <p className="text-text-muted text-sm">Conversations</p>
          <p className="text-2xl font-bold text-text-primary">{stats?.conversations_count || 0}</p>
        </div>
        <div className="bg-core-card border border-core-border rounded-xl p-4">
          <p className="text-text-muted text-sm">Messages</p>
          <p className="text-2xl font-bold text-text-primary">{stats?.messages_count || 0}</p>
        </div>
        <div className="bg-core-card border border-core-border rounded-xl p-4">
          <p className="text-text-muted text-sm">API Requests</p>
          <p className="text-2xl font-bold text-text-primary">—</p>
        </div>
      </div>

      <div className="bg-core-card border border-core-border rounded-xl p-6">
        <h3 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
          <Key className="w-5 h-5 text-electric-blue" />
          API Key
        </h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-electric-blue/10 rounded-lg flex items-center justify-center">
                <Key className="w-5 h-5 text-electric-blue" />
              </div>
              <div>
                <p className="font-medium text-text-primary">CoreAI3D API Key</p>
                <p className="text-sm text-text-muted">Used for programmatic access</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopyApiKey}
                className={cn(
                  'p-2 rounded-lg transition-colors flex items-center gap-2',
                  copied ? 'bg-green-500/20 text-green-400' : 'text-text-muted hover:text-text-primary hover:bg-core-cardHover'
                )}
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                <span className="hidden sm:inline">{copied ? 'Copied!' : 'Copy'}</span>
              </button>
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-core-cardHover transition-colors"
              >
                {showApiKey ? <Shield className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
              </button>
              <button
                onClick={handleRegenerateApiKey}
                disabled={regenerating}
                className="px-3 py-1.5 text-sm bg-core-cardHover border border-core-border text-text-muted hover:text-red-400 hover:border-red-500/50 rounded-lg transition-colors disabled:opacity-50"
              >
                {regenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Regenerate'}
              </button>
            </div>
          </div>
          <div className={cn(
            'font-mono text-sm bg-core-bgSecondary border border-core-border rounded-lg p-4',
            'break-all'
          )}>
            {showApiKey && user?.api_key ? user.api_key : '••••••••••••••••••••••••••••••••'}
          </div>
          <p className="text-xs text-text-muted">
            Keep your API key secret. Anyone with this key can make requests on your behalf.
          </p>
        </div>
      </div>
    </div>
  );
}