'use client';

import { useState, useEffect } from 'react';
import { 
  Plus, 
  Trash2, 
  Copy, 
  Check, 
  Loader2, 
  Search,
  Filter,
  Download
} from 'lucide-react';
import { cn, formatDateTime } from '@/lib/utils';
import { api, betaKeysApi } from '@/lib/api-client';

interface BetaKey {
  id: string;
  key: string;
  created_by: string;
  created_at: string;
  used_by: string | null;
  used_at: string | null;
  is_active: number;
  notes: string | null;
  creator_email?: string;
  user_email?: string;
}

export function AdminBetaKeys() {
  const [keys, setKeys] = useState<BetaKey[]>([]);
  const [stats, setStats] = useState({ total: 0, used: 0, active: 0, revoked: 0 });
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [generateCount, setGenerateCount] = useState(1);
  const [generateNotes, setGenerateNotes] = useState('');
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<'all' | 'active' | 'used' | 'revoked'>('all');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const fetchKeys = async () => {
    setLoading(true);
    try {
      const data = await betaKeysApi.list(200, 0);
      setKeys(data.keys);
      setStats(data.stats);
    } catch (error) {
      console.error('Failed to fetch keys:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKeys();
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const data = await betaKeysApi.create(generateCount, generateNotes);
      setKeys(prev => [...data.keys, ...prev]);
      setStats(prev => ({ ...prev, total: prev.total + data.keys.length, active: prev.active + data.keys.length }));
      setShowGenerateModal(false);
      setGenerateCount(1);
      setGenerateNotes('');
    } catch (error) {
      console.error('Failed to generate keys:', error);
      alert('Failed to generate keys');
    } finally {
      setGenerating(false);
    }
  };

  const handleRevoke = async (id: string) => {
    if (!confirm('Revoke this beta key? This action cannot be undone.')) return;
    try {
      await betaKeysApi.revoke(id);
      setKeys(prev => prev.map(k => k.id === id ? { ...k, is_active: 0 } : k));
      setStats(prev => ({ ...prev, active: prev.active - 1, revoked: prev.revoked + 1 }));
    } catch (error) {
      console.error('Failed to revoke key:', error);
      alert('Failed to revoke key');
    }
  };

  const handleCopy = async (key: string) => {
    await navigator.clipboard.writeText(key);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const filteredKeys = keys.filter(k => {
    if (filter === 'active') return k.is_active === 1 && !k.used_by;
    if (filter === 'used') return k.used_by !== null;
    if (filter === 'revoked') return k.is_active === 0 && !k.used_by;
    return true;
  }).filter(k => 
    k.key.toLowerCase().includes(search.toLowerCase()) ||
    k.creator_email?.toLowerCase().includes(search.toLowerCase()) ||
    k.user_email?.toLowerCase().includes(search.toLowerCase()) ||
    k.notes?.toLowerCase().includes(search.toLowerCase())
  );

  const getStatusBadge = (key: BetaKey) => {
    if (key.used_by) return <span className="px-2 py-1 text-xs bg-green-500/20 text-green-400 rounded-full">Used</span>;
    if (!key.is_active) return <span className="px-2 py-1 text-xs bg-red-500/20 text-red-400 rounded-full">Revoked</span>;
    return <span className="px-2 py-1 text-xs bg-electric-blue/20 text-electric-blue rounded-full">Active</span>;
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-core-border">
        <h2 className="text-xl font-bold text-text-primary">Beta Key Management</h2>
        <button
          onClick={() => setShowGenerateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-electric-blue text-white rounded-lg hover:bg-electric-blue/90 transition-colors font-medium"
        >
          <Plus className="w-4 h-4" />
          Generate Keys
        </button>
      </div>

      <div className="p-4 border-b border-core-border grid grid-cols-4 gap-4">
        <div className="bg-core-card border border-core-border rounded-lg p-4">
          <p className="text-text-muted text-sm">Total Generated</p>
          <p className="text-2xl font-bold text-text-primary">{stats.total}</p>
        </div>
        <div className="bg-core-card border border-core-border rounded-lg p-4">
          <p className="text-text-muted text-sm">Active</p>
          <p className="text-2xl font-bold text-electric-blue">{stats.active}</p>
        </div>
        <div className="bg-core-card border border-core-border rounded-lg p-4">
          <p className="text-text-muted text-sm">Used</p>
          <p className="text-2xl font-bold text-green-400">{stats.used}</p>
        </div>
        <div className="bg-core-card border border-core-border rounded-lg p-4">
          <p className="text-text-muted text-sm">Revoked</p>
          <p className="text-2xl font-bold text-red-400">{stats.revoked}</p>
        </div>
      </div>

      <div className="p-4 border-b border-core-border flex gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search keys, emails, notes..."
            className="w-full pl-10 pr-4 py-2 bg-core-card border border-core-border rounded-lg text-text-primary placeholder-text-muted focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue"
          />
        </div>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as any)}
          className="px-4 py-2 bg-core-card border border-core-border rounded-lg text-text-primary focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue"
        >
          <option value="all">All Keys</option>
          <option value="active">Active</option>
          <option value="used">Used</option>
          <option value="revoked">Revoked</option>
        </select>
      </div>

      <div className="flex-1 overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-core-border text-left text-sm font-medium text-text-muted">
              <th className="pb-3 px-4">Key</th>
              <th className="pb-3 px-4">Status</th>
              <th className="pb-3 px-4">Created By</th>
              <th className="pb-3 px-4">Used By</th>
              <th className="pb-3 px-4">Created</th>
              <th className="pb-3 px-4">Notes</th>
              <th className="pb-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-text-muted">
                  <Loader2 className="w-6 h-6 mx-auto animate-spin text-electric-blue" />
                </td>
              </tr>
            ) : filteredKeys.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-text-muted">
                  No beta keys found
                </td>
              </tr>
            ) : (
              filteredKeys.map((key) => (
                <tr key={key.id} className="border-b border-core-border/50 hover:bg-core-cardHover/50">
                  <td className="py-3 px-4 font-mono text-sm text-text-primary max-w-[200px] truncate">
                    {key.key}
                  </td>
                  <td className="py-3 px-4">{getStatusBadge(key)}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{key.creator_email || 'Unknown'}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">
                    {key.user_email || (key.used_by ? 'Unknown' : '-')}
                  </td>
                  <td className="py-3 px-4 text-sm text-text-muted">{formatDateTime(key.created_at)}</td>
                  <td className="py-3 px-4 text-sm text-text-muted max-w-[150px] truncate">
                    {key.notes || '-'}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => handleCopy(key.key)}
                        className={cn(
                          'p-2 rounded-lg transition-colors',
                          copiedKey === key.key
                            ? 'bg-green-500/20 text-green-400'
                            : 'text-text-muted hover:text-text-primary hover:bg-core-cardHover'
                        )}
                        title="Copy key"
                      >
                        {copiedKey === key.key ? (
                          <Check className="w-4 h-4" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </button>
                      {key.is_active === 1 && !key.used_by && (
                        <button
                          onClick={() => handleRevoke(key.id)}
                          className="p-2 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
                          title="Revoke key"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {showGenerateModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-core-card border border-core-border rounded-xl p-6 w-full max-w-md animate-slide-up">
            <h3 className="text-lg font-bold text-text-primary mb-4">Generate Beta Keys</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">
                  Number of Keys
                </label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={generateCount}
                  onChange={(e) => setGenerateCount(parseInt(e.target.value) || 1)}
                  className="w-full px-4 py-2 bg-core-bgSecondary border border-core-border rounded-lg text-text-primary focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">
                  Notes (optional)
                </label>
                <textarea
                  value={generateNotes}
                  onChange={(e) => setGenerateNotes(e.target.value)}
                  rows={3}
                  placeholder="Optional notes about these keys..."
                  className="w-full px-4 py-2 bg-core-bgSecondary border border-core-border rounded-lg text-text-primary focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowGenerateModal(false)}
                className="px-4 py-2 bg-core-cardHover border border-core-border rounded-lg text-text-primary hover:bg-core-border transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleGenerate}
                disabled={generating}
                className="flex items-center gap-2 px-4 py-2 bg-electric-blue text-white rounded-lg hover:bg-electric-blue/90 transition-colors disabled:opacity-50"
              >
                {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                Generate
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}