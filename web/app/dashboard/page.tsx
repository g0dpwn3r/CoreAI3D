'use client';

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { userApi, adminApi } from '@/lib/api-client';
import { ProfileCard } from '@/components/ProfileCard';
import { BillingSection } from '@/components/BillingSection';
import { AdminBetaKeys } from '@/components/AdminBetaKeys';
import { Loader2, Users, Key, LayoutDashboard, CreditCard, User } from 'lucide-react';
import { cn } from '@/lib/utils';

const TABS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'billing', label: 'Billing', icon: CreditCard },
] as const;

const ADMIN_TABS = [
  { id: 'users', label: 'Users', icon: Users },
  { id: 'beta-keys', label: 'Beta Keys', icon: Key },
] as const;

function DashboardFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-core-bg">
      <Loader2 className="w-8 h-8 text-electric-blue animate-spin" />
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<DashboardFallback />}>
      <DashboardPageContent />
    </Suspense>
  );
}

function DashboardPageContent() {
  const searchParams = useSearchParams();
  const { user, loading: authLoading } = useAuth();
  const [profileData, setProfileData] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(
    searchParams.get('tab') || 'profile'
  );

  useEffect(() => {
    if (!authLoading && !user) return;
  }, [user, authLoading]);

  useEffect(() => {
    if (user) {
      loadData();
    }
  }, [user]);

  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && (TABS.some(t => t.id === tab) || ADMIN_TABS.some(t => t.id === tab))) {
      setActiveTab(tab);
    }
  }, [searchParams]);

  const loadData = async () => {
    try {
      const [profileRes, usersRes] = await Promise.all([
        userApi.getProfile(),
        user?.role === 'admin' ? adminApi.getUsers() : Promise.resolve({ users: [] }),
      ]);
      setProfileData(profileRes.user);
      setStats(profileRes.stats);
      if (usersRes.users) setUsers(usersRes.users);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleProfileUpdate = (updatedUser: any) => {
    setProfileData(updatedUser);
  };

  const allTabs = user?.role === 'admin' ? [...TABS, ...ADMIN_TABS] : TABS;

  if (authLoading || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-core-bg">
        <Loader2 className="w-8 h-8 text-electric-blue animate-spin" />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="min-h-screen bg-core-bg pt-16 pb-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-text-primary mb-2">Dashboard</h1>
          <p className="text-text-muted">Manage your account, billing, and settings</p>
        </div>

        <div className="flex flex-col lg:flex-row gap-8">
          <nav className="lg:w-48 flex-shrink-0">
            <div className="bg-core-card border border-core-border rounded-xl overflow-hidden">
              <ul className="p-2 space-y-1" role="tablist">
                {allTabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <li key={tab.id} role="presentation">
                      <button
                        onClick={() => setActiveTab(tab.id)}
                        role="tab"
                        aria-selected={isActive}
                        className={cn(
                          'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                          isActive
                            ? 'bg-electric-blue/10 text-electric-blue border border-electric-blue/30'
                            : 'text-text-muted hover:text-text-primary hover:bg-core-cardHover'
                        )}
                      >
                        <Icon className="w-5 h-5 flex-shrink-0" />
                        {tab.label}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          </nav>

          <div className="flex-1 min-w-0">
            {activeTab === 'profile' && (
              <ProfileCard 
                user={profileData} 
                stats={stats} 
                onUpdate={handleProfileUpdate} 
              />
            )}
            {activeTab === 'billing' && <BillingSection />}
            {activeTab === 'users' && user?.role === 'admin' && (
              <div className="bg-core-card border border-core-border rounded-xl overflow-hidden">
                <div className="p-4 border-b border-core-border">
                  <h2 className="text-lg font-bold text-text-primary">User Management</h2>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-core-border text-left text-sm font-medium text-text-muted">
                        <th className="pb-3 px-4">User</th>
                        <th className="pb-3 px-4">Role</th>
                        <th className="pb-3 px-4">Conversations</th>
                        <th className="pb-3 px-4">Keys Used</th>
                        <th className="pb-3 px-4">Joined</th>
                      </tr>
                    </thead>
                    <tbody>
                      {users.map((u) => (
                        <tr key={u.id} className="border-b border-core-border/50 hover:bg-core-cardHover/50">
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 bg-electric-blue/20 rounded-lg flex items-center justify-center">
                                <span className="text-sm font-medium text-electric-blue">
                                  {u.name ? u.name.split(' ').map((n: string) => n[0]).join('').toUpperCase() : u.email.slice(0, 2).toUpperCase()}
                                </span>
                              </div>
                              <div>
                                <p className="font-medium text-text-primary">{u.name || 'Unnamed'}</p>
                                <p className="text-sm text-text-muted">{u.email}</p>
                              </div>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <span className={cn(
                              'px-2 py-1 text-xs rounded-full',
                              u.role === 'admin' ? 'bg-purple-500/20 text-purple-400' : 'bg-electric-blue/20 text-electric-blue'
                            )}>
                              {u.role}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-sm text-text-secondary">{u.conversations_count || 0}</td>
                          <td className="py-3 px-4 text-sm text-text-secondary">{u.keys_used || 0}</td>
                          <td className="py-3 px-4 text-sm text-text-muted">{new Date(u.created_at).toLocaleDateString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {activeTab === 'beta-keys' && user?.role === 'admin' && <AdminBetaKeys />}
          </div>
        </div>
      </div>
    </div>
  );
}