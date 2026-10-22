'use client';

import { cn, formatDateTime } from '@/lib/utils';
import { 
  MessageSquare, 
  Plus, 
  X, 
  Trash2, 
  ChevronLeft,
  Settings,
  LayoutDashboard,
  BookOpen,
  CreditCard,
  Users,
  Key,
  LogOut
} from 'lucide-react';
import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface Conversation {
  id: string;
  title: string;
  model: string;
  created_at: string;
  updated_at: string;
  last_message?: string;
  message_count?: number;
}

interface SidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  user?: { role: string } | null;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

const NAV_ITEMS = [
  { href: '/chat', label: 'Chat', icon: MessageSquare },
  { href: '/models', label: 'Models', icon: BookOpen },
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
] as const;

const ADMIN_NAV_ITEMS = [
  { href: '/dashboard?tab=users', label: 'Users', icon: Users },
  { href: '/dashboard?tab=beta-keys', label: 'Beta Keys', icon: Key },
] as const;

export function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  user,
  collapsed = false,
  onToggleCollapse,
}: SidebarProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-full bg-core-bgSecondary border-r border-core-border',
        'flex flex-col transition-all duration-300 z-40',
        collapsed ? 'w-16' : 'w-72'
      )}
    >
      <div className="flex items-center justify-between h-16 px-4 border-b border-core-border">
        {!collapsed && (
          <Link href="/chat" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-electric-blue rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">CA</span>
            </div>
            <span className="font-semibold text-xl text-text-primary">CoreAI3D</span>
          </Link>
        )}
        <button
          onClick={onToggleCollapse}
          className={cn(
            'p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-core-cardHover transition-colors',
            collapsed && 'ml-auto'
          )}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronLeft className="w-5 h-5 rotate-180" /> : <ChevronLeft className="w-5 h-5" />}
        </button>
      </div>

      {!collapsed && (
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="flex items-center gap-2 mb-4">
            <button
              onClick={onNewConversation}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-electric-blue text-white rounded-lg hover:bg-electric-blue/90 transition-colors font-medium"
            >
              <Plus className="w-4 h-4" />
              New Chat
            </button>
          </div>

          <div className="border-t border-core-border pt-3">
            <h3 className="px-3 text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              Conversations
            </h3>
            {conversations.length === 0 ? (
              <p className="px-3 py-4 text-center text-text-muted text-sm">
                No conversations yet. Start a new chat!
              </p>
            ) : (
              <ul className="space-y-1" role="listbox">
                {conversations.map((conv) => (
                  <li key={conv.id} role="option" aria-selected={currentConversationId === conv.id}>
                    <button
                      onClick={() => onSelectConversation(conv.id)}
                      onMouseEnter={() => setHoveredId(conv.id)}
                      onMouseLeave={() => setHoveredId(null)}
                      className={cn(
                        'w-full px-3 py-2.5 rounded-lg text-left transition-colors',
                        'group flex items-start gap-2',
                        currentConversationId === conv.id
                          ? 'bg-electric-blue/10 border border-electric-blue/30'
                          : 'hover:bg-core-cardHover'
                      )}
                    >
                      <div className="flex-1 min-w-0">
                        <p className={cn(
                          'font-medium truncate',
                          currentConversationId === conv.id ? 'text-electric-blue' : 'text-text-primary'
                        )}>
                          {conv.title || 'Untitled'}
                        </p>
                        <div className="flex items-center gap-2 mt-1 text-xs text-text-muted">
                          <span className="truncate flex-1">{conv.last_message || 'No messages'}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1 text-xs text-text-muted">
                          <span>{formatDateTime(conv.updated_at)}</span>
                          <span className="px-1.5 py-0.5 bg-core-bg text-text-muted rounded">{conv.message_count || 0} msgs</span>
                        </div>
                      </div>
                      {hoveredId === conv.id && currentConversationId !== conv.id && (
                        <button
                          onClick={(e) => { e.stopPropagation(); onDeleteConversation(conv.id); }}
                          className="p-1.5 text-text-muted hover:text-red-400 hover:bg-red-400/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                          aria-label="Delete conversation"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </nav>
      )}

      <nav className={cn('p-3 space-y-1', collapsed && 'items-center')}>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || (item.href !== '/chat' && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                'group',
                isActive
                  ? 'bg-electric-blue/10 text-electric-blue border border-electric-blue/30'
                  : 'text-text-muted hover:text-text-primary hover:bg-core-cardHover'
              )}
              title={collapsed ? item.label : undefined}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && <span className="font-medium">{item.label}</span>}
            </Link>
          );
        })}

        {user?.role === 'admin' && (
          <>
            <div className="border-t border-core-border my-2" />
            {ADMIN_NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname.includes(item.href.split('=')[1] || '');
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                    'group',
                    isActive
                      ? 'bg-electric-blue/10 text-electric-blue border border-electric-blue/30'
                      : 'text-text-muted hover:text-text-primary hover:bg-core-cardHover'
                  )}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {!collapsed && <span className="font-medium">{item.label}</span>}
                </Link>
              );
            })}
          </>
        )}

        <div className="border-t border-core-border mt-2 pt-2" />
        <Link
          href="/login"
          className={cn(
            'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
            'group text-text-muted hover:text-red-400 hover:bg-red-400/10'
          )}
          title={collapsed ? 'Logout' : undefined}
        >
          <LogOut className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span className="font-medium">Logout</span>}
        </Link>
      </nav>
    </aside>
  );
}