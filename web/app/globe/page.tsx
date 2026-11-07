'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { Sidebar } from '@/components/Sidebar';
import { GlobeFrame, GlobeControl, GlobeMIC } from '@/components/GlobeFrame';
import { useGlobeBridge } from '@/lib/gep-bridge';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { chatApi } from '@/lib/api-client';

function GlobePageFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-core-bg">
      <Loader2 className="w-8 h-8 text-electric-blue animate-spin" />
    </div>
  );
}

export default function GlobePage() {
  return (
    <Suspense fallback={<GlobePageFallback />}>
      <GlobePageContent />
    </Suspense>
  );
}

function GlobePageContent() {
  const router = useRouter();
  const pathname = usePathname();
  const { user, loading: authLoading } = useAuth();
  const [conversations, setConversations] = useState<any[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);

  const bridge = useGlobeBridge({
    onEvent: (event) => {
      console.log('[globe] event', event);
    },
  });

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      chatApi.getConversations().then((data) => setConversations(data.conversations || [])).catch(() => {});
    }
  }, [user]);

  const handleSelectConversation = (id: string | null) => {
    setCurrentConversationId(id);
    router.push(`/chat?conversation=${id}`);
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-core-bg">
        <Loader2 className="w-8 h-8 text-electric-blue animate-spin" />
      </div>
    );
  }
  if (!user) return null;

  return (
    <div className="min-h-screen bg-core-bg">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={() => {}}
        onDeleteConversation={() => {}}
        user={user}
        collapsed={false}
      />

      <div className={cn('transition-all duration-300 lg:ml-72')}>
        <header className="fixed top-0 right-0 h-16 bg-core-bgSecondary/80 backdrop-blur-sm border-b border-core-border z-30 lg:ml-72" style={{ left: '288px' }}>
          <div className="flex items-center justify-between h-full px-4">
            <div className="flex items-center gap-4">
              <h1 className="text-lg font-semibold text-text-primary">God's Eye View</h1>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => router.push('/chat')}
                className="p-2 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-lg transition-colors" title="Back to chat">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1H5z" /></svg>
              </button>
            </div>
          </div>
        </header>

        <main className="pt-16 h-[calc(100vh-4rem)] flex flex-col lg:flex-row">
          <div className="flex-1 min-w-0">
            <GlobeFrame />
          </div>
          <aside className="w-80 border-l border-core-border flex flex-col">
            <GlobeControl
              send={bridge.send}
              sendAndAwait={bridge.sendAndAwait}
              ready={bridge.ready}
            />
            <div className="flex-1" />
            <div className="p-4 border-t border-core-border">
              <GlobeMIC send={bridge.send} ready={bridge.ready} />
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}
