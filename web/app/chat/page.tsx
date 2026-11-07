'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { ChatWindow } from '@/components/ChatWindow';
import { Sidebar } from '@/components/Sidebar';
import { useAuth } from '@/lib/auth-context';
import { chatApi, modelsApi } from '@/lib/api-client';
import { Loader2, Menu, X, ChevronLeft } from 'lucide-react';
import { cn } from '@/lib/utils';
import { freellm } from '@/lib/freellm';

function ChatPageFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-core-bg">
      <Loader2 className="w-8 h-8 text-electric-blue animate-spin" />
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<ChatPageFallback />}>
      <ChatPageContent />
    </Suspense>
  );
}

function ChatPageContent() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { user, loading: authLoading, logout } = useAuth();
  const [conversations, setConversations] = useState<any[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [model, setModel] = useState('coreai3d');
  const [models, setModels] = useState<any[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      loadData();
    }
  }, [user]);

  useEffect(() => {
    const convId = searchParams.get('conversation');
    if (convId) {
      setCurrentConversationId(convId);
    }
  }, [searchParams]);

  const loadData = async () => {
    try {
      const [convData, modelsData] = await Promise.all([
        chatApi.getConversations(),
        modelsApi.list(),
      ]);
      let models = modelsData.models;
      if (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_FREELLM_MODELS === 'true') {
        try {
          const freellmModels = await freellm.models();
          models = [
            ...modelsData.models,
            ...freellmModels.map((m) => ({
              ...m,
              provider: 'freellm',
              name: m.name || m.id,
              description: m.description || `FreeLLMAPI route: ${m.id}`,
              capabilities: m.capabilities || [],
              context_window: m.context_window || 0,
              max_output: m.max_output || 0,
              is_default: false,
            })),
          ];
        } catch {
          // FreeLLMAPI unreachable; keep CoreAI3D models only.
        }
      }
      setConversations(convData.conversations);
      setModels(models);
      
      if (!currentConversationId && convData.conversations.length > 0) {
        setCurrentConversationId(convData.conversations[0].id);
      }
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewConversation = async () => {
    try {
      const data = await chatApi.createConversation('New Chat', model);
      setConversations(prev => [data.conversation, ...prev]);
      setCurrentConversationId(data.conversation.id);
      router.push(`/chat?conversation=${data.conversation.id}`);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id: string | null) => {
    setCurrentConversationId(id);
    router.push(`/chat?conversation=${id}`);
    setMobileSidebarOpen(false);
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await chatApi.deleteConversation(id);
      setConversations(prev => prev.filter(c => c.id !== id));
      if (currentConversationId === id) {
        setCurrentConversationId(null);
        router.push('/chat');
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const handleModelChange = (newModel: string) => {
    setModel(newModel);
  };

  if (authLoading || loading) {
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
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        user={user}
        collapsed={!sidebarOpen}
        onToggleCollapse={() => setSidebarOpen(!sidebarOpen)}
      />

      <div className={cn(
        'transition-all duration-300',
        sidebarOpen ? 'lg:ml-72' : 'lg:ml-16'
      )}>
        <header className="fixed top-0 right-0 h-16 bg-core-bgSecondary/80 backdrop-blur-sm border-b border-core-border z-30 lg:ml-72 transition-all duration-300" style={{ left: sidebarOpen ? '288px' : '64px' }}>
          <div className="flex items-center justify-between h-full px-4">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setMobileSidebarOpen(true)}
                className="lg:hidden p-2 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-lg"
              >
                <Menu className="w-5 h-5" />
              </button>
              <h1 className="text-lg font-semibold text-text-primary hidden sm:block">CoreAI3D Chat</h1>
            </div>
            <div className="flex items-center gap-4">
              <button onClick={logout} className="p-2 text-text-muted hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors" title="Logout">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
              </button>
            </div>
          </div>
        </header>

        <main className="pt-16 h-[calc(100vh-4rem)]">
          <ChatWindow
            conversationId={currentConversationId}
            model={model}
            onModelChange={handleModelChange}
            onConversationChange={handleSelectConversation}
            models={models}
          />
        </main>
      </div>

      {mobileSidebarOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setMobileSidebarOpen(false)} />
          <Sidebar
            conversations={conversations}
            currentConversationId={currentConversationId}
            onSelectConversation={handleSelectConversation}
            onNewConversation={handleNewConversation}
            onDeleteConversation={handleDeleteConversation}
            user={user}
            collapsed={false}
            onToggleCollapse={() => {}}
          />
        </div>
      )}
    </div>
  );
}