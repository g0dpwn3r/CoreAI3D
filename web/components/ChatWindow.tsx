'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Paperclip, Mic, Image, FileText, X } from 'lucide-react';
import { MessageBubble } from './MessageBubble';
import { ThinkingIndicator } from './ThinkingIndicator';
import { ModelSelector } from './ModelSelector';
import { cn, generateId } from '@/lib/utils';
import { api, chatApi } from '@/lib/api-client';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  isStreaming?: boolean;
}

interface ChatWindowProps {
  conversationId: string | null;
  model: string;
  onModelChange: (model: string) => void;
  onConversationChange: (id: string | null) => void;
  models: any[];
}

export function ChatWindow({ 
  conversationId, 
  model, 
  onModelChange, 
  onConversationChange,
  models 
}: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamState, setStreamState] = useState<'idle' | 'thinking' | 'generating'>('idle');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (conversationId) {
      loadConversation(conversationId);
    } else {
      setMessages([]);
    }
  }, [conversationId]);

  const loadConversation = async (id: string) => {
    try {
      const data = await chatApi.getConversation(id);
      setMessages(data.messages.map((m: any) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.created_at,
      })));
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && attachedFiles.length === 0) return;
    if (isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setAttachedFiles([]);
    setIsLoading(true);
    setStreamState('thinking');

    const tempId = generateId();
    setMessages(prev => [...prev, {
      id: tempId,
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
    }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          message: userMessage,
          model,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      let assistantContent = '';
      let assistantId = generateId();

      setMessages(prev => [...prev, {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
      }]);

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.substring(6);
            try {
              const parsed = JSON.parse(data);
              
              if (parsed.type === 'thinking') {
                continue;
              }
              
              if (parsed.type === 'token' && parsed.content) {
                assistantContent += parsed.content;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantId 
                    ? { ...msg, content: assistantContent, isStreaming: true }
                    : msg
                ));
              } else if (parsed.type === 'done') {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantId 
                    ? { ...msg, content: assistantContent, isStreaming: false }
                    : msg
                ));
                setStreamState('idle');
                if (parsed.conversation_id && !conversationId) {
                  onConversationChange(parsed.conversation_id);
                }
              } else if (parsed.type === 'error') {
                throw new Error(parsed.error);
              }
            } catch {
              assistantContent += data;
              setMessages(prev => prev.map(msg => 
                msg.id === assistantId 
                  ? { ...msg, content: assistantContent, isStreaming: true }
                  : msg
              ));
            }
          }
        }
      }

      setStreamState('idle');
      setIsLoading(false);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => prev.filter(msg => msg.id !== tempId));
      setMessages(prev => [...prev, {
        id: generateId(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      }]);
      setStreamState('idle');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      if (files.length + attachedFiles.length > 5) {
        alert('Maximum 5 files allowed');
        return;
      }
      setAttachedFiles(prev => [...prev, ...files]);
    }
  };

  const removeFile = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="flex flex-col h-full bg-core-bg">
      <div className="flex items-center justify-between h-16 px-4 border-b border-core-border bg-core-bgSecondary">
        <div className="flex items-center gap-4 flex-1">
          <ModelSelector
            models={models}
            selectedModel={model}
            onSelect={onModelChange}
          />
          {conversationId && (
            <span className="text-sm text-text-muted">Conversation: {conversationId.slice(0, 8)}...</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={cn(
            'px-2 py-1 rounded-full text-xs font-medium',
            streamState === 'thinking' && 'bg-yellow-500/20 text-yellow-400',
            streamState === 'generating' && 'bg-electric-blue/20 text-electric-blue',
            streamState === 'idle' && 'bg-green-500/20 text-green-400'
          )}>
            {streamState === 'thinking' && 'Thinking'}
            {streamState === 'generating' && 'Generating'}
            {streamState === 'idle' && 'Ready'}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && !conversationId && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-electric-blue/10 rounded-2xl flex items-center justify-center mb-6">
              <span className="text-3xl">🤖</span>
            </div>
            <h1 className="text-3xl font-bold text-text-primary mb-2">Welcome to CoreAI3D</h1>
            <p className="text-text-muted max-w-md mb-8">
              Start a conversation by selecting a model and typing a message below.
              CoreAI3D features calculus-based activation functions for enhanced reasoning.
            </p>
            <div className="flex flex-wrap gap-3 justify-center text-sm text-text-muted">
              <span className="px-3 py-1 bg-core-cardHover border border-core-border rounded-full">Try: "Explain calculus activation"</span>
              <span className="px-3 py-1 bg-core-cardHover border border-core-border rounded-full">Try: "Write a Python script"</span>
              <span className="px-3 py-1 bg-core-cardHover border border-core-border rounded-full">Try: "Analyze this code"</span>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            role={message.role}
            content={message.content}
            timestamp={message.timestamp}
            isStreaming={message.isStreaming}
          />
        ))}
       {isLoading && messages.length > 0 && (() => {
          const last = messages[messages.length - 1];
          return last?.role === 'assistant' && last?.isStreaming === true && (last?.content ?? '') === '';
        })() && (
           <MessageBubble
             role="assistant"
             content=""
             isStreaming={true}
           >
             <ThinkingIndicator state={streamState as 'thinking' | 'generating'} />
           </MessageBubble>
         )}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-core-border bg-core-bgSecondary p-4">
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {attachedFiles.map((file, index) => (
              <span key={index} className="flex items-center gap-1 px-2 py-1 bg-core-card border border-core-border rounded-lg text-sm">
                <FileText className="w-4 h-4 text-electric-blue" />
                {file.name}
                <button onClick={() => removeFile(index)} className="text-text-muted hover:text-red-400">
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="flex items-end gap-3">
          <div className="flex items-center gap-1 p-1 bg-core-card border border-core-border rounded-lg">
            <label className="p-2 text-text-muted hover:text-electric-blue cursor-pointer">
              <input type="file" multiple onChange={handleFileChange} className="hidden" accept=".txt,.md,.py,.js,.ts,.json,.pdf,.png,.jpg,.jpeg" />
              <Paperclip className="w-5 h-5" />
            </label>
            <label className="p-2 text-text-muted hover:text-electric-blue cursor-pointer">
              <Image className="w-5 h-5" />
            </label>
            <label className="p-2 text-text-muted hover:text-electric-blue cursor-pointer">
              <Mic className="w-5 h-5" />
            </label>
          </div>

          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message CoreAI3D..."
              className={cn(
                'w-full px-4 py-3 pr-12 bg-core-card border border-core-border rounded-xl',
                'text-text-primary placeholder-text-muted',
                'focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue',
                'resize-none min-h-[52px] max-h-48',
                'font-medium'
              )}
              rows={1}
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() && attachedFiles.length === 0 || isLoading}
              className={cn(
                'absolute right-2 bottom-2 p-2 rounded-lg transition-colors',
                'flex items-center justify-center',
                isLoading || (!input.trim() && attachedFiles.length === 0)
                  ? 'text-text-muted cursor-not-allowed'
                  : 'text-electric-blue hover:bg-electric-blue/10'
              )}
              aria-label="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>

        <p className="text-xs text-text-muted text-center mt-2">
          Press Enter to send, Shift+Enter for new line. CoreAI3D may make mistakes.
        </p>
      </div>
    </div>
  );
}