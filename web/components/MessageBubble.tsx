'use client';

import { cn, formatDateTime } from '@/lib/utils';
import { Copy, Check } from 'lucide-react';
import { useState, ReactNode } from 'react';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  onCopy?: () => void;
  isStreaming?: boolean;
  children?: ReactNode; // <--- Add this property
}

export function MessageBubble({ role, content, timestamp, isStreaming, onCopy, children }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    onCopy?.();
  };

  return (
    <div
      className={cn(
        'flex gap-3 max-w-3xl animate-fade-in',
        role === 'user' ? 'flex-row-reverse justify-end' : 'flex-row'
      )}
    >
      <div
        className={cn(
          'flex-1 min-w-0',
          role === 'user' ? 'text-right' : 'text-left'
        )}
      >
        <div
          className={cn(
            'inline-block max-w-[85%] px-4 py-3 rounded-2xl',
            role === 'user'
              ? 'bg-electric-blue text-white rounded-tr-sm'
              : 'bg-core-card border border-core-border rounded-tl-sm'
          )}
        >
          <div className="whitespace-pre-wrap text-base leading-relaxed">
            {content}
            {isStreaming && <span className="inline-block w-1 h-5 bg-electric-blue animate-pulse ml-1 align-bottom" />}
          </div>
          {children}
        </div>
        
        <div className="flex items-center gap-2 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {role === 'assistant' && (
            <button
              onClick={handleCopy}
              className="p-1.5 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-lg transition-colors"
              aria-label="Copy message"
            >
              {copied ? (
                <Check className="w-4 h-4 text-electric-blue" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>
          )}
          {timestamp && (
            <span className="text-xs text-text-muted">
              {formatDateTime(timestamp)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}