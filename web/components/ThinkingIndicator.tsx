'use client';

import { cn } from '@/lib/utils';

interface ThinkingIndicatorProps {
  state: 'thinking' | 'generating';
  detail?: string;
  className?: string;
}

export function ThinkingIndicator({ state, detail, className }: ThinkingIndicatorProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 text-electric-blueLight/80 font-medium',
        'animate-thinking',
        className
      )}
      role="status"
      aria-live="polite"
    >
      <div className="flex gap-1">
        <span className="w-2 h-2 rounded-full bg-electric-blue animate-bounce" style={{ animationDelay: '0ms' }}></span>
        <span className="w-2 h-2 rounded-full bg-electric-blue animate-bounce" style={{ animationDelay: '150ms' }}></span>
        <span className="w-2 h-2 rounded-full bg-electric-blue animate-bounce" style={{ animationDelay: '300ms' }}></span>
      </div>
      <span className="text-sm">
        {state === 'thinking' ? 'Thinking' : 'Generating'}
        {detail && <span className="text-text-muted ml-1">— {detail}</span>}
      </span>
      {state === 'thinking' && (
        <span className="text-text-muted text-sm animate-pulse">...</span>
      )}
    </div>
  );
}
