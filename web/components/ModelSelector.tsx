'use client';

import { cn } from '@/lib/utils';
import { ChevronDown, Sparkles, Zap, Brain } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

interface Model {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  context_window: number;
  max_output: number;
  is_default: boolean;
}

interface ModelSelectorProps {
  models: Model[];
  selectedModel: string;
  onSelect: (modelId: string) => void;
  className?: string;
}

export function ModelSelector({ models, selectedModel, onSelect, className }: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node) &&
          buttonRef.current && !buttonRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selected = models.find(m => m.id === selectedModel) || models[0];
 const filteredModels = models.filter(m => 
  m.name.toLowerCase().includes(search.toLowerCase()) ||
  (m.description?.toLowerCase().includes(search.toLowerCase()) ?? false)
);

  const getCapabilityIcon = (cap: string) => {
    switch (cap) {
      case 'reasoning': return <Brain className="w-3 h-3" />;
      case 'code': return <Zap className="w-3 h-3" />;
      default: return <Sparkles className="w-3 h-3" />;
    }
  };

  return (
    <div className={cn('relative', className)}>
      <button
        ref={buttonRef}
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-2 px-3 py-2 bg-core-cardHover border border-core-border rounded-lg',
          'hover:border-electric-blue/50 transition-colors',
          'text-sm font-medium text-text-primary'
        )}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="flex items-center gap-1 text-electric-blue">
          {selected.is_default && <Sparkles className="w-4 h-4" />}
        </span>
        <span className="truncate max-w-[150px]">{selected.name}</span>
        <ChevronDown className={cn('w-4 h-4 text-text-muted transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div
          ref={dropdownRef}
          className="absolute right-0 top-full mt-2 w-72 bg-core-card border border-core-border rounded-xl shadow-blue-glow overflow-hidden z-50 animate-slide-up"
          role="listbox"
        >
          <div className="p-2 border-b border-core-border">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="w-full px-3 py-2 bg-core-bgSecondary border border-core-border rounded-lg text-text-primary placeholder-text-muted focus:border-electric-blue focus:outline-none focus:ring-1 focus:ring-electric-blue"
            />
          </div>
          <div className="max-h-60 overflow-y-auto">
            {filteredModels.map((model) => (
              <button
                key={model.id}
                onClick={() => { onSelect(model.id); setOpen(false); }}
                className={cn(
                  'w-full px-3 py-3 text-left hover:bg-core-cardHover transition-colors',
                  'flex flex-col gap-1',
                  selectedModel === model.id && 'bg-electric-blue/10 border-l-2 border-electric-blue'
                )}
                role="option"
                aria-selected={selectedModel === model.id}
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium text-text-primary">{model.name}</span>
                  {model.is_default && (
                    <span className="px-1.5 py-0.5 text-xs bg-electric-blue/20 text-electric-blue rounded-full">
                      Default
                    </span>
                  )}
                </div>
                <p className="text-xs text-text-muted line-clamp-1">{model.description}</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {model.capabilities.slice(0, 4).map((cap) => (
                    <span key={cap} className="flex items-center gap-1 px-2 py-0.5 text-xs bg-core-bgSecondary text-text-muted rounded-full">
                      {getCapabilityIcon(cap)}
                      <span className="capitalize">{cap}</span>
                    </span>
                  ))}
                  {model.capabilities.length > 4 && (
                    <span className="px-2 py-0.5 text-xs text-text-muted">+{model.capabilities.length - 4} more</span>
                  )}
                </div>
                <div className="flex gap-4 mt-2 text-xs text-text-muted">
                  <span>Context: {model.context_window.toLocaleString()}</span>
                  <span>Max output: {model.max_output.toLocaleString()}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}