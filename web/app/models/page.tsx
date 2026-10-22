'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { modelsApi } from '@/lib/api-client';
import { Loader2, Sparkles, Zap, Brain, Check, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Model {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  context_window: number;
  max_output: number;
  is_default: boolean;
}

export default function ModelsPage() {
  const { user, loading: authLoading } = useAuth();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('coreai3d');

  useEffect(() => {
    if (!authLoading && !user) return;
  }, [user, authLoading]);

  useEffect(() => {
    if (user) {
      loadModels();
    }
  }, [user]);

  const loadModels = async () => {
    try {
      const data = await modelsApi.list();
      setModels(data.models);
      setSelectedModel(data.models.find((m: Model) => m.is_default)?.id || data.models[0]?.id);
    } catch (error) {
      console.error('Failed to load models:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCapabilityIcon = (cap: string) => {
    switch (cap) {
      case 'reasoning': return <Brain className="w-4 h-4" />;
      case 'code': return <Zap className="w-4 h-4" />;
      default: return <Sparkles className="w-4 h-4" />;
    }
  };

  const getCapabilityColor = (cap: string) => {
    switch (cap) {
      case 'reasoning': return 'text-purple-400';
      case 'code': return 'text-yellow-400';
      case 'multimodal': return 'text-pink-400';
      default: return 'text-electric-blue';
    }
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
    <div className="min-h-screen bg-core-bg pt-16 pb-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-text-primary mb-4">Available Models</h1>
          <p className="text-text-muted text-lg">
            Choose the right model for your task. Each model has unique capabilities and trade-offs.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {models.map((model) => (
            <div
              key={model.id}
              className={cn(
                'relative bg-core-card border rounded-xl p-6 transition-all',
                selectedModel === model.id
                  ? 'border-electric-blue/50 shadow-blue-glow'
                  : 'border-core-border hover:border-electric-blue/30'
              )}
            >
              {model.is_default && (
                <div className="absolute -top-3 left-4 px-3 py-1 bg-electric-blue text-white text-xs font-medium rounded-full">
                  Default
                </div>
              )}

              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-xl font-bold text-text-primary">{model.name}</h3>
                  <p className="text-text-muted text-sm mt-1">{model.description}</p>
                </div>
                {selectedModel === model.id && (
                  <div className="w-6 h-6 bg-electric-blue rounded-full flex items-center justify-center">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2 mb-6">
                {model.capabilities.map((cap) => (
                  <span
                    key={cap}
                    className={cn(
                      'flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full',
                      `bg-${getCapabilityColor(cap).replace('text-', '').replace('-400', '-500/20')} ${getCapabilityColor(cap)}`
                    )}
                  >
                    {getCapabilityIcon(cap)}
                    <span className="capitalize">{cap}</span>
                  </span>
                ))}
              </div>

              <div className="space-y-3 mb-6 pt-4 border-t border-core-border">
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Context Window</span>
                  <span className="font-medium text-text-primary">{model.context_window.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Max Output</span>
                  <span className="font-medium text-text-primary">{model.max_output.toLocaleString()} tokens</span>
                </div>
              </div>

              <button
                onClick={() => setSelectedModel(model.id)}
                className={cn(
                  'w-full py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
                  selectedModel === model.id
                    ? 'bg-electric-blue/10 text-electric-blue border border-electric-blue/30 cursor-default'
                    : 'bg-electric-blue text-white hover:bg-electric-blue/90'
                )}
                disabled={selectedModel === model.id}
              >
                {selectedModel === model.id ? (
                  <>
                    <Check className="w-4 h-4" />
                    Selected
                  </>
                ) : (
                  <>
                    Use this model
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          ))}
        </div>

        {models.length === 0 && (
          <div className="text-center py-12">
            <Loader2 className="w-8 h-8 text-electric-blue animate-spin mx-auto mb-4" />
            <p className="text-text-muted">No models available</p>
          </div>
        )}
      </div>
    </div>
  );
}