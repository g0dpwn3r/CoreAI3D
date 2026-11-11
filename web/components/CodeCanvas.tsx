'use client';

import { cn } from '@/lib/utils';
import { useState, useEffect, useRef } from 'react';
import { Copy, Check, Download, RotateCcw, ChevronDown, ChevronUp, FileCode2 } from 'lucide-react';
import { CodeHighlight } from './CodeHighlight';

interface CodeCanvasProps {
  language: string;
  filename: string;
  code: string;
}

const LANGUAGE_LABELS: Record<string, string> = {
  py: 'Python',
  python: 'Python',
  js: 'JavaScript',
  javascript: 'JavaScript',
  jsx: 'JavaScript JSX',
  ts: 'TypeScript',
  typescript: 'TypeScript',
  tsx: 'TypeScript TSX',
  cpp: 'C++',
  c: 'C',
  h: 'C Header',
  hpp: 'C++ Header',
  java: 'Java',
  php: 'PHP',
  go: 'Go',
  rust: 'Rust',
  rs: 'Rust',
  cs: 'C#',
  csharp: 'C#',
  rb: 'Ruby',
  ruby: 'Ruby',
  swift: 'Swift',
  kotlin: 'Kotlin',
  scala: 'Scala',
  sql: 'SQL',
  json: 'JSON',
  xml: 'XML',
  html: 'HTML',
  css: 'CSS',
  scss: 'SCSS',
  less: 'LESS',
  sh: 'Shell',
  bash: 'Bash',
  zsh: 'Zsh',
  ps1: 'PowerShell',
  powershell: 'PowerShell',
  yaml: 'YAML',
  yml: 'YAML',
  toml: 'TOML',
  ini: 'INI',
  md: 'Markdown',
  markdown: 'Markdown',
  r: 'R',
  matlab: 'MATLAB',
  perl: 'Perl',
  lua: 'Lua',
  dart: 'Dart',
  dockerfile: 'Dockerfile',
  graphql: 'GraphQL',
  gql: 'GraphQL',
  text: 'Plain Text',
  txt: 'Plain Text',
};

const getLanguageLabel = (lang: string) => {
  const key = lang.toLowerCase().trim();
  return LANGUAGE_LABELS[key] || lang.toUpperCase();
};

export function CodeCanvas({ language, filename, code }: CodeCanvasProps) {
  const [mounted, setMounted] = useState(false);
  const [edited, setEdited] = useState(code);
  const [expanded, setExpanded] = useState(true);
  const [mode, setMode] = useState<'view' | 'edit'>('view');
  const [copied, setCopied] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Update edited state when the code prop changes (e.g. conversation reload, streaming)
  useEffect(() => {
    setEdited(code);
  }, [code]);

  useEffect(() => {
    setMounted(true);
    // Initialize ResizeObserver for responsive layout
    if (containerRef.current && window.ResizeObserver) {
      const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          // Handle container resize if needed
          // This ensures the canvas adapts to parent container changes
        }
      });
      observer.observe(containerRef.current);
      return () => observer.disconnect();
    }
  }, []);

  const label = getLanguageLabel(language);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(edited);
      setCopied(true);
      setMessage('Copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
      setTimeout(() => setMessage(null), 2000);
    } catch {
      setMessage('Copy failed');
      setTimeout(() => setMessage(null), 2000);
    }
  };

  const handleDownload = () => {
    try {
      const blob = new Blob([edited], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      setMessage('Downloaded');
      setTimeout(() => setMessage(null), 2000);
    } catch {
      setMessage('Download failed');
      setTimeout(() => setMessage(null), 2000);
    }
  };

  const handleReset = () => {
    setEdited(code);
    setMessage('Reset to original');
    setTimeout(() => setMessage(null), 2000);
  };

  return (
    <div ref={containerRef} className="relative w-full border border-core-border rounded-xl overflow-hidden bg-core-bgSecondary shadow-lg">
      {!mounted ? (
        <div className="p-4 bg-gray-900/50 text-gray-400 rounded-lg animate-pulse">
          Initializing Code Canvas...
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between px-4 py-2 border-b border-core-border bg-core-card">
            <div className="flex items-center gap-2 text-sm">
              <FileCode2 className="w-4 h-4 text-electric-blue" />
              <span className="font-mono text-text-primary">{filename}</span>
              <span className="px-1.5 py-0.5 text-xs bg-electric-blue/10 text-electric-blue rounded-md border border-electric-blue/30">
                {label}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={handleCopy} className="p-1.5 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-md transition-colors" aria-label="Copy code">
                {copied ? <Check className="w-4 h-4 text-electric-blue" /> : <Copy className="w-4 h-4" />}
              </button>
              <button onClick={handleDownload} className="p-1.5 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-md transition-colors" aria-label="Download code">
                <Download className="w-4 h-4" />
              </button>
              <button onClick={handleReset} className="p-1.5 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-md transition-colors" aria-label="Reset code">
                <RotateCcw className="w-4 h-4" />
              </button>
              <button onClick={() => setExpanded((v) => !v)} className="p-1.5 text-text-muted hover:text-text-primary hover:bg-core-cardHover rounded-md transition-colors" aria-label="Toggle canvas">
                {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
          </div>
          {expanded && (
            <div className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={() => setMode('view')}
                  className={cn('px-2.5 py-1 text-xs rounded-md transition-colors', mode === 'view' ? 'bg-electric-blue text-white' : 'bg-core-card text-text-muted hover:text-text-primary')}
                >
                  View
                </button>
                <button
                  onClick={() => setMode('edit')}
                  className={cn('px-2.5 py-1 text-xs rounded-md transition-colors', mode === 'edit' ? 'bg-electric-blue text-white' : 'bg-core-card text-text-muted hover:text-text-primary')}
                >
                  Edit
                </button>
              </div>
              {mode === 'view' ? (
                <div className="bg-core-bg border border-core-border rounded-lg p-3 overflow-x-auto">
                  <CodeHighlight language={language} code={edited} />
                </div>
              ) : (
              <textarea
                value={edited}
                onChange={(e) => setEdited(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    const target = e.target as HTMLTextAreaElement;
                    const start = target.selectionStart;
                    const end = target.selectionEnd;
                    const newValue = edited.slice(0, start) + '\n' + edited.slice(end);
                    setEdited(newValue);
                    // Move cursor to the new line
                    requestAnimationFrame(() => {
                      target.selectionStart = target.selectionEnd = start + 1;
                    });
                  } else if (e.key === 'Tab') {
                    e.preventDefault();
                    const target = e.target as HTMLTextAreaElement;
                    const start = target.selectionStart;
                    const end = target.selectionEnd;
                    const newValue = edited.slice(0, start) + '    ' + edited.slice(end);
                    setEdited(newValue);
                    // Move cursor after the inserted spaces
                    requestAnimationFrame(() => {
                      target.selectionStart = target.selectionEnd = start + 4;
                    });
                  }
                }}
                className={cn(
                  'w-full min-h-[200px] max-h-[600px] resize-y font-mono text-sm bg-core-bg text-text-primary',
                  'border border-core-border rounded-lg p-3 focus:outline-none focus:ring-1 focus:ring-electric-blue'
                )}
                spellCheck={false}
              />
              )}
              {message && <p className="mt-2 text-xs text-text-muted">{message}</p>}
            </div>
          )}
        </>
      )}
    </div>
  );
}
