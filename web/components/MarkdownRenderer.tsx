'use client';

import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CodeCanvas } from './CodeCanvas';
import SpreadsheetCanvas from './SpreadsheetCanvas';
import { parseMessage, type MessageSegment } from '@/lib/code-detect';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  const segments = useMemo(() => parseMessage(content), [content]);

  return (
    <div className={className}>
      {segments.map((segment, index) => {
        if (segment.type === 'text') {
          return (
            <div key={index} className="markdown-body">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                  code({ node, className: codeClass, children, ...props }: any) {
                    const codeText = String(children);
                    const isBlock = codeClass?.includes('language-');
                    const lang = codeClass?.replace('language-', '') || 'text';

                    if (isBlock) {
                      return <CodeCanvas
                        language={lang}
                        filename={`snippet-${index + 1}.${lang}`}
                        code={codeText.replace(/\n$/, '')}
                      />;
                    }
                    return (
                      <code className="px-1.5 py-0.5 bg-core-bg rounded text-sm font-mono text-electric-blue">
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {segment.content}
              </ReactMarkdown>
            </div>
          );
        }

        if (segment.type === 'code' && segment.block) {
          return (
            <CodeCanvas
              key={index}
              language={segment.block.language}
              filename={segment.block.filename}
              code={segment.block.code}
            />
          );
        }

        if (segment.type === 'data' && segment.dataBlock) {
          return (
            <SpreadsheetCanvas
              key={index}
              csv={segment.dataBlock.data}
              title={segment.dataBlock.filename}
            />
          );
        }

        return null;
      })}
    </div>
  );
}
