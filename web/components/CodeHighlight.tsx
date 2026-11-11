'use client';

import { useMemo } from 'react';

const KEYWORDS: Record<string, string[]> = {
  python: ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or', 'True', 'False', 'None', 'lambda', 'async', 'await', 'pass', 'break', 'continue', 'raise', 'yield'],
  py: ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or', 'True', 'False', 'None', 'lambda', 'async', 'await', 'pass', 'break', 'continue', 'raise', 'yield'],
  javascript: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'new', 'this', 'true', 'false', 'null', 'undefined', 'try', 'catch', 'finally', 'throw', 'switch', 'case', 'default', 'break', 'continue'],
  js: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'new', 'this', 'true', 'false', 'null', 'undefined', 'try', 'catch', 'finally', 'throw', 'switch', 'case', 'default', 'break', 'continue'],
  jsx: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'new', 'this', 'true', 'false', 'null', 'undefined', 'try', 'catch', 'finally', 'throw', 'switch', 'case', 'default', 'break', 'continue'],
  typescript: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'new', 'this', 'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'extends', 'implements', 'public', 'private', 'protected', 'static', 'readonly', 'try', 'catch', 'finally', 'throw', 'switch', 'case', 'default', 'break', 'continue'],
  ts: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'new', 'this', 'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'extends', 'implements', 'public', 'private', 'protected', 'static', 'readonly', 'try', 'catch', 'finally', 'throw', 'switch', 'case', 'default', 'break', 'continue'],
  tsx: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'new', 'this', 'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'extends', 'implements', 'public', 'private', 'protected', 'static', 'readonly', 'try', 'catch', 'finally', 'throw', 'switch', 'case', 'default', 'break', 'continue'],
  cpp: ['int', 'float', 'double', 'char', 'bool', 'void', 'long', 'short', 'unsigned', 'signed', 'const', 'static', 'return', 'if', 'else', 'for', 'while', 'do', 'class', 'struct', 'include', 'using', 'namespace', 'std', 'true', 'false', 'nullptr', 'public', 'private', 'protected', 'virtual', 'override', 'new', 'delete', 'try', 'catch', 'throw', 'template', 'typename', 'auto'],
  c: ['int', 'float', 'double', 'char', 'bool', 'void', 'long', 'short', 'unsigned', 'signed', 'const', 'static', 'return', 'if', 'else', 'for', 'while', 'do', 'struct', 'include', 'typedef', 'extern', 'sizeof', 'true', 'false', 'NULL'],
  java: ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'int', 'String', 'boolean', 'return', 'if', 'else', 'for', 'while', 'do', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'import', 'package', 'true', 'false', 'null'],
  php: ['function', 'class', 'public', 'private', 'protected', 'static', 'return', 'if', 'else', 'elseif', 'for', 'while', 'foreach', 'try', 'catch', 'echo', 'print', 'use', 'namespace', 'new', 'true', 'false', 'null', 'array', 'as'],
  go: ['func', 'package', 'import', 'type', 'struct', 'interface', 'return', 'if', 'else', 'for', 'range', 'var', 'const', 'true', 'false', 'nil', 'chan', 'map', 'go', 'defer'],
  rust: ['fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'return', 'if', 'else', 'for', 'while', 'match', 'use', 'mod', 'pub', 'true', 'false', 'None', 'Some', 'Ok', 'Err'],
  rs: ['fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'return', 'if', 'else', 'for', 'while', 'match', 'use', 'mod', 'pub', 'true', 'false', 'None', 'Some', 'Ok', 'Err'],
};

export function CodeHighlight({ language, code }: { language: string; code: string }) {
  const lines = useMemo(() => code.split('\n'), [code]);
  const keywords = KEYWORDS[language.toLowerCase()] || [];

  return (
    <pre className="overflow-x-auto text-sm font-mono leading-relaxed whitespace-pre">
      {lines.map((line, i) => (
        <div key={i} className="flex">
          <span className="select-none text-text-muted w-10 text-right pr-3">{i + 1}</span>
          <code className="flex-1 whitespace-pre">{highlightLine(line, keywords)}</code>
        </div>
      ))}
    </pre>
  );
}

function highlightLine(line: string, keywords: string[]) {
  const tokens = line.split(/(\s+|[(){}[\],;:.<>\/\*+-=!?&|%^])/);
  return tokens.map((token, i) => {
    if (keywords.includes(token)) {
      return <span key={i} className="text-electric-blue font-semibold">{token}</span>;
    }
    if (/^["'`].*["'`]$/.test(token)) {
      return <span key={i} className="text-green-400">{token}</span>;
    }
    if (/^\/\/.*$/.test(line) && token.startsWith('//')) {
      return <span key={i} className="text-text-muted italic">{line.substring(line.indexOf(token))}</span>;
    }
    if (line.trim().startsWith('#') && token.startsWith('#')) {
      return <span key={i} className="text-text-muted italic">{line.substring(line.indexOf(token))}</span>;
    }
    if (/^\d+$/.test(token)) {
      return <span key={i} className="text-yellow-400">{token}</span>;
    }
    return <span key={i}>{token}</span>;
  });
}
