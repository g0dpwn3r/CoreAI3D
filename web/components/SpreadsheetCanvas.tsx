'use client';

import React, { useMemo, useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Download, Table, Plus, Trash2, Edit2, Check, X, RefreshCw } from 'lucide-react';

interface SpreadsheetCanvasProps {
  csv: string;
  title?: string;
  onDataChange?: (data: string) => void;
}

function parseCSV(csv: string): string[][] {
  const rows: string[][] = [];
  const lines = csv.trim().split(/\r?\n/);

  for (const line of lines) {
    // Simple CSV parser (handles quoted fields)
    const row: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        row.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    row.push(current.trim());
    rows.push(row);
  }
  return rows;
}

function serializeCSV(rows: string[][]): string {
  return rows
    .map((row) => {
      const escaped = row.map((cell) => {
        const escapedCell = String(cell ?? '').replace(/"/g, '""');
        // Quote if contains comma, quote, or newline
        if (escapedCell.includes(',') || escapedCell.includes('"') || escapedCell.includes('\n')) {
          return `"${escapedCell}"`;
        }
        return escapedCell;
      });
      return escaped.join(',');
    })
    .join('\n');
}

function downloadFile(content: string, filename: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function SpreadsheetCanvas({ csv, title, onDataChange }: SpreadsheetCanvasProps) {
  const data = useMemo(() => parseCSV(csv), [csv]);
  const [rows, setRows] = useState<string[][]>(data);
  const [headers, setHeaders] = useState<string[]>(data[0] || []);
  const [editingCell, setEditingCell] = useState<{ row: number; col: number } | null>(null);
  const [editValue, setEditValue] = useState('');
  const [copied, setCopied] = useState(false);

  // Sync state when csv prop changes (e.g., during streaming)
  useEffect(() => {
    const parsed = parseCSV(csv);
    setRows(parsed);
    if (parsed.length > 0) {
      setHeaders(parsed[0]);
    }
  }, [csv]);

  const handleDownload = () => {
    const csvContent = serializeCSV(rows);
    downloadFile(csvContent, `${title || 'data'}.csv`, 'text/csv');
    if (onDataChange) onDataChange(csvContent);
  };

  const handleCellEdit = (row: number, col: number) => {
    const cell = rows[row]?.[col] ?? '';
    setEditingCell({ row, col });
    setEditValue(cell);
  };

  const handleCellSave = () => {
    if (editingCell) {
      const { row, col } = editingCell;
      const newRows = [...rows];
      newRows[row] = [...newRows[row]];
      newRows[row][col] = editValue;
      setRows(newRows);
      if (onDataChange) {
        onDataChange(serializeCSV(newRows));
      }
    }
    setEditingCell(null);
  };

  const handleCellCancel = () => {
    setEditingCell(null);
    setEditValue('');
  };

  const handleAddRow = () => {
    const newRow = Array(headers.length).fill('');
    setRows([...rows, newRow]);
    if (onDataChange) {
      onDataChange(serializeCSV([...rows, newRow]));
    }
  };

  const handleAddColumn = () => {
    const header = `column-${headers.length + 1}`;
    setHeaders([...headers, header]);
    const newRows = rows.map((row) => [...row, '']);
    setRows(newRows);
    if (onDataChange) {
      onDataChange(serializeCSV(newRows));
    }
  };

  const handleDeleteRow = (rowIndex: number) => {
    const newRows = rows.filter((_, i) => i !== rowIndex);
    setRows(newRows);
    if (onDataChange) {
      onDataChange(serializeCSV(newRows));
    }
  };

  const handleDeleteColumn = (colIndex: number) => {
    const newHeaders = headers.filter((_, i) => i !== colIndex);
    const newRows = rows.map((row) => row.filter((_, i) => i !== colIndex));
    setHeaders(newHeaders);
    setRows(newRows);
    if (onDataChange) {
      const csvContent = serializeCSV([newHeaders, ...newRows]);
      onDataChange(csvContent);
    }
  };

  const handleCopy = async () => {
    const csvContent = serializeCSV(rows);
    try {
      await navigator.clipboard.writeText(csvContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard failed, ignore
    }
  };

  if (data.length === 0) return null;

  const displayHeaders = rows.length > 0 ? rows[0] : headers;
  const displayRows = rows.length > 1 ? rows.slice(1) : [];

  return (
    <div className="my-4 rounded-xl overflow-hidden border border-zinc-700 bg-[#0d1117] shadow-lg">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/90 border-b border-zinc-700">
        <div className="flex items-center gap-2 text-zinc-400">
          <Table className="w-4 h-4" />
          <span className="text-xs font-medium uppercase tracking-wider">
            {title || 'Spreadsheet'}
          </span>
          <span className="text-xs text-zinc-500">
            {displayRows.length} rows × {displayHeaders.length} cols
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Check className="w-3.5 h-3.5" />}
            Copy
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download
          </button>
          <button
            onClick={handleAddRow}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Row
          </button>
          <button
            onClick={handleAddColumn}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Col
          </button>
          <button
            onClick={() => setRows(data)}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-8 rounded-md transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto max-h-[420px]">
        <table className="w-full text-sm border-collapse">
          <thead className="sticky top-0 bg-zinc-900">
            <tr>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-zinc-300 border-b border-zinc-700 whitespace-nowrap">
                # 
              </th>
              {displayHeaders.map((h, i) => (
                <th
                  key={i}
                  className="px-4 py-2.5 text-left text-xs font-semibold text-zinc-300 border-b border-zinc-700 whitespace-nowrap"
                >
                  {h || `Column ${i + 1}`}
                </th>
              ))}
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-zinc-300 border-b border-zinc-700"></th>
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, ri) => (
              <tr
                key={ri}
                className="hover:bg-zinc-800/50 border-b border-zinc-800/60"
              >
                <td className="px-4 py-2 text-zinc-500 font-mono text-[13px]">
                  {ri + 1}
                </td>
                {row.map((cell, ci) => (
                  <td
                    key={ci}
                    onDoubleClick={() => handleCellEdit(ri + 1, ci + 1)}
                    className={cn(
                      'px-4 py-2 text-zinc-200 whitespace-nowrap font-mono text-[13px] border-r border-zinc-800/60 cursor-cell',
                      editingCell?.row === ri + 1 && editingCell?.col === ci + 1 && 'bg-zinc-800/70'
                    )}
                  >
                    {editingCell?.row === ri + 1 && editingCell?.col === ci + 1 ? (
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onBlur={handleCellSave}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleCellSave();
                          if (e.key === 'Escape') handleCellCancel();
                        }}
                        autoFocus
                        className="w-full bg-zinc-800 text-white px-2 py-0.5 rounded border border-electric-blue focus:outline-none"
                      />
                    ) : (
                      cell
                    )}
                  </td>
                ))}
                <td className="px-2 py-2">
                  <button
                    onClick={() => handleDeleteRow(ri)}
                    className="p-1.5 text-zinc-500 hover:text-red-400 rounded-md transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="px-4 py-2 bg-zinc-900/60 border-t border-zinc-700 flex items-center gap-3 text-xs text-zinc-500">
        <span>Double-click cell to edit</span>
        <span>•</span>
        <span>Double-click a cell to edit</span>
        <span>•</span>
        <span>Use Add Row / Add Col to modify structure</span>
      </div>
    </div>
  );
}
