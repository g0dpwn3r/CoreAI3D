import { NextResponse } from 'next/server';

const UPSTREAM = process.env.MCP_UPSTREAM || 'http://127.0.0.1:3001/mcp';
const KEY = process.env.MCP_API_KEY || process.env.FREELLM_API_KEY || '';

function headers() {
  const h: Record<string, string> = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
  if (KEY) h['Authorization'] = `Bearer ${KEY}`;
  return h;
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const res = await fetch(`${UPSTREAM}`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify(body),
    });
    const text = await res.text();
    const ct = res.headers.get('content-type') || 'application/json';
    return new NextResponse(text, { status: res.status, headers: { 'content-type': ct } });
  } catch (err) {
    return NextResponse.json({ error: err instanceof Error ? err.message : 'mcp proxy error' }, { status: 500 });
  }
}

export async function GET() {
  // Discovery: list tools via MCP tools/list on the upstream.
  try {
    const res = await fetch(`${UPSTREAM}`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} }),
    });
    const text = await res.text();
    return new NextResponse(text, { status: res.status, headers: { 'content-type': res.headers.get('content-type') || 'application/json' } });
  } catch (err) {
    return NextResponse.json({ error: err instanceof Error ? err.message : 'mcp proxy error' }, { status: 500 });
  }
}
