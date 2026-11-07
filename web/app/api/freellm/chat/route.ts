import { NextResponse } from 'next/server';

const BASE = process.env.FREELLM_API_URL || 'http://127.0.0.1:3001/v1';
const KEY = process.env.FREELLM_API_KEY || '';

export async function POST(req: Request) {
  if (!KEY) {
    return NextResponse.json(
      { error: 'FREELLM_API_KEY is not configured on the server' },
      { status: 500 },
    );
  }
  try {
    const body = await req.json();
    const res = await fetch(`${BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${KEY}`,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: { 'content-type': res.headers.get('content-type') || 'application/json' },
    });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'proxy error' },
      { status: 500 },
    );
  }
}

export async function GET() {
  return NextResponse.json({ error: 'use POST' }, { status: 405 });
}
