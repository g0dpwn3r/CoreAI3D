import { NextRequest, NextResponse } from 'next/server';
import { verifyToken, getTokenFromHeader } from '@/lib/auth';

const BACKEND_URL = process.env.COREAI3D_API_URL || 'http://coreai3d:8080';

function getUserFromRequest(request: NextRequest) {
  const authHeader = request.headers.get('authorization');
  const cookieToken = request.cookies.get('auth_token')?.value;
  const token = getTokenFromHeader(authHeader) || cookieToken;
  
  if (!token) return null;
  return verifyToken(token);
}

async function proxyToBackend(request: NextRequest, endpoint: string, method: string, body?: any) {
  const url = `${BACKEND_URL}/api/v1/beta${endpoint}`;
  
  const response = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': request.headers.get('authorization') || '',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.error || 'Backend request failed');
  }
  
  return data;
}

export async function GET(request: NextRequest) {
  try {
    const user = getUserFromRequest(request);
    if (!user || user.role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '100';
    const offset = searchParams.get('offset') || '0';

    const data = await proxyToBackend(request, `/list?limit=${limit}&offset=${offset}`, 'GET');
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Get beta keys error:', error);
    return NextResponse.json({ error: 'Failed to fetch beta keys' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const user = getUserFromRequest(request);
    if (!user || user.role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const body = await request.json();
    const { count = 1, notes = '', email = '' } = body;

    if (count < 1 || count > 100) {
      return NextResponse.json({ error: 'Count must be between 1 and 100' }, { status: 400 });
    }

    const data = await proxyToBackend(request, '/generate', 'POST', { count, description: notes, email });
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Create beta keys error:', error);
    return NextResponse.json({ error: 'Failed to create beta keys' }, { status: 500 });
  }
}