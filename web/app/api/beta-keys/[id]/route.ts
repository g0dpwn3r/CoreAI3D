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

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = getUserFromRequest(request);
    if (!user || user.role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const { id } = await params;
    const data = await proxyToBackend(request, '/toggle', 'POST', { id: parseInt(id), enable: false });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Revoke beta key error:', error);
    return NextResponse.json({ error: 'Failed to revoke beta key' }, { status: 500 });
  }
}