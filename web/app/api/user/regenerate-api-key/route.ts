import { NextRequest, NextResponse } from 'next/server';
import { verifyToken, getTokenFromHeader } from '@/lib/auth';
import db from '@/lib/db';
import { v4 as uuidv4 } from 'uuid';

function getUserFromRequest(request: NextRequest) {
  const authHeader = request.headers.get('authorization');
  const cookieToken = request.cookies.get('auth_token')?.value;
  const token = getTokenFromHeader(authHeader) || cookieToken;
  
  if (!token) return null;
  return verifyToken(token);
}

export async function POST(request: NextRequest) {
  try {
    const user = getUserFromRequest(request);
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const newApiKey = `ca3d_${uuidv4().replace(/-/g, '')}`;
    
    db.prepare(`
      UPDATE users SET api_key = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
    `).run(newApiKey, user.id);

    return NextResponse.json({ api_key: newApiKey });
  } catch (error) {
    console.error('Regenerate API key error:', error);
    return NextResponse.json({ error: 'Failed to regenerate API key' }, { status: 500 });
  }
}