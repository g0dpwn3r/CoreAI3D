import { NextRequest, NextResponse } from 'next/server';
import { verifyToken, getTokenFromHeader } from '@/lib/auth';
import db from '@/lib/db';

function getUserFromRequest(request: NextRequest) {
  const authHeader = request.headers.get('authorization');
  const cookieToken = request.cookies.get('auth_token')?.value;
  const token = getTokenFromHeader(authHeader) || cookieToken;
  
  if (!token) return null;
  return verifyToken(token);
}

export async function GET(request: NextRequest) {
  try {
    const user = getUserFromRequest(request);
    if (!user || user.role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const users = db.prepare(`
      SELECT id, email, role, name, avatar_url, created_at,
        (SELECT COUNT(*) FROM conversations WHERE user_id = users.id) as conversations_count,
        (SELECT COUNT(*) FROM beta_keys WHERE used_by = users.id) as keys_used
      FROM users
      ORDER BY created_at DESC
    `).all();

    return NextResponse.json({ users });
  } catch (error) {
    console.error('Get users error:', error);
    return NextResponse.json({ error: 'Failed to fetch users' }, { status: 500 });
  }
}
export const dynamic = 'force-dynamic';
