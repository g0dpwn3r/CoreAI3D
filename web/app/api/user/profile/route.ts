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
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userData = db.prepare(`
      SELECT id, email, role, name, avatar_url, api_key, created_at
      FROM users WHERE id = ?
    `).get(user.id);

    const stats = db.prepare(`
      SELECT 
        (SELECT COUNT(*) FROM conversations WHERE user_id = ?) as conversations_count,
        (SELECT COUNT(*) FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)) as messages_count
    `).get(user.id, user.id);

    return NextResponse.json({ user: userData, stats });
  } catch (error) {
    console.error('Get profile error:', error);
    return NextResponse.json({ error: 'Failed to fetch profile' }, { status: 500 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const user = getUserFromRequest(request);
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await request.json();
    const { name, avatar_url } = body;

    db.prepare(`
      UPDATE users SET name = ?, avatar_url = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
    `).run(name, avatar_url, user.id);

    const updatedUser = db.prepare(`
      SELECT id, email, role, name, avatar_url, api_key, created_at
      FROM users WHERE id = ?
    `).get(user.id);

    return NextResponse.json({ user: updatedUser });
  } catch (error) {
    console.error('Update profile error:', error);
    return NextResponse.json({ error: 'Failed to update profile' }, { status: 500 });
  }
}
export const dynamic = 'force-dynamic';
