import { NextRequest, NextResponse } from 'next/server';
import { verifyToken, getTokenFromHeader } from '@/lib/auth';
import db from '@/lib/db';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    const cookieToken = request.cookies.get('auth_token')?.value;
    const token = getTokenFromHeader(authHeader) || cookieToken;

    if (!token) {
      return NextResponse.json({ valid: false, error: 'No token provided' }, { status: 401 });
    }

    const decoded = verifyToken(token);
    if (!decoded) {
      return NextResponse.json({ valid: false, error: 'Invalid token' }, { status: 401 });
    }

    const user = db.prepare('SELECT id, email, role, name, avatar_url, api_key FROM users WHERE id = ?').get(decoded.id) as {
      id: string;
      email: string;
      role: string;
      name: string | null;
      avatar_url: string | null;
      api_key: string | null;
    } | undefined;

    if (!user) {
      return NextResponse.json({ valid: false, error: 'User not found' }, { status: 401 });
    }

    return NextResponse.json({ valid: true, user });
  } catch (error) {
    console.error('Validation error:', error);
    return NextResponse.json({ valid: false, error: 'Validation failed' }, { status: 500 });
  }
}
export const dynamic = 'force-dynamic';
