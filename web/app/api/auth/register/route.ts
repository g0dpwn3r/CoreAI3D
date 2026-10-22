import { NextRequest, NextResponse } from 'next/server';
import db from '@/lib/db';
import { hashPassword, generateToken, verifyToken, getTokenFromHeader } from '@/lib/auth';
import { v4 as uuidv4 } from 'uuid';

const BACKEND_URL = process.env.COREAI3D_API_URL || 'http://coreai3d:8080';

async function validateBetaKeyWithBackend(betaKey: string): Promise<{ valid: boolean; error?: string; keyData?: any }> {
  try {
    const response = await fetch(`${BACKEND_URL}/api/v1/beta/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: betaKey }),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Beta key validation error:', error);
    return { valid: false, error: 'Validation service unavailable' };
  }
}

async function useBetaKeyWithBackend(betaKey: string, userId: string): Promise<boolean> {
  try {
    const response = await fetch(`${BACKEND_URL}/api/v1/beta/use`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: betaKey, user_id: userId }),
    });
    
    const data = await response.json();
    return data.success === true;
  } catch (error) {
    console.error('Beta key usage error:', error);
    return false;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password, beta_key } = body;

    if (!email || !password) {
      return NextResponse.json({ error: 'Email and password are required' }, { status: 400 });
    }

    if (!beta_key) {
      return NextResponse.json({ error: 'Beta key is required for registration' }, { status: 400 });
    }

    const validation = await validateBetaKeyWithBackend(beta_key);
    if (!validation.valid) {
      return NextResponse.json({ error: validation.error }, { status: 400 });
    }

    const existingUser = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
    if (existingUser) {
      return NextResponse.json({ error: 'Email already registered' }, { status: 400 });
    }

    const userId = uuidv4();
    const passwordHash = hashPassword(password);
    const apiKey = `ca3d_${uuidv4().replace(/-/g, '')}`;

    db.prepare(`
      INSERT INTO users (id, email, password_hash, api_key, role)
      VALUES (?, ?, ?, ?, 'user')
    `).run(userId, email, passwordHash, apiKey);

    const used = await useBetaKeyWithBackend(beta_key, userId);
    if (!used) {
      console.warn('Failed to mark beta key as used in backend, but user created');
    }

    const token = generateToken({ id: userId, email, role: 'user' });

    const response = NextResponse.json({
      user: { id: userId, email, role: 'user' },
      token,
    });

    response.cookies.set('auth_token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7,
      path: '/',
    });

    return response;
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json({ error: 'Registration failed' }, { status: 500 });
  }
}