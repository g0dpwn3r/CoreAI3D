import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.COREAI3D_API_URL || 'http://coreai3d:8080';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { beta_key } = body;

    if (!beta_key) {
      return NextResponse.json({ valid: false, error: 'Beta key is required' }, { status: 400 });
    }

    const response = await fetch(`${BACKEND_URL}/api/v1/beta/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: beta_key }),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Validate beta key error:', error);
    return NextResponse.json({ valid: false, error: 'Validation failed' }, { status: 500 });
  }
}