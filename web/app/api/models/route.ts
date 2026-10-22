import { NextRequest, NextResponse } from 'next/server';
import { verifyToken, getTokenFromHeader } from '@/lib/auth';

function getUserFromRequest(request: NextRequest) {
  const authHeader = request.headers.get('authorization');
  const cookieToken = request.cookies.get('auth_token')?.value;
  const token = getTokenFromHeader(authHeader) || cookieToken;
  
  if (!token) return null;
  return verifyToken(token);
}

const MODELS = [
  {
    id: 'coreai3d',
    name: 'CoreAI3D',
    description: 'CoreAI3D flagship model with calculus-based activation functions',
    capabilities: ['chat', 'reasoning', 'code', 'analysis', 'multimodal'],
    context_window: 128000,
    max_output: 4096,
    is_default: true,
  },
  {
    id: 'coreai3d-fast',
    name: 'CoreAI3D Fast',
    description: 'Optimized for speed with reduced latency',
    capabilities: ['chat', 'code', 'quick-tasks'],
    context_window: 32000,
    max_output: 2048,
    is_default: false,
  },
  {
    id: 'coreai3d-reasoning',
    name: 'CoreAI3D Reasoning',
    description: 'Enhanced reasoning capabilities for complex problem solving',
    capabilities: ['reasoning', 'math', 'logic', 'analysis'],
    context_window: 64000,
    max_output: 8192,
    is_default: false,
  },
];

export async function GET(request: NextRequest) {
  try {
    const user = getUserFromRequest(request);
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    return NextResponse.json({ models: MODELS });
  } catch (error) {
    console.error('Get models error:', error);
    return NextResponse.json({ error: 'Failed to fetch models' }, { status: 500 });
  }
}
export const dynamic = 'force-dynamic';
