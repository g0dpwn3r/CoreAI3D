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

    const body = await request.json();
    const { message, model, conversation_id, files } = body;

    if (!message || !message.trim()) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    let conversationId = conversation_id;
    
    if (!conversationId) {
      conversationId = uuidv4();
      const title = message.substring(0, 50) + (message.length > 50 ? '...' : '');
      db.prepare(`
        INSERT INTO conversations (id, user_id, title, model)
        VALUES (?, ?, ?, ?)
      `).run(conversationId, user.id, title, model || 'coreai3d');
    }

    db.prepare(`
      INSERT INTO messages (id, conversation_id, role, content)
      VALUES (?, ?, 'user', ?)
    `).run(uuidv4(), conversationId, message);

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          const backendUrl = process.env.COREAI3D_API_URL || 'http://coreai3d:8080';
          
          const response = await fetch(`${backendUrl}/api/chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${user.id}`,
            },
            body: JSON.stringify({
              message,
              model: model || 'coreai3d',
              conversation_id: conversationId,
              stream: true,
            }),
          });

          if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body');
          }

          let fullResponse = '';
          let isThinking = true;
          let thinkingSent = false;

          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'thinking', conversation_id: conversationId })}\n\n`));
          thinkingSent = true;

          // Collect all raw chunks from the backend
          const rawChunks: string[] = [];

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = new TextDecoder().decode(value);
            rawChunks.push(chunk);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.substring(6);
                try {
                  const parsed = JSON.parse(data);
                  
                  if (parsed.type === 'thinking' && isThinking) {
                    continue;
                  }
                  
                  if (parsed.type === 'token' || parsed.content) {
                    if (isThinking) {
                      isThinking = false;
                      controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'generating' })}\n\n`));
                    }
                    fullResponse += parsed.content || parsed.token || '';
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'token', content: parsed.content || parsed.token })}\n\n`));
                  } else if (parsed.type === 'done') {
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'done', conversation_id: conversationId })}\n\n`));
                  }
                } catch {
                  if (isThinking) {
                    isThinking = false;
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'generating' })}\n\n`));
                  }
                  fullResponse += data;
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'token', content: data })}\n\n`));
                }
              }
            }
          }

          // If the backend returned regular JSON (not SSE), parse it and send the response as a token
          if (!fullResponse && rawChunks.length > 0) {
            const rawText = rawChunks.join('');
            try {
              const parsed = JSON.parse(rawText);
              if (parsed.response) {
                if (isThinking) {
                  isThinking = false;
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'generating' })}\n\n`));
                }
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'token', content: parsed.response })}\n\n`));
                fullResponse = parsed.response;
              } else if (parsed.message) {
                if (isThinking) {
                  isThinking = false;
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'generating' })}\n\n`));
                }
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'token', content: parsed.message })}\n\n`));
                fullResponse = parsed.message;
              }
            } catch {
              // Not JSON either, skip
            }
          }

          if (fullResponse) {
            db.prepare(`
              INSERT INTO messages (id, conversation_id, role, content)
              VALUES (?, ?, 'assistant', ?)
            `).run(uuidv4(), conversationId, fullResponse);
            
            db.prepare(`
              UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
            `).run(conversationId);
          }

        } catch (error) {
          console.error('Chat stream error:', error);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'error', error: 'Failed to get response from AI' })}\n\n`));
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('Chat error:', error);
    return NextResponse.json({ error: 'Chat failed' }, { status: 500 });
  }
}