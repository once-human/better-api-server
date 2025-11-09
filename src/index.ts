import { Hono } from 'hono';
import { cors } from 'hono/cors';

type Bindings = {
  GROQ_API_KEY?: string;
  GEMINI_API_KEY?: string;
  RATE_KV: KVNamespace;
};

const app = new Hono<{ Bindings: Bindings }>();

app.use('*', cors({
  origin: '*',                    // tighten to your domain in prod
  allowMethods: ['POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'X-Client-Id'],
}));

async function rateLimit(env: Bindings, id: string) {
  const bucket = `rl:${id || 'anon'}:${Math.floor(Date.now()/600000)}`; // 10-min window
  const current = parseInt(await env.RATE_KV.get(bucket) ?? '0', 10) + 1;
  await env.RATE_KV.put(bucket, String(current), { expirationTtl: 660 });
  return current <= 60; // 60 requests / 10 minutes
}

function toOpenAIResponse(text: string) {
  return { choices: [ { message: { content: text } } ] };
}

function openAIToGeminiBody(body: any) {
  // Map OpenAI messages → Gemini "contents"
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  let systemPrefix = '';
  const contents: any[] = [];

  for (const m of messages) {
    if (!m?.role || !m?.content) continue;
    if (m.role === 'system') {
      systemPrefix += (systemPrefix ? '\n' : '') + String(m.content);
      continue;
    }
    const role = m.role === 'assistant' ? 'model' : 'user';
    let content = String(m.content);
    if (systemPrefix && role === 'user') {
      content = `${systemPrefix}\n\n${content}`;
      systemPrefix = ''; // inject once
    }
    contents.push({ role, parts: [{ text: content }] });
  }
  if (systemPrefix) {
    // no user message appeared; create one
    contents.push({ role: 'user', parts: [{ text: systemPrefix }] });
  }

  const generationConfig: any = {};
  if (typeof body?.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body?.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;

  return { contents, generationConfig };
}

async function callGroq(env: Bindings, body: any) {
  if (!env.GROQ_API_KEY) throw new Error('groq_key_missing');
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GROQ_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: body?.model ?? 'llama-3.1-70b-versatile',
      messages: body?.messages ?? [],
      temperature: body?.temperature ?? 0.7,
      max_tokens: body?.max_tokens,
      stream: false,
    }),
  });
  if (!res.ok) throw new Error(`groq_http_${res.status}`);
  const json: any = await res.json();
  const txt = json?.choices?.[0]?.message?.content;
  if (typeof txt !== 'string' || !txt.trim()) throw new Error('groq_empty');
  return toOpenAIResponse(txt);
}

async function callGemini(env: Bindings, body: any) {
  if (!env.GEMINI_API_KEY) throw new Error('gemini_key_missing');
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(body?.model ?? 'gemini-1.5-pro')}:generateContent?key=${env.GEMINI_API_KEY}`;
  const gemBody = openAIToGeminiBody(body);

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(gemBody),
  });
  if (!res.ok) throw new Error(`gemini_http_${res.status}`);
  const json: any = await res.json();

  const parts = json?.candidates?.[0]?.content?.parts;
  let txt = '';
  if (Array.isArray(parts)) {
    txt = parts.map((p: any) => p?.text ?? '').filter(Boolean).join('\n').trim();
  }
  if (!txt) throw new Error('gemini_empty');
  return toOpenAIResponse(txt);
}

app.post('/v1/chat', async (c) => {
  try {
    const env = c.env;
    const clientId = c.req.header('X-Client-Id') ?? 'anon';
    if (!(await rateLimit(env, clientId))) {
      return c.json({ error: 'rate_limited' }, 429);
    }

    const body = await c.req.json();
    const provider = (body?.provider ?? 'auto') as 'auto'|'groq'|'gemini';
    const fallback = Boolean(body?.enable_fallback ?? true);

    if (provider === 'groq') {
      return c.json(await callGroq(env, body));
    }
    if (provider === 'gemini') {
      return c.json(await callGemini(env, body));
    }

    // auto: try Groq → fallback Gemini (if enabled)
    try {
      const out = await callGroq(env, body);
      return c.json(out);
    } catch (_) {
      if (fallback && env.GEMINI_API_KEY) {
        try {
          const out = await callGemini(env, body);
          return c.json(out);
        } catch {
          return c.json({ error: 'both_providers_failed' }, 500);
        }
      }
      return c.json({ error: 'groq_failed_no_fallback' }, 502);
    }
  } catch (e) {
    return c.json({ error: 'server_error' }, 500);
  }
});

export default app;
