// src/index.ts
import { Hono } from 'hono';
import { cors } from 'hono/cors';

type Bindings = {
  GROQ_API_KEY?: string;
  GEMINI_API_KEY?: string;
  RATE_KV: KVNamespace;
};

const app = new Hono<{ Bindings: Bindings }>();

// ---------- CORS ----------
app.use(
  '*',
  cors({
    origin: '*', // tighten to your app domains in prod
    allowMethods: ['POST', 'GET', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'X-Client-Id'],
  })
);

// ---------- Helpers ----------
async function rateLimit(env: Bindings, id: string) {
  const bucket = `rl:${id || 'anon'}:${Math.floor(Date.now() / 600000)}`; // 10-min window
  const current = parseInt((await env.RATE_KV.get(bucket)) ?? '0', 10) + 1;
  await env.RATE_KV.put(bucket, String(current), { expirationTtl: 660 });
  return current <= 60; // 60 req / 10 min
}

function toOpenAIResponse(text: string) {
  return { choices: [{ message: { content: text } }] };
}

// Map OpenAI-style messages -> Gemini "contents"
function openAIToGeminiBody(body: any) {
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
      systemPrefix = '';
    }
    contents.push({ role, parts: [{ text: content }] });
  }
  if (systemPrefix) contents.push({ role: 'user', parts: [{ text: systemPrefix }] });

  const generationConfig: any = {};
  if (typeof body?.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body?.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;

  return { contents, generationConfig };
}

// ---------- Providers ----------
async function callGroq(env: Bindings, body: any) {
  if (!env.GROQ_API_KEY) throw new Error('groq_key_missing');

  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.GROQ_API_KEY}`,
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

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`groq_http_${res.status}:${t}`);
  }
  const json: any = await res.json().catch((e) => {
    throw new Error(`groq_json:${String(e)}`);
  });

  const txt = json?.choices?.[0]?.message?.content;
  if (!txt || typeof txt !== 'string') throw new Error('groq_empty');
  return toOpenAIResponse(txt);
}

async function callGemini(env: Bindings, body: any) {
  if (!env.GEMINI_API_KEY) throw new Error('gemini_key_missing');

  const model = encodeURIComponent(body?.model ?? 'gemini-1.5-pro');
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${env.GEMINI_API_KEY}`;
  const gemBody = openAIToGeminiBody(body);

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(gemBody),
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`gemini_http_${res.status}:${t}`);
  }
  const json: any = await res.json().catch((e) => {
    throw new Error(`gemini_json:${String(e)}`);
  });

  const parts = json?.candidates?.[0]?.content?.parts;
  const txt =
    Array.isArray(parts) ? parts.map((p: any) => p?.text ?? '').filter(Boolean).join('\n').trim() : '';

  if (!txt) throw new Error('gemini_empty');
  return toOpenAIResponse(txt);
}

// ---------- Health ----------
app.get('/health', async (c) => {
  const out: any = { ok: true };

  try {
    await c.env.RATE_KV.put('health', '1', { expirationTtl: 30 });
    out.kv = 'ok';
  } catch (e) {
    out.kv = 'error';
    out.kv_error = String(e);
  }

  out.groq_secret = !!c.env.GROQ_API_KEY;
  out.gemini_secret = !!c.env.GEMINI_API_KEY;

  // Light provider probes (don’t block if they fail)
  if (c.env.GROQ_API_KEY) {
    try {
      const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${c.env.GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'llama-3.1-70b-versatile',
          messages: [{ role: 'user', content: 'ping' }],
        }),
      });
      out.groq_status = r.status;
      if (!r.ok) out.groq_body = await r.text();
    } catch (e) {
      out.groq_error = String(e);
    }
  }

  if (c.env.GEMINI_API_KEY) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=${c.env.GEMINI_API_KEY}`;
      const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: 'user', parts: [{ text: 'ping' }] }],
        }),
      });
      out.gemini_status = r.status;
      if (!r.ok) out.gemini_body = await r.text();
    } catch (e) {
      out.gemini_error = String(e);
    }
  }

  return c.json(out);
});

// ---------- Chat endpoint ----------
app.post('/v1/chat', async (c) => {
  try {
    const env = c.env;
    const clientId = c.req.header('X-Client-Id') ?? 'anon';

    // Rate limit
    const allowed = await rateLimit(env, clientId);
    if (!allowed) return c.json({ error: 'rate_limited' }, 429);

    const body = await c.req.json();
    const provider = (body?.provider ?? 'auto') as 'auto' | 'groq' | 'gemini';
    const fallback = Boolean(body?.enable_fallback ?? true);

    if (provider === 'groq') {
      try {
        return c.json(await callGroq(env, body));
      } catch (e) {
        return c.json({ error: `groq_failed:${String(e)}` }, 502);
      }
    }

    if (provider === 'gemini') {
      try {
        return c.json(await callGemini(env, body));
      } catch (e) {
        return c.json({ error: `gemini_failed:${String(e)}` }, 502);
      }
    }

    // auto: Groq → Gemini fallback
    try {
      const out = await callGroq(env, body);
      return c.json(out);
    } catch (e1) {
      if (fallback && env.GEMINI_API_KEY) {
        try {
          const out = await callGemini(env, body);
          return c.json(out);
        } catch (e2) {
          return c.json({ error: `fallback_failed:${String(e2)}` }, 500);
        }
      }
      return c.json({ error: `groq_failed_no_fallback:${String(e1)}` }, 502);
    }
  } catch (e) {
    return c.json({ error: `server_error:${String(e)}` }, 500);
  }
});

export default app;
