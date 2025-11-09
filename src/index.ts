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
    origin: '*', // lock to your domains in prod
    allowMethods: ['POST', 'GET', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'X-Client-Id'],
  })
);

// ---------- Config / Priorities ----------
const GROQ_SPEED_PRIORITY = [
  'llama-3.1-8b-instant',
  'llama-3.2-11b-vision',
  'llama-3.2-3b-instruct',
  'llama-3.3-70b-versatile', // fallback to quality if speed SKUs missing
];
const GROQ_QUALITY_PRIORITY = [
  'llama-3.3-70b-versatile',
  'llama-3.1-70b-versatile',
  'llama-3.1-8b-instant',
];

const GEMINI_SPEED_PRIORITY = [
  'gemini-2.5-flash-lite',
  'gemini-2.5-flash',
  'gemini-2.0-flash',
  'gemini-2.5-pro', // fallback to quality if flashes missing
];
const GEMINI_QUALITY_PRIORITY = [
  'gemini-2.5-pro',
  'gemini-2.0-pro',
  'gemini-2.5-flash',
];

type Preset = 'speed' | 'quality';

// ---------- Helpers ----------
async function rateLimit(env: Bindings, id: string) {
  const bucket = `rl:${id || 'anon'}:${Math.floor(Date.now() / 600000)}`; // 10-min window
  const current = parseInt((await env.RATE_KV.get(bucket)) ?? '0', 10) + 1;
  await env.RATE_KV.put(bucket, String(current), { expirationTtl: 660 }); // TTL >= 60
  return current <= 60; // 60 req / 10 min
}

function toOpenAIResponse(text: string) {
  return { choices: [{ message: { content: text } }] };
}

function isDecommissionedHttp(status: number, body: string) {
  if (status === 404) return true;
  if (status === 400 && /model.*(decommissioned|not found)/i.test(body)) return true;
  return false;
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

// ---------- Live model discovery ----------
async function listGroqModels(apiKey: string): Promise<string[]> {
  const r = await fetch('https://api.groq.com/openai/v1/models', {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!r.ok) return [];
  const j = await r.json().catch(() => ({}));
  const ids: string[] = Array.isArray(j?.data)
    ? j.data.map((m: any) => m?.id).filter((s: any) => typeof s === 'string')
    : [];
  return ids;
}

async function listGeminiModels(apiKey: string): Promise<string[]> {
  const r = await fetch(`https://generativelanguage.googleapis.com/v1/models?key=${apiKey}`);
  if (!r.ok) return [];
  const j = await r.json().catch(() => ({}));
  const ids: string[] = Array.isArray(j?.models)
    ? j.models.map((m: any) => m?.name).filter((s: any) => typeof s === 'string')
    : [];
  // names are like "models/gemini-2.5-pro" -> normalize to "gemini-2.5-pro"
  return ids.map((s) => s.replace(/^models\//, ''));
}

function pickFirstAvailable(priority: string[], available: string[]): string | null {
  for (const p of priority) if (available.includes(p)) return p;
  return null;
}

// ---------- Providers (with auto model selection) ----------
async function resolveGroqModel(env: Bindings, requested: string | undefined, preset: Preset) {
  const available = await listGroqModels(env.GROQ_API_KEY!);
  if (requested && available.includes(requested)) return requested;
  const priority = preset === 'quality' ? GROQ_QUALITY_PRIORITY : GROQ_SPEED_PRIORITY;
  return pickFirstAvailable(priority, available) ?? (available[0] || 'llama-3.1-8b-instant');
}

async function resolveGeminiModel(env: Bindings, requested: string | undefined, preset: Preset) {
  const available = await listGeminiModels(env.GEMINI_API_KEY!);
  if (requested && available.includes(requested)) return requested;
  const priority = preset === 'quality' ? GEMINI_QUALITY_PRIORITY : GEMINI_SPEED_PRIORITY;
  return pickFirstAvailable(priority, available) ?? (available[0] || 'gemini-2.5-flash');
}

async function callGroq(env: Bindings, body: any, preset: Preset) {
  if (!env.GROQ_API_KEY) throw new Error('groq_key_missing');

  let model = await resolveGroqModel(env, body?.model, preset);
  const req = {
    model,
    messages: body?.messages ?? [],
    temperature: body?.temperature ?? 0.7,
    max_tokens: body?.max_tokens,
    stream: false,
  };

  let res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.GROQ_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    // If the requested model was invalid/decommissioned, retry with fresh resolution
    if (isDecommissionedHttp(res.status, t)) {
      model = await resolveGroqModel(env, undefined, preset);
      const retryReq = { ...req, model };
      res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${env.GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(retryReq),
      });
      if (!res.ok) {
        const t2 = await res.text().catch(() => '');
        throw new Error(`groq_http_${res.status}:${t2}`);
      }
    } else {
      throw new Error(`groq_http_${res.status}:${t}`);
    }
  }

  const json: any = await res.json().catch((e) => {
    throw new Error(`groq_json:${String(e)}`);
  });

  const txt = json?.choices?.[0]?.message?.content;
  if (!txt || typeof txt !== 'string') throw new Error('groq_empty');
  return toOpenAIResponse(txt);
}

async function callGemini(env: Bindings, body: any, preset: Preset) {
  if (!env.GEMINI_API_KEY) throw new Error('gemini_key_missing');

  let model = await resolveGeminiModel(env, body?.model, preset);
  const url = (m: string) =>
    `https://generativelanguage.googleapis.com/v1/models/${encodeURIComponent(m)}:generateContent?key=${env.GEMINI_API_KEY}`;
  const gemBody = openAIToGeminiBody(body);

  let res = await fetch(url(model), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(gemBody),
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    if (isDecommissionedHttp(res.status, t)) {
      model = await resolveGeminiModel(env, undefined, preset);
      res = await fetch(url(model), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gemBody),
      });
      if (!res.ok) {
        const t2 = await res.text().catch(() => '');
        throw new Error(`gemini_http_${res.status}:${t2}`);
      }
    } else {
      throw new Error(`gemini_http_${res.status}:${t}`);
    }
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
    await c.env.RATE_KV.put('health', '1', { expirationTtl: 60 });
    out.kv = 'ok';
  } catch (e) {
    out.kv = 'error';
    out.kv_error = String(e);
  }

  out.groq_secret = !!c.env.GROQ_API_KEY;
  out.gemini_secret = !!c.env.GEMINI_API_KEY;

  // Optional light pings removed to keep health quick & cheap.

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
    const preset: Preset = (body?.preset === 'quality' ? 'quality' : 'speed');

    if (provider === 'groq') {
      try {
        return c.json(await callGroq(env, body, preset));
      } catch (e) {
        return c.json({ error: `groq_failed:${String(e)}` }, 502);
      }
    }

    if (provider === 'gemini') {
      try {
        return c.json(await callGemini(env, body, preset));
      } catch (e) {
        return c.json({ error: `gemini_failed:${String(e)}` }, 502);
      }
    }

    // auto: Groq → Gemini fallback
    try {
      const out = await callGroq(env, body, preset);
      return c.json(out);
    } catch (e1) {
      if (env.GEMINI_API_KEY) {
        try {
          const out = await callGemini(env, body, preset);
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
