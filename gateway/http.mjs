import { createServer } from 'node:http';

const MAX_BODY_BYTES = 512 * 1024;

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function send(response, status, payload) {
  if (response.destroyed || response.headersSent) return;
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
  });
  response.end(JSON.stringify(payload));
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let exceeded = false;
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        if (!exceeded) reject(new HttpError(413, 'Corpo maior que 512 KiB.'));
        exceeded = true;
        chunks.length = 0;
      } else if (!exceeded) {
        chunks.push(chunk);
      }
    });
    request.once('end', () => {
      if (exceeded) return;
      try {
        const text = new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks));
        resolve(JSON.parse(text));
      } catch {
        reject(new HttpError(400, 'JSON inválido.'));
      }
    });
    request.once('error', () => reject(new HttpError(400, 'Falha ao ler a requisição.')));
    request.once('aborted', () => reject(new HttpError(400, 'Requisição interrompida.')));
  });
}

export function validatePayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)
      || Object.keys(payload).length !== 1 || !Object.hasOwn(payload, 'texts')
      || !Array.isArray(payload.texts) || payload.texts.length < 1 || payload.texts.length > 32) {
    throw new HttpError(422, 'Envie somente texts com 1 a 32 mensagens.');
  }
  const texts = payload.texts.map((text) => {
    if (typeof text !== 'string' || [...text.trim()].length < 1 || [...text.trim()].length > 4000) {
      throw new HttpError(422, 'Cada mensagem deve conter 1 a 4000 caracteres.');
    }
    return text.trim();
  });
  return { texts };
}

function validPrediction(value) {
  const p = value?.probabilities;
  return ['suporte', 'venda'].includes(value?.label)
    && Number.isFinite(value.confidence) && value.confidence >= 0 && value.confidence <= 1
    && typeof value.needs_review === 'boolean'
    && p && Object.keys(p).length === 2
    && ['suporte', 'venda'].every((label) => Number.isFinite(p[label]) && p[label] >= 0 && p[label] <= 1)
    && Math.abs(p.suporte + p.venda - 1) < 0.0001;
}

export function createGateway({ upstreamUrl, timeoutMs = 30000 }) {
  const upstream = new URL(upstreamUrl);
  if (!['http:', 'https:'].includes(upstream.protocol) || upstream.username || upstream.password
      || upstream.pathname !== '/' || upstream.search || upstream.hash) {
    throw new Error('UPSTREAM_URL deve ser uma origem HTTP(S) sem credenciais ou caminho.');
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 120000) {
    throw new Error('REQUEST_TIMEOUT_MS deve estar entre 100 e 120000.');
  }
  const server = createServer(async (request, response) => {
    try {
      const route = request.url;
      if (!['/health', '/classify'].includes(route)) {
        request.resume();
        return send(response, 404, { error: 'Caminho não encontrado.' });
      }
      const method = route === '/health' ? 'GET' : 'POST';
      if (request.method !== method) {
        request.resume();
        response.setHeader('allow', method);
        return send(response, 405, { error: 'Método não permitido.' });
      }
      let payload;
      if (route === '/classify') {
        if (request.headers['content-type']?.split(';', 1)[0].trim().toLowerCase() !== 'application/json') {
          request.resume();
          return send(response, 415, { error: 'Use Content-Type application/json.' });
        }
        if (Number(request.headers['content-length']) > MAX_BODY_BYTES) {
          request.resume();
          return send(response, 413, { error: 'Corpo maior que 512 KiB.' });
        }
        payload = validatePayload(await readJson(request));
      } else {
        request.resume();
      }
      const signal = AbortSignal.timeout(timeoutMs);
      try {
        const result = await fetch(new URL(route, upstream), {
          method, signal, redirect: 'error',
          headers: payload ? { 'content-type': 'application/json' } : {},
          body: payload ? JSON.stringify(payload) : undefined,
        });
        if (!result.ok) {
          await result.body?.cancel();
          return send(response, result.status === 422 ? 422 : 502, { error: 'O serviço de classificação recusou a requisição.' });
        }
        const body = await result.json();
        const valid = route === '/health'
          ? body?.status === 'ok' && ['smoke', 'fine_tuning'].includes(body.model_purpose)
          : Array.isArray(body?.predictions) && body.predictions.length === payload.texts.length
            && body.predictions.every(validPrediction);
        if (!valid) throw new Error('Resposta incompatível.');
        send(response, 200, body);
      } catch {
        send(response, signal.aborted ? 504 : 502, {
          error: signal.aborted ? 'Tempo limite do classificador excedido.' : 'Classificador indisponível ou resposta inválida.',
        });
      }
    } catch (error) {
      send(response, error instanceof HttpError ? error.status : 500, {
        error: error instanceof HttpError ? error.message : 'Erro interno.',
      });
    }
  });
  server.requestTimeout = 15000;
  server.headersTimeout = 10000;
  server.keepAliveTimeout = 5000;
  return server;
}
