import { createGateway } from './http.mjs';

try {
  process.loadEnvFile(new URL('../.env', import.meta.url));
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}

const port = Number(process.env.GATEWAY_PORT ?? 3000);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('GATEWAY_PORT inválida.');
const host = process.env.GATEWAY_HOST ?? '127.0.0.1';
const server = createGateway({
  upstreamUrl: process.env.UPSTREAM_URL ?? 'http://127.0.0.1:8000',
  timeoutMs: Number(process.env.REQUEST_TIMEOUT_MS ?? 30000),
});
server.on('error', (error) => {
  console.error(`Não foi possível iniciar o gateway: ${error.code ?? 'erro'}.`);
  process.exitCode = 1;
});
server.listen(port, host, () => console.info(`Gateway em http://${host}:${port}`));

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => {
    server.close(() => process.exit(0));
    setTimeout(() => {
      server.closeAllConnections();
      process.exit(0);
    }, 10000).unref();
  });
}
