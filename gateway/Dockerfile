FROM node:22-bookworm-slim
WORKDIR /app/gateway
COPY --chown=node:node gateway/package.json gateway/http.mjs gateway/server.mjs ./
USER node
EXPOSE 3000
CMD ["node", "server.mjs"]
