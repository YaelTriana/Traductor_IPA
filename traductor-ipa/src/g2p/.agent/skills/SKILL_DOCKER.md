# SKILL — Docker Compose (Frontend + Backend)

## Contexto
Este skill aplica cuando se trabaja en `docker-compose.yml`, `Dockerfile`, o cualquier configuración de contenedores.

## Estructura de archivos Docker

```
Traductor_IPA/
├── docker-compose.yml
├── .env.example
├── traductor-ipa/
│   └── Dockerfile
└── g2p/
    └── Dockerfile
```

## docker-compose.yml

```yaml
services:
  backend:
    build:
      context: ./g2p
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./g2p:/app
    environment:
      - MODEL_PATH=/app/g2p_model.pt
      - VOCAB_PATH=/app/g2p_vocab.json
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./traductor-ipa
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - G2P_API_URL=http://backend:8000
    depends_on:
      backend:
        condition: service_healthy
```

## g2p/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## traductor-ipa/Dockerfile

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
```

## next.config.ts (agregar output standalone)

```typescript
// traductor-ipa/next.config.ts
import type { NextConfig } from 'next'

const config: NextConfig = {
  output: 'standalone',
}

export default config
```

## .env.example

```env
# Backend (g2p/)
MODEL_PATH=./g2p_model.pt
VOCAB_PATH=./g2p_vocab.json
PORT=8000

# Frontend (traductor-ipa/)
G2P_API_URL=http://localhost:8000
```

## Comandos clave

```bash
# Levantar todo
docker-compose up --build

# Solo el backend
docker-compose up backend --build

# Ver logs
docker-compose logs -f backend

# Limpiar y reconstruir
docker-compose down -v && docker-compose up --build

# Verificar health del backend
curl http://localhost:8000/health
```

## Endpoint /health requerido en FastAPI

```python
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "model" in ml}
```
