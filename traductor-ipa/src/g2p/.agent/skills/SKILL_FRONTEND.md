# SKILL — Frontend Next.js (React 19 + SCSS Modules)

## Contexto
Este skill aplica cuando se trabaja en `traductor-ipa/src/`.

## Archivos clave

```
traductor-ipa/src/app/
├── api/
│   └── translate/
│       └── route.ts         ← CREAR — proxy al backend Python
├── components/
│   └── translator/
│       ├── translator.tsx   ← MODIFICAR — conectar a la API
│       └── translator.module.scss
├── translator/
│   └── page.tsx             ← ya existe, no modificar
└── layout.tsx
```

## API Route (proxy interno)

```typescript
// src/app/api/translate/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const { text } = await req.json();

  if (!text?.trim()) {
    return NextResponse.json({ error: 'Texto vacío' }, { status: 422 });
  }

  const backendUrl = process.env.G2P_API_URL ?? 'http://localhost:8000';

  try {
    const res = await fetch(`${backendUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      return NextResponse.json({ error: 'Error del modelo' }, { status: 500 });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: 'Backend no disponible' }, { status: 503 });
  }
}
```

## Tipos del proyecto

```typescript
// src/types/ipa.ts
export interface WordResult {
  word: string;
  ipa: string;
}

export interface IPAResponse {
  input: string;
  results: WordResult[];
  phrase_ipa: string;
}
```

## Patrón de fetch en translator.tsx

```typescript
'use client';
import { useState } from 'react';
import type { IPAResponse } from '@/types/ipa';

export default function Translator() {
  const [input, setInput]     = useState('');
  const [result, setResult]   = useState<IPAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const translate = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      });
      if (!res.ok) throw new Error('Error al traducir');
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); translate(); }
    if (e.key === 'Escape') { setInput(''); setResult(null); }
  };

  // ... render
}
```

## Variables de entorno

```env
# .env.local (frontend — solo para comunicación interna Next.js)
G2P_API_URL=http://localhost:8000

# NOTA: no uses NEXT_PUBLIC_ para la URL del backend
# porque la API Route corre en el servidor, no en el navegador
```

## Checklist antes de hacer merge

- [ ] `translator.tsx` tiene estados: `loading`, `error`, `result`
- [ ] `route.ts` existe en `src/app/api/translate/`
- [ ] `G2P_API_URL` está en `.env.local` (y en `.env.example` documentado)
- [ ] El resultado IPA se puede copiar al portapapeles
- [ ] El componente maneja el caso de backend caído con mensaje claro
