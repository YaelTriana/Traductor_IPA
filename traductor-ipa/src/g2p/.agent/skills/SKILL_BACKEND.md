# SKILL — Backend Python (FastAPI + PyTorch)

## Contexto
Este skill aplica cuando se trabaja en cualquier archivo dentro de `g2p/` o cuando se crea la API FastAPI.

## Estructura objetivo del backend

```
g2p/
├── api/
│   ├── __init__.py
│   └── main.py          ← FastAPI app
├── dataset.csv
├── entrenar.py
├── evaluar.py
├── g2p_cli.py
├── g2p_core.py          ← NO modificar sin avisar
├── g2p_model.pt         ← NUNCA modificar
├── g2p_vocab.json       ← NUNCA modificar
└── requirements.txt     ← SIEMPRE actualizar al agregar deps
```

## Patrón de carga del modelo (singleton)

```python
# g2p/api/main.py
from pathlib import Path
from contextlib import asynccontextmanager
import torch
import json
from fastapi import FastAPI
from g2p_core import G2PModel, load_vocab, predict as model_predict

BASE = Path(__file__).parent.parent  # → carpeta g2p/

ml = {}  # estado global del modelo

@asynccontextmanager
async def lifespan(app: FastAPI):
    vocab = load_vocab(str(BASE / "g2p_vocab.json"))
    with open(BASE / "g2p_vocab.json") as f:
        vdata = json.load(f)
    model = G2PModel(len(vocab["letters"]), len(vocab["ipa_symbols"]))
    model.load_state_dict(torch.load(BASE / "g2p_model.pt", map_location="cpu"))
    model.eval()
    ml["model"] = model
    ml["vocab"] = vocab
    ml["max_src"] = vdata.get("max_src", 30)
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)
```

## Contrato de respuesta

```python
from pydantic import BaseModel

class WordResult(BaseModel):
    word: str
    ipa: str

class IPAResponse(BaseModel):
    input: str
    results: list[WordResult]
    phrase_ipa: str

class PredictRequest(BaseModel):
    text: str
```

## Soporte de frases

```python
def predict_phrase(text: str) -> IPAResponse:
    words = text.strip().lower().split()
    results = [
        WordResult(
            word=w,
            ipa=model_predict(w, ml["model"], ml["vocab"], ml["max_src"])
        )
        for w in words
    ]
    return IPAResponse(
        input=text,
        results=results,
        phrase_ipa=" ".join(r.ipa for r in results)
    )
```

## CORS para desarrollo local

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)
```

## requirements.txt mínimo

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
torch==2.3.1
pydantic==2.7.4
python-dotenv==1.0.1
pandas==2.2.2
```

## Comandos clave

```bash
# Desarrollo
uvicorn api.main:app --reload --port 8000

# Verificar que el modelo carga
python -c "from g2p_core import load_vocab; v=load_vocab(); print(len(v['letters']), 'letras')"

# Probar endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "asynchronous middleware"}'

# Ver documentación automática
open http://localhost:8000/docs
```

## Errores comunes y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: g2p_core` | Ejecutar uvicorn desde directorio incorrecto | Ejecutar desde `g2p/` o agregar `g2p/` al PYTHONPATH |
| `RuntimeError: PytorchStreamReader` | Versión de PyTorch distinta a la del entrenamiento | Usar `map_location="cpu"` en `torch.load()` |
| `KeyError: max_src` | vocab JSON viejo sin ese campo | Usar `.get("max_src", 30)` como fallback |
| CORS error en frontend | Origins no configurados | Verificar que el dominio de Vercel está en `allow_origins` |
