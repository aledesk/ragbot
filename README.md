# RAG Mayorista — MVP

Sistema de consulta inteligente sobre catálogos PDF para distribuidores mayoristas.  
Tecnologías: Python · FastAPI · ChromaDB · Claude API · PyMuPDF

---

## Requisitos

- Python 3.10 o superior
- API Key de Anthropic → https://console.anthropic.com

---

## Instalación (una sola vez)

```bash
# 1. Clonar / copiar el proyecto
cd rag-mayorista

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key
export ANTHROPIC_API_KEY="sk-ant-..."   # Mac / Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
```

---

## Uso

### Opción A — Inicio automático (recomendado)
```bash
python backend/main.py
```
Esto indexa el PDF automáticamente si es la primera vez, y levanta el servidor.

### Opción B — Paso a paso
```bash
# Paso 1: Indexar el PDF
python backend/ingest.py

# Paso 2: Levantar el servidor
uvicorn backend.rag:app --reload --port 8000
```

Luego abrí: **http://localhost:8000**

---

## Reemplazar el PDF del cliente

1. Colocá el PDF en `data/`
2. En `backend/ingest.py` cambiá la última línea:
   ```python
   ingestar_pdf("./data/TU_ARCHIVO.pdf")
   ```
3. Ejecutá `python backend/ingest.py` para re-indexar.

---

## Estructura del proyecto

```
rag-mayorista/
├── data/
│   └── lista_precios_mayorista.pdf   ← PDF del catálogo
├── backend/
│   ├── ingest.py    ← extrae, chunkea e indexa el PDF
│   ├── rag.py       ← motor de búsqueda + API FastAPI
│   └── main.py      ← punto de entrada único
├── scripts/
│   └── generar_pdf_demo.py   ← genera el PDF de prueba
├── chroma_db/       ← base vectorial (se crea automáticamente)
├── requirements.txt
└── README.md
```

---

## Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/chat` | Enviar pregunta, recibir respuesta RAG |
| GET | `/api/health` | Estado del servidor y cantidad de chunks |

### Ejemplo de llamada
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cuál es el precio de la freidora de aire?"}'
```

---

## Arquitectura RAG

```
PDF → extracción (PyMuPDF) → chunks 400 chars
    → embeddings TF-IDF (offline) → ChromaDB

Pregunta → embedding → búsqueda semántica (top 4 chunks)
         → contexto + pregunta → Claude API → respuesta
```

---

## Para producción

- Reemplazar TF-IDF por OpenAI embeddings o `text-embedding-3-small`
- Agregar autenticación (Bearer token o API key)
- Desplegar en Railway, Fly.io o VPS con Docker
- Separar frontend como app React/Next.js

