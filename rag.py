import chromadb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "mayorista_docs"
TOP_K           = 4
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Sos el asistente de ventas de Distribuidora Norte.
Respondés preguntas de clientes y revendedores usando ÚNICAMENTE la información del catálogo y las condiciones comerciales que se te proporcionan como contexto.

Reglas:
- Si la información está en el contexto, respondé con precisión y detalle.
- Si no está en el contexto, decí claramente: "No tengo esa información en el catálogo actual."
- Nunca inventes precios, stocks o condiciones.
- Usá un tono profesional pero cercano.
- Si hay precios, siempre aclarás la moneda (USD).
- Respondé en español rioplatense (vos, ustedes)."""

app = FastAPI(title="RAG Mayorista API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Cargando modelo de embeddings...")
embedder = SentenceTransformer(MODEL_NAME)
print("✅ Modelo cargado")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_collection(name=COLLECTION_NAME)
groq_client   = Groq()

print("✅ RAG backend listo con embeddings semánticos + Groq")


def embed_query(query: str) -> list[float]:
    return embedder.encode([query])[0].tolist()


def recuperar_contexto(query: str) -> tuple[str, list[dict]]:
    embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks     = results["documents"][0]
    metas      = results["metadatas"][0]
    distancias = results["distances"][0]

    fuentes, contexto_parts = [], []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metas, distancias)):
        contexto_parts.append(f"[Fragmento {i+1} — Página {meta['page']}]\n{chunk}")
        fuentes.append({"page": meta["page"], "source": meta["source"], "relevancia": round(1 - dist, 3)})

    return "\n\n---\n\n".join(contexto_parts), fuentes


def generar_respuesta(pregunta: str, contexto: str) -> str:
    prompt = f"Contexto del catálogo:\n\n{contexto}\n\n---\n\nPregunta del cliente: {pregunta}"
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content


class QueryRequest(BaseModel):
    pregunta: str

class QueryResponse(BaseModel):
    respuesta: str
    fuentes:   list[dict]
    pregunta:  str


@app.post("/api/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    if not req.pregunta.strip():
        raise HTTPException(400, "La pregunta no puede estar vacía")
    contexto, fuentes = recuperar_contexto(req.pregunta)
    respuesta = generar_respuesta(req.pregunta, contexto)
    return QueryResponse(respuesta=respuesta, fuentes=fuentes, pregunta=req.pregunta)


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks_indexados": collection.count()}


@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/index.html") as f:
        return f.read()