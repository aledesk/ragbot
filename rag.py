import chromadb
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "mayorista_docs"
VECTOR_DIM      = 384
TOP_K           = 6
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
groq_client = Groq()

def get_vectorizer():
    with open(f"{CHROMA_PATH}/vectorizer.pkl", "rb") as f:
        return pickle.load(f)

def embed_query(query, vectorizer):
    mat = vectorizer.transform([query])
    arr = np.array(mat.todense()).flatten()
    arr = arr[:VECTOR_DIM] if len(arr) >= VECTOR_DIM else np.pad(arr, (0, VECTOR_DIM - len(arr)))
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).tolist()

def recuperar_contexto(query):
    vectorizer = get_vectorizer()
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION_NAME)
    embedding = embed_query(query, vectorizer)
    results = collection.query(query_embeddings=[embedding], n_results=TOP_K, include=["documents","metadatas","distances"])
    chunks, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
    fuentes, partes = [], []
    for i, (c, m, d) in enumerate(zip(chunks, metas, dists)):
        partes.append(f"[Fragmento {i+1} — Página {m['page']}]\n{c}")
        fuentes.append({"page": m["page"], "source": m["source"], "relevancia": round(1-d, 3)})
    return "\n\n---\n\n".join(partes), fuentes

def generar_respuesta(pregunta, contexto):
    prompt = f"Contexto del catálogo:\n\n{contexto}\n\n---\n\nPregunta del cliente: {pregunta}"
    r = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        max_tokens=800,
    )
    return r.choices[0].message.content

class QueryRequest(BaseModel):
    pregunta: str

class QueryResponse(BaseModel):
    respuesta: str
    fuentes: list[dict]
    pregunta: str

@app.post("/api/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    if not req.pregunta.strip():
        raise HTTPException(400, "Pregunta vacía")
    contexto, fuentes = recuperar_contexto(req.pregunta)
    respuesta = generar_respuesta(req.pregunta, contexto)
    return QueryResponse(respuesta=respuesta, fuentes=fuentes, pregunta=req.pregunta)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/index.html") as f:
        return f.read()
