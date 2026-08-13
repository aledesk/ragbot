import os
import pickle
import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq
from pathlib import Path

app = FastAPI()

# Configuración de CORS corregida para permitir peticiones desde cualquier origen (Render, Local y tu Hosting)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Forzamos la ruta absoluta apuntando a la raíz del proyecto en Render
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "mayorista_docs"
VECTOR_DIM = 384

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# Cargamos el archivo vectorizer.pkl asegurando su ubicación exacta
with open(BASE_DIR / "chroma_db" / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class QueryRequest(BaseModel):
    pregunta: str

class SourceItem(BaseModel):
    page: int
    relevancia: float

class QueryResponse(BaseModel):
    respuesta: str
    fuentes: List[SourceItem]
    pregunta: str

def transformar_pregunta_a_embedding(texto: str, dim=VECTOR_DIM):
    row = vectorizer.transform([texto])
    arr = np.array(row.todense()).flatten()
    arr = arr[:dim] if len(arr) >= dim else np.pad(arr, (0, dim - len(arr)))
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).tolist()

def recuperar_contexto(pregunta: str):
    query_vector = transformar_pregunta_a_embedding(pregunta)
    res = collection.query(query_embeddings=[query_vector], n_results=3)
    
    if not res["documents"] or not res["documents"]:
        return "", []
        
    contexto = "\n".join(res["documents"])
    fuentes = []
    for i in range(len(res["documents"])):
        distancia = res["distances"][i] if res["distances"] else 0.5
        relevancia = max(0.0, min(1.0, 1.0 - distancia))
        page = res["metadatas"][i]["page"] if res["metadatas"] else 1
        fuentes.append(SourceItem(page=page, relevancia=relevancia))
        
    return contexto, fuentes

def generar_respuesta(pregunta: str, contexto: str):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = f"""Sos el asistente virtual de Distribuidora Norte. 
Responde la pregunta basándote únicamente en el contexto provisto abajo.
Si no encontrás la respuesta en el contexto, decí amablemente que no tenés esa información.

Contexto:
{contexto}

Pregunta: {pregunta}
Respuesta:"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices.message.content

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
    # Corregimos la ruta para abrir el HTML desde la raíz
    html_path = BASE_DIR / "frontend" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

