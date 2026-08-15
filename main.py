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

# Configuración de CORS corregida y ampliada para procesar correctamente las peticiones OPTIONS previas del navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "mayorista_docs"
VECTOR_DIM = 384

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

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
    
    # CORRECCIÓN DE EXTRACCIÓN: Accedemos al primer elemento [0] para obtener las listas planas
    if not res["documents"] or not res["documents"][0]:
        return "", []
        
    docs_lista = res["documents"][0]
    distances_lista = res["distances"][0] if res["distances"] else []
    metadatas_lista = res["metadatas"][0] if res["metadatas"] else []
    
    contexto = "\n".join(docs_lista)
    fuentes = []
    for i in range(len(docs_lista)):
        distancia = distances_lista[i] if i < len(distances_lista) else 0.5
        relevancia = max(0.0, min(1.0, 1.0 - distancia))
        page = metadatas_lista[i]["page"] if (i < len(metadatas_lista) and "page" in metadatas_lista[i]) else 1
        fuentes.append(SourceItem(page=page, relevancia=relevancia))
        
    return contexto, fuentes

# Configuración del Plan del Cliente (Cambiar según lo contratado)
# Opciones válidas: "Emprendedor", "Profesional", "Empresa"
PLAN_ACTUAL = "Emprendedor" 

PLAN_ACTUAL = "Emprendedor" 

def generar_respuesta(pregunta: str, contexto: str):
    import os
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # REGLA ANCLA: Si el usuario pide hablar con un humano, asesor, vendedor, o si el producto NO existe.
    if PLAN_ACTUAL == "Emprendedor":
        discurso_insuficiencia = """
        SITUACIÓN DE DERIVACIÓN O FALTA DE DATOS:
        Si el usuario te pregunta por un producto que NO está en el catálogo (como televisores), o si explícitamente te pide hablar con un humano, persona, vendedor, asesor o contacto, tienes PROHIBIDO darle correos electrónicos o teléfonos que leas en el documento. 
        En su lugar, debes responder EXACTAMENTE con este discurso muy profesional e inflado para capturar el lead:
        '¡Excelente elección! Para gestionar tu solicitud de forma prioritaria, brindar de manera exacta las condiciones mayoristas actualizadas al minuto junto con los costos de logística para tu zona, y asignarte un asesor exclusivo de Distribuidora Norte, necesitamos registrar tus datos en el sistema de atención. Por favor, facilítame tu Nombre completo y un número de celular de respaldo. Con esa información, un personal de la firma se pondrá en contacto con vos a la brevedad para cerrar tu cotización personalizada.'
        """
    
    elif PLAN_ACTUAL == "Profesional":
        discurso_insuficiencia = """
        Si el producto no existe o piden hablar con un humano, responde EXACTAMENTE:
        'Para brindarte los precios mayoristas exactos, promociones vigentes y las condiciones en este mismo momento, te voy a transferir en vivo y en directo con uno de nuestros asesores comerciales que se encuentra en línea. Aguardame un instante por favor, ya tomamos tu consulta...'
        """
    
    else:
        discurso_insuficiencia = """
        Si el producto no existe o piden hablar con un humano, responde EXACTAMENTE:
        'Entendido. Estoy derivando tu consulta de forma prioritaria a nuestro Departamento de Cuentas Empresa. Un ejecutivo de cuentas senior tomará el control de este canal de manera inmediata.'
        """

    system_prompt = f"""Sos el asistente virtual de atención al cliente de Distribuidora Norte.
Tu objetivo principal es ayudar al usuario con precios, stock o condiciones comerciales usando el contexto brindado.

REGLAS ESTRICTAS DE COMPORTAMIENTO:
1. Responde de forma amable, clara y concisa (estilo vendedor argentino neutral-cordial). Usa viñetas para listar productos de forma ordenada.
2. Si los productos y precios están en el contexto (como los auriculares), bríndalos con seguridad.
3. NUNCA inventes datos que no estén explícitamente en el texto.
4. CONTROL DE LEADS: {discurso_insuficiencia}
"""

    user_content = f"""Contexto del catálogo:\n{contexto}\n\nPregunta del usuario: {pregunta}"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2, # Bajamos un pelín la temperatura para que sea más obediente a las reglas estrictas
    )
    return completion.choices[0].message.content


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

@app.api_route("/", response_class=HTMLResponse, methods=["GET", "HEAD"])
def index():
    html_path = BASE_DIR / "frontend" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    puerto = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=puerto)
