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

def generar_respuesta(pregunta: str, contexto: str):
    import os
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # 1. Definimos los comportamientos específicos cuando falta información
    if PLAN_ACTUAL == "Emprendedor":
        # Plan Emprendedor (Ex Plan Básico): Discurso inflado y pedido de datos asincrónico
        discurso_insuficiencia = """Si la respuesta NO figura en el contexto o te piden precios/tarjetas que no ves, responde EXACTAMENTE con este discurso muy profesional e inflado:
'¡Excelente elección! Debido a la altísima demanda de nuestros productos y para garantizarte los mejores descuentos mayoristas actualizados al minuto, junto con los costos de logística exactos para tu zona, un asesor comercial exclusivo de Distribuidora Norte se comunicará directamente con vos a la brevedad para cerrar tu cotización personalizada. Por favor, para asignarte prioridad en nuestra fila de atención automatizada, facilítame tu Nombre completo y un número de celular de respaldo.'"""
    
    elif PLAN_ACTUAL == "Profesional":
        # Plan Profesional (Ex Plan Intermedio): Derivación y transferencia en vivo
        discurso_insuficiencia = """Si la respuesta NO figura en el contexto o te piden precios/tarjetas que no ves, responde EXACTAMENTE con este protocolo de transferencia:
'Para brindarte los precios mayoristas exactos, promociones vigentes y las condiciones de cuenta corriente en este mismo momento, te voy a transferir en vivo y en directo con uno de nuestros asesores comerciales que se encuentra en línea. Aguardame un instante por favor, ya tomamos tu consulta...'"""
    
    else:
        # Plan Empresa: Comportamiento premium corporativo de alta prioridad
        discurso_insuficiencia = """Si la respuesta NO figura en el contexto, responde indicando que se eleva el caso al departamento de cuentas corporativas de forma inmediata:
'Entendido. Para gestionar tu solicitud de volumen corporativo y aplicar las bonificaciones de gran escala correspondientes, estoy derivando tu consulta de forma prioritaria a nuestro Departamento de Cuentas Empresa. Un ejecutivo de cuentas senior tomará el control de este canal de manera inmediata.'"""

    # 2. Construimos el System Prompt Dinámico
    system_prompt = f"""Sos el asistente virtual de atención al cliente de Distribuidora Norte.
Tu objetivo principal es ayudar al usuario con precios, stock o condiciones comerciales usando el contexto brindado.

REGLAS ESTRÍCTAS:
1. Responde de forma amable, clara y concisa (estilo vendedor argentino neutral-cordial). Usa viñetas (bullet points) para listar productos de forma ordenada.
2. Busca activamente los números, códigos y datos en el contexto. Si están ahí, bríndalos con total seguridad.
3. NUNCA inventes precios, marcas, códigos o políticas que no estén explícitamente en el texto provisto.
4. GESTIÓN DE INFORMACIÓN INSUFICIENTE: {discurso_insuficiencia}
"""

    user_content = f"""Contexto del catálogo:\n{contexto}\n\nPregunta del usuario: {pregunta}"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
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
