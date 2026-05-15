import fitz
import chromadb
import hashlib
import re
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "mayorista_docs"
CHUNK_SIZE      = 200
CHUNK_OVERLAP   = 40
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"

print("⏳ Cargando modelo de embeddings...")
model = SentenceTransformer(MODEL_NAME)
print("✅ Modelo cargado")


def extraer_texto_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paginas = []
    for i, page in enumerate(doc):
        texto = page.get_text("text")
        texto = re.sub(r'\n{3,}', '\n\n', texto).strip()
        if texto:
            paginas.append({"page": i + 1, "text": texto})
    doc.close()
    print(f"  → {len(paginas)} páginas extraídas")
    return paginas


def chunking(paginas, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for p in paginas:
        texto = p["text"]
        inicio, parte = 0, 0
        while inicio < len(texto):
            fin = inicio + size
            chunk_texto = texto[inicio:fin]
            if fin < len(texto):
                ultimo_nl = chunk_texto.rfind('\n')
                if ultimo_nl > size // 2:
                    fin = inicio + ultimo_nl
                    chunk_texto = texto[inicio:fin]
            cid = hashlib.md5(f"{p['page']}-{parte}-{chunk_texto[:40]}".encode()).hexdigest()[:12]
            chunks.append({"id": cid, "text": chunk_texto.strip(), "page": p["page"], "part": parte})
            inicio = fin - overlap
            parte += 1
    print(f"  → {len(chunks)} chunks generados")
    return chunks


def indexar(chunks, pdf_nombre):
    textos = [c["text"] for c in chunks]
    print("  → Generando embeddings semánticos...")
    embeddings = model.encode(textos, show_progress_bar=True).tolist()

    Path(CHROMA_PATH).mkdir(exist_ok=True)

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    col.add(
        ids=[c["id"] for c in chunks],
        documents=textos,
        embeddings=embeddings,
        metadatas=[{"page": c["page"], "part": c["part"], "source": pdf_nombre} for c in chunks],
    )
    print(f"  → {len(chunks)} chunks indexados con embeddings semánticos")


def ingestar_pdf(pdf_path):
    path = Path(pdf_path)
    print(f"\n🔄 Ingesta: {path.name}")
    paginas = extraer_texto_pdf(pdf_path)
    chunks  = chunking(paginas)
    indexar(chunks, path.name)
    print(f"\n✅ {len(chunks)} chunks listos.\n")


if __name__ == "__main__":
    ingestar_pdf("./data/lista_precios_mayorista.pdf")