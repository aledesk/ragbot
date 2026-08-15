import os
import chromadb
import pypdf

def actualizar_base_de_datos():
    # Ruta donde Chroma guarda sus datos vectoriales
    persist_directory = "./chroma_db" 
    nombre_coleccion = "distribuidora"
    
    print("🔄 Conectando con ChromaDB...")
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        print(f"🗑️ Eliminando colección antigua '{nombre_coleccion}'...")
        client.delete_collection(name=nombre_coleccion)
        print("✅ Colección eliminada con éxito.")
    except Exception as e:
        print(f"⚠️ Nota: No se pudo borrar la colección (puede que no existiera): {e}")
    
    print(f"🆕 Creando nueva colección '{nombre_coleccion}'...")
    collection = client.get_or_create_collection(name=nombre_coleccion)
    
    # Archivo original correcto
    pdf_path = "data/lista_precios_mayorista.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: No se encontró el archivo en la ruta: {pdf_path}")
        return
        
    print(f"📄 Procesando el archivo: {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    
    documents = []
    ids = []
    metadatas = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(text)
            ids.append(f"id_pag_{i+1}")
            metadatas.append({"page": i + 1, "source": "lista_precios_mayorista.pdf"})
            
    if documents:
        print(f"📥 Guardando {len(documents)} páginas en ChromaDB...")
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print("🎉 ¡Base de datos vectorial actualizada con el PDF original!")
    else:
        print("⚠️ El PDF no contenía texto extraíble.")

if __name__ == "__main__":
    actualizar_base_de_datos()
