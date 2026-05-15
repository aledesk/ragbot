"""
main.py — Punto de entrada único
Sirve la UI estática + la API RAG desde un solo proceso
"""
import subprocess, sys, os
from pathlib import Path

def run():
    os.chdir(Path(__file__).parent.parent)
    # 1. Ingestar PDF si no existe la DB
    if not Path("chroma_db/vectorizer.pkl").exists():
        print("⚙️  Primera ejecución: indexando el PDF...")
        subprocess.run([sys.executable, "backend/ingest.py"], check=True)
    # 2. Levantar el servidor
    print("\n🚀 Servidor iniciado en http://localhost:8000\n")
    subprocess.run(["uvicorn", "backend.rag:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    run()
