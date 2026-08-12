import uvicorn
import os

if __name__ == "__main__":
    # Obtenemos el puerto que nos asigne el servidor gratuito (por defecto 8000)
    puerto = int(os.environ.get("PORT", 8000))
    # Arrancamos uvicorn apuntando directamente al archivo rag.py
    uvicorn.run("rag:app", host="0.0.0.0", port=puerto, reload=False)
