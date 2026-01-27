# app
# Questo modulo implementa il backend del sistema RAG tramite Flask.
# Espone:
#   - un endpoint HTTP per interrogare il grafo LangGraph
#   - il serving delle immagini associate alle figure del PDF
#   - la gestione delle sessioni di chat tramite thread_id
# Funziona da ponte tra l’interfaccia utente (frontend)
# e il core del sistema RAG.

# Import principali
from flask import Flask, request, jsonify, send_from_directory
import os, uuid
from rag_graph import graph, IMAGES_ROOT
from langchain.schema import Document

# Inizializza app Flask
app = Flask(__name__)

# Route: pagina principale
@app.route("/")
def home():
    """Serve la pagina HTML dell'interfaccia RAG."""
    return send_from_directory('.', 'rag_interface.html')



# Route: serve le immagini
@app.route("/images/<path:filename>")
def serve_image(filename):
    """Restituisce le immagini richieste dall'interfaccia."""
    full_path = os.path.join(IMAGES_ROOT, filename)
    print(f"[FLASK] Richiesta immagine: {filename}")
    print(f"[FLASK] Percorso assoluto: {os.path.abspath(full_path)}")

    if not os.path.exists(full_path):
        print(f"[FLASK]  IMMAGINE NON TROVATA")
    else:
        print(f"[FLASK]  Immagine trovata, la invio")
    return send_from_directory(IMAGES_ROOT, filename)



# Funzione ausiliaria: normalizzazione chunk
def normalize_chunk(d):
    """Restituisce un dizionario standardizzato dal chunk del grafo/documento."""

    # Caso 1: LangChain Document
    if isinstance(d, Document):
        return {
            "doc": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", None),
            "text": d.page_content
        }

    # Caso 2: già dict (LangGraph serializzato)
    if isinstance(d, dict):
        metadata = d.get("metadata", {})
        return {
            "doc": metadata.get("source", "unknown"),
            "page": metadata.get("page", None),
            "text": d.get("page_content", "")
        }

    # Fallback di sicurezza
    return {
        "doc": "unknown",
        "page": None,
        "text": str(d)
    }


# Endpoint principale: ask
@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint principale per ricevere domande e restituire risposta RAG."""
    data = request.json
    question = data.get("question", "")
    thread_id = data.get("thread_id")  # riceve un ID per la chat

    if not thread_id:
        thread_id = str(uuid.uuid4())  # genera un nuovo ID se non esiste

    # Configurazione passata al grafo (LangGraph MemorySaver usa thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    # Pipeline del grafo (query_or_respond → tools → generate → END)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config
    )

    # Stampa l'ultima risposta nel terminale
    print("\n=== Risposta Modello ===\n")
    print(result["messages"][-1].content)
    print("\n=========================\n")

    # Recupera i chunk rilevanti dall'ultimo messaggio tool
    chunks = []
    for msg in reversed(result["messages"]):
        if msg.type == "tool" and hasattr(msg, "artifact"):
            chunks = [normalize_chunk(d) for d in msg.artifact]
            break

    print("CHUNKS RETURNED:", len(chunks))
    for c in chunks:
        print("-", c["doc"])

    # Restituisce JSON alla UI con risposta, thread_id e chunks
    return jsonify({
        "answer": result["messages"][-1].content,
        "thread_id": thread_id,
        "chunks": chunks
    })


# Avvio server
if __name__ == "__main__":
    app.run(debug=True, port=5017)