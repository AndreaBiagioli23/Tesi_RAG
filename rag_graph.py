# rag_graph
# Questo modulo implementa il core del sistema Retrieval-Augmented Generation (RAG).
# Definisce il grafo LangGraph, i nodi (query, retrieval, generate) e la logica
# di orchestrazione tra LLM, vector store e strumenti di recupero.

import os
import re
import numpy as np
# LangChain / Ollama: gestione LLM, embeddings e strumenti
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# Loader e text splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph: gestione del workflow / grafo
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver



# Funzione: auto_latex
# Scopo:
#   Normalizzare il testo estratto dagli OCR, correggendo simboli matematici corrotti,
#   ricostruendo frazioni, complessità asintotiche, lettere greche, e proteggendo URL 
#   o blocchi già in LaTeX. Il risultato è un testo più coerente e pronto per la 
#   conversione in markup matematico valido.

def auto_latex(text: str) -> str:
    # correzione di simboli semplici corrotti dall’OCR
    # alcune sequenze come "pi", "rr", "pp", "ro" tendono ad apparire in luogo di π
    text = re.sub(r'\bpi\b', r'\\pi', text)          # pi minuscolo
    text = re.sub(r'\bPi\b', r'\\Pi', text)          # Pi maiuscolo
    text = re.sub(r'\brr\b', r'\\pi', text)
    text = re.sub(r'\bpp\b', r'\\pi', text)
    text = re.sub(r'\bro\b', r'\\pi', text)

    # mappa delle lettere greche
    greek_map = {
        "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
        "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
        "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
        "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
        "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi", "χ": r"\chi",
        "ψ": r"\psi", "ω": r"\omega",
        "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta",
        "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi",
        "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    }
    # conversione effettiva delle lettere
    for key, val in greek_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', lambda m: val, text)

    # correzione dei "7" corrotti nei PDF
    seven_aliases = [r'ni', r'n1', r'nI', r'nl', r'ri', r'ηι', r'ηl', r'η1']
    for alias in seven_aliases:
        text = re.sub(rf'\b{alias}\b', '7PLACEHOLDER', text, flags=re.IGNORECASE)

    # gestione esponenti frazionari
    text = re.sub(r'n\s*7PLACEHOLDER\s*/\s*(\d+)', r'n^{7/\1}', text, flags=re.IGNORECASE)
    text = text.replace('7PLACEHOLDER', '7')

    # gestione complessità
    def repl_complexity(match):
        letter = match.group(1)
        inner = match.group(2)
        inner = re.sub(r'\blog\b', r'\\log ', inner)
        inner = re.sub(r'(\w|\d)\^(\w+)', r'\1^{\2}', inner)
        return f"${letter}({inner})$"

    text = re.sub(r'\b([OΘΩ])\(([^)]+)\)', repl_complexity, text)

   # Questo blocco incapsula logiche complesse che non sono direttamente
   # esprimibili in una singola operazione o struttura dati.
    def wrap_if_missing_dollars(t):
        pattern = r'(?<!\$)([OΘΩ]\([^()]+\))(?!\$)'
        return re.sub(pattern, r'$\1$', t)

    text = wrap_if_missing_dollars(text)

    # protezione URL e tag HTML <img>
    URL_PLACEHOLDERS = {}
    def protect_url(match):
        url = match.group(0)
        key = f"__URL_PROT_{len(URL_PLACEHOLDERS)}__"
        URL_PLACEHOLDERS[key] = url
        return key

    text = re.sub(r'<img[^>]+>', protect_url, text)
    text = re.sub(r'/images/[^ \n\t"]+', protect_url, text)

    # protezione di and/or, input/output, left/right
    text = re.sub(r'\band\s*/\s*or\b', '__ANDOR__', text, flags=re.IGNORECASE)
    text = re.sub(r'\binput\s*/\s*output\b', '__INOUT__', text, flags=re.IGNORECASE)
    text = re.sub(r'\bleft\s*/\s*right\b', '__LEFTRIGHT__', text, flags=re.IGNORECASE)

    # conversione frazioni a/b in \frac{a}{b}
    fraction_pattern = r'(\\?[a-zA-Z]+|\d+)\s*/\s*(\\?[a-zA-Z]+|\d+)'
    text = re.sub(
        fraction_pattern,
        lambda m: f"$\\frac{{{m.group(1)}}}{{{m.group(2)}}}$",
        text
    )

    # ripristino and/or
    text = text.replace('__ANDOR__', 'and/or')
    text = text.replace('__INOUT__', 'input/output')
    text = text.replace('__LEFTRIGHT__', 'left/right')

    # somme
    text = text.replace('∑', r"$\sum$")
    text = re.sub(r'(?<=\w)\s*~\s*(?=\w)', lambda m: ' $\\sum$ ', text)
    text = re.sub(r'\s~\s', lambda m: ' $\\sum$ ', text)
    text = re.sub(r'(~\s*){2,}', lambda m: ' $\\sum$ ', text)
    text = re.sub(r'\$(.*?)~(.*?)\$', lambda m: f"${m.group(1)}\\sum {m.group(2)}$", text)

    # rimozione di coppie di dollari vuoti e normalizzazione esponenti
    text = re.sub(
        r'([a-zA-Z\\]+)\s*\^\s*([a-zA-Z0-9]+)',
        r'\1^{\2}',
        text
    )
    text = text.replace('\\\\', '\\')
    text = re.sub(r'\${3}([^$]+)\${3}', r'$$\1$$', text)
    text = re.sub(r'\${3}([^$]+)\${2}', r'$$\1$$', text)
    text = re.sub(r'\${2}([^$]+)\${3}', r'$$\1$$', text)
    text = re.sub(r'\$(\s*)\$', r'\1', text)

    # maschera blocchi matematici già presenti
    math_placeholders = {}
    def protect_math(match):
        k = f"__MATH_PROT_{len(math_placeholders)}__"
        math_placeholders[k] = match.group(0)
        return k

    text = re.sub(r'\$\$.*?\$\$', protect_math, text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', protect_math, text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', protect_math, text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', protect_math, text, flags=re.DOTALL)

    # Incapsulamento di operazioni matematiche elementari
    def wrap_latex_commands(t):
        t = re.sub(r'(\\frac\{[^}]+\}\{[^}]+\})', r'$\1$', t)
        t = re.sub(r'(?<!\$)(?<!\\)(\\(?:pi|Pi|Gamma|Delta|Sigma|Omega|sum|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|rho|sigma|tau|phi|psi|omega))', r'$\1$', t)
        t = re.sub(r'(?<!\$)(\d+\s*\\pi\s*/\s*\d+)', r'$\1$', t)
        return t

    text = wrap_latex_commands(text)

    # ripristino blocchi matematici originali
    for k, orig in math_placeholders.items():
        text = text.replace(k, orig)

    for key, url in URL_PLACEHOLDERS.items():
        text = text.replace(key, url)

    return text

# Inizializzazione modello LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# Inizializza embeddings testuali e vectorstore
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = InMemoryVectorStore(embeddings)

# Lista dei PDF da caricare e relative pagine
pdf_files = [
    ("GraphDrawing.pdf", [range(0, 49), range(144, 178)]),  # selezione speciale
    ("WalkOnTheWildSide.pdf", "all"),                       # tutte le pagine
    ("OnTurnRegularOrthogonalRepresentation.pdf", "all"),
    ("ComputingBendMinimumOrthogonalDrawingsOfPlaneSeriesParallelGraphInLinearTime.pdf","all"),
    ("OnTheComplexityOfHVRectilinearPlanarityTesting.pdf","all"),
    ("OnRectilinearDrawingofGraphs.pdf","all"),
    ("ComplexityOfFindingNonPlanarRectilinearDrawingsofGraph.pdf","all"),
    ("LinearAlgorithmForOptimalOrthogonalDrawingsOfTriconnectedCubicPlaneGraphs.pdf","all"),
    ("OrthogonalDrawingsOfPlaneGraphsWithoutBends.pdf","all"),
    ("RectangularDrawingAlgorithms.pdf","all"),
    ("PlanarOrthogonalandPolylineDrawingAlgorithms.pdf","all"),
    ("OnEmbeddingAGraphInTheGridWithTheMinimumNumberOfBends.pdf","all")
]

all_docs = []  # lista accumulo di tutti i documenti

# Caricamento compatto di tutti i PDF e selezione pagine
for filename, pages in pdf_files:
    loader = PyPDFLoader(filename)
    docs = loader.load()
    if pages == "all":
        selected = docs[:]  # tutte le pagine
    else:
        selected = []
        for interval in pages:
            selected.extend(docs[interval.start: interval.stop])
    all_docs.extend(selected)

# Splitting dei documenti in chunk sovrapposti
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    add_start_index=True
)
all_splits = text_splitter.split_documents(all_docs)
_ = vector_store.add_documents(all_splits)  # aggiunge chunk al vectorstore

# Impostazione struttura cartelle immagini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.join(BASE_DIR, "pdf_images")
IMAGE_MAPS = {}  # dizionario mapping PDF -> figure

# Scansione cartelle PDF e creazione mapping figure
for folder in os.listdir(IMAGES_ROOT):
    full_path = os.path.join(IMAGES_ROOT, folder)
    if not os.path.isdir(full_path):
        continue
    pdf_name = folder.lower().replace("_", "").replace("-", "")
    IMAGE_MAPS[folder] = {}  # inizializza mapping figure
    for fname in os.listdir(full_path):
        if fname.lower().endswith(".png"):
            match = re.match(r"(?:figure|fig)_(\d+(?:\.\d+)?)\.png", fname, re.IGNORECASE)
            if match:
                fig_num = match.group(1)
                IMAGE_MAPS[folder][fig_num] = f"{folder}/{fname}"

# Stampa riepilogo immagini trovate
print("IMAGE MAPS")
for pdf, mapping in IMAGE_MAPS.items():
    print(pdf, " => ", mapping)
print("========================")

# TOOL: retrieve
# Recupera dai vettori i chunk di testo più rilevanti.
# Funziona in due fasi:
#   1) Ricerca deterministica di figure, lemma, theorem, algorithm
#   2) Fallback: ricerca semantica tramite embeddings se non ci sono match esatti
@tool(response_format="content_and_artifact", description="Retrieve information related to a query")
def retrieve(query: str):
    """Recupera dai documenti chunk rilevanti in base alla query."""
    # Parole chiave per ricerca diretta
    special_marks = ["Figure", "Fig.", "Theorem", "Lemma", "Algorithm"]
    direct_hits = []
    query_lower = query.lower()

    # Ricerca esatta: esempio "Theorem 5.3"
    for mark in special_marks:
        if mark.lower() in query_lower:
            match = re.search(r"\d+(\.\d+)*", query)
            if not match:
                continue
            num = match.group()

            # Pattern da cercare nei chunk (varianti con punto o due punti)
            pattern1 = f"{mark} {num}".lower()
            pattern2 = f"{mark} {num}.".lower()
            pattern3 = f"{mark} {num}:".lower()

            for doc in all_splits:
                text = doc.page_content.lower()
                if pattern1 in text or pattern2 in text or pattern3 in text:
                    direct_hits.append(doc)

    # Se match esatti trovati, restituisce subito
    if direct_hits:
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent:\n{doc.page_content}" for doc in direct_hits
        )
        return serialized, direct_hits

    # FALLBACK: ricerca semantica tramite embeddings (k=7)
    retrieved_docs = vector_store.similarity_search(query, k=7)

    # Stampa chunks recuperati per debug
    print("\n CHUNK RECUPERATI (k=7)")
    for i, d in enumerate(retrieved_docs, 1):
        print(f"\n---- CHUNK {i} ----")
        print(f"Source: {d.metadata}")
        print(d.page_content)
    print("== Fine recupero chunk ==\n")

    # Serializzazione finale dei chunk
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent:\n{doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Nodo decisionale del grafo
# è il punto di ingresso del workflow.
# Qui il modello decide se:
#   - rispondere direttamente all’utente, oppure
#   - invocare il tool "retrieve" per raccogliere informazioni dai documenti.
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools_node = ToolNode([retrieve])


# Funzione extract_figure_context:estrazione del contesto attorno a una figura
def extract_figure_context(text, fig_num, window=500):
    """Restituisce un frammento di testo centrato sulla figura fig_num, 
    con lunghezza circa 'window' caratteri."""
    pattern = rf"(Figure|Fig\.?)\s+{re.escape(fig_num)}"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return ""
    start = match.start()
    return text[start:start + window]

# Funzione embed:embedding di una query/testo tramite modello embeddings
def embed(text):
    """Restituisce il vettore embedding del testo passato usando OllamaEmbeddings."""
    return embeddings.embed_query(text)

# Funzione:similarità coseno tra due vettori
def cosine_similarity(a, b):
    """Calcola la similarità coseno tra due vettori a e b."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Nodo finale generate()
# Funzione principale che produce la risposta finale all’utente.
# Oltre a generare il testo:
#   - riformatta LaTeX tramite auto_latex()
#   - identifica figure citate
#   - individua il PDF corretto
#   - allega automaticamente l’immagine HTML se disponibile
def generate(state: MessagesState):
    image_file = None  #inizializzo immagine vuota

    # Recupera i documenti generati dai tools 
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Contenuto dei documenti dei tools per il prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    # Recupera l’ultimo messaggio dell’utente
    user_msg = None
    for m in reversed(state["messages"]):
        if m.type == "human":
            user_msg = m.content
            break

    # 1) Individuazione del PDF nominato esplicitamente dall'utente
    user_pdf_key = None
    if user_msg:
        q = user_msg.lower().replace("?", "").strip()

        # Match esatto con le cartelle dei PDF
        for key in IMAGE_MAPS.keys():
            k_norm = key.lower().replace("_", "").replace("-", "")
            if k_norm in q.replace(" ", ""):
                user_pdf_key = key
                break

        # Match approssimato se quello diretto non funziona
        if user_pdf_key is None:
            import difflib
            candidates = list(IMAGE_MAPS.keys())
            match = difflib.get_close_matches(
                q.replace(" ", ""),
                [c.replace(" ", "").lower() for c in candidates],
                n=1,
                cutoff=0.65
            )
            if match:
                idx = [c.replace(" ", "").lower() for c in candidates].index(match[0])
                user_pdf_key = candidates[idx]

    if user_pdf_key:
        print(f"📌 L’utente ha nominato il PDF: {user_pdf_key}")

    # Preparazione del system prompt per il modello
    # Impone regole severe su formattazione e uso delle figure
    system_message_content = (
        "You must answer the user's question using ONLY the information retrieved from the documents below.\n"
        "IF the question is a simple definition, a short fact, or does not require detailed explanation (e.g., 'What is X?'), "
        "provide a concise, direct answer in one paragraph without following the title-summary-detailed schema.\n"
        "If the user mentions a figure (e.g., “Figure 5.18”), do NOT say that you cannot show images."
        "Simply describe the figure, and do NOT mention limitations about image display."
        "If an image is automatically appended by the system (HTML <img>), you MUST NOT mention any limitation about showing images.NEVER say “I cannot display images” in any case."
        "DO NOT invent diagrams, ASCII drawings, examples, or approximations of the figure.\n"
        "Otherwise, FORMAT YOUR ANSWER STRICTLY USING MARKDOWN, in the following structure:\n\n"
        "##  **Title (short and descriptive)**\n"
        "- Write it as a level-2 header (##), bold, and visually separated.\n\n"
        "_**Summary (3–4 lines)**_\n"
        "- Immediately below the title, in italic.\n\n"
        "### **Detailed Explanation**\n"
        "- Break content into short paragraphs.\n"
        "- Use bullet lists when appropriate.\n"
        "- Use descriptive mini-headings when useful.\n\n"
        "-------------------------------\n"
        "Relevant documents:\n"
        f"{docs_content}"
    )

    # Messaggi della conversazione da passare al modello LLM
    conversation_messages = [
        m for m in state["messages"]
        if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
    ]

    # Prompt finale: system + conversazione
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)

    # 2) Riformatta LaTeX nella risposta
    response.content = auto_latex(response.content)

    # 3) Controllo se l’utente vuole un’immagine
    user_wants_image = False
    if user_msg:
        image_keywords = [
            "figure", "fig", "image", "show", "display",
            "illustrate", "diagram", "picture"
        ]
        q_lower = user_msg.lower()
        user_wants_image = any(kw in q_lower for kw in image_keywords)

    # Identifica figure candidate dai documenti dei tools
    candidate_figures = set()
    if user_wants_image:
        for msg in tool_messages:
            for d in msg.artifact:
                matches = re.findall(
                    r"(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)?)",
                    d.page_content,
                    re.IGNORECASE
                )
                for m in matches:
                    candidate_figures.add(m)

    # 4) Individuazione della figura citata dal modello nella risposta
    fig_num = None
    match = re.search(r"(?:Figure|Fig\.?|Figs?)\s+(\d+(?:\.\d+)?)", response.content, re.IGNORECASE)
    if match:
        fig_num = match.group(1)

    # 5) Selezione semantica della figura più rilevante (se non esplicitamente citata)
    if user_wants_image and fig_num is None and candidate_figures:
        query_emb = embed(user_msg)
        best_score = -1
        best_fig = None
        best_pdf = None
        for msg in tool_messages:
            for d in msg.artifact:
                for fig in candidate_figures:
                    context = extract_figure_context(d.page_content, fig)
                    if not context:
                        continue
                    fig_emb = embed(context)
                    score = cosine_similarity(query_emb, fig_emb)
                    if score > best_score:
                        best_score = score
                        best_fig = fig
                        best_pdf = d.metadata["source"]

        fig_num = best_fig
        pdf_name = best_pdf
        print(f"🔁 Figura selezionata semanticamente: Figure {fig_num}")

    # 6) Ricerca PDF e immagine associata
    image_file = None
    if user_wants_image and fig_num:
        pdf_name = None
        # Cerca il PDF nei chunk se non specificato dall’utente
        if user_pdf_key is None:
            for msg in tool_messages:
                for d in msg.artifact:
                    text = d.page_content.lower()
                    if f"figure {fig_num}" in text or f"fig. {fig_num}" in text:
                        pdf_name = d.metadata["source"]
                        break
                if pdf_name:
                    break

        # Determina PDF finale
        if user_pdf_key:
            key = user_pdf_key
        elif pdf_name:
            key = os.path.splitext(pdf_name)[0]
        else:
            key = None

        # Trova immagine nella mappa FIG -> PNG
        if key and key in IMAGE_MAPS and fig_num in IMAGE_MAPS[key]:
            image_file = IMAGE_MAPS[key][fig_num]
            print(f" Immagine trovata: {image_file}")

    # 7) Allega immagine in fondo alla risposta (HTML)
    if user_wants_image and image_file:
        image_html = (
             '<br><br>'
            '<div style="text-align:center;">'
            f'<img src="/images/{image_file}" '
            'style="max-width:600px;border:1px solid #ccc;">'
            '</div>'
        )
        response.content += image_html

    return {"messages": [response]}



# Costruzione del grafo e memoria

# Inizializza il salvataggio dello stato (MemorySaver)
memory = MemorySaver()

# Crea il builder del grafo basato sullo stato dei messaggi
graph_builder = StateGraph(MessagesState)

# Aggiunge i nodi principali del workflow
graph_builder.add_node(query_or_respond)  # nodo di ingresso: decide se usare tools o rispondere
graph_builder.add_node(tools_node)        # nodo tool: recupera informazioni dai documenti
graph_builder.add_node(generate)          # nodo finale: genera la risposta utente con eventuali immagini

# Imposta il nodo di ingresso del grafo
graph_builder.set_entry_point("query_or_respond")

# Logica condizionale: instrada i messaggi tra i nodi
#   - se il modello decide "END", il grafo termina
#   - se il modello indica "tools", passa al nodo dei tools
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)

# Flusso finale sequenziale: dopo tools -> generate -> risposta
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# Compila il grafo e associa il checkpointer per persistenza dello stato
graph = graph_builder.compile(checkpointer=memory)