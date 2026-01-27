# Evals
# Questo modulo implementa la pipeline di valutazione del sistema RAG.
# Definisce:
#   - il dataset di domande di test
#   - le metriche di valutazione (correcteness, faithfulness, coverage, context_precision)
#   - un LLM giudice (Ollama) per lo scoring automatico
# Confronta le risposte del RAG
# e salva i risultati per successive analisi statistiche.
import os
import csv

from pydantic import BaseModel
from ragas import Dataset
from ragas.metrics import DiscreteMetric
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from rag_graph import graph


# Percorsi vari per salvataggio esperimenti
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "rag_eval", "experiments")
LOG_DIR = os.path.join(BASE_DIR, "rag_eval", "logs")
NOTEBOOK_LLM_PATH = os.path.join(BASE_DIR, "rag_eval", "datasets", "notebook_llm_answers.csv")

# Creazione directory se non esistono
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# 1. LLM giudice 
class JudgeOutput(BaseModel):
    value: float

class OllamaJudge:
    """LLM judge basato su Ollama per la valutazione automatica delle risposte RAG."""
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0)

    def generate(self, prompt: str, **kwargs):
        response = self.llm.invoke(prompt).content.strip()
        try:
            value = float(response)
        except ValueError:
            value = 0.0
        # Forza valori tra 0 e 1
        value = max(0.0, min(1.0, value))
        return JudgeOutput(value=value)

# Istanza globale per il giudice LLM
judge_llm = OllamaJudge()


# 2. Dataset con domande di riferimento
def load_dataset():
    """Carica e costruisce il dataset di valutazione"""
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir="evals/datasets",
    )

    samples = [
        {
            "question": "What is meant by an orthogonal representation?",
            "grading_notes": ("""An orthogonal representation H is a dimensionless description of the "shape" of an orthogonal drawing, defining an equivalence class of drawings that share the same abstract structure. 
It is defined by the planar embedding, the clockwise sequence of geometric angles (from {90°, 180°, 270°, 360°}) around each vertex, and the ordered sequence of bends (left/right turns) along each edge."""),
        },
        {
            "question": "What is a bend in an orthogonal drawing?",
            "grading_notes": ("""A bend is a point along an edge in an orthogonal drawing where the edge changes direction. Specifically, it is the point where a horizontal and a vertical line segment meet. 
Geometrically, a bend determines an angle of 90° in one face incident to the edge and 270° in the other. In algorithms, a bend corresponds to a dummy vertex inserted along the edge to model the drawing process."""),
        },
        {
            "question": "What does bend minimization mean?",
            "grading_notes": ("""Bend minimization is an optimization objective that seeks to find an orthogonal drawing containing the minimum possible number of bends. 
For a graph whose planar embedding is fixed (plane graph), this optimization problem is equivalent to inserting the minimum necessary number of subdivision vertices (dummy vertices) along the edges to make the resulting graph rectilinear planar (drawable without bends)."""),
        },
        {
            "question": "What is the maximum vertex degree for a graph to admit an orthogonal drawing?",
            "grading_notes": (r"""For a standard orthogonal drawing where vertices are represented as points, the graph's maximum vertex degree Delta must be at most four. 
This constraint exists because the smallest angle between incident edges must be 90° ($\pi/2$), limiting the number of edges to four. 
However, graphs with a degree greater than four can be accommodated by allowing vertices to be drawn as rectangular boxes."""),
        },
        {
            "question": "What is a rectilinear drawing and how does it relate to orthogonal drawings?",
            "grading_notes": ("""A rectilinear drawing is a special type of orthogonal drawing characterized by the absence of bends. 
In a rectilinear drawing, every edge is represented by a single straight-line segment that is either horizontal or vertical. 
Any orthogonal drawing of a graph G can be mathematically viewed as a rectilinear drawing of a subdivision of G, where the original drawing's bends become the subdivision's dummy vertices."""),
        },
        {
            "question": "What is the topology–shape–metrics framework?",
            "grading_notes": ("""The Topology–Shape–Metrics (TSM) framework is a widely used three-phase methodology for constructing orthogonal drawings. The phases are executed consecutively:

Topology (Planarization): Determines the planar embedding, introducing dummy vertices if necessary to represent edge crossings.
Shape (Orthogonalization): Computes the dimensionless orthogonal representation consistent with the defined topology, often prioritizing bend minimization.
Metrics (Compaction): Assigns precise integer coordinates to the vertices and bends to produce the final drawing, typically optimizing area or edge length."""),
        },
        {
            "question": "What is a planar embedding and why is it relevant to orthogonal drawings?",
            "grading_notes": ("""A planar embedding is the specific circular ordering of incident edges around every vertex (defining the set of faces) and the specific choice of an outer face in a planar drawing of the graph. 
It defines equivalency classes among the planar drawings of a graph on a plane. 
A graph provided with a fixed planar embedding is called a plane graph. 
It is relevant because the first stage of the TSM framework is dedicated to determining or fixing this embedding. For a plane graph input, the drawing algorithm must preserve the given embedding, maintaining the same set of faces as the input graph."""),
        },
        {
            "question": "What is an angle assignment?",
            "grading_notes": ("""An angle assignment (or angle labeling) refers to the specification of the geometric angle value for every angle around the vertices of the graph, forming part of the overall orthogonal representation. 
These values are multiples of 90° (from {90°, 180°, 270°, 360°}) and must collectively satisfy the condition that the sum of the angles around any vertex equals 360°."""),
        },
        {
            "question": "What is the difference between a drawing and a representation?",
            "grading_notes": ("""The difference lies in the level of detail specified: a representation (or orthogonal representation) is dimensionless, describing the abstract shape, angles, and bend structure, but not coordinates or segment lengths. 
A drawing is the geometrical realization of the graph, assigning concrete coordinate and metric values to vertices and edge segments."""),
        },
        {
            "question": "What is an HV-rectilinear drawing?",
            "grading_notes": ("""An HV-rectilinear drawing (or HV-drawing) is a rectilinear orthogonal drawing of an HV-restricted planar graph (HV-graph), which is a graph where every edge is pre-labeled as either H (Horizontal) or V (Vertical). 
In the resulting drawing, all H-labeled edges must be drawn as horizontal segments, and all V-labeled edges must be drawn as vertical segments."""),
        },
        {
            "question": "What role does the planar embedding play in the existence of an orthogonal drawing?",
            "grading_notes": ("""The existence of a standard orthogonal drawing requires the graph to be planar and have a maximum vertex degree of at most four. 
However, the specific planar embedding (or representation) plays a crucial role in determining the complexity of minimizing bends and realizing the drawing. 
If the algorithm can freely choose the planar embedding (variable embedding setting), finding an orthogonal drawing with the minimum number of bends is NP-complete. 
Conversely, if the graph is given as a plane graph with a fixed embedding, the minimum-bend problem is polynomial-time solvable."""),
        },
        {
            "question": "Does a 4-graph always admit a rectilinear drawing?",
            "grading_notes": ("""TA 4-graph (a graph with maximum degree at most 4) does not always admit a rectilinear drawing (an orthogonal drawing without bends). In fact, determining whether a 4-graph admits a planar rectilinear drawing (rectilinear planarity testing) is an NP-complete problem. """),
        },
        {
            "question": "Are self-loops or multiple edges allowed in classical orthogonal drawing models?",
            "grading_notes": ("""In general graph definitions used in graph drawing, graphs are sometimes assumed to be simple (no self-loops or multiple edges), unless specified otherwise. However, the network flow model developed by Tamassia for minimum bend orthogonal drawing explicitly allows the graphs to have multiple edges and self-loops. Furthermore, the dual graph of a planar embedding may contain self-loops and multiple edges. """),
        },
        {
            "question": "What is the main idea behind Tamassia’s algorithm for minimum bend orthogonal drawings?",
            "grading_notes": (r"""Tamassia's seminal work reduces the minimum-bend orthogonal drawing problem for a given plane graph to a minimum cost flow problem on a specially constructed network. The core idea is to model the angles ($90^\circ$ units) as a "commodity" that is supplied by the vertices and consumed by the faces. """),
        },
        {
            "question": "How is bend minimization reduced to a flow problem in Tamassia’s approach?",
            "grading_notes": (r"""The reduction involves building a flow network $N(P)$ whose nodes correspond to the vertices and faces of the graph $G$. 

Vertex Nodes: Each vertex node supplies 4 units of flow (representing the $360^\circ$ angle constraint around a vertex). 

Face Nodes: Each face node consumes a fixed amount of flow determined by its degree (Euler's formula ensures supply equals demand). 

Vertex-to-Face Arcs: Arcs between a vertex and its incident face represent the angle formed on the vertex by the two edges incident to the face and the vertex. The units of flow represent multiples of $90^\circ$ and these units of flow have no cost. 

Face-to-Face Arcs: Arcs between adjacent face nodes represent edges in $G$ and their flow dictates the number of bends along that edge. 

The cost associated with these flow units is set to 1 per unit of flow, meaning that minimizing the total cost of the flow minimizes the total number of bends. """),
        },
        {
            "question": "What is the computational complexity of Tamassia’s bend minimization algorithm?",
            "grading_notes": (r"""The complexity of Tamassia’s original algorithm, which uses the minimum cost flow formulation, was $O(n^2 \log n)$. Subsequent improvements using more advanced minimum cost flow techniques later reduced the complexity to $O(n^{7/4} \sqrt{\log n})$ and later to $O(n^{3/2} \log n)$. """),
        },
        {
            "question": "What is optimized in the flow network constructed by Tamassia?",
            "grading_notes": ("""The algorithm optimizes the cost of a feasible flow of a fixed value $z(P)$ (which equals the total available angular commodity) in the constructed network $N(P)$. This minimization of flow cost directly corresponds to minimizing the total number of bends in the orthogonal representation. """),
        },
        {
            "question": "What assumptions are required for Tamassia’s algorithm to work?",
            "grading_notes": ("""Tamassia's algorithm requires the input graph $G$ to be a 4-planar graph (planar with a maximum vertex degree of at most 4) and be provided with a fixed planar representation ($P$). The algorithm computes a region-preserving grid embedding with respect to this fixed representation $P$. """),
        },
        {
            "question": "Does Tamassia’s algorithm require a fixed planar embedding?",
            "grading_notes": ("""Yes, Tamassia's algorithm requires a fixed planar embedding $P$.Operating in this fixed embedding setting allows the problem to be solved in polynomial time; minimizing bends over all possible embeddings is NP-complete. """),
        },
        {
            "question": "What is the idea behind the algorithm to compute an orthogonal drawing of a series-parallel graph?",
            "grading_notes": ("""For plane series-parallel 4-graphs, a linear-time algorithm exists to compute a bend-minimum orthogonal drawing. The algorithm operates incrementally using a series–parallel decomposition tree ($SPQ^*$-tree). A key concept used is orthogonal spirality, which measures how much each component of the graph is "rolled-up" in the drawing, allowing for a combinatorial description of the optimal shape. """),
        },
        {
            "question": "What is the approach of the algorithm to compute an orthogonal drawing of a triconnected cubic graph?",
            "grading_notes": ("""The approach for computing a bend-minimum orthogonal drawing of a 3-connected cubic plane graph uses an elegant linear-time algorithm. The key idea is to reduce the orthogonal drawing problem to the rectangular drawing problem, where a rectangular drawing is an orthogonal drawing with no bends and every face drawn as a rectangle. The process involves several steps: 

Four dummy vertices (of degree two) are added to the outer boundary $C_o(G)$ to create a graph $G'$. 

The algorithm identifies and contracts specific "maximal" bad cycles (3-legged cycles containing no vertex of degree two) and the subgraphs inside them into single vertices, resulting in a graph $G''$ which admits a rectangular drawing. 

The algorithm recursively finds orthogonal drawings for the contracted subgraphs (cycles and their insides). 

These component drawings are patched together into the rectangular drawing of $G''$ to form an orthogonal drawing of $G'$. 

Finally, the four initial dummy vertices in the drawing of $G'$ are replaced by bends to yield the minimum-bend orthogonal drawing of $G$. """),
        },
        {
            "question": "Is the problem of finding a minimum bend orthogonal drawing NP-hard?",
            "grading_notes": ("""Yes, the problem of finding an orthogonal drawing with the minimum number of bends is in general NP-hard. This NP-hardness applies specifically to general planar 4-graphs (graphs with maximum degree four) with no prescribed planar embedding. Furthermore, the specific case of checking if a graph admits a rectilinear drawing (an orthogonal drawing with zero bends) is also NP-complete. """),
        },
        {
            "question": "For which variants is the bend minimization problem polynomial-time solvable?",
            "grading_notes": (r"""The bend minimization problem is polynomial-time solvable for several variants: 

Fixed Embedding (Plane Graphs): If the graph $G$ is a plane 4-graph (meaning the planar embedding is fixed), the problem is polynomial-time solvable. The original solution used a minimum cost flow network model running in $O(n^2 \log n)$ time, which has since been improved to $O(n^{3/2} \log n)$. 

Series-Parallel Graphs: A linear-time ($O(n)$) algorithm exists to compute bend-minimum orthogonal drawings for plane series-parallel 4-graphs. 

Maximum Degree 3: An optimal $O(n)$-time algorithm is known for planar 3-graphs (graphs with maximum degree three). 

Triconnected Cubic Plane Graphs: A linear-time algorithm finds the minimum number of bends for 3-connected cubic plane graphs. """),
        },
        {
            "question": "Are there known lower bounds on the number of bends?",
            "grading_notes": (r"""Yes. For any orthogonal drawing of a graph with $n$ vertices, the minimum number of bends is $O(n)$. More specific lower bounds include: 

There exists a planar degree-four graph that requires at least $2n-2$ bends in any orthogonal drawing. 

An orthogonal drawing of the complete graph of 4 vertices $K_4$ requires at least $\lfloor n/2 \rfloor + 2$. For planar degree-three graph, except $K_4$, the lower bound on the number of bends is $\lfloor n/2 \rfloor + 1$. """),
        },
        {
            "question": "Are there known upper bounds on the number of bends?",
            "grading_notes": (r"""Yes. For an orthogonal drawing of a planar 4-graph with $n$ vertices, there exists an orthogonal representation with at most $2n+2$ bends in total, with no edge having more than two bends. Also, if the graph has maximum degree 3, there is an algorithm that produces a drawing with at most $\lfloor n/2 \rfloor+1$ bends, with at most one edge that bends twice. """),
        },
        {
            "question": "Does allowing variable embeddings change the complexity?",
            "grading_notes": ("""Yes, allowing variable embeddings significantly changes the complexity of the bend minimization problem. 

In the variable embedding setting (where the planar embedding can be freely chosen), finding a minimum-bend orthogonal drawing for a general planar 4-graph is NP-hard. 

In the fixed embedding setting (where the graph is supplied as a plane graph), the minimum-bend problem is polynomial-time solvable. """),
        },
        {
            "question": "Does allowing crossings in the drawing change the complexity?",
            "grading_notes": ("""For the problem of determining the existence of a rectilinear drawing (zero bends), allowing crossings does not simplify the complexity for unrestricted graphs; the problem remains NP-complete. """),
        },
        {
            "question": "Are FPT algorithms discussed in the selected papers?",
            "grading_notes": (r"""Yes, Fixed-Parameter Tractable (FPT) algorithms are discussed for the problem of finding non-planar rectilinear drawings (zero bends). 

A linear-time FPT algorithm exists to test whether a degree-4 graph has a rectilinear drawing, where the parameter $k$ is the number of vertices of degree 3 or 4. 

The complexity for the unrestricted case is $O(24^k \cdot k^{2k+1} + n)$. The problem is also FPT for cyclic-restricted and HV-restricted graphs, with complexity bounds related to $k$. """),
        },
        {
            "question": "Are there applications of orthogonal graph drawing?",
            "grading_notes": ("""Orthogonal graph drawing is a fundamental topic due to its many practical applications. These applications include: 

Software engineering, such as for data flow diagrams, subroutine-call graphs, and program nesting trees. 

Databases, specifically entity-relationship diagrams. 

Information systems (e.g., organization charts). 

VLSI circuit layout and circuit schematics. 

Architectural floor plan layout. 

Transportation problems. """),
        },
        {
            "question": "Why is bend minimization considered an important optimization goal?",
            "grading_notes": ("""Bend minimization is crucial because the number of bends directly impacts the drawing's readability. Historically, algorithms prioritized minimizing crossings, which often resulted in drawings with unnecessarily long edge paths, a high number of bends, large area, and poor geometric uniformity. To produce drawings that are clearer and easier for the viewer to follow, it is important to keep the number of bends low. """),
        },
        {
            "question": "What aspects of the drawing are decided before metrics are assigned?",
            "grading_notes": (r"""In the Topology–Shape–Metrics (TSM) framework, the phase before assigning metrics (coordinates/lengths) is the Shape phase (Orthogonalization). This phase determines the Orthogonal Representation of the graph, which decides the following aspects: 

The planar embedding (Topology) is chosen. 

The ordered sequence of bends along the edges (left/right turns). 

The geometric angles at each vertex, typically as a clockwise sequence of values from ${90^\circ, 180^\circ, 270^\circ, 360^\circ}$. 

These aspects define the abstract shape of the drawing, independent of segment lengths or coordinates. 

 """),
        },
        {
            "question": "Why might two drawings of the same graph have different numbers of bends?",
            "grading_notes": ("""Two drawings of the same graph may have different numbers of bends because bend minimization often conflicts with other aesthetic criteria, such as crossing minimization. Graph drawing methodologies typically establish a precedence relation among aesthetics, and different algorithms prioritize these criteria differently. For instance, the traditional Topology–Shape–Metrics (TSM) approach prioritizes minimizing edge crossings, which often leads to "unnecessarily long edge paths with many bends". A drawing produced by a technique prioritizing bends will therefore likely have fewer bends than one prioritizing crossings. """),
        },
        {
            "question": "Is there a known linear-time algorithm for minimum bend orthogonal drawing of all planar graphs?",
            "grading_notes": ("""No. While computing a bend-minimum orthogonal drawing for a general planar 4-graph (where the embedding can be chosen freely) is NP-hard, finding an $O(n)$-time algorithm for minimum bend orthogonal drawing of a plane 4-graph (where the planar embedding is fixed) is a "longstanding open question". However, linear-time algorithms are known for specific subclasses, such as planar 3-graphs, and plane series–parallel 4-graphs. """),
        },
        {
            "question": "Can orthogonal drawing techniques be directly applied to non-planar graphs?",
            "grading_notes": ("""Yes, orthogonal drawing techniques can be applied to non-planar graphs. In the conventional Topology–Shape–Metrics (TSM) approach, non-planar graphs are first subjected to a planarization step where dummy vertices are introduced to represent edge crossings, resulting in a planar graph that can then be processed. TSM has also been extended to general graphs by representing high-degree vertices (which often occur in non-planar contexts) as boxes. Additionally, other methods can construct orthogonal drawings of non-planar graphs incrementally without a full planarization step."""),
        },
        {
            "question": "Do the selected papers discuss 3D orthogonal graph drawing?",
            "grading_notes": ("""Yes, 3D orthogonal graph drawing is discussed. The topic is noted as a relevant area of research in graph drawing, particularly as 3D drawings become more popular. The potential of extending concepts like rectilinear drawing and 4-cycle blocks to three dimensions is mentioned as an interesting open problem. """),
        },
        {
            "question": "Is edge crossing minimization addressed in these papers?",
            "grading_notes": ("""Yes, edge crossing minimization is addressed extensively. It is identified as one of the fundamental aesthetic criteria. Historically, algorithms, including those following the traditional TSM framework, prioritized crossing minimization. However, some recent research challenges this priority, noting that orthogonal crossings have "minimal impact on readability" and that aggressively minimizing them often negatively affects other metrics like the number of bends and area. """),
        },
        {
            "question": "Do these papers propose machine-learning-based approaches?",
            "grading_notes": ("""The papers do not explicitly propose machine-learning-based approaches. """),
        },
        {
            "question": "Which paper introduced the flow-based approach to bend minimization?",
            "grading_notes": ("""The flow-based approach for bend minimization was introduced by Roberto Tamassia. His work, presented in 1987, demonstrated how to transform the problem of finding a region-preserving grid embedding with the minimum number of bends into a minimum cost flow problem on a derived network. """),
        },
        {
            "question": "Are bend minimization and area minimization addressed by the same algorithms?",
            "grading_notes": ("""Generally, they are addressed by sequential phases rather than the same core algorithm within the Topology–Shape–Metrics (TSM) framework. Bend minimization occurs in the "Shape" phase, determining the dimensionless orthogonal representation. Area minimization occurs later in the "Metrics" (or Compaction) phase, where coordinates are assigned to the realized shape. For a specific subclass of layouts known as turn-regular orthogonal representations, the area minimization problem (compaction) can be solved optimally in linear time. """),
        },
        {
            "question": "Which papers address the problem of bend minimization having in input an orthogonal representation or part of it?",
            "grading_notes": ("""The work by Didimo, Liotta, and Patrignani (2014 discusses HV-restricted planar graphs, where edges are pre-labeled as horizontal or vertical. It notes that HV-rectilinear planarity testing is a specialized version of the general bend minimization problem. """),
        },
        {
            "question": "Which papers allow non-planar drawings?",
            "grading_notes": ("""Several papers discuss methods designed to produce or that result in non-planar drawings (i.e., drawings with crossings): 

Algorithms focused on constructing orthogonal grid drawings of nonplanar graphs are described. 

The Shape-Metrics (SM) methodology is based on creating non-planar orthogonal drawings if a rectilinear drawing is not feasible, actively subverting the TSM pipeline's prioritization of planarity. """),
        },
        {
            "question": "Which results rely on a fixed embedding, and which do not?",
            "grading_notes": ("""Results relying on a fixed embedding (Plane Graphs): The problem of finding an orthogonal drawing with the minimum number of bends is polynomial-time solvable only when a fixed planar representation (embedding) is specified. This requirement applies to algorithms for series–parallel graphs and triconnected cubic plane graphs. 

Results not relying on a fixed embedding (Variable/Unrestricted Graphs): The minimum-bend orthogonal drawing problem is NP-hard if the embedding can be freely chosen, with the exception of planar 3-graphs. Methods like the Shape-Metrics (SM) approach address problems in the variable embedding setting, often resulting in non-planar solutions.  """),
        },
        
    ]

    for s in samples:
        dataset.append(s)

    dataset.save()
    return dataset


# 3. Metriche di valutazione
# Valori ammessi da 0.0 a 1.0 con step 0.01
allowed = [round(x * 0.01, 2) for x in range(0, 101)]

# Faithfulness: quanto la risposta è supportata dal contesto
faithfulness_metric = DiscreteMetric(
    name="faithfulness",
    prompt=(
        "Evaluate the faithfulness of the answer with respect to the context.\n\n"
        "Give a score between 0 and 1 where:\n"
        "1.0 = all statements are fully supported by the context\n"
        "0.0 = most statements are unsupported or hallucinated\n\n"
        "Return ONLY the numeric score.\n\n"
        "Answer:\n{response}\n\n"
        "Context:\n{context}"
    ),
    allowed_values=allowed,
)

# Context precision: quanto il contesto recuperato è rilevante
context_precision_metric = DiscreteMetric(
    name="context_precision",
    prompt=(
        "Evaluate how relevant the retrieved context is for the question.\n\n"
        "Give a score between 0 and 1 where:\n"
        "1.0 = almost all retrieved context is relevant\n"
        "0.0 = most context is irrelevant or noisy\n\n"
        "Return ONLY the numeric score.\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{context}"
    ),
    allowed_values=allowed,
)

# Coverage: quanto la risposta copre completamente la domanda
coverage_metric = DiscreteMetric(
    name="coverage",
    prompt=(
        "Evaluate how completely the answer addresses the question.\n\n"
        "Give a score between 0 and 1 where:\n"
        "1.0 = all key aspects are covered\n"
        "0.0 = important aspects are missing\n\n"
        "Return ONLY the numeric score.\n\n"
        "Question:\n{question}\n\n"
        "Answer:\n{response}"
    ),
    allowed_values=allowed,
)

# Correctness: correttezza rispetto alla risposta di riferimento
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt=(
        "Evaluate the correctness of the answer with respect to the reference answer.\n\n"
        "Do not penalize additional details unless they contradict the reference.\n"
        "Give a score between 0 and 1 where:\n"
        "1.0 = the answer is fully correct\n"
        "0.0 = the answer is mostly incorrect\n\n"
        "Return ONLY the numeric score.\n\n"
        "Answer:\n{response}\n\n"
        "Reference:\n{context}"
    ),
    allowed_values=allowed,
)


# 4. Query al RAG per avere le risposte

def load_notebook_llm_answers(csv_path):
    """Carica risposte LLM da CSV per valutazione"""
    answers = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers[row["question"]] = row["response"]
    return answers

def query_notebook_llm(question: str, notebook_answers: dict):
    """Recupera risposta LLM dal notebook"""
    return notebook_answers.get(question, "")

def query_rag(question: str):
    """Interroga il RAG e restituisce risposta e chunk recuperati"""
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={
            "configurable": {
                "thread_id": "eval",
                "checkpoint_ns": "evals",
                "checkpoint_id": "0",
            }
        },
    )
    response = result["messages"][-1].content

    chunks = []
    for msg in result.get("messages", []):
        if getattr(msg, "type", None) == "tool" and hasattr(msg, "artifact") and msg.artifact:
            for d in msg.artifact:
                if hasattr(d, "page_content"):
                    chunks.append(d.page_content)

    # Debug
    print("\n=== Debug dei chun trovati ===")
    print("QUESTION:", question)
    print("Numero chunk RETRIEVED:", len(chunks))
    print("=== FINE DEBUG ===\n")

    return response, chunks


# 5. Funzione per esecuzione esperimento 
def run_experiment():
    """Valutazione automatica RAG sulle domande del dataset"""
    dataset = load_dataset()
    results = []

    for row in dataset:
        response, chunks = query_rag(row["question"])
        context_text = "\n\n".join(chunks)
        grading_notes = row.get("grading_notes", "")

        scores = {
            "correctness": correctness_metric.score(
                llm=judge_llm, response=response, context=grading_notes
            ).value,
            "faithfulness": faithfulness_metric.score(
                llm=judge_llm, response=response, context=context_text
            ).value,
            "context_precision": context_precision_metric.score(
                llm=judge_llm, question=row["question"], context=context_text
            ).value,
            "coverage": coverage_metric.score(
                llm=judge_llm, question=row["question"], response=response
            ).value,
        }

        results.append(
            {
                "question": row["question"],
                "response": response,
                "correctness": scores["correctness"],
                "faithfulness": scores["faithfulness"],
                "context_precision": scores["context_precision"],
                "coverage": scores["coverage"],
            }
        )

        # Log per debug
        log_path = os.path.join(
            LOG_DIR, f"{row['question'][:20].replace(' ', '_')}.log"
        )
        with open(log_path, "w") as f:
            f.write(f"QUESTION:\n{row['question']}\n\nANSWER:\n{response}\n\nCONTEXT:\n{context_text}")

    # Salvataggio CSV
    results_path = os.path.join(EXPERIMENT_DIR, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "response", "correctness", "faithfulness", "context_precision", "coverage"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Console output
    for r in results:
        print("\n---")
        for k, v in r.items():
            print(f"{k}: {v}")

def run_notebook_llm_experiment():
    """Valutazione delle risposte già generate dal Notebook LLM"""
    dataset = load_dataset()
    notebook_answers = load_notebook_llm_answers(NOTEBOOK_LLM_PATH)
    results = []

    for row in dataset:
        question = row["question"]
        if question not in notebook_answers:
            continue

        response = notebook_answers[question]
        grading_notes = row.get("grading_notes", "")

        scores = {
            "correctness": correctness_metric.score(
                llm=judge_llm, response=response, context=grading_notes
            ).value,
            "coverage": coverage_metric.score(
                llm=judge_llm, question=question, response=response
            ).value,
        }

        results.append({
            "question": question,
            "response": response,
            **scores,
        })

    # Salvataggio CSV separato
    results_path = os.path.join(EXPERIMENT_DIR, "results_notebook_llm.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "response", "correctness", "coverage"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Console output
    for r in results:
        print("\n---")
        for k, v in r.items():
            print(f"{k}: {v}")

    print("\n=== Valutazione NOTEBOOK LLM completata ===")
    print(f"Results saved to: {results_path}")

# 6. Avvio esperimenti di valutazione

if __name__ == "__main__":
    run_notebook_llm_experiment()  # Notebook LLM
    run_experiment()               # RAG