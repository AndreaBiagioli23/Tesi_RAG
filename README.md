# Tesi_RAG
Questo repository contiene il codice sviluppato a supporto di una tesi accademica incentrata sulla progettazione e implementazione di un sistema di Retrieval-Augmented Generation (RAG) per documenti scientifici.

Il repository include:
- il codice per la costruzione del sistema RAG,
- gli script utilizzati per la valutazione sperimentale,
- le classi impiegate per la generazione dei grafici riportati nella tesi.

L’obiettivo del progetto è fornire un prototipo di sistema RAG applicato
a documenti scientifici, con particolare attenzione sulla riproducibilità
degli esperimenti e sulla separazione tra contenuto testuale e risorse visive.

## Struttura del repository

Il repository contiene un archivio `.zip` con il progetto completo:

- progetto_tesi.zip
  - Codice/
    - app.py -> Avvio dell’applicazione (backend). 
    - rag_graph.py -> Definizione della pipeline RAG (workflow/graph).
    - evals.py -> modulo per la valutazione sperimentale (RAGAS).
    - analysis.py -> Analisi dei risultati e generazione dei grafici.
    - rag_interface.html  -> Interfaccia HTML per l’interrogazione del sistema.
    - static/  -> Risorse statiche dell’interfaccia (logo del sistema).    
    - pdf_images/  -> Sottocartelle (una per PDF) contenenti le immagini estratte dai documenti.        
      **Nota:** queste immagini non sono incluse nel repository per motivi di copyright.
    - rag_eval/ -> Output della valutazione sperimentale (file CSV e grafici generati).


## Dataset e risorse non incluse (Copyright)
Per motivi di copyright, NON sono inclusi nel repository:
- i documenti PDF utilizzati come dataset (paper scientifici, capitoli di libri, ecc.),
- immagini/figure estratte dai documenti.

Di conseguenza, per ripetere gli esperimenti è necessario procurarsi autonomamente i documenti originali e inserirli nel percorso previsto dagli script.
Per garantire la riproducibilità, l’elenco completo dei documenti utilizzati è riportato nella tesi (Appendice A).
