# analysis
# Questo modulo analizza i risultati sperimentali prodotti dalla valutazione del RAG.
# Carica i file CSV delle metriche, calcola statistiche aggregate
# (media, deviazione standard, min, max) e genera grafici comparativi.
# Supporta:
#   - confronto tra diversi numeri di chunk recuperati
#   - confronto tra RAG e NotebookLM
# I plot prodotti sono utilizzati per l’analisi quantitativa delle prestazioni.
import os
import csv
import statistics
import matplotlib.pyplot as plt



#Percorsi vari per i risultati
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_PATH = os.path.join(BASE_DIR, "rag_eval", "experiments", "results.csv")
NOTEBOOK_RESULTS_PATH = os.path.join(BASE_DIR, "rag_eval", "experiments", "results_notebook_llm.csv")
RESULTS_3_PATH = os.path.join(BASE_DIR, "rag_eval", "experiments", "results_3chunk.csv")
RESULTS_5_PATH = os.path.join(BASE_DIR, "rag_eval", "experiments", "results_5chunk.csv")
RESULTS_13_PATH = os.path.join(BASE_DIR, "rag_eval", "experiments", "results_13chunk.csv")
RESULTS_15_PATH = os.path.join(BASE_DIR, "rag_eval", "experiments", "results_15chunk.csv")


# Analisi di NotebookLM

def analyze_notebook_llm(results_path):
    """Carica i risultati del Notebook LLM e calcola media e deviazione standard"""
    metrics = ["correctness", "coverage"]
    rows = []

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Notebook LLM results not found: {results_path}")

    # Lettura CSV
    with open(results_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({m: float(row[m]) for m in metrics})

    # Statistiche
    stats = {
        m: {
            "mean": statistics.mean(r[m] for r in rows),
            "std": statistics.stdev(r[m] for r in rows),
        }
        for m in metrics
    }

    print("\n== NOTEBOOK LLM STATISTICS ==")
    for m, v in stats.items():
        print(f"{m.upper():15s} mean={v['mean']:.3f} | std={v['std']:.3f}")

    return stats


# Classe per analizzare i risultati
class ResultsAnalyzer:
    METRICS = ["correctness", "faithfulness", "context_precision", "coverage"]

    def __init__(self, results_path: str):
        self.results_path = results_path
        self.rows = []

    def load_results(self):
        """Carica i risultati da CSV e converte le metriche in float"""
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        with open(self.results_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    parsed[k] = float(v) if k in self.METRICS else v
                self.rows.append(parsed)

    def compute_means(self):
        """Calcola solo le medie delle metriche"""
        return {metric: statistics.mean(r[metric] for r in self.rows) for metric in self.METRICS}

    def compute_stats(self):
        """Calcola media, std, min e max per ogni metrica(max e min non usati poi nei grafici)"""
        return {
            metric: {
                "mean": statistics.mean(r[metric] for r in self.rows),
                "std": statistics.stdev(r[metric] for r in self.rows),
                "min": min(r[metric] for r in self.rows),
                "max": max(r[metric] for r in self.rows),
            }
            for metric in self.METRICS
        }


    # Funzione di plot per il sistema RAG a 7 chunk
    def plot_metric_stats_grouped(self, save_path=None):
        """Grafico a barre con media e deviazione standard per ogni metrica"""
        stats = self.compute_stats()
        metrics = list(stats.keys())
        means = [stats[m]["mean"] for m in metrics]
        stds  = [stats[m]["std"]  for m in metrics]
        x = range(len(metrics))
        width = 0.35

        plt.figure(figsize=(10, 5))
        bars_mean = plt.bar([i - width / 2 for i in x], means, width, label="Media")
        bars_std = plt.bar([i + width / 2 for i in x], stds, width, label="Dev.Standard")

        # Aggiunge i valori sopra le barre
        for bar in bars_mean + bars_std:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.xticks(x, metrics)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"\n📊 Plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()


# Funzione di stampa per vedere risultati al variare del numero di chunk
def print_chunk_stats(stats, chunk_label):
    """Stampa le statistiche principali per ogni metrica"""
    print(f"\n=== STATISTICHE RAG ({chunk_label}) ===")
    for metric, values in stats.items():
        print(
            f"{metric.upper():20s} "
            f"mean={values['mean']:.3f} | "
            f"std={values['std']:.3f} | "
            f"min={values['min']:.3f} | "
            f"max={values['max']:.3f}"
        )

def plot_chunk_comparison_mean_std(stats_3, stats_5, stats_7, stats_13, stats_15, save_path=None):
    """Confronto dei valori medi per diverse configurazioni di chunk"""
    metrics = ResultsAnalyzer.METRICS
    x = range(len(metrics))
    width = 0.15

    means_3 = [stats_3[m]["mean"] for m in metrics]
    means_5 = [stats_5[m]["mean"] for m in metrics]
    means_7 = [stats_7[m]["mean"] for m in metrics]
    means_13 = [stats_13[m]["mean"] for m in metrics]
    means_15 = [stats_15[m]["mean"] for m in metrics]

    plt.figure(figsize=(12, 5))

    bars = [
        plt.bar([i - 2*width for i in x], means_3, width, label="3 chunks"),
        plt.bar([i - width for i in x], means_5, width, label="5 chunks"),
        plt.bar(x, means_7, width, label="7 chunks"),
        plt.bar([i + width for i in x], means_13, width, label="13 chunks"),
        plt.bar([i + 2*width for i in x], means_15, width, label="15 chunks"),
    ]

    # Valori sopra le barre
    for group in bars:
        for bar in group:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n📊 Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_std_bars_vs_chunks(stats_3, stats_5, stats_7, stats_13, stats_15, save_path=None):
    """Confronto della deviazione standard per diverse configurazioni di chunk"""
    metrics = ResultsAnalyzer.METRICS
    x = range(len(metrics))
    width = 0.15

    std_3 = [stats_3[m]["std"] for m in metrics]
    std_5 = [stats_5[m]["std"] for m in metrics]
    std_7 = [stats_7[m]["std"] for m in metrics]
    std_13 = [stats_13[m]["std"] for m in metrics]
    std_15 = [stats_15[m]["std"] for m in metrics]

    plt.figure(figsize=(12, 5))
    bars = [
        plt.bar([i - 2*width for i in x], std_3, width, label="3 chunks"),
        plt.bar([i - width for i in x], std_5, width, label="5 chunks"),
        plt.bar(x, std_7, width, label="7 chunks"),
        plt.bar([i + width for i in x], std_13, width, label="13 chunks"),
        plt.bar([i + 2*width for i in x], std_15, width, label="15 chunks"),
    ]

    for group in bars:
        for bar in group:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, metrics)
    plt.ylim(0, 0.5)
    plt.ylabel("Dev.Standard")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n📊 Std bars plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# Funzioni per i ocnfronti tra il sistema RAG e NotebookLM
def plot_rag_vs_notebook(rag_means, notebook_means, save_path=None):
    """Confronto dei valori medi tra RAG e Notebook LLM"""
    metrics = ["correctness", "coverage"]
    rag_values = [rag_means[m] for m in metrics]
    notebook_values = [notebook_means[m]["mean"] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars_rag = plt.bar([i - width/2 for i in x], rag_values, width, label="RAG", color="green")
    bars_notebook = plt.bar([i + width/2 for i in x], notebook_values, width, label="NotebookLM", color="red")

    # Aggiungi valori sopra le barre
    for bar in bars_rag + bars_notebook:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylim(0, 1)
    plt.ylabel("Valore Medio")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n📊 Comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_rag_vs_notebook_std(rag_stats, notebook_stats, save_path=None):
    """Confronto della deviazione standard tra RAG e Notebook LLM"""
    metrics = ["correctness", "coverage"]
    rag_std = [rag_stats[m]["std"] for m in metrics]
    notebook_std = [notebook_stats[m]["std"] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars_rag = plt.bar([i - width/2 for i in x], rag_std, width, label="RAG", color="green")
    bars_notebook = plt.bar([i + width/2 for i in x], notebook_std, width, label="NotebookLM", color="red")

    # Imposta limite massimo leggermente sopra il valore massimo
    plt.ylim(0, max(rag_std + notebook_std) + 0.04)

    for bar in bars_rag + bars_notebook:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylabel("Dev.Standard")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n📊 Std comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# Avvio programma

if __name__ == "__main__":
    # Caricamento e analisi dei diversi chunk
    analyzers = {}
    stats = {}

    for label, path in [("3 CHUNK", RESULTS_3_PATH),
                        ("5 CHUNK", RESULTS_5_PATH),
                        ("7 CHUNK", RESULTS_PATH),
                        ("13 CHUNK", RESULTS_13_PATH),
                        ("15 CHUNK", RESULTS_15_PATH)]:
        analyzer = ResultsAnalyzer(path)
        analyzer.load_results()
        analyzers[label] = analyzer
        stats[label] = analyzer.compute_stats()
        print_chunk_stats(stats[label], label)

    # Statistiche aggregate RAG 7 chunk
    analyzer_7 = analyzers["7 CHUNK"]
    rag_means = analyzer_7.compute_means()
    rag_stats = analyzer_7.compute_stats()

    # Notebook LLM
    notebook_stats = analyze_notebook_llm(NOTEBOOK_RESULTS_PATH)

    # Plot comparativi
    plot_chunk_comparison_mean_std(stats["3 CHUNK"], stats["5 CHUNK"], stats["7 CHUNK"],
                                   stats["13 CHUNK"], stats["15 CHUNK"],
                                   save_path=os.path.join(BASE_DIR, "rag_eval", "experiments", "chunk_comparison_mean_std.png"))

    plot_std_bars_vs_chunks(stats["3 CHUNK"], stats["5 CHUNK"], stats["7 CHUNK"],
                            stats["13 CHUNK"], stats["15 CHUNK"],
                            save_path=os.path.join(BASE_DIR, "rag_eval", "experiments", "std_bars_vs_chunks.png"))

    plot_rag_vs_notebook(rag_means, notebook_stats,
                         save_path=os.path.join(BASE_DIR, "rag_eval", "experiments", "rag_vs_notebook.png"))

    plot_rag_vs_notebook_std(rag_stats, notebook_stats,
                             save_path=os.path.join(BASE_DIR, "rag_eval", "experiments", "rag_vs_notebook_std.png"))

    analyzer_7.plot_metric_stats_grouped(save_path=os.path.join(BASE_DIR, "rag_eval", "experiments", "metrics_grouped.png"))