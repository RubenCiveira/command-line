"""
Generación de embeddings local usando sentence-transformers, sin depender de Ollama.

Incluye una selección de modelos de distintos tamaños y capacidades.

Uso:
    python src/embeddings_local.py
    python src/embeddings_local.py --model BAAI/bge-small-en-v1.5
"""

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt

console = Console()

MODELS = [
    ("BAAI/bge-small-en-v1.5", "33M", "~130 MB", 384, "Inglés, muy ligero y rápido"),
    ("BAAI/bge-base-en-v1.5", "109M", "~440 MB", 768, "Inglés, buen equilibrio"),
    ("sentence-transformers/all-MiniLM-L6-v2", "22M", "~90 MB", 384, "Inglés, clásico ligero"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "118M", "~470 MB", 384, "Multilingüe, compacto"),
    ("BAAI/bge-m3", "567M", "~2.2 GB", 1024, "Multilingüe 100+ idiomas, alta calidad"),
    ("intfloat/multilingual-e5-large", "560M", "~2.2 GB", 1024, "Multilingüe, rendimiento top"),
]

DEFAULT_MODEL_IDX = 3  # paraphrase-multilingual-MiniLM-L12-v2

SAMPLE_TEXTS = [
    "How do I fix a segmentation fault in C++?",
    "Configure nginx as a reverse proxy",
    "What is the derivative of x squared?",
    "Solve a system of linear equations",
    "Messi scored a hat-trick yesterday",
    "The team won the championship last season",
    "Add salt and pepper, then simmer for 20 minutes",
    "I have a headache and a sore throat",
]


def select_model() -> str:
    console.print("\n[bold]Modelos de embeddings disponibles:[/bold]\n")
    for i, (name, params, ram, dim, desc) in enumerate(MODELS, 1):
        console.print(f"  [{i}] {name}")
        console.print(f"      [dim]{params} params · {ram} · dim={dim} · {desc}[/dim]")
    console.print()

    idx = IntPrompt.ask("Selecciona modelo", default=DEFAULT_MODEL_IDX + 1)
    if 1 <= idx <= len(MODELS):
        return MODELS[idx - 1][0]
    console.print("[red]Opción no válida, usando modelo por defecto.[/red]")
    return MODELS[DEFAULT_MODEL_IDX][0]


def load_model(model_name: str) -> SentenceTransformer:
    console.print(f"\n[yellow]Cargando modelo:[/yellow] {model_name}")
    console.print("[dim]Primera ejecución descarga los pesos...[/dim]\n")

    model = SentenceTransformer(model_name)

    console.print(f"[green]Modelo cargado.[/green]")
    console.print(f"  Dimensión embedding: [cyan]{model.get_sentence_embedding_dimension()}[/cyan]\n")
    return model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def demo_embeddings(model: SentenceTransformer):
    console.print("[bold]═══ Generación de Embeddings ═══[/bold]\n")

    # Generar embeddings para todos los textos de ejemplo
    console.print("[yellow]Generando embeddings para textos de ejemplo...[/yellow]\n")
    embeddings = model.encode(SAMPLE_TEXTS, normalize_embeddings=True)

    # Mostrar dimensiones
    for i, (text, emb) in enumerate(zip(SAMPLE_TEXTS, embeddings)):
        preview = text[:60] + "..." if len(text) > 60 else text
        console.print(f"  [{i}] {preview}")
        console.print(f"      shape={emb.shape}  primeros 5 valores: {emb[:5].round(4).tolist()}")
    console.print()

    # Matriz de similitud
    console.print("[bold]Matriz de similitud coseno:[/bold]\n")
    table = Table(show_lines=True)
    table.add_column("", style="bold", min_width=6)
    for i in range(len(SAMPLE_TEXTS)):
        table.add_column(f"[{i}]", justify="right", min_width=6)

    for i, emb_i in enumerate(embeddings):
        row = []
        for j, emb_j in enumerate(embeddings):
            sim = cosine_similarity(emb_i, emb_j)
            if i == j:
                row.append("[dim]1.000[/dim]")
            elif sim > 0.7:
                row.append(f"[bold green]{sim:.3f}[/bold green]")
            elif sim > 0.4:
                row.append(f"[yellow]{sim:.3f}[/yellow]")
            else:
                row.append(f"[dim]{sim:.3f}[/dim]")
        table.add_row(f"[{i}]", *row)

    console.print(table)
    console.print()

    # Pares más similares (excluyendo self)
    console.print("[bold]Top 5 pares más similares:[/bold]\n")
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            pairs.append((i, j, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, sim in pairs[:5]:
        console.print(f"  {sim:.4f}  [{i}] {SAMPLE_TEXTS[i][:50]}")
        console.print(f"           [{j}] {SAMPLE_TEXTS[j][:50]}")
        console.print()


def interactive_mode(model: SentenceTransformer):
    console.print("[bold]Modo interactivo[/bold] — compara dos textos (vacío para salir)\n")
    while True:
        text_a = console.input("[cyan]Texto A:[/cyan] ")
        if not text_a.strip():
            break
        text_b = console.input("[cyan]Texto B:[/cyan] ")
        if not text_b.strip():
            break

        embs = model.encode([text_a, text_b], normalize_embeddings=True)
        sim = cosine_similarity(embs[0], embs[1])

        if sim > 0.7:
            color = "bold green"
        elif sim > 0.4:
            color = "yellow"
        else:
            color = "red"

        console.print(f"\n  Similitud coseno: [{color}]{sim:.4f}[/{color}]\n")


def search_mode(model: SentenceTransformer):
    console.print("[bold]Modo búsqueda[/bold] — busca en los textos de ejemplo (vacío para salir)\n")
    corpus_embeddings = model.encode(SAMPLE_TEXTS, normalize_embeddings=True)

    while True:
        query = console.input("[cyan]Query:[/cyan] ")
        if not query.strip():
            break

        query_emb = model.encode([query], normalize_embeddings=True)[0]

        scores = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(corpus_embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)

        console.print()
        table = Table(title="Resultados", show_lines=False)
        table.add_column("#", style="bold", min_width=3)
        table.add_column("Score", style="magenta", justify="right", min_width=8)
        table.add_column("Texto", style="cyan")

        for i, (idx, score) in enumerate(scores[:5], 1):
            table.add_row(str(i), f"{score:.4f}", SAMPLE_TEXTS[idx])

        console.print(table)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Embeddings local con sentence-transformers")
    parser.add_argument("--model", default=None, help="Modelo HF Hub (omitir para menú interactivo)")
    args = parser.parse_args()

    if args.model:
        model_name = args.model
    else:
        model_name = select_model()

    model = load_model(model_name)

    while True:
        console.print("\n[bold]═══ Embeddings Local ═══[/bold]\n")
        console.print("  [1] Demo con textos de ejemplo + matriz de similitud")
        console.print("  [2] Comparar dos textos")
        console.print("  [3] Buscar en corpus de ejemplo")
        console.print("  [4] Cambiar modelo")
        console.print("  [5] Salir")
        console.print()

        choice = Prompt.ask("Opción", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "1":
            demo_embeddings(model)
        elif choice == "2":
            interactive_mode(model)
        elif choice == "3":
            search_mode(model)
        elif choice == "4":
            model_name = select_model()
            model = load_model(model_name)
        elif choice == "5":
            console.print("[dim]Saliendo...[/dim]")
            break


if __name__ == "__main__":
    main()
