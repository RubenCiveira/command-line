"""
Evaluador interactivo de clasificación zero-shot jerárquica por categorías temáticas.

Clasifica textos en un árbol de categorías multinivel usando modelos
de Hugging Face Hub. Primero clasifica en nivel 1, luego baja al subnivel
del ganador, y así sucesivamente.

Uso:
    python src/zero_shot_eval.py
"""

import json
from pathlib import Path
from transformers import pipeline
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.prompt import Prompt, IntPrompt

console = Console()
SCRIPT_DIR = Path(__file__).parent

MODELS = [
    ("facebook/bart-large-mnli", "BART large, estándar de facto para zero-shot (solo inglés)"),
    ("MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "ModernBERT, multilingüe y reciente"),
    ("DAMO-NLP-SG/zero-shot-classify-SSTuning-base", "RoBERTa base, ligero (2-20 labels)"),
]

# Árbol jerárquico de categorías.
# Se carga desde iptc_categories.json (taxonomía IPTC Media Topics).
# Cada nodo es un dict: las claves son categorías y los valores son
# sus subcategorías (dict) o None si es hoja.
def _load_category_tree() -> dict:
    json_path = SCRIPT_DIR / "iptc_categories.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    console.print(f"[red]No se encontró {json_path}[/red]")
    raise SystemExit(1)

CATEGORY_TREE = _load_category_tree()

EXAMPLE_QUERIES = [
    "How do I fix a NullPointerException in Java?",
    "What is the derivative of x squared?",
    "The mitochondria is the powerhouse of the cell",
    "Our Q3 revenue exceeded expectations by 15%",
    "Messi scored a hat-trick in yesterday's match",
    "I have a headache and a sore throat",
    "Add salt and pepper to taste, then simmer for 20 minutes",
    "The new album features a mix of jazz and electronic music",
    "The parliament approved the new tax reform",
    "How do I teach fractions to a 10 year old?",
    "Implement a binary search tree in Python",
    "Solve the integral of sin(x) * cos(x)",
    "What are the side effects of ibuprofen?",
    "Configure nginx as a reverse proxy for a Node.js app",
    "Explain quantum entanglement in simple terms",
]

MIN_CONFIDENCE = 0.3  # Umbral mínimo para bajar al siguiente nivel


def load_model(model_name: str):
    console.print(f"\n[yellow]Cargando modelo:[/yellow] {model_name}")
    console.print("[dim]Esto puede tardar la primera vez (descarga de pesos)...[/dim]")
    try:
        clf = pipeline("zero-shot-classification", model=model_name)
        console.print("[green]Modelo cargado correctamente.[/green]\n")
        return clf
    except Exception as e:
        console.print(f"[red]Error cargando modelo: {e}[/red]\n")
        return None


def classify_flat(clf, text: str, categories: list[str]) -> dict:
    return clf(text, candidate_labels=categories)


def classify_hierarchical(clf, text: str, tree: dict, min_confidence: float = MIN_CONFIDENCE) -> list[dict]:
    """
    Clasifica un texto bajando por el árbol de categorías nivel a nivel.

    Retorna una lista de dicts, uno por nivel, con:
        {"level": int, "label": str, "score": float, "all": dict}
    """
    path = []
    current = tree
    level = 0

    while current is not None:
        labels = list(current.keys())
        if len(labels) < 2:
            # Un solo hijo: se selecciona directamente sin clasificar
            if labels:
                only_label = labels[0]
                path.append({"level": level, "label": only_label, "score": 1.0, "all": {only_label: 1.0}})
                current = current[only_label]
                level += 1
            else:
                break
            continue

        result = clf(text, candidate_labels=labels)
        scores = dict(zip(result["labels"], result["scores"]))
        best_label = result["labels"][0]
        best_score = result["scores"][0]

        path.append({"level": level, "label": best_label, "score": best_score, "all": scores})

        if best_score < min_confidence:
            break

        subtree = current.get(best_label)
        if subtree is None:
            break

        current = subtree
        level += 1

    return path


def show_hierarchical_result(query: str, path: list[dict]):
    full_path = " > ".join(step["label"] for step in path)
    console.print(f"\n[bold]{query}[/bold]")
    console.print(f"  Ruta: [green]{full_path}[/green]\n")

    for step in path:
        level_label = f"Nivel {step['level']}"
        table = Table(title=level_label, show_lines=False, title_style="bold yellow", padding=(0, 1))
        table.add_column("Categoría", style="cyan", min_width=25)
        table.add_column("Score", style="magenta", justify="right", min_width=8)
        table.add_column("", min_width=25)

        sorted_scores = sorted(step["all"].items(), key=lambda x: x[1], reverse=True)
        for label, score in sorted_scores[:5]:
            bar_len = int(score * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            if label == step["label"]:
                table.add_row(f"[bold green]{label}[/bold green]", f"{score:.4f}", bar)
            else:
                table.add_row(label, f"{score:.4f}", bar)

        console.print(table)

    console.print()


def show_tree(tree: dict, title: str = "Árbol de categorías"):
    rich_tree = Tree(f"[bold]{title}[/bold]")
    _build_rich_tree(rich_tree, tree)
    console.print(rich_tree)
    console.print()


def _build_rich_tree(parent, node: dict):
    for key, children in node.items():
        branch = parent.add(f"[cyan]{key}[/cyan]")
        if children:
            _build_rich_tree(branch, children)


def run_examples(clf, tree: dict):
    console.print("\n[bold]Ejecutando queries de ejemplo (clasificación jerárquica)...[/bold]\n")
    for query in EXAMPLE_QUERIES:
        path = classify_hierarchical(clf, query, tree)
        show_hierarchical_result(query, path)


def interactive_mode(clf, tree: dict):
    console.print("\n[bold]Modo interactivo[/bold] — escribe un texto para clasificar (vacío para volver)\n")
    while True:
        query = Prompt.ask("[cyan]Texto[/cyan]")
        if not query.strip():
            break
        path = classify_hierarchical(clf, query, tree)
        show_hierarchical_result(query, path)


def main():
    clf = None
    current_model = None
    tree = CATEGORY_TREE

    while True:
        console.print("\n[bold]═══ Evaluador Zero-Shot Jerárquico ═══[/bold]\n")
        if current_model:
            console.print(f"  Modelo activo: [green]{current_model}[/green]")

        top_count = len(tree)
        total = _count_leaves(tree)
        console.print(f"  Categorías:    [cyan]{top_count}[/cyan] raíz, [cyan]{total}[/cyan] hojas totales")
        console.print(f"  Confianza mín: [yellow]{MIN_CONFIDENCE}[/yellow]\n")

        console.print("  [1] Seleccionar modelo")
        console.print("  [2] Ver árbol de categorías")
        console.print("  [3] Ejecutar queries de ejemplo")
        console.print("  [4] Modo interactivo")
        console.print("  [5] Salir")
        console.print()

        choice = Prompt.ask("Opción", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "1":
            console.print("\n[bold]Modelos disponibles:[/bold]\n")
            for i, (name, desc) in enumerate(MODELS, 1):
                console.print(f"  [{i}] {name}")
                console.print(f"      [dim]{desc}[/dim]")
            console.print()

            idx = IntPrompt.ask("Selecciona modelo", default=1)
            if 1 <= idx <= len(MODELS):
                model_name = MODELS[idx - 1][0]
                clf = load_model(model_name)
                if clf:
                    current_model = model_name
            else:
                console.print("[red]Opción no válida.[/red]")

        elif choice == "2":
            show_tree(tree)

        elif choice == "3":
            if clf is None:
                console.print("[red]Primero selecciona un modelo (opción 1).[/red]")
            else:
                run_examples(clf, tree)

        elif choice == "4":
            if clf is None:
                console.print("[red]Primero selecciona un modelo (opción 1).[/red]")
            else:
                interactive_mode(clf, tree)

        elif choice == "5":
            console.print("[dim]Saliendo...[/dim]")
            break


def _count_leaves(node: dict) -> int:
    count = 0
    for children in node.values():
        if children is None:
            count += 1
        else:
            count += _count_leaves(children)
    return count


if __name__ == "__main__":
    main()
