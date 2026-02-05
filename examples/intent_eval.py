"""
Evaluador interactivo de modelos de clasificación de intenciones de Hugging Face Hub.

Uso:
    python examples/intent_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt

from ai.intention import Intention

console = Console()

MODELS = [
    ("Falconsai/intent_classification", "DistilBERT fine-tuned para intenciones genéricas"),
    ("Serj/intent-classifier", "Clasificador general de intenciones"),
]

EXAMPLE_QUERIES = [
    "I want to book a flight to Madrid",
    "What's the weather like today?",
    "Cancel my subscription please",
    "Tell me a joke",
    "How do I reset my password?",
    "I need help with my order",
    "Play some music",
    "Set an alarm for 7am",
    "What is the capital of France?",
    "I want to talk to a human agent",
]


def load_classifier(model_name: str) -> Intention | None:
    console.print(f"\n[yellow]Cargando modelo:[/yellow] {model_name}")
    console.print("[dim]Esto puede tardar la primera vez (descarga de pesos)...[/dim]")
    try:
        clf = Intention(model_name=model_name)
        # Force pipeline load
        clf.classify("test")
        console.print(f"[green]Modelo cargado correctamente.[/green]\n")
        return clf
    except Exception as e:
        console.print(f"[red]Error cargando modelo: {e}[/red]\n")
        return None


def show_results(query: str, results: list[dict], max_labels: int = 5):
    table = Table(title=f"Query: {query}", show_lines=True)
    table.add_column("Intent", style="cyan", min_width=20)
    table.add_column("Score", style="magenta", justify="right", min_width=10)

    for r in results[:max_labels]:
        score = f"{r['score']:.4f}"
        table.add_row(r["label"], score)

    console.print(table)


def run_examples(clf: Intention):
    console.print("\n[bold]Ejecutando queries de ejemplo...[/bold]\n")
    for query in EXAMPLE_QUERIES:
        results = clf.classify(query)
        show_results(query, results)
    console.print()


def interactive_mode(clf: Intention):
    console.print("\n[bold]Modo interactivo[/bold] — escribe una query para clasificar (vacío para volver)\n")
    while True:
        query = Prompt.ask("[cyan]Query[/cyan]")
        if not query.strip():
            break
        results = clf.classify(query)
        show_results(query, results)


def main():
    clf = None
    current_model = None

    while True:
        console.print("\n[bold]═══ Evaluador de Intent Classification ═══[/bold]\n")
        if current_model:
            console.print(f"  Modelo activo: [green]{current_model}[/green]\n")

        console.print("  [1] Seleccionar modelo")
        console.print("  [2] Ejecutar queries de ejemplo")
        console.print("  [3] Modo interactivo")
        console.print("  [4] Salir")
        console.print()

        choice = Prompt.ask("Opción", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            console.print("\n[bold]Modelos disponibles:[/bold]\n")
            for i, (name, desc) in enumerate(MODELS, 1):
                console.print(f"  [{i}] {name}")
                console.print(f"      [dim]{desc}[/dim]")
            console.print()

            idx = IntPrompt.ask("Selecciona modelo", default=1)
            if 1 <= idx <= len(MODELS):
                model_name = MODELS[idx - 1][0]
                clf = load_classifier(model_name)
                if clf:
                    current_model = model_name
            else:
                console.print("[red]Opción no válida.[/red]")

        elif choice == "2":
            if clf is None:
                console.print("[red]Primero selecciona un modelo (opción 1).[/red]")
            else:
                run_examples(clf)

        elif choice == "3":
            if clf is None:
                console.print("[red]Primero selecciona un modelo (opción 1).[/red]")
            else:
                interactive_mode(clf)

        elif choice == "4":
            console.print("[dim]Saliendo...[/dim]")
            break


if __name__ == "__main__":
    main()
