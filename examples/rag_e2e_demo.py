"""
Demo end-to-end del pipeline RAG: ingesta, clasificacion y busqueda.

Procesa los documentos de un directorio usando RagIngest (que gestiona
la extraccion, chunking, embeddings y clasificacion), y muestra los
resultados por consola.

Uso:
    python examples/rag_e2e_demo.py /ruta/a/directorio/con/pdfs
    python examples/rag_e2e_demo.py /ruta/a/pdfs --db /tmp/test_rag.db
    python examples/rag_e2e_demo.py /ruta/a/pdfs --query "machine learning"
"""

import argparse
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to sys.path so we can import the ai package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ai.rag.sqlite_rag_setup import SqliteRagSetup
from ai.rag.rag_ingest import RagIngest
from ai.rag.rag_retriever import RagRetriever
from ai.classificator import Classificator
from ai.user.user_config import UserConfig

console = Console()


def find_categories_file() -> Path:
    """Look for iptc_categories.json in standard locations."""
    candidates = [
        UserConfig().ragCategoriesPath(),
        Path(__file__).parent / ".." / "resources" / "iptc_categories.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]  # default path even if missing


class DemoConfig(UserConfig):
    """UserConfig override that points to the demo directory and DB."""

    def __init__(self, directory: Path, db_path: Path, categories_path: Path):
        self._directory = directory
        self._db_path = db_path
        self._categories_path = categories_path

    def ragDatabasePath(self) -> Path:
        return self._db_path

    def ragCategoriesPath(self) -> Path:
        return self._categories_path

    def ragTopics(self) -> list[dict]:
        return [{"name": "demo", "path": str(self._directory)}]


def read_results_from_db(db_path: Path) -> dict:
    """Read ingestion results from the database for display."""
    setup = SqliteRagSetup(db_path)
    conn = setup.connect()
    try:
        # Documents with chunk count
        docs = conn.execute(
            "SELECT d.id, d.topic, d.path, length(d.content), d.created_at, "
            "  (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) "
            "FROM documents d ORDER BY d.topic, d.path"
        ).fetchall()

        # Categories
        categories = conn.execute(
            "SELECT dc.document_id, d.path, dc.level, dc.category, dc.score "
            "FROM document_categories dc "
            "JOIN documents d ON d.id = dc.document_id "
            "ORDER BY dc.document_id, dc.level"
        ).fetchall()

        # Totals
        total_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        total_categories = conn.execute(
            "SELECT count(*) FROM document_categories"
        ).fetchone()[0]

        return {
            "documents": docs,
            "categories": categories,
            "total_docs": len(docs),
            "total_chunks": total_chunks,
            "total_categories": total_categories,
            "db_path": db_path,
        }
    finally:
        conn.close()


def _format_timestamp(ts: float | None) -> str:
    """Format a Unix timestamp as a readable date string."""
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def show_summary(summary: dict, ingested: int, elapsed: float) -> None:
    """Display formatted summary of the ingestion."""
    console.print()
    console.print(Panel("[bold]Resumen de ingesta[/bold]", expand=False))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Documentos ingestados", f"[green]{ingested}[/green]")
    table.add_row("Total en base de datos", str(summary["total_docs"]))
    table.add_row("Chunks generados", str(summary["total_chunks"]))
    table.add_row("Clasificaciones", str(summary["total_categories"]))
    table.add_row("Tiempo total", f"{elapsed:.1f}s")
    table.add_row("Base de datos", str(summary["db_path"]))
    console.print(table)


def show_documents_table(summary: dict) -> None:
    """Display the documents table contents."""
    docs = summary["documents"]
    if not docs:
        console.print("\n[dim]Tabla documents vacia.[/dim]")
        return

    console.print()
    console.print(Panel("[bold]Tabla: documents[/bold]", expand=False))

    table = Table(show_lines=False)
    table.add_column("id", style="dim", justify="right")
    table.add_column("topic", style="cyan")
    table.add_column("path")
    table.add_column("chars", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("created_at", style="dim")

    for doc_id, topic, path, content_len, created_at, chunk_count in docs:
        table.add_row(
            str(doc_id),
            topic,
            path,
            str(content_len or 0),
            str(chunk_count),
            _format_timestamp(created_at),
        )
    console.print(table)


def show_categories_table(summary: dict) -> None:
    """Display the document_categories table contents."""
    categories = summary["categories"]
    if not categories:
        console.print("\n[dim]Tabla document_categories vacia.[/dim]")
        return

    console.print()
    console.print(Panel("[bold]Tabla: document_categories[/bold]", expand=False))

    table = Table(show_lines=False)
    table.add_column("doc_id", style="dim", justify="right")
    table.add_column("documento")
    table.add_column("level", justify="center")
    table.add_column("category", style="cyan")
    table.add_column("score", justify="right")

    for doc_id, path, level, category, score in categories:
        if score > 0.7:
            score_style = "green"
        elif score > 0.4:
            score_style = "yellow"
        else:
            score_style = "red"
        table.add_row(
            str(doc_id),
            path,
            str(level),
            category,
            f"[{score_style}]{score:.3f}[/{score_style}]",
        )
    console.print(table)

    # Category frequency
    console.print()
    cat_freq: dict[str, int] = {}
    for _, _, _, category, _ in categories:
        cat_freq[category] = cat_freq.get(category, 0) + 1

    freq_table = Table(title="Frecuencia de categorias", show_lines=False)
    freq_table.add_column("Categoria", style="cyan")
    freq_table.add_column("Docs", style="magenta", justify="right")
    for label, count in sorted(
        cat_freq.items(), key=lambda x: x[1], reverse=True,
    )[:15]:
        freq_table.add_row(label, str(count))
    console.print(freq_table)


def run_query(
    config: DemoConfig,
    query: str,
    retriever: RagRetriever,
    classificator: Classificator | None,
) -> None:
    """Run a similarity search query against the ingested database."""
    console.print()
    console.print(Panel(f"[bold]Busqueda: [cyan]{query}[/cyan][/bold]", expand=False))

    # Classify the query
    classification: list[dict] | None = None
    if classificator is not None:
        try:
            classification = classificator.classify(query)
            if classification:
                path = " > ".join(s["label"] for s in classification)
                scores = ", ".join(f'{s["score"]:.3f}' for s in classification)
                console.print(f"  [bold]Clasificacion:[/bold] [cyan]{path}[/cyan]")
                console.print(f"  [dim]Scores: {scores}[/dim]")
                level0 = classification[0]["label"] if classification else None
                console.print(f"  [dim]Filtro nivel 0: {level0}[/dim]")
            else:
                console.print("  [dim]Sin clasificacion para la query.[/dim]")
        except Exception as exc:
            console.print(f"  [dim]Error clasificando query: {exc}[/dim]")

    # Retrieve with category-aware filtering
    results = retriever.retrieve_with_sources(
        query, query_classification=classification, k=5,
    )

    if not results:
        console.print("[dim]Sin resultados.[/dim]")
        return

    for i, entry in enumerate(results, 1):
        preview = entry["content"]
        if len(preview) > 300:
            preview = preview[:300] + "..."
        distance = entry["distance"]
        overlap = entry.get("category_overlap")
        dist_info = f"dist={distance:.4f}"
        if overlap is not None:
            dist_info += f"  cat_overlap={overlap}"
        console.print(f"\n  [bold]#{i}[/bold]  [dim]{dist_info}[/dim]")
        console.print(f"  [bold]Documento:[/bold] [cyan]{entry['path']}[/cyan]  [dim]({entry['topic']})[/dim]")
        console.print(f"  {preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo E2E del pipeline RAG: ingesta + clasificacion + busqueda",
    )
    parser.add_argument(
        "directory",
        help="Directorio con documentos (PDF, DOCX, TXT, etc.)",
    )
    parser.add_argument(
        "--db", default=None,
        help="Ruta a la base de datos SQLite (default: temporal)",
    )
    parser.add_argument(
        "--query", "-q", default=None,
        help="Query de busqueda para probar tras la ingesta",
    )
    parser.add_argument(
        "--keep-db", action="store_true",
        help="No borrar la base de datos temporal al salir",
    )
    args = parser.parse_args()

    # Configure logging so RagIngest progress is visible
    logging.basicConfig(
        level=logging.INFO,
        format="  %(message)s",
    )

    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        console.print(f"[red]No es un directorio: {directory}[/red]")
        raise SystemExit(1)

    # Database path
    if args.db:
        db_path = Path(args.db).resolve()
        use_temp = False
    else:
        tmp_dir = tempfile.mkdtemp(prefix="rag_demo_")
        db_path = Path(tmp_dir) / "demo.db"
        use_temp = True

    categories_path = find_categories_file()
    if categories_path.exists():
        console.print(f"[green]Categorias:[/green] {categories_path}")
    else:
        console.print(
            "[yellow]iptc_categories.json no encontrado, "
            "se omite clasificacion.[/yellow]"
        )

    console.print(f"[green]Directorio:[/green] {directory}")
    console.print(f"[green]Base datos:[/green]  {db_path}")

    # Build config and run ingestion via RagIngest
    config = DemoConfig(directory, db_path, categories_path)

    console.print("\n[bold]Procesando documentos...[/bold]\n")
    t_start = time.time()
    ingest = RagIngest(config)
    ingested = ingest.ingest()
    elapsed = time.time() - t_start

    # Read results from DB and display
    summary = read_results_from_db(db_path)
    show_summary(summary, ingested, elapsed)
    show_documents_table(summary)
    show_categories_table(summary)

    # Prepare retriever and classificator for queries
    retriever = RagRetriever(config)
    classificator = None
    if categories_path.exists():
        classificator = Classificator(categories_path)

    # Query
    if args.query:
        run_query(config, args.query, retriever, classificator)

    if not args.query:
        console.print()
        console.print("[bold]Modo busqueda[/bold] (vacio para salir)\n")
        while True:
            query = console.input("[cyan]Query:[/cyan] ")
            if not query.strip():
                break
            run_query(config, query, retriever, classificator)

    # Cleanup
    if use_temp and not args.keep_db:
        shutil.rmtree(db_path.parent, ignore_errors=True)
        console.print(f"\n[dim]Base de datos temporal eliminada.[/dim]")
    elif use_temp and args.keep_db:
        console.print(f"\n[dim]Base de datos conservada en: {db_path}[/dim]")

    console.print("[dim]Fin.[/dim]")


if __name__ == "__main__":
    main()
