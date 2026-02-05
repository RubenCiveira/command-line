"""
Evaluador interactivo del ProjectResolver.

Simula un conjunto de proyectos en distintas categorias (programacion,
marketing, datos, documentacion, infraestructura) y prueba la resolucion
de proyectos a partir de mensajes de usuario.

Uso:
    python examples/project_resolver_eval.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from ai.project_resolver import ProjectResolver
from ai.model_pool import ModelPool
from ai.user.user_config import UserConfig
from ai.user.project_topic import ProjectTopic
from ai.ui.cli_user_interface import CliUserInterface

console = Console()

CLASS_TECH = [
    "science and technology",
    "technology and engineering",
    "information technology and computer science",
]
CLASS_MARKETING = [
    "economy, business and finance",
    "business information",
    "business strategy and marketing",
]
CLASS_MED_RESEARCH = [
    "science and technology",
    "scientific research",
    "medical research",
]
CLASS_PHYSICS = [
    "science and technology",
    "natural science",
    "physics",
]
CLASS_COSMOLOGY = [
    "science and technology",
    "natural science",
    "cosmology",
]
CLASS_CANCER = [
    "health",
    "disease and condition",
    "cancer",
]

# ── Modelos zero-shot disponibles ─────────────────────────────────

MODELS = [
    (
        "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        "ModernBERT large, multilingue y reciente (recomendado)",
    ),
    (
        "facebook/bart-large-mnli",
        "BART large, estandar de facto para zero-shot (solo ingles)",
    ),
    (
        "DAMO-NLP-SG/zero-shot-classify-SSTuning-base",
        "RoBERTa base, ligero (2-20 labels)",
    ),
]

# ── Conjuntos de proyectos simulados ─────────────────────────────

PROJECT_SETS = {
    "Programacion": [
        ProjectTopic(name="backend", path="/home/user/work/backend",
                     description="REST API built with FastAPI and PostgreSQL",
                     classification=CLASS_TECH),
        ProjectTopic(name="frontend", path="/home/user/work/frontend",
                     description="React SPA with TypeScript and Tailwind",
                     classification=CLASS_TECH),
        ProjectTopic(name="mobile", path="/home/user/work/mobile",
                     description="React Native app for iOS and Android",
                     classification=CLASS_TECH),
        ProjectTopic(name="shell", path="/home/user/work/shell",
                     description="CLI utilities and shell scripts",
                     classification=CLASS_TECH),
        ProjectTopic(name="sdk", path="/home/user/work/sdk",
                     description="Python SDK for the public API",
                     classification=CLASS_TECH),
    ],
    "Marketing": [
        ProjectTopic(name="campaigns", path="/home/user/marketing/campaigns",
                     description="Marketing campaigns, funnels and analytics",
                     classification=CLASS_MARKETING),
        ProjectTopic(name="brand", path="/home/user/marketing/brand",
                     description="Brand guidelines, logos and design assets",
                     classification=CLASS_MARKETING),
        ProjectTopic(name="social", path="/home/user/marketing/social",
                     description="Social media content and scheduling",
                     classification=CLASS_MARKETING),
        ProjectTopic(name="website", path="/home/user/marketing/website",
                     description="Corporate website with WordPress",
                     classification=CLASS_MARKETING),
        ProjectTopic(name="newsletter", path="/home/user/marketing/newsletter",
                     description="Email newsletter with Mailchimp",
                     classification=CLASS_MARKETING),
    ],
    "Investigacion medica": [
        ProjectTopic(name="oncologia", path="/home/user/research/oncologia",
                     description="Modelos predictivos de respuesta a inmunoterapia",
                     classification=CLASS_CANCER),
        ProjectTopic(name="neuroimagen", path="/home/user/research/neuroimagen",
                     description="Segmentacion de resonancias magneticas cerebrales con deep learning",
                     classification=CLASS_MED_RESEARCH),
        ProjectTopic(name="genomica", path="/home/user/research/genomica",
                     description="Analisis de variantes geneticas y GWAS en cohortes poblacionales",
                     classification=CLASS_MED_RESEARCH),
        ProjectTopic(name="farma", path="/home/user/research/farma",
                     description="Ensayos clinicos fase III y analisis farmacocinetico",
                     classification=CLASS_MED_RESEARCH),
        ProjectTopic(name="epidemio", path="/home/user/research/epidemio",
                     description="Modelado epidemiologico y vigilancia de brotes",
                     classification=CLASS_MED_RESEARCH),
    ],
    "Fisica teorica": [
        ProjectTopic(name="cuerdas", path="/home/user/research/cuerdas",
                     description="Compactificaciones de teoria de cuerdas y paisaje de vacios",
                     classification=CLASS_PHYSICS),
        ProjectTopic(name="qcd-lattice", path="/home/user/research/qcd-lattice",
                     description="Simulaciones de QCD en red y espectro hadronico",
                     classification=CLASS_PHYSICS),
        ProjectTopic(name="cosmologia", path="/home/user/research/cosmologia",
                     description="Energia oscura, inflacion y perturbaciones primordiales",
                     classification=CLASS_COSMOLOGY),
        ProjectTopic(name="cuantica", path="/home/user/research/cuantica",
                     description="Informacion cuantica, entrelazamiento y codigos de correccion",
                     classification=CLASS_PHYSICS),
        ProjectTopic(name="gravedad", path="/home/user/research/gravedad",
                     description="Gravedad cuantica de lazos y termodinamica de agujeros negros",
                     classification=CLASS_PHYSICS),
    ],
    "Mixto (todos)": [
        # Programacion
        ProjectTopic(name="backend", path="/home/user/work/backend",
                     description="REST API built with FastAPI and PostgreSQL",
                     classification=CLASS_TECH),
        ProjectTopic(name="frontend", path="/home/user/work/frontend",
                     description="React SPA with TypeScript and Tailwind",
                     classification=CLASS_TECH),
        ProjectTopic(name="mobile", path="/home/user/work/mobile",
                     description="React Native app for iOS and Android",
                     classification=CLASS_TECH),
        ProjectTopic(name="shell", path="/home/user/work/shell",
                     description="CLI utilities and shell scripts",
                     classification=CLASS_TECH),
        # Marketing
        ProjectTopic(name="campaigns", path="/home/user/marketing/campaigns",
                     description="Marketing campaigns, funnels and analytics",
                     classification=CLASS_MARKETING),
        ProjectTopic(name="brand", path="/home/user/marketing/brand",
                     description="Brand guidelines, logos and design assets",
                     classification=CLASS_MARKETING),
        ProjectTopic(name="social", path="/home/user/marketing/social",
                     description="Social media content and scheduling",
                     classification=CLASS_MARKETING),
        # Datos e infra
        ProjectTopic(name="data", path="/home/user/work/data",
                     description="Data pipelines and analytics with dbt",
                     classification=CLASS_TECH),
        ProjectTopic(name="infra", path="/home/user/work/infra",
                     description="Infrastructure as code with Terraform and K8s",
                     classification=CLASS_TECH),
        ProjectTopic(name="docs", path="/home/user/work/docs",
                     description="Technical documentation with MkDocs",
                     classification=CLASS_TECH),
        # Investigacion medica
        ProjectTopic(name="oncologia", path="/home/user/research/oncologia",
                     description="Modelos predictivos de respuesta a inmunoterapia",
                     classification=CLASS_CANCER),
        ProjectTopic(name="genomica", path="/home/user/research/genomica",
                     description="Analisis de variantes geneticas y GWAS en cohortes poblacionales",
                     classification=CLASS_MED_RESEARCH),
        ProjectTopic(name="epidemio", path="/home/user/research/epidemio",
                     description="Modelado epidemiologico y vigilancia de brotes",
                     classification=CLASS_MED_RESEARCH),
        # Fisica teorica
        ProjectTopic(name="cosmologia", path="/home/user/research/cosmologia",
                     description="Energia oscura, inflacion y perturbaciones primordiales",
                     classification=CLASS_COSMOLOGY),
        ProjectTopic(name="cuantica", path="/home/user/research/cuantica",
                     description="Informacion cuantica, entrelazamiento y codigos de correccion",
                     classification=CLASS_PHYSICS),
    ],
}

# ── Queries de ejemplo ────────────────────────────────────────────

EXAMPLE_QUERIES = [
    # Referencia directa por nombre
    ("haz commit de los ultimos cambios de shell", "shell"),
    ("despliega el backend en staging", "backend"),
    ("abre el proyecto frontend", "frontend"),
    # Referencia indirecta / semantica
    ("actualiza la landing page", "frontend / website"),
    ("revisa las metricas de la ultima campana", "campaigns"),
    ("publica el post de Instagram de manana", "social"),
    ("necesito cambiar el logo corporativo", "brand"),
    ("configura los pods de Kubernetes", "infra"),
    ("actualiza la documentacion del API", "docs"),
    ("anade un endpoint para crear usuarios", "backend"),
    ("corrige el bug del login en la app movil", "mobile"),
    ("envia la newsletter de esta semana", "newsletter"),
    ("ejecuta el pipeline de transformacion de datos", "data"),
    ("genera el paquete del SDK para PyPI", "sdk"),
    # Investigacion medica
    ("analiza los resultados del ultimo ensayo clinico", "farma"),
    ("revisa la segmentacion de las resonancias", "neuroimagen"),
    ("lanza el pipeline de GWAS con la nueva cohorte", "genomica"),
    ("actualiza el modelo SIR con los datos de esta semana", "epidemio"),
    ("compara la supervivencia entre los brazos de inmunoterapia", "oncologia"),
    # Fisica teorica
    ("recalcula el espectro de masas en la red", "qcd-lattice"),
    ("revisa las perturbaciones primordiales del modelo inflacionario", "cosmologia"),
    ("verifica la cota de entrelazamiento del codigo de superficie", "cuantica"),
    ("explora las compactificaciones de Calabi-Yau a 6 dimensiones", "cuerdas"),
    ("calcula la entropia de Bekenstein-Hawking para el caso rotante", "gravedad"),
    # Ambiguas
    ("haz push de los ultimos cambios", "ambiguo"),
    ("revisa los tests", "ambiguo"),
    # Con typos
    ("actualiza el bakend", "backend (fuzzy)"),
    ("revisa el frotend", "frontend (fuzzy)"),
]


# ── Logica de evaluacion ──────────────────────────────────────────


def _parse_expected(raw: str) -> tuple[set[str], bool]:
    """Parse the expected value into (acceptable_names, is_inherently_ambiguous).

    Examples::

        "shell"              -> ({"shell"}, False)
        "frontend / website" -> ({"frontend", "website"}, False)
        "backend (fuzzy)"    -> ({"backend"}, False)
        "ambiguo"            -> (set(), True)
    """
    if raw == "ambiguo":
        return set(), True
    clean = re.sub(r"\s*\([^)]*\)", "", raw)
    names = {n.strip() for n in clean.split("/")}
    return names, False


def _effective_expected(
    expected_names: set[str],
    is_ambiguous: bool,
    active_names: set[str],
) -> tuple[set[str], bool]:
    """Compute the effective expected given the active project set.

    Returns (acceptable_names, should_be_ambiguous).

    If none of the expected projects exist in the active set the query
    becomes effectively ambiguous: the resolver cannot know the right
    answer.
    """
    if is_ambiguous:
        return set(), True
    available = expected_names & active_names
    if available:
        return available, False
    return set(), True


def show_resolve_result(
    query: str,
    conclusion,
    raw_expected: str,
    active_names: set[str],
) -> bool:
    """Display and evaluate the result of resolving a single query.

    Returns ``True`` when the resolver's answer is correct.
    """
    project = conclusion.proposal or ""
    has_doubts = conclusion.doubts is not None

    expected_names, is_ambiguous = _parse_expected(raw_expected)
    effective_names, should_be_ambiguous = _effective_expected(
        expected_names, is_ambiguous, active_names,
    )

    if should_be_ambiguous:
        match_ok = has_doubts
        effective_label = "ambiguo"
    else:
        match_ok = project in effective_names
        effective_label = " / ".join(sorted(effective_names))

    status = "[yellow]ambiguo[/yellow]" if has_doubts else "[green]seguro[/green]"
    icon = "[green]OK[/green]" if match_ok else "[red]MISS[/red]"

    if effective_label != raw_expected:
        expected_display = f"{effective_label} [dim](orig: {raw_expected})[/dim]"
    else:
        expected_display = effective_label

    console.print(
        f"  {icon}  {status}  "
        f"[bold]{project or '(ninguno)':12s}[/bold]  "
        f"[dim](esperado: {expected_display})[/dim]  "
        f"[cyan]{query}[/cyan]"
    )

    if has_doubts:
        options = conclusion.doubts["properties"]["project"]["oneOf"]
        for opt in options[:3]:
            console.print(f"          [dim]{opt['title']}[/dim]")

    return match_ok


def show_projects_table(projects: list[ProjectTopic]):
    """Display the active project set."""
    table = Table(show_lines=False, title="Proyectos activos")
    table.add_column("Nombre", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Descripcion")

    for p in projects:
        table.add_row(p.name, p.path, p.description)

    console.print(table)


def show_pool_status(pool: ModelPool):
    cached = pool.cached_keys
    if not cached:
        console.print("  [dim]Pool vacio[/dim]\n")
        return
    total_mib = pool.total_memory_bytes / 1024**2
    max_mib = pool.max_memory_bytes / 1024**2
    console.print(
        f"  Pool: [cyan]{len(cached)}[/cyan] modelos, "
        f"[cyan]{total_mib:.0f}[/cyan] / [cyan]{max_mib:.0f}[/cyan] MiB"
    )
    for task, model in cached:
        console.print(f"    [dim]{task}:[/dim] {model}")
    console.print()


# ── Config override para el eval ──────────────────────────────────

class EvalConfig(UserConfig):
    """UserConfig override that injects a custom project list."""

    def __init__(self, projects: list[ProjectTopic]):
        self._projects = projects

    def projectTopics(self) -> list[ProjectTopic]:
        return self._projects


# ── Acciones principales ──────────────────────────────────────────

def load_resolver(
    projects: list[ProjectTopic],
    model_name: str,
    pool: ModelPool,
) -> ProjectResolver | None:
    console.print(f"\n[yellow]Cargando modelo:[/yellow] {model_name}")
    console.print("[dim]Esto puede tardar la primera vez (descarga de pesos)...[/dim]")
    try:
        config = EvalConfig(projects)
        resolver = ProjectResolver(
            config,
            memory=None,
            model_pool=pool,
            ui=CliUserInterface(console),
        )
        # Force pipeline load with a warmup query
        resolver.resolve("test")
        console.print("[green]Modelo cargado correctamente.[/green]\n")
        return resolver
    except Exception as e:
        console.print(f"[red]Error cargando modelo: {e}[/red]\n")
        return None


def run_examples(resolver: ProjectResolver, active_projects: list[ProjectTopic]):
    active_names = {p.name for p in active_projects}
    console.print("\n[bold]Ejecutando queries de ejemplo...[/bold]\n")

    ok = 0
    total = len(EXAMPLE_QUERIES)
    for query, expected in EXAMPLE_QUERIES:
        def _handle_conclusion(conclusion):
            nonlocal ok
            if show_resolve_result(query, conclusion, expected, active_names):
                ok += 1

        def _handle_conclusion_with_notice(conclusion):
            _handle_conclusion(conclusion)

        if resolver._ui is not None:
            original_request_form = resolver._ui.request_form

            def _wrap_request_form(schema, on_complete):
                console.print(
                    f"  [dim]Formulario para:[/dim] [cyan]{query}[/cyan]"
                )
                original_request_form(schema, on_complete)

            resolver._ui.request_form = _wrap_request_form
            try:
                resolver.resolve_with_ui(query, _handle_conclusion_with_notice)
            finally:
                resolver._ui.request_form = original_request_form
        else:
            resolver.resolve_with_ui(query, _handle_conclusion_with_notice)

    miss = total - ok
    console.print()
    console.print(
        f"  [bold]Resultado:[/bold] {ok}/{total} "
        f"([green]{ok} OK[/green], [red]{miss} MISS[/red])"
    )
    console.print()


def interactive_mode(resolver: ProjectResolver):
    console.print(
        "\n[bold]Modo interactivo[/bold] "
        "-- escribe un mensaje para resolver el proyecto (vacio para volver)\n"
    )
    while True:
        query = Prompt.ask("[cyan]Mensaje[/cyan]")
        if not query.strip():
            break
        def _handle_conclusion(conclusion):
            project = conclusion.proposal or "(ninguno)"
            has_doubts = conclusion.doubts is not None

            if has_doubts:
                console.print(
                    f"  [yellow]Ambiguo[/yellow] -> [bold]{project}[/bold]"
                )
                options = conclusion.doubts["properties"]["project"]["oneOf"]
                for opt in options:
                    console.print(f"    [dim]{opt['title']}[/dim]")
            else:
                console.print(f"  [green]Seguro[/green]  -> [bold]{project}[/bold]")
            console.print()

        if resolver._ui is not None:
            original_request_form = resolver._ui.request_form

            def _wrap_request_form(schema, on_complete):
                console.print(
                    f"  [dim]Formulario para:[/dim] [cyan]{query}[/cyan]"
                )
                original_request_form(schema, on_complete)

            resolver._ui.request_form = _wrap_request_form
            try:
                resolver.resolve_with_ui(query, _handle_conclusion)
            finally:
                resolver._ui.request_form = original_request_form
        else:
            resolver.resolve_with_ui(query, _handle_conclusion)


# ── Main ──────────────────────────────────────────────────────────

def main():
    pool = ModelPool()
    resolver = None
    current_model = None
    current_set_name = None
    current_projects: list[ProjectTopic] = []

    while True:
        console.print("\n[bold]═══ Evaluador de Project Resolver ═══[/bold]\n")
        if current_model:
            console.print(f"  Modelo activo:   [green]{current_model}[/green]")
        if current_set_name:
            console.print(
                f"  Set de proyectos: [cyan]{current_set_name}[/cyan] "
                f"({len(current_projects)} proyectos)"
            )
        show_pool_status(pool)

        console.print("  [1] Seleccionar modelo")
        console.print("  [2] Seleccionar set de proyectos")
        console.print("  [3] Ver proyectos activos")
        console.print("  [4] Ejecutar queries de ejemplo")
        console.print("  [5] Modo interactivo")
        console.print("  [6] Salir")
        console.print()

        choice = Prompt.ask(
            "Opcion", choices=["1", "2", "3", "4", "5", "6"], default="1",
        )

        if choice == "1":
            console.print("\n[bold]Modelos disponibles:[/bold]\n")
            for i, (name, desc) in enumerate(MODELS, 1):
                console.print(f"  [{i}] {name}")
                console.print(f"      [dim]{desc}[/dim]")
            console.print()

            idx = IntPrompt.ask("Selecciona modelo", default=1)
            if 1 <= idx <= len(MODELS):
                model_name = MODELS[idx - 1][0]
                if not current_projects:
                    # Auto-select the mixed set
                    current_set_name = "Mixto (todos)"
                    current_projects = PROJECT_SETS[current_set_name]
                    console.print(
                        f"  [dim]Auto-seleccionado set: {current_set_name}[/dim]"
                    )
                resolver = load_resolver(current_projects, model_name, pool)
                if resolver:
                    current_model = model_name
            else:
                console.print("[red]Opcion no valida.[/red]")

        elif choice == "2":
            console.print("\n[bold]Sets de proyectos disponibles:[/bold]\n")
            set_names = list(PROJECT_SETS.keys())
            for i, name in enumerate(set_names, 1):
                count = len(PROJECT_SETS[name])
                console.print(f"  [{i}] {name} ({count} proyectos)")
            console.print()

            idx = IntPrompt.ask("Selecciona set", default=1)
            if 1 <= idx <= len(set_names):
                current_set_name = set_names[idx - 1]
                current_projects = PROJECT_SETS[current_set_name]
                console.print(
                    f"  [green]Set activo: {current_set_name}[/green]"
                )
                # Reload resolver with new projects if model is selected
                if current_model:
                    resolver = load_resolver(
                        current_projects, current_model, pool,
                    )
            else:
                console.print("[red]Opcion no valida.[/red]")

        elif choice == "3":
            if not current_projects:
                console.print("[dim]No hay proyectos seleccionados.[/dim]")
            else:
                console.print()
                show_projects_table(current_projects)

        elif choice == "4":
            if resolver is None:
                console.print(
                    "[red]Primero selecciona un modelo (opcion 1).[/red]"
                )
            else:
                run_examples(resolver, current_projects)

        elif choice == "5":
            if resolver is None:
                console.print(
                    "[red]Primero selecciona un modelo (opcion 1).[/red]"
                )
            else:
                interactive_mode(resolver)

        elif choice == "6":
            console.print("[dim]Saliendo...[/dim]")
            break


if __name__ == "__main__":
    main()
