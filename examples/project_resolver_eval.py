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
import tempfile
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
from ai.rag.project_detail_indexer import ProjectDetailIndexer

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


def _detail(lines: list[str]) -> str:
    return "\n".join(lines)


DETAILS = {
    "backend": _detail([
        "overview: api para operaciones core del negocio",
        "stack: fastapi, python, postgres",
        "domain: usuarios, permisos, pagos",
        "services: auth, billing, metrics",
        "data: migraciones y seeds",
        "infra: docker compose local",
        "tests: pytest y coverage",
        "docs: openapi y postman",
        "ops: despliegue a staging",
        "notes: foco en seguridad",
    ]),
    "frontend": _detail([
        "overview: web app principal",
        "stack: react, typescript",
        "ui: tailwind y componentes",
        "routing: react router",
        "state: context y hooks",
        "api: integracion con backend",
        "tests: unit y e2e",
        "build: vite",
        "deploy: static hosting",
        "notes: optimizacion de bundle",
    ]),
    "mobile": _detail([
        "overview: app movil ios y android",
        "stack: react native",
        "features: login, perfil, notificaciones",
        "api: consume endpoints REST",
        "storage: secure storage",
        "build: fastlane",
        "tests: detox y unit",
        "release: app store y play",
        "ux: flows simplificados",
        "notes: soporte offline",
    ]),
    "shell": _detail([
        "overview: utilidades cli internas",
        "stack: bash y python",
        "commands: deploy, logs, cleanup",
        "targets: dev y staging",
        "config: env por proyecto",
        "docs: help en cada comando",
        "tests: scripts de smoke",
        "release: versionado semantico",
        "ops: soporte para equipos",
        "notes: seguridad en sudo",
    ]),
    "sdk": _detail([
        "overview: sdk para clientes",
        "stack: python",
        "features: auth, clientes, recursos",
        "api: wrapper REST",
        "docs: ejemplos y quickstart",
        "tests: unit y integration",
        "build: publish a pypi",
        "versioning: semver",
        "compat: python 3.10+",
        "notes: errores tipados",
    ]),
    "campaigns": _detail([
        "overview: campanas de marketing",
        "stack: dashboards y reports",
        "funnels: conversion y leads",
        "ads: google y meta",
        "tracking: utm y eventos",
        "kpi: cpa y roi",
        "tests: validacion de datos",
        "docs: playbooks",
        "ops: calendario editorial",
        "notes: segmentacion",
    ]),
    "brand": _detail([
        "overview: identidad de marca",
        "assets: logo, colores, tipografia",
        "guidelines: usos y restricciones",
        "templates: presentaciones",
        "design: iconografia",
        "files: svg y png",
        "docs: manual de marca",
        "ops: aprobaciones",
        "notes: consistencia visual",
        "updates: rebrand anual",
    ]),
    "social": _detail([
        "overview: contenido redes sociales",
        "channels: instagram, twitter",
        "calendar: planificacion semanal",
        "assets: reels y posts",
        "copy: mensajes y hashtags",
        "metrics: alcance y engagement",
        "ops: approvals",
        "tools: scheduling",
        "notes: respuesta a comentarios",
        "experiments: a/b",
    ]),
    "website": _detail([
        "overview: sitio corporativo",
        "stack: wordpress",
        "pages: landing y pricing",
        "seo: keywords y metadata",
        "forms: leads",
        "assets: imagenes optimizadas",
        "content: blog",
        "ops: backups",
        "notes: performance",
        "analytics: GA",
    ]),
    "newsletter": _detail([
        "overview: boletin semanal",
        "platform: mailchimp",
        "segments: usuarios activos",
        "content: resumen de novedades",
        "templates: branding",
        "metrics: open y click rate",
        "ops: listas y bajas",
        "compliance: gdpr",
        "notes: entregabilidad",
        "automation: triggers",
    ]),
    "oncologia": _detail([
        "overview: investigacion en oncologia",
        "focus: inmunoterapia",
        "data: cohortes clinicas",
        "models: prediccion de respuesta",
        "pipeline: preprocessing",
        "metrics: supervivencia",
        "tools: python y r",
        "docs: informes",
        "collab: hospital",
        "notes: privacidad",
    ]),
    "neuroimagen": _detail([
        "overview: analisis de neuroimagen",
        "data: mri y ct",
        "task: segmentacion",
        "models: unet",
        "pipeline: normalizacion",
        "metrics: dice",
        "tools: python",
        "docs: protocolos",
        "ops: gpu",
        "notes: calidad de datos",
    ]),
    "genomica": _detail([
        "overview: analisis genomico",
        "data: variantes y gwas",
        "pipeline: qc",
        "tools: plink",
        "models: asociacion",
        "metrics: pvalues",
        "docs: reportes",
        "ops: storage",
        "notes: reproducibilidad",
        "collab: bioinfo",
    ]),
    "farma": _detail([
        "overview: ensayos clinicos",
        "phase: fase III",
        "data: farmacocinetica",
        "analysis: cohortes",
        "metrics: eficacia",
        "tools: sas",
        "docs: protocolos",
        "ops: auditoria",
        "notes: regulatorio",
        "reporting: clinico",
    ]),
    "epidemio": _detail([
        "overview: epidemiologia",
        "models: sir",
        "data: vigilancia",
        "analysis: brotes",
        "tools: python",
        "metrics: r0",
        "docs: informes",
        "ops: dashboards",
        "notes: alertas",
        "collab: salud publica",
    ]),
    "cuerdas": _detail([
        "overview: teoria de cuerdas",
        "focus: compactificaciones",
        "math: calabi yau",
        "tools: mathematica",
        "analysis: espectro",
        "notes: landscape",
        "docs: papers",
        "collab: grupo teorico",
        "ops: simulaciones",
        "targets: vacios",
    ]),
    "qcd-lattice": _detail([
        "overview: qcd en red",
        "sim: lattice",
        "analysis: espectro hadronico",
        "tools: c++",
        "infra: cluster",
        "metrics: observables",
        "docs: notas",
        "ops: colas",
        "notes: precision",
        "targets: masa",
    ]),
    "cosmologia": _detail([
        "overview: cosmologia",
        "focus: inflacion",
        "data: perturbaciones",
        "models: lcdm",
        "tools: python",
        "metrics: espectro",
        "docs: articulos",
        "ops: simulaciones",
        "notes: energia oscura",
        "targets: parametros",
    ]),
    "cuantica": _detail([
        "overview: informacion cuantica",
        "focus: entrelazamiento",
        "codes: correccion",
        "tools: python",
        "analysis: cota",
        "metrics: fidelidad",
        "docs: notebooks",
        "ops: simulaciones",
        "notes: ruido",
        "targets: superficies",
    ]),
    "gravedad": _detail([
        "overview: gravedad cuantica",
        "focus: lazos",
        "analysis: agujeros negros",
        "math: entropia",
        "tools: python",
        "docs: papers",
        "ops: calculos",
        "notes: termodinamica",
        "targets: rotacion",
        "collab: grupo teorico",
    ]),
    "data": _detail([
        "overview: pipelines de datos",
        "tools: dbt",
        "sources: raw y staged",
        "models: marts",
        "quality: tests",
        "ops: orquestacion",
        "docs: lineage",
        "metrics: freshness",
        "notes: performance",
        "targets: dashboards",
    ]),
    "infra": _detail([
        "overview: infraestructura",
        "tools: terraform",
        "cluster: kubernetes",
        "ops: autoscaling",
        "security: iam",
        "monitoring: prometheus",
        "docs: runbooks",
        "notes: costes",
        "deploy: ci",
        "targets: reliability",
    ]),
    "docs": _detail([
        "overview: documentacion tecnica",
        "tools: mkdocs",
        "content: guias",
        "api: referencias",
        "ops: builds",
        "reviews: semanales",
        "notes: versionado",
        "format: markdown",
        "targets: onboarding",
        "extras: diagramas",
    ]),
}


def _project(name: str, path: str, description: str, classification: list[str]) -> ProjectTopic:
    return ProjectTopic(
        name=name,
        path=path,
        description=description,
        classification=classification,
        detail=DETAILS.get(name, ""),
    )

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
        _project(
            "backend",
            "/home/user/work/backend",
            "REST API built with FastAPI and PostgreSQL",
            CLASS_TECH,
        ),
        _project(
            "frontend",
            "/home/user/work/frontend",
            "React SPA with TypeScript and Tailwind",
            CLASS_TECH,
        ),
        _project(
            "mobile",
            "/home/user/work/mobile",
            "React Native app for iOS and Android",
            CLASS_TECH,
        ),
        _project(
            "shell",
            "/home/user/work/shell",
            "CLI utilities and shell scripts",
            CLASS_TECH,
        ),
        _project(
            "sdk",
            "/home/user/work/sdk",
            "Python SDK for the public API",
            CLASS_TECH,
        ),
    ],
    "Marketing": [
        _project(
            "campaigns",
            "/home/user/marketing/campaigns",
            "Marketing campaigns, funnels and analytics",
            CLASS_MARKETING,
        ),
        _project(
            "brand",
            "/home/user/marketing/brand",
            "Brand guidelines, logos and design assets",
            CLASS_MARKETING,
        ),
        _project(
            "social",
            "/home/user/marketing/social",
            "Social media content and scheduling",
            CLASS_MARKETING,
        ),
        _project(
            "website",
            "/home/user/marketing/website",
            "Corporate website with WordPress",
            CLASS_MARKETING,
        ),
        _project(
            "newsletter",
            "/home/user/marketing/newsletter",
            "Email newsletter with Mailchimp",
            CLASS_MARKETING,
        ),
    ],
    "Investigacion medica": [
        _project(
            "oncologia",
            "/home/user/research/oncologia",
            "Modelos predictivos de respuesta a inmunoterapia",
            CLASS_CANCER,
        ),
        _project(
            "neuroimagen",
            "/home/user/research/neuroimagen",
            "Segmentacion de resonancias magneticas cerebrales con deep learning",
            CLASS_MED_RESEARCH,
        ),
        _project(
            "genomica",
            "/home/user/research/genomica",
            "Analisis de variantes geneticas y GWAS en cohortes poblacionales",
            CLASS_MED_RESEARCH,
        ),
        _project(
            "farma",
            "/home/user/research/farma",
            "Ensayos clinicos fase III y analisis farmacocinetico",
            CLASS_MED_RESEARCH,
        ),
        _project(
            "epidemio",
            "/home/user/research/epidemio",
            "Modelado epidemiologico y vigilancia de brotes",
            CLASS_MED_RESEARCH,
        ),
    ],
    "Fisica teorica": [
        _project(
            "cuerdas",
            "/home/user/research/cuerdas",
            "Compactificaciones de teoria de cuerdas y paisaje de vacios",
            CLASS_PHYSICS,
        ),
        _project(
            "qcd-lattice",
            "/home/user/research/qcd-lattice",
            "Simulaciones de QCD en red y espectro hadronico",
            CLASS_PHYSICS,
        ),
        _project(
            "cosmologia",
            "/home/user/research/cosmologia",
            "Energia oscura, inflacion y perturbaciones primordiales",
            CLASS_COSMOLOGY,
        ),
        _project(
            "cuantica",
            "/home/user/research/cuantica",
            "Informacion cuantica, entrelazamiento y codigos de correccion",
            CLASS_PHYSICS,
        ),
        _project(
            "gravedad",
            "/home/user/research/gravedad",
            "Gravedad cuantica de lazos y termodinamica de agujeros negros",
            CLASS_PHYSICS,
        ),
    ],
    "Mixto (todos)": [
        # Programacion
        _project(
            "backend",
            "/home/user/work/backend",
            "REST API built with FastAPI and PostgreSQL",
            CLASS_TECH,
        ),
        _project(
            "frontend",
            "/home/user/work/frontend",
            "React SPA with TypeScript and Tailwind",
            CLASS_TECH,
        ),
        _project(
            "mobile",
            "/home/user/work/mobile",
            "React Native app for iOS and Android",
            CLASS_TECH,
        ),
        _project(
            "shell",
            "/home/user/work/shell",
            "CLI utilities and shell scripts",
            CLASS_TECH,
        ),
        # Marketing
        _project(
            "campaigns",
            "/home/user/marketing/campaigns",
            "Marketing campaigns, funnels and analytics",
            CLASS_MARKETING,
        ),
        _project(
            "brand",
            "/home/user/marketing/brand",
            "Brand guidelines, logos and design assets",
            CLASS_MARKETING,
        ),
        _project(
            "social",
            "/home/user/marketing/social",
            "Social media content and scheduling",
            CLASS_MARKETING,
        ),
        # Datos e infra
        _project(
            "data",
            "/home/user/work/data",
            "Data pipelines and analytics with dbt",
            CLASS_TECH,
        ),
        _project(
            "infra",
            "/home/user/work/infra",
            "Infrastructure as code with Terraform and K8s",
            CLASS_TECH,
        ),
        _project(
            "docs",
            "/home/user/work/docs",
            "Technical documentation with MkDocs",
            CLASS_TECH,
        ),
        # Investigacion medica
        _project(
            "oncologia",
            "/home/user/research/oncologia",
            "Modelos predictivos de respuesta a inmunoterapia",
            CLASS_CANCER,
        ),
        _project(
            "genomica",
            "/home/user/research/genomica",
            "Analisis de variantes geneticas y GWAS en cohortes poblacionales",
            CLASS_MED_RESEARCH,
        ),
        _project(
            "epidemio",
            "/home/user/research/epidemio",
            "Modelado epidemiologico y vigilancia de brotes",
            CLASS_MED_RESEARCH,
        ),
        # Fisica teorica
        _project(
            "cosmologia",
            "/home/user/research/cosmologia",
            "Energia oscura, inflacion y perturbaciones primordiales",
            CLASS_COSMOLOGY,
        ),
        _project(
            "cuantica",
            "/home/user/research/cuantica",
            "Informacion cuantica, entrelazamiento y codigos de correccion",
            CLASS_PHYSICS,
        ),
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

    def __init__(self, projects: list[ProjectTopic], db_path: Path):
        self._projects = projects
        self._db_path = db_path

    def projectTopics(self) -> list[ProjectTopic]:
        return self._projects

    def projectDatabasePath(self) -> Path:
        return self._db_path


# ── Acciones principales ──────────────────────────────────────────

def load_resolver(
    projects: list[ProjectTopic],
    model_name: str,
    pool: ModelPool,
    db_path: Path,
) -> ProjectResolver | None:
    console.print(f"\n[yellow]Cargando modelo:[/yellow] {model_name}")
    console.print("[dim]Esto puede tardar la primera vez (descarga de pesos)...[/dim]")
    try:
        config = EvalConfig(projects, db_path)
        resolver = ProjectResolver(
            config,
            memory=None,
            model_pool=pool,
            ui=CliUserInterface(console),
        )
        indexer = ProjectDetailIndexer(config.projectDatabasePath())
        indexed = indexer.index_projects(projects)
        if indexed:
            console.print(
                f"[dim]Detail embeddings indexados: {indexed}[/dim]"
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
    temp_dir = tempfile.TemporaryDirectory(prefix="project-resolver-")
    temp_db_path = Path(temp_dir.name) / "project_details.db"
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
                resolver = load_resolver(
                    current_projects, model_name, pool, temp_db_path,
                )
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
                        current_projects, current_model, pool, temp_db_path,
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
            temp_dir.cleanup()
            break


if __name__ == "__main__":
    main()
