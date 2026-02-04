# Plan: Agente Multi-Capacidades para Command-Line

## Resumen

Transformar el CLI actual (que solo reacciona a comandos fallidos) en un agente extensible
multi-capacidad con: deteccion de intencion, sistema de skills dinamico, RAG con indexacion
en background, contexto de workspace con fuzzy matching, tools/MCP, interaccion JSON-Schema,
y arquitectura basada en interfaces (protocols) para extensibilidad futura.

## Arquitectura de Paquetes

```
command-line/src/
  main.py                              (entry point - modificado)
  aa/
    __init__.py
    domain/
      __init__.py
      ports/                           (Protocol interfaces)
        __init__.py
        llm.py
        skill_registry.py
        intent_detector.py
        knowledge_store.py
        knowledge_indexer.py
        workspace.py
        tool_provider.py
        interaction.py
        memory.py
      model/                           (Entidades y value objects)
        __init__.py
        intent.py
        skill.py
        thought.py
        workspace.py
        knowledge.py
        tool.py
        interaction.py
        context.py
    application/                       (Use cases / orquestacion)
      __init__.py
      pipeline.py
      intent_resolution.py
      skill_dispatch.py
      knowledge_retrieval.py
      workspace_resolution.py
      interaction_flow.py
      background_indexing.py
    infrastructure/                    (Implementaciones concretas)
      __init__.py
      llm/
        __init__.py
        llm_factory.py
      skill/
        __init__.py
        file_skill_registry.py
        parser/
          __init__.py
          parser.py
          markdown_parser.py
      intent/
        __init__.py
        llm_intent_detector.py
        intent_prompt_builder.py
        intent_schema.py
      knowledge/
        __init__.py
        pgvector_store.py
        pgvector_indexer.py
        content_extractor.py
        postgres_setup.py
      workspace/
        __init__.py
        directory_workspace.py
        fuzzy_match.py
      tools/
        __init__.py
        tool_registry.py
        builtin/
          __init__.py
          shell.py
          file_read.py
          file_write.py
      mcp/
        __init__.py
        mcp_client.py
        mcp_tool_wrapper.py
      interaction/
        __init__.py
        cli_interaction.py
        schema_prompter.py
      memory/
        __init__.py
        shell_memory.py
      thought/
        __init__.py
        llm_thought.py
        knowledge_thought.py
        skill_thought.py
        interaction_thought.py
      config/
        __init__.py
        app_config.py
    presentation/
      __init__.py
      cli.py
      output.py
  tests/
    __init__.py
    test_intent_detector.py
    test_skill_registry.py
    test_knowledge_store.py
    test_workspace_provider.py
    test_pipeline.py
    test_fuzzy_match.py
    fixtures/
      skills/
        test_skill.md
```

---

## Issues

---

### Issue 1: Domain Layer - Protocols (Ports)

**Objetivo:** Crear todas las interfaces Protocol que definen los contratos del sistema.
Sin dependencias externas, solo `typing.Protocol` y `abc.ABC`.

**Ficheros:**

#### `src/aa/domain/__init__.py` - Package marker

#### `src/aa/domain/ports/__init__.py` - Re-exporta todos los protocols

#### `src/aa/domain/ports/llm.py`
```python
class LLMProvider(Protocol):
    def create(self, model: str, temperature: float = 0.0) -> BaseChatModel: ...
```

#### `src/aa/domain/ports/skill_registry.py`
```python
class SkillRegistry(Protocol):
    def list_skills(self) -> list[SkillMetadata]: ...
    def load_skill(self, name: str) -> SkillConfig: ...
    def find_skills_for_intent(self, intent: Intent) -> list[SkillMetadata]: ...
```

#### `src/aa/domain/ports/intent_detector.py`
```python
class IntentDetector(Protocol):
    def detect(self, user_input: str, context: ExecutionContext,
               available_skills: list[SkillMetadata] | None = None) -> Intent: ...
```

#### `src/aa/domain/ports/knowledge_store.py`
```python
class KnowledgeStore(Protocol):
    def search(self, query: KnowledgeQuery) -> list[KnowledgeResult]: ...
```

#### `src/aa/domain/ports/knowledge_indexer.py`
```python
class KnowledgeIndexer(Protocol):
    def index_directory(self, topic: str, path: Path,
                        progress: Callable[[str], None] | None = None) -> int: ...
    def is_indexed(self, topic: str, relative_path: str) -> bool: ...
```

#### `src/aa/domain/ports/workspace.py`
```python
class WorkspaceProvider(Protocol):
    def resolve_reference(self, reference: str) -> WorkspaceInfo | None: ...
    def current_workspace(self) -> WorkspaceInfo | None: ...
    def list_projects(self, workspace: WorkspaceInfo) -> list[ProjectInfo]: ...
    def find_project(self, workspace: WorkspaceInfo, query: str) -> ProjectInfo | None: ...
```

#### `src/aa/domain/ports/tool_provider.py`
```python
class ToolProvider(Protocol):
    def list_tools(self) -> list[ToolSpec]: ...
    def get_tool(self, name: str) -> BaseTool | None: ...
    def register_tool(self, spec: ToolSpec, tool: BaseTool) -> None: ...
```

#### `src/aa/domain/ports/interaction.py`
```python
class UserInteraction(Protocol):
    def ask(self, question: Question) -> Answer | None: ...
    def inform(self, message: str) -> None: ...
```

#### `src/aa/domain/ports/memory.py`
```python
class Memory(Protocol):
    def append(self, entry: dict) -> None: ...
    def tail_session(self, n: int = 20) -> list[dict]: ...
    def tail_global(self, n: int = 20) -> list[dict]: ...
```

**Dependencias:** Ninguna.

---

### Issue 2: Domain Layer - Modelos y Entidades

**Objetivo:** Crear los dataclasses y enums del dominio.

**Ficheros:**

#### `src/aa/domain/model/__init__.py` - Re-exporta modelos

#### `src/aa/domain/model/intent.py`
- `IntentCategory(str, Enum)`: SHELL_COMMAND, SHELL_SUDO, AI_QUESTION, SKILL_TASK, CLARIFICATION_NEEDED
- `Intent(frozen dataclass)`: category, confidence, raw_input, corrected_command?, target_skill?, clarification_schema?, metadata

#### `src/aa/domain/model/skill.py`
- `SkillMetadata(frozen dataclass)`: name, description, category, tags: list[str], model?
- `SkillConfig(dataclass)`: metadata, mode, temperature?, tools, permissions, prompt, knowledge_topics: list[str]
- Los `tags` permiten al detector de intencion matchear "haz un commit convencional" con skill `conventional-commits` que tiene tags `[commit, conventional, git]`
- `knowledge_topics` indica que topics del RAG necesita este skill

#### `src/aa/domain/model/thought.py`
- `Thought(ABC)`: action: str, resolve() -> Conclusion
- `Conclusion(dataclass)`: proposal: str, context?: str, next_step?: Thought
- Evoluciona el patron actual. `next_step` reemplaza el stub `and_then()`

#### `src/aa/domain/model/workspace.py`
- `WorkspaceInfo(frozen dataclass)`: name, root_dir: Path, topics: list[str]
- `ProjectInfo(frozen dataclass)`: name, root_dir: Path, topics: list[str], aliases: list[str]

#### `src/aa/domain/model/knowledge.py`
- `KnowledgeQuery(frozen dataclass)`: text, topics: list[str], top_k: int = 5
- `KnowledgeResult(frozen dataclass)`: content, topic, source_path, score: float

#### `src/aa/domain/model/tool.py`
- `ToolSpec(frozen dataclass)`: name, description, parameters_schema: dict, source: str ("builtin"|"mcp"|"plugin")
- `ToolResult(frozen dataclass)`: success: bool, output?, error?

#### `src/aa/domain/model/interaction.py`
- `Question(frozen dataclass)`: prompt: str, schema: dict (JSON Schema), timeout_seconds: float = 30.0, default?: dict
- `Answer(frozen dataclass)`: values: dict, answered: bool = True

#### `src/aa/domain/model/context.py`
- `ExecutionContext(dataclass)`: user_input, argv, cwd: Path, stderr, history: list[dict], workspace?, project?, session_id

**Dependencias:** Ninguna.

---

### Issue 3: Infraestructura - LLM Factory

**Objetivo:** Migrar `src/ai/llm/llm_factory.py` a nueva ubicacion implementando `LLMProvider`.

**Ficheros:**

#### `src/aa/infrastructure/llm/__init__.py`

#### `src/aa/infrastructure/llm/llm_factory.py`
```python
class LangChainLLMProvider:
    """Implementa LLMProvider. Soporta ollama/ y openai/ como prefijos de modelo."""
    _providers: dict[str, Callable] = {
        "ollama": lambda name, temp: ChatOllama(model=name, temperature=temp),
        "openai": lambda name, temp: ChatOpenAI(model=name, temperature=temp),
    }
    def create(self, model: str, temperature: float = 0.0) -> BaseChatModel: ...
    @classmethod
    def register_provider(cls, name: str, factory: Callable) -> None: ...
```
- Misma logica que el actual `LLMFactory.create()` pero como instancia
- `register_provider` permite extensibilidad (anadir Anthropic, etc.)

**Dependencias:** Issue 1.

---

### Issue 4: Infraestructura - Skill Registry y Parsers

**Objetivo:** Evolucionar `AgentFactory` + `Parser` en un `FileSkillRegistry` que implementa `SkillRegistry`.
Extender el parser Markdown para leer campos nuevos: `tags`, `knowledge_topics`.

**Ficheros:**

#### `src/aa/infrastructure/skill/__init__.py`

#### `src/aa/infrastructure/skill/parser/__init__.py`

#### `src/aa/infrastructure/skill/parser/parser.py`
```python
class SkillParser(ABC):
    def parse_metadata(self, path: Path) -> SkillMetadata: ...  # Solo frontmatter (rapido)
    def parse(self, path: Path) -> SkillConfig: ...             # Completo
```

#### `src/aa/infrastructure/skill/parser/markdown_parser.py`
- Evoluciona `OpenCodeMarkdownParser`
- Parsea YAML frontmatter extendido:
```yaml
---
name: Conventional Commits
description: Genera mensajes de commit convencionales
category: development
tags: [commit, conventional, git, semver]
knowledge_topics: [git-conventions]
mode: primary
temperature: 0.0
tools: {}
---
<system prompt como markdown body>
```

#### `src/aa/infrastructure/skill/file_skill_registry.py`
```python
class FileSkillRegistry:
    def __init__(self, agents_dir: Path = ~/.config/asistente/agents/):
        self._parsers = {".md": MarkdownSkillParser()}
        self._index: dict[str, SkillMetadata] = {}
        self._scan()  # Build metadata index on init

    def list_skills(self) -> list[SkillMetadata]: ...
    def load_skill(self, name: str) -> SkillConfig: ...  # Full parse on demand
    def find_skills_for_intent(self, intent: Intent) -> list[SkillMetadata]:
        # Match intent.raw_input tokens contra skill tags
        # Retorna skills ordenados por relevancia
```

**Dependencias:** Issues 1, 2.

---

### Issue 5: Infraestructura - Detector de Intencion LLM

**Objetivo:** Crear un `LLMIntentDetector` que clasifica input del usuario con conocimiento
de los skills disponibles. Reemplaza el enfoque estatico de `root.md`.

**Ficheros:**

#### `src/aa/infrastructure/intent/__init__.py`

#### `src/aa/infrastructure/intent/intent_schema.py`
```python
class IntentDetectionResult(BaseModel):
    """Modelo Pydantic para structured output del LLM."""
    action: str  # shell_command | shell_sudo | ai_question | skill_task | clarification_needed
    command: str | None = None
    target_skill: str | None = None
    confidence: float
    clarification_schema: dict | None = None
```

#### `src/aa/infrastructure/intent/intent_prompt_builder.py`
```python
class IntentPromptBuilder:
    def build_system_prompt(self, skills: list[SkillMetadata]) -> str:
        # Genera prompt dinamico que incluye catalogo de skills disponibles
        # Logica similar a root.md pero con seccion de skills generada
    def build_user_prompt(self, context: ExecutionContext) -> str:
        # Formatea user_input, stderr, history
```

#### `src/aa/infrastructure/intent/llm_intent_detector.py`
```python
class LLMIntentDetector:
    def __init__(self, llm_provider: LLMProvider, model: str): ...
    def detect(self, user_input: str, context: ExecutionContext,
               available_skills: list[SkillMetadata] | None = None) -> Intent:
        # 1. Construir prompt con skills disponibles
        # 2. Invocar LLM con structured output (Pydantic)
        # 3. Mapear resultado a Intent del dominio
```

**Dependencias:** Issues 1, 2, 3, 4.

---

### Issue 6: Infraestructura - Knowledge Store (RAG lectura)

**Objetivo:** Portar la logica de retrieval de `asistente` a una clase que implementa `KnowledgeStore`.

**Referencia:** `asistente/src/app/rag/rag_ingest.py` (patron pgvector + OllamaEmbeddings bge-m3)

**Ficheros:**

#### `src/aa/infrastructure/knowledge/__init__.py`

#### `src/aa/infrastructure/knowledge/postgres_setup.py`
- Portado de `asistente`: asegura extension pgvector y tablas (documents, embeddings)

#### `src/aa/infrastructure/knowledge/pgvector_store.py`
```python
class PgVectorKnowledgeStore:
    def __init__(self, host, port, database, user, password,
                 table_prefix="", embedding_model="bge-m3",
                 embedding_base_url="http://localhost:11434"): ...
    def search(self, query: KnowledgeQuery) -> list[KnowledgeResult]:
        # Embed query, SQL similarity search con pgvector, filtrar por topics
```

**Dependencias:** Issues 1, 2.

---

### Issue 7: Infraestructura - Knowledge Indexer (RAG escritura)

**Objetivo:** Portar la logica de ingestion de `asistente/src/app/rag/rag_ingest.py` para
indexacion incremental.

**Ficheros:**

#### `src/aa/infrastructure/knowledge/content_extractor.py`
- Portado de `asistente`: extraccion de texto de PDF, DOCX, PPTX, ODT, RTF, MSG, HTML

#### `src/aa/infrastructure/knowledge/pgvector_indexer.py`
```python
class PgVectorKnowledgeIndexer:
    def __init__(self, ...pg params..., embedding_model, embedding_base_url): ...
    def index_directory(self, topic, path, progress=None) -> int:
        # Walk dir, skip indexed, extract text, split chunks, embed, insert
        # Logica de RagIngest.ingest() adaptada
    def is_indexed(self, topic, relative_path) -> bool: ...
```

**Dependencias:** Issues 1, 2, 6.

---

### Issue 8: Infraestructura - Workspace Provider con Fuzzy Matching

**Objetivo:** Crear `DirectoryWorkspaceProvider` que implementa `WorkspaceProvider`.
Capacidad clave: `find_project("shell-back")` matchea contra `sgt-shellapi`.

**Ficheros:**

#### `src/aa/infrastructure/workspace/__init__.py`

#### `src/aa/infrastructure/workspace/fuzzy_match.py`
```python
def fuzzy_match(query: str, candidates: list[str], threshold: float = 0.4) -> list[tuple[str, float]]:
    # Estrategia 1: Substring containment (mayor peso)
    # Estrategia 2: Token overlap (split en -, _, etc.) "shell-back" -> "shell" matchea "sgt-shellapi"
    # Estrategia 3: difflib.SequenceMatcher ratio
    # Estrategia 4: Abbreviation matching
    # Retorna candidatos ordenados por score, filtrados por threshold
```
- Usa solo stdlib (`difflib.SequenceMatcher`)

#### `src/aa/infrastructure/workspace/directory_workspace.py`
```python
class DirectoryWorkspaceProvider:
    def __init__(self, workspaces_dir: Path, config_path: Path | None = None): ...
    def resolve_reference(self, reference: str) -> WorkspaceInfo | None:
        # Fuzzy match reference contra nombres de directorio en workspaces_dir
    def current_workspace(self) -> WorkspaceInfo | None:
        # Lee config activa
    def list_projects(self, workspace: WorkspaceInfo) -> list[ProjectInfo]:
        # Enumera subdirectorios del workspace
    def find_project(self, workspace: WorkspaceInfo, query: str) -> ProjectInfo | None:
        # Fuzzy match query contra project names y aliases
```

**Dependencias:** Issues 1, 2.

---

### Issue 9: Infraestructura - Tool Registry y Built-in Tools

**Objetivo:** Crear `LangChainToolProvider` con tools built-in para shell, lectura y escritura de ficheros.

**Ficheros:**

#### `src/aa/infrastructure/tools/__init__.py`

#### `src/aa/infrastructure/tools/tool_registry.py`
```python
class LangChainToolProvider:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._register_builtins()
    def list_tools(self) -> list[ToolSpec]: ...
    def get_tool(self, name: str) -> BaseTool | None: ...
    def register_tool(self, spec: ToolSpec, tool: BaseTool) -> None: ...
```

#### `src/aa/infrastructure/tools/builtin/__init__.py`

#### `src/aa/infrastructure/tools/builtin/shell.py`
- `ShellExecuteTool(BaseTool)`: ejecuta comando shell, retorna stdout/stderr

#### `src/aa/infrastructure/tools/builtin/file_read.py`
- `FileReadTool(BaseTool)`: lee contenido de fichero

#### `src/aa/infrastructure/tools/builtin/file_write.py`
- `FileWriteTool(BaseTool)`: escribe contenido a fichero

**Dependencias:** Issues 1, 2.

---

### Issue 10: Infraestructura - MCP Client Adapter

**Objetivo:** Conectar a servidores MCP, descubrir sus tools y registrarlos como LangChain BaseTool.

**Ficheros:**

#### `src/aa/infrastructure/mcp/__init__.py`

#### `src/aa/infrastructure/mcp/mcp_client.py`
```python
@dataclass
class MCPServerConfig:
    name: str
    transport: str        # "stdio" | "http"
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)

class MCPClientAdapter:
    def __init__(self, server_config: MCPServerConfig): ...
    async def connect(self) -> None:
        # Inicializar sesion MCP, listar tools, wrappear como BaseTool
    def get_tools(self) -> list[BaseTool]: ...
    async def disconnect(self) -> None: ...
```

#### `src/aa/infrastructure/mcp/mcp_tool_wrapper.py`
```python
class MCPToolWrapper(BaseTool):
    """Envuelve un tool MCP individual como LangChain BaseTool."""
    def _run(self, **kwargs) -> str: ...
    async def _arun(self, **kwargs) -> str: ...
```

**Nueva dependencia:** Anadir `mcp>=1.0.0` a `requirements.txt`.

**Dependencias:** Issues 1, 2, 9.

---

### Issue 11: Infraestructura - Interaccion CLI con JSON Schema

**Objetivo:** Crear `CLIUserInteraction` que implementa `UserInteraction`.
Usa `prompt_toolkit` + `rich` para renderizar preguntas basadas en JSON Schema.

**Ficheros:**

#### `src/aa/infrastructure/interaction/__init__.py`

#### `src/aa/infrastructure/interaction/schema_prompter.py`
```python
class SchemaPrompter:
    def prompt_for_schema(self, schema: dict, timeout: float = 30.0) -> dict | None:
        # Recorre propiedades del schema:
        #   string -> input simple
        #   number -> input con validacion
        #   boolean -> yes/no
        #   enum -> lista de seleccion
        #   object -> recursion
        # Retorna valores o None en timeout
```

#### `src/aa/infrastructure/interaction/cli_interaction.py`
```python
class CLIUserInteraction:
    def __init__(self, console: Console | None = None): ...
    def ask(self, question: Question) -> Answer | None:
        # Renderiza question.prompt con rich
        # Delega a SchemaPrompter para recoger valores
        # Valida contra schema con jsonschema
        # Retorna Answer o None si timeout/skip
    def inform(self, message: str) -> None:
        # rich print
```

**Dependencias:** Issues 1, 2.

---

### Issue 12: Infraestructura - Memory (migrar ShellMemory)

**Objetivo:** Mover `src/ai/user/shell_memory.py` a nueva ubicacion, conformando al protocol `Memory`.

**Ficheros:**

#### `src/aa/infrastructure/memory/__init__.py`

#### `src/aa/infrastructure/memory/shell_memory.py`
- Misma implementacion que la actual (JSONL, session/global, auto-cleanup, rotacion)
- Ahora conforma al protocol `Memory`

**Dependencias:** Issue 1.

---

### Issue 13: Infraestructura - Configuracion Unificada

**Objetivo:** Crear `AppConfig` que consolida toda la configuracion del sistema.
Persistida como JSON en `~/.config/asistente/asistente.json`.

**Ficheros:**

#### `src/aa/infrastructure/config/__init__.py`

#### `src/aa/infrastructure/config/app_config.py`
```python
@dataclass
class AgentModelConfig:
    default_model: str = "ollama/llama3.2:3b"
    intent_model: str = "ollama/llama3.2:3b"

@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    table_prefix: str = ""

@dataclass
class TopicConfig:
    name: str
    type: str = "directory"  # extensible a "cloud", "api", etc.
    path: str = ""

@dataclass
class AppConfig:
    config_path: Path
    agents_dir: Path            # donde buscar skills (.md)
    workspaces_dir: Path        # directorio de workspaces
    models: AgentModelConfig
    postgres: PostgresConfig
    topics: list[TopicConfig]
    mcp_servers: list[MCPServerConfig]
    active_workspace: Path | None

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig": ...
    def save(self) -> None: ...
```

**Dependencias:** Issues 1, 10 (para MCPServerConfig).

---

### Issue 14: Infraestructura - Evolucion del Thought Chain

**Objetivo:** Extender el patron Thought/Conclusion para soportar cadenas multi-paso.

**Ficheros:**

#### `src/aa/infrastructure/thought/__init__.py`

#### `src/aa/infrastructure/thought/llm_thought.py`
- Evolucion del actual `LLMThought`. Misma logica: invoke LLM, retorna Conclusion.

#### `src/aa/infrastructure/thought/knowledge_thought.py`
```python
class KnowledgeThought(Thought):
    """Busca conocimiento en RAG y lo pasa al siguiente thought."""
    def __init__(self, query, store, next_thought_factory): ...
    def resolve(self) -> Conclusion:
        results = self._store.search(self._query)
        context = "\n\n".join(r.content for r in results)
        next_thought = self._next_factory(context)
        return Conclusion(proposal=f"Retrieved {len(results)} chunks",
                          context=context, next_step=next_thought)
```

#### `src/aa/infrastructure/thought/skill_thought.py`
```python
class SkillThought(Thought):
    """Ejecuta un skill: prompt + context + LLM con tools opcionales."""
    def __init__(self, skill, context, knowledge_context, llm, tools): ...
    def resolve(self) -> Conclusion:
        # Si hay tools: crear LangChain agent (create_react_agent)
        # Si no: invoke LLM directo
```

#### `src/aa/infrastructure/thought/interaction_thought.py`
```python
class InteractionThought(Thought):
    """Pide info al usuario via JSON Schema."""
    def __init__(self, question, interaction): ...
    def resolve(self) -> Conclusion:
        answer = self._interaction.ask(self._question)
        # Manejar caso de no-respuesta
```

**Dependencias:** Issues 1, 2, 3, 6, 9.

---

### Issue 15: Application Layer - Use Cases

**Objetivo:** Implementar los casos de uso que orquestan el flujo. Dependen solo de protocols del dominio.

**Ficheros:**

#### `src/aa/application/__init__.py`

#### `src/aa/application/intent_resolution.py`
```python
class ResolveIntentUseCase:
    def __init__(self, detector: IntentDetector, skill_registry: SkillRegistry): ...
    def execute(self, context: ExecutionContext) -> Intent:
        skills = self._skill_registry.list_skills()
        return self._detector.detect(context.user_input, context, skills)
```

#### `src/aa/application/workspace_resolution.py`
```python
class ResolveWorkspaceUseCase:
    def __init__(self, workspace_provider: WorkspaceProvider): ...
    def execute(self, context: ExecutionContext) -> ExecutionContext:
        # Resolver workspace desde cwd
        # Escanear user_input para referencias a proyectos
        # Fuzzy match contra proyectos conocidos
        # Retornar context enriquecido
```

#### `src/aa/application/knowledge_retrieval.py`
```python
class RetrieveKnowledgeUseCase:
    def __init__(self, knowledge_store: KnowledgeStore): ...
    def execute(self, question: str, topics: list[str], k: int = 5) -> str:
        # Construir KnowledgeQuery, buscar, formatear como string de contexto
```

#### `src/aa/application/skill_dispatch.py`
```python
class DispatchToSkillUseCase:
    def __init__(self, skill_registry, knowledge_store, llm_provider, tool_provider): ...
    def execute(self, skill_name: str, context: ExecutionContext) -> Conclusion:
        # 1. Cargar skill config
        # 2. Obtener knowledge para los topics del skill
        # 3. Construir prompt (skill.prompt + knowledge + user_input)
        # 4. Crear LLM con modelo del skill
        # 5. Si skill tiene tools, bind tools al LLM (agente LangChain)
        # 6. Invocar y retornar Conclusion
```

#### `src/aa/application/interaction_flow.py`
```python
class HandleInteractionUseCase:
    def __init__(self, interaction: UserInteraction): ...
    def execute(self, question: Question) -> Answer:
        answer = self._interaction.ask(question)
        if answer is None or not answer.answered:
            return Answer(values=question.default or {}, answered=False)
        return answer
```

#### `src/aa/application/background_indexing.py`
```python
class BackgroundIndexService:
    def __init__(self, indexer: KnowledgeIndexer, config: AppConfig): ...
    async def run(self) -> None:
        # Para cada topic configurado: escanear ficheros nuevos, indexar
    def start_background(self) -> asyncio.Task:
        return asyncio.create_task(self.run())
```

#### `src/aa/application/pipeline.py`
```python
class RequestPipeline:
    """Orquestador principal. Compone todos los use cases."""
    def __init__(self, intent_detector, skill_registry, knowledge_store,
                 workspace_provider, tool_provider, interaction,
                 llm_provider, memory): ...

    def execute(self, context: ExecutionContext) -> Conclusion:
        # 1. workspace_resolution.execute(context) -> context enriquecido
        # 2. intent_resolution.execute(context) -> Intent
        # 3. match intent.category:
        #    CLARIFICATION_NEEDED -> interaction_flow con JSON Schema
        #    SKILL_TASK -> skill_dispatch (carga skill, RAG, ejecuta)
        #    SHELL_COMMAND/SUDO -> Conclusion con comando
        #    AI_QUESTION -> knowledge_retrieval + LLM invoke
```

**Dependencias:** Issues 1, 2.

---

### Issue 16: Presentation Layer - CLI y Output

**Objetivo:** Crear la capa de presentacion que reemplaza `main.py`. Cablea el grafo de dependencias.

**Ficheros:**

#### `src/aa/presentation/__init__.py`

#### `src/aa/presentation/output.py`
```python
class OutputFormatter:
    def __init__(self, console: Console | None = None): ...
    def print_conclusion(self, conclusion: Conclusion) -> None: ...
    def print_command(self, command: str, sudo: bool = False) -> None: ...
    def print_error(self, message: str) -> None: ...
```

#### `src/aa/presentation/cli.py`
```python
class CLI:
    def __init__(self):
        self._config = AppConfig.load()
        self._build_dependencies()

    def _build_dependencies(self) -> None:
        # Construir grafo completo:
        # LLMProvider, Memory, SkillRegistry, KnowledgeStore,
        # WorkspaceProvider, ToolProvider, UserInteraction,
        # IntentDetector, MCPClients -> ToolProvider
        # Pipeline(todo lo anterior)

    def run(self, argv: list[str]) -> int:
        # Construir ExecutionContext
        # Fast path: try_run_command (mantener comportamiento actual)
        # Si falla: pipeline.execute(context)
        # Formatear y mostrar conclusion
        # Lanzar background indexing si configurado
```

#### Modificar `src/main.py`
```python
from aa.presentation.cli import CLI

def main() -> int:
    cli = CLI()
    return cli.run(sys.argv[1:])
```
- Mantiene `try_run_command` como fast-path
- Delega al pipeline completo cuando el comando falla o se detecta intencion AI

**Dependencias:** Issues 1-15 (todo debe estar disponible).

---

### Issue 17: Tests

**Objetivo:** Tests unitarios e integracion para los componentes criticos.

**Ficheros:**

#### `tests/__init__.py`

#### `tests/test_fuzzy_match.py`
- `test_exact_match`, `test_substring_match`
- `test_token_overlap`: "shell-back" matchea "sgt-shellapi"
- `test_abbreviation`, `test_below_threshold`

#### `tests/test_skill_registry.py`
- Test scan de directorio, extraccion de metadata, match por tags

#### `tests/test_intent_detector.py`
- Test clasificacion con mocks de LLM

#### `tests/test_knowledge_store.py`
- Test search con mock de postgres

#### `tests/test_workspace_provider.py`
- Test fuzzy project matching con directorios temporales

#### `tests/test_pipeline.py`
- Integracion: mock de todos los protocols, ejecutar pipeline completo
- Escenarios: correccion de comando, pregunta AI, dispatch a skill, clarificacion

#### `tests/fixtures/skills/test_skill.md`
- Skill de ejemplo para tests

**Dependencias:** Issues 1-16.

---

### Issue 18: Actualizacion de Dependencias y Limpieza

**Objetivo:** Actualizar requirements.txt, crear pyproject.toml, eliminar codigo viejo.

**Ficheros:**

#### Modificar `requirements.txt`
- Anadir: `mcp>=1.0.0`, `pydantic>=2.0.0` (explicita)

#### Crear `pyproject.toml`
- Definir paquete `aa`, Python >= 3.12

#### Eliminar `src/ai/` (todo el directorio)
- Solo despues de que la migracion este completa y tests pasen

**Dependencias:** Issues 1-17.

---

## Orden de Implementacion Recomendado

```
Fase 1 - Fundamentos:
  Issue 1  (Domain Ports)
  Issue 2  (Domain Models)

Fase 2 - Infraestructura basica (en paralelo):
  Issue 3  (LLM Factory)
  Issue 12 (Memory)
  Issue 13 (Config)

Fase 3 - Infraestructura core (en paralelo):
  Issue 4  (Skill Registry)
  Issue 8  (Workspace Provider)
  Issue 6  (Knowledge Store - RAG read)
  Issue 9  (Tool Registry)
  Issue 11 (CLI Interaction)

Fase 4 - Infraestructura avanzada:
  Issue 5  (Intent Detector) - necesita 3, 4
  Issue 7  (Knowledge Indexer) - necesita 6
  Issue 10 (MCP Client) - necesita 9
  Issue 14 (Thought Chain) - necesita 3, 6, 9

Fase 5 - Orquestacion:
  Issue 15 (Application Layer / Use Cases)
  Issue 16 (Presentation / CLI)

Fase 6 - Cierre:
  Issue 17 (Tests)
  Issue 18 (Dependencias y limpieza)
```

## Grafo de Dependencias

```
Issue  1 (Domain Ports)
  |
  +---> Issue  2 (Domain Models)
  |       |
  |       +---> Issue  3 (LLM Factory)
  |       +---> Issue 12 (Memory)
  |       +---> Issue 13 (Config) -------- necesita 10 para MCPServerConfig
  |       +---> Issue  4 (Skill Registry)
  |       +---> Issue  8 (Workspace)
  |       +---> Issue  6 (Knowledge Store)
  |       +---> Issue  9 (Tool Registry)
  |       +---> Issue 11 (CLI Interaction)
  |       +---> Issue 15 (Application Layer)
  |
  +---> Issue  5 (Intent Detector) ------- necesita 3, 4
  +---> Issue  7 (Knowledge Indexer) ----- necesita 6
  +---> Issue 10 (MCP Client) ------------ necesita 9
  +---> Issue 14 (Thought Chain) --------- necesita 3, 6, 9
  +---> Issue 16 (Presentation / CLI) ---- necesita 1-15
  +---> Issue 17 (Tests) ----------------- necesita 1-16
  +---> Issue 18 (Limpieza) -------------- necesita 1-17
```

## Verificacion

Para validar que el sistema funciona end-to-end:

1. **Unit tests:** `pytest tests/` - todos los tests deben pasar
2. **Smoke test basico:** `python src/main.py als` - debe corregir a `ls` (fast-path actual)
3. **Intent con skill:** Crear skill `conventional-commits.md` con tags `[commit, conventional]`, ejecutar `python src/main.py "haz un commit convencional"` - debe detectar intent SKILL_TASK y cargar el skill
4. **Fuzzy workspace:** Configurar workspace con proyecto `sgt-shellapi`, ejecutar con referencia `shell-back` - debe resolver correctamente
5. **RAG:** Configurar topic con directorio de documentos, ejecutar background indexing, luego query que requiera ese conocimiento
6. **Interaccion:** Ejecutar comando ambiguo tipo `delete file` - debe pedir clarificacion via JSON Schema
7. **Ruff/lint:** `ruff check src/aa/` sin errores
