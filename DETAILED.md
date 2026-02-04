# Plan detallado — @command-line (basado en PLAN.md)

## Fase 1 — Capa de Dominio (Ports + Modelos)
### Tarea 1.1 — Protocols (Ports)
#### Issue 1 — Domain Layer: Protocols (Ports)
**Objetivo:** definir contratos del sistema con `typing.Protocol`.
**Informacion tecnica:** crear interfaces en `src/aa/domain/ports/` para LLM, skills, intent, knowledge, workspace, tools, interaccion y memoria.
**Definicion de hecho:** todos los ports existen y son re-exportados en `src/aa/domain/ports/__init__.py`.
**Como verificar:** import desde `aa.domain.ports` funciona y no hay dependencias externas.

### Tarea 1.2 — Modelos y entidades
#### Issue 2 — Domain Layer: Modelos y Entidades
**Objetivo:** crear dataclasses/enums de dominio.
**Informacion tecnica:** implementar `Intent`, `Skill`, `Thought`, `Workspace`, `Knowledge`, `Tool`, `Interaction` y `ExecutionContext` en `src/aa/domain/model/`.
**Definicion de hecho:** todos los modelos estan disponibles y tipados.
**Como verificar:** importacion y uso en un test simple sin errores.

## Fase 2 — Infraestructura base
### Tarea 2.1 — LLM Provider
#### Issue 3 — Infraestructura: LLM Factory
**Objetivo:** migrar `LLMFactory` a `LangChainLLMProvider`.
**Informacion tecnica:** `src/aa/infrastructure/llm/llm_factory.py` con `create()` y `register_provider()`.
**Definicion de hecho:** se puede instanciar LLM con `ollama/` y `openai/`.
**Como verificar:** test con modelos dummy o mock.

### Tarea 2.2 — Memoria
#### Issue 12 — Infraestructura: Memory
**Objetivo:** mover `ShellMemory` a infraestructura.
**Informacion tecnica:** `src/aa/infrastructure/memory/shell_memory.py` implementa `Memory`.
**Definicion de hecho:** misma funcionalidad que la actual, pero bajo `aa`.
**Como verificar:** lectura/escritura JSONL y rotacion funcionan.

### Tarea 2.3 — Configuracion
#### Issue 13 — Infraestructura: AppConfig
**Objetivo:** consolidar configuracion global.
**Informacion tecnica:** `src/aa/infrastructure/config/app_config.py` con dataclasses y `load/save`.
**Definicion de hecho:** config se carga y guarda en `~/.config/asistente/asistente.json`.
**Como verificar:** crear config base y recargarla.

## Fase 3 — Infraestructura core (Skills, Workspace, Knowledge, Tools)
### Tarea 3.1 — Skills
#### Issue 4 — Infraestructura: Skill Registry y Parsers
**Objetivo:** crear `FileSkillRegistry` y parser markdown extendido.
**Informacion tecnica:** parser con frontmatter YAML (tags, knowledge_topics).
**Definicion de hecho:** registry indexa y carga skills bajo demanda.
**Como verificar:** usar `tests/fixtures/skills/test_skill.md`.

### Tarea 3.2 — Workspace
#### Issue 8 — Infraestructura: Workspace Provider con Fuzzy Matching
**Objetivo:** mapear referencias naturales a proyectos reales.
**Informacion tecnica:** `DirectoryWorkspaceProvider` + `fuzzy_match`.
**Definicion de hecho:** "shell-back" resuelve `sgt-shellapi`.
**Como verificar:** test con directorios temporales.

### Tarea 3.3 — Knowledge (lectura)
#### Issue 6 — Infraestructura: Knowledge Store (RAG lectura)
**Objetivo:** implementar busqueda RAG con pgvector.
**Informacion tecnica:** `PgVectorKnowledgeStore` + `postgres_setup`.
**Definicion de hecho:** consulta devuelve `KnowledgeResult`.
**Como verificar:** test con DB local o mock.

### Tarea 3.4 — Tools
#### Issue 9 — Infraestructura: Tool Registry y Builtins
**Objetivo:** registry + tools builtin (shell/read/write).
**Informacion tecnica:** `LangChainToolProvider` y tools en `builtin/`.
**Definicion de hecho:** tools listables y ejecutables.
**Como verificar:** ejecutar tool shell en test.

### Tarea 3.5 — Interaccion CLI
#### Issue 11 — Infraestructura: Interaccion JSON Schema
**Objetivo:** interaccion de usuario con JSON schema.
**Informacion tecnica:** `CLIUserInteraction` + `SchemaPrompter`.
**Definicion de hecho:** preguntas se renderizan y validan.
**Como verificar:** test manual de prompt.

## Fase 4 — Infraestructura avanzada
### Tarea 4.1 — Deteccion de intencion
#### Issue 5 — Infraestructura: Intent Detector LLM
**Objetivo:** clasificar intencion usando LLM + skills disponibles.
**Informacion tecnica:** `IntentPromptBuilder`, `IntentSchema`, `LLMIntentDetector`.
**Definicion de hecho:** intent mapeado a `Intent` con confidence.
**Como verificar:** mock de LLM con structured output.

### Tarea 4.2 — Knowledge (indexacion)
#### Issue 7 — Infraestructura: Knowledge Indexer
**Objetivo:** ingestion incremental en pgvector.
**Informacion tecnica:** `content_extractor` + `PgVectorKnowledgeIndexer`.
**Definicion de hecho:** indexa solo nuevos o modificados.
**Como verificar:** cambiar un fichero y reindexar.

### Tarea 4.3 — MCP
#### Issue 10 — Infraestructura: MCP Client Adapter
**Objetivo:** conectar MCPs y registrarlos como tools.
**Informacion tecnica:** `MCPClientAdapter` y `MCPToolWrapper`.
**Definicion de hecho:** tools MCP aparecen en registry.
**Como verificar:** conectar a servidor MCP de prueba.

### Tarea 4.4 — Thought Chain
#### Issue 14 — Infraestructura: Thought Chain
**Objetivo:** soportar cadena multi-paso de pensamientos.
**Informacion tecnica:** `KnowledgeThought`, `SkillThought`, `InteractionThought`.
**Definicion de hecho:** `Conclusion.next_step` encadena pasos.
**Como verificar:** test unitario de chaining.

## Fase 5 — Application Layer
### Tarea 5.1 — Use Cases
#### Issue 15 — Application Layer: Use Cases
**Objetivo:** orquestar el flujo completo en casos de uso.
**Informacion tecnica:** `ResolveIntent`, `ResolveWorkspace`, `RetrieveKnowledge`, `DispatchToSkill`, `HandleInteraction`, `BackgroundIndexService`, `RequestPipeline`.
**Definicion de hecho:** pipeline compone todos los ports.
**Como verificar:** test de `RequestPipeline` con mocks.

## Fase 6 — Presentation Layer
### Tarea 6.1 — CLI y output
#### Issue 16 — Presentation Layer: CLI y Output
**Objetivo:** reemplazar `main.py` por CLI nuevo.
**Informacion tecnica:** `aa.presentation.cli` + `OutputFormatter`.
**Definicion de hecho:** `python src/main.py` usa pipeline.
**Como verificar:** smoke test con comando fallido.

## Fase 7 — Tests
### Tarea 7.1 — Suites unitarias e integracion
#### Issue 17 — Tests
**Objetivo:** cobertura minima de componentes criticos.
**Informacion tecnica:** tests para intent, skills, knowledge, workspace y pipeline.
**Definicion de hecho:** `pytest tests/` verde.
**Como verificar:** ejecutar tests localmente.

## Fase 8 — Dependencias y limpieza
### Tarea 8.1 — Actualizacion del entorno
#### Issue 18 — Dependencias y limpieza
**Objetivo:** actualizar requirements y limpiar `src/ai/`.
**Informacion tecnica:** anadir `mcp`, `pydantic` y crear `pyproject.toml`.
**Definicion de hecho:** dependencias sincronizadas y codigo antiguo eliminado.
**Como verificar:** instalacion limpia + tests OK.

## Verificacion end-to-end
### Escenario 1 — Fast path
**Objetivo:** mantener la ejecucion directa de comandos.
**Como verificar:** `python src/main.py ls` ejecuta `ls`.

### Escenario 2 — Skill intent
**Objetivo:** detectar skill por tags.
**Como verificar:** skill `conventional-commits` y comando "haz un commit convencional".

### Escenario 3 — Workspace fuzzy
**Objetivo:** resolver proyecto por alias.
**Como verificar:** "shell-back" -> `sgt-shellapi`.

### Escenario 4 — RAG
**Objetivo:** recuperar contexto de conocimiento indexado.
**Como verificar:** query que use documento indexado.

### Escenario 5 — Clarificacion
**Objetivo:** pedir datos faltantes y seguir flujo.
**Como verificar:** input ambiguo produce JSON-schema prompt.
