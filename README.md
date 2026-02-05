# command-line

Asistente de linea de comandos con capacidades de IA local: RAG con SQLite, clasificacion zero-shot y embeddings multilingues.

## Requisitos

- Python 3.11+
- macOS (Apple Silicon) o Linux

## Instalacion

```bash
make build
```

Esto crea un virtualenv en `.venv/` e instala todas las dependencias de `requirements.txt`.

## Estructura del proyecto

```
command-line/
  src/
    ai/
      agent/
        thought/
          thought.py            # ABC base para pensamientos
          conclusion.py         # Resultado de un Thought (con doubts JSON schema)
          response.py           # Respuestas del usuario a doubts
          llm_thought.py        # Thought generico que invoca un LLM
          internal/
            project_resolver_thought.py  # Resolucion de proyecto (4 capas)
      llm/                      # Integracion con modelos LLM
      rag/                      # RAG: ingesta, retrieval y setup SQLite
        sqlite_rag_setup.py
        rag_ingest.py
        rag_retriever.py
        content_extractor.py
      user/                     # Configuracion de usuario
        user_config.py
        user_memory.py
        project_topic.py
        rag_topic.py
      classificator.py          # Clasificacion zero-shot jerarquica (IPTC)
      guardrails.py             # Deteccion de prompt injection
      intention.py              # Clasificacion de intenciones
      model_pool.py             # Cache LRU de pipelines de transformers
      project_resolver.py       # Fachada para resolver proyecto por mensaje
    tools/
      build_iptc_tree.py        # Genera el arbol de categorias IPTC
  examples/
    rag_e2e_demo.py             # Demo end-to-end del pipeline RAG
    project_resolver_eval.py    # Evaluacion de resolucion de proyectos
    zero_shot_eval.py           # Evaluacion de clasificacion zero-shot
    intent_eval.py              # Evaluacion de deteccion de intents
    guardrail_eval.py           # Evaluacion de guardrails (prompt injection)
    chat_local.py               # Chat con modelos locales
    embeddings_local.py         # Pruebas de embeddings
  resources/
    iptc_categories.json        # Taxonomia IPTC Media Topics
```

## Demo RAG end-to-end

El script `examples/rag_e2e_demo.py` ejecuta el pipeline RAG completo: ingesta de documentos, generacion de embeddings, clasificacion por categorias IPTC y busqueda por similitud.

### Preparar las categorias IPTC

Si no existe `resources/iptc_categories.json`, generarlo con:

```bash
PYTHONPATH=src .venv/bin/python src/tools/build_iptc_tree.py
```

El fichero se genera en `resources/iptc_categories.json`. Tambien se puede copiar a `~/.config/.asistente/iptc_categories.json` para uso global.

### Ejecutar la demo

```bash
# Ingesta + modo busqueda interactivo
PYTHONPATH=src .venv/bin/python examples/rag_e2e_demo.py /ruta/a/directorio/con/pdfs

# Con base de datos persistente
PYTHONPATH=src .venv/bin/python examples/rag_e2e_demo.py /ruta/a/pdfs --db /tmp/rag_test.db

# Con query directa
PYTHONPATH=src .venv/bin/python examples/rag_e2e_demo.py /ruta/a/pdfs --query "machine learning"

# Conservar la base de datos temporal al salir
PYTHONPATH=src .venv/bin/python examples/rag_e2e_demo.py /ruta/a/pdfs --keep-db
```

La demo:

1. Procesa todos los documentos del directorio (PDF, DOCX, PPTX, TXT, etc.)
2. Genera chunks y embeddings con `paraphrase-multilingual-MiniLM-L12-v2`
3. Clasifica cada documento en la taxonomia IPTC con zero-shot (`ModernBERT-large-zeroshot-v2.0`)
4. Muestra tablas resumen de `documents` y `document_categories`
5. Permite buscar por similitud; las queries se clasifican y se filtran por categoria nivel 0

### Formatos soportados

PDF, DOCX, PPTX, ODT, RTF, EML, MSG, HTML y cualquier fichero de texto plano.

## Ejemplos de evaluacion

Todos los evaluadores tienen un menu interactivo con seleccion de modelo, bateria de queries de ejemplo y modo interactivo para probar textos libres.

### Project Resolver

```bash
PYTHONPATH=src .venv/bin/python examples/project_resolver_eval.py
```

Evalua la resolucion de proyectos: dado un mensaje del usuario y un conjunto de proyectos configurados, determina a que proyecto se refiere. Usa una estrategia en 4 capas:

1. **Keyword / fuzzy match** — coincidencia por nombre de proyecto en el mensaje
2. **Zero-shot classification** — clasificacion semantica con nombres de proyecto como etiquetas
3. **History boost** — sesgo hacia proyectos mencionados recientemente en la sesion
4. **Doubts** — si el resultado es ambiguo, genera un JSON schema para pedir confirmacion

El evaluador incluye 5 sets de proyectos simulados (programacion, marketing, investigacion medica, fisica teorica, y un set mixto con todos). La evaluacion cruza el proyecto esperado con el set activo: si el proyecto esperado no existe en el set, lo correcto es que el resolver lo marque como ambiguo.

### Guardrails (prompt injection)

```bash
PYTHONPATH=src .venv/bin/python examples/guardrail_eval.py
```

Evalua modelos de deteccion de inyeccion de prompts y jailbreak. Incluye bateria de pruebas en multiples idiomas (ingles, espanol, frances, aleman, italiano, portugues) con frases benignas y maliciosas. Permite ajustar el threshold de deteccion.

### Intent detection

```bash
PYTHONPATH=src .venv/bin/python examples/intent_eval.py
```

Evalua clasificacion de intenciones del usuario (booking, weather, cancel, etc.) con modelos fine-tuned de Hugging Face.

### Zero-shot classification

```bash
PYTHONPATH=src .venv/bin/python examples/zero_shot_eval.py
```

Evalua clasificacion zero-shot jerarquica con la taxonomia IPTC Media Topics. Clasifica textos nivel a nivel en el arbol de categorias, mostrando scores y distribuciones en cada nivel. Requiere `resources/iptc_categories.json` (ver seccion "Preparar las categorias IPTC").

## Nota sobre modelos de meta-llama en Hugging Face

El sistema actual no envia cabeceras de autorizacion para modelos de Hugging Face.
Los modelos de `meta-llama` requieren autenticacion remota, por lo que no estan
soportados en este momento.

## Procesos principales

### Categorizacion (IPTC)

Se usa clasificacion zero-shot jerarquica basada en la taxonomia IPTC para:
- etiquetar contenidos de RAG (ingesta y retrieval),
- inferir el dominio de una consulta y filtrar contexto relevante.

El pipeline baja nivel por nivel y se detiene si la confianza cae por debajo del umbral.

### Deteccion de intencion

Se clasifica la entrada del usuario para decidir si es:
- un comando de shell,
- una pregunta general,
- o una tarea que debe enviarse a un skill.

Este proceso permite enrutar la ejecucion hacia el agente adecuado.

### Guardrail (prompt injection)

Antes de enviar prompts al LLM, se aplica un clasificador de inyeccion que
marca entradas sospechosas como bloqueadas. Incluye una evaluacion interactiva
en `examples/guardrail_eval.py` para medir falsos positivos/negativos.

### Seleccion de proyecto

El resolutor de proyectos combina varias senales:
- coincidencia por nombre y fuzzy matching,
- clasificacion IPTC para filtrar proyectos por dominio,
- embeddings del `detail` del proyecto para reordenar candidatos,
- y confirmacion del usuario cuando hay dudas.

El objetivo es mapear frases como "shell-back" o "actualiza la landing" al
proyecto correcto de forma robusta.

## Tests

```bash
make test
```

## Lint y formato

```bash
make lint
make format
```
