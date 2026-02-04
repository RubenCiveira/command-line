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
      agent/          # Agentes conversacionales
      llm/            # Integracion con modelos LLM
      rag/            # RAG: ingesta, retrieval y setup SQLite
        sqlite_rag_setup.py
        rag_ingest.py
        rag_retriever.py
        content_extractor.py
      user/            # Configuracion de usuario
        user_config.py
      classificator.py # Clasificacion zero-shot jerarquica (IPTC)
    tools/
      build_iptc_tree.py  # Genera el arbol de categorias IPTC
  examples/
    rag_e2e_demo.py    # Demo end-to-end del pipeline RAG
    chat_local.py      # Chat con modelos locales
    embeddings_local.py # Pruebas de embeddings
    zero_shot_eval.py  # Evaluacion de clasificacion zero-shot
    intent_eval.py     # Evaluacion de deteccion de intents
  resources/
    iptc_categories.json  # Taxonomia IPTC Media Topics
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

## Tests

```bash
make test
```

## Lint y formato

```bash
make lint
make format
```
