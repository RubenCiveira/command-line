---
description: Analyze the project structure in two parallel passes, then summarize
agents:
  analyst: agent.md
steps:
  - name: list_files
    agent: analyst
    prompt: >
      Run the list tool with path "." and output ONLY the raw file names returned,
      one per line. No explanations, no formatting, just the names.

  - parallel:
      - name: explore_src
        agent: analyst
        prompt: >
          The project root directory contains: {list_files}.
          These are ROOT-level entries. Do NOT look for them inside src/.
          Step 1: run the list tool with path "src" to see what is directly inside src/.
          Step 2: for each subdirectory found inside src/, run list again using its full
          path from the project root (e.g. "src/forge", "src/ai").
          Step 3: read 1-2 key files using their full path from the root (e.g. "src/main.py").
          Describe what the src/ directory contains and what its purpose is.
      - name: explore_examples
        agent: analyst
        prompt: >
          The project root directory contains: {list_files}.
          These are ROOT-level entries. Do NOT look for them inside examples/.
          Step 1: run the list tool with path "examples" to see what is directly inside examples/.
          Step 2: for each subdirectory found inside examples/, run list again using its full
          path from the project root (e.g. "examples/forge").
          Step 3: read 1-2 key files using their full path from the root (e.g. "examples/forge/agent.md").
          Describe what the examples/ directory contains and what its purpose is.

  - name: summary
    agent: analyst
    prompt: >
      Based on the src analysis: {explore_src}
      And the examples analysis: {explore_examples}
      Write a concise description of what this project is and how it is organized.
---
