---
description: Analyze the project structure in two parallel passes, then summarize
agents:
  analyst: agent.md
steps:
  - name: list_files
    agent: analyst
    prompt: "List the files and folders in the current directory."

  - parallel:
      - name: explore_src
        agent: analyst
        prompt: >
          The project contains these entries: {list_files}.
          Read and describe what is inside the src/ directory.
      - name: explore_examples
        agent: analyst
        prompt: >
          The project contains these entries: {list_files}.
          Read and describe what is inside the examples/ directory.

  - name: summary
    agent: analyst
    prompt: >
      Based on the src analysis: {explore_src}
      And the examples analysis: {explore_examples}
      Write a concise description of what this project is and how it is organized.
---
