from pathlib import Path


class UserConfig:
    def rootAgentName(self) -> str:
        return "root"

    def currentAgentModel(self) -> str:
        return "ollama/llama3.2:3b"

    def ragDatabaseType(self) -> str:
        """Return the RAG database type. Currently only 'sqlite' is supported."""
        return "sqlite"

    def ragDatabasePath(self) -> Path:
        """Return the path to the SQLite RAG database file."""
        return Path.home() / ".config" / ".asistente" / "rag" / "knowledge.db"

    def ragCategoriesPath(self) -> Path:
        """Return the path to the IPTC categories JSON tree.

        The file can be generated with build_iptc_tree.py and should be
        copied to this location.
        """
        return Path.home() / ".config" / ".asistente" / "iptc_categories.json"

    def ragTopics(self) -> list[dict]:
        """Return the list of directories to watch for RAG ingestion.

        Each entry is a dict with 'name' (topic identifier) and
        'path' (absolute directory path).

        Example:
            [
                {"name": "docs", "path": "/home/user/Documents/knowledge"},
                {"name": "notes", "path": "/home/user/Notes"},
            ]
        """
        return []