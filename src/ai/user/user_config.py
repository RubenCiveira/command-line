import json
import logging
from pathlib import Path

from ai.user.project_topic import ProjectTopic
from ai.user.rag_topic import RagTopic

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / ".asistente"
CONFIG_FILE = CONFIG_DIR / "user-config.json"


class UserConfig:

    def __init__(self, config_path: Path = CONFIG_FILE) -> None:
        self._config_path = config_path
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if not self._config_path.exists():
            logger.debug("Config file not found: %s", self._config_path)
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load config %s: %s", self._config_path, exc)

    def rootAgentName(self) -> str:
        return "root"

    def currentAgentModel(self) -> str:
        return "ollama/llama3.2:3b"

    def ragDatabaseType(self) -> str:
        """Return the RAG database type. Currently only 'sqlite' is supported."""
        return "sqlite"

    def ragDatabasePath(self) -> Path:
        """Return the path to the SQLite RAG database file."""
        return Path.home() / ".cache" / ".asistente" / "rag" / "knowledge.db"

    def ragCategoriesPath(self) -> Path:
        """Return the path to the IPTC categories JSON tree.

        The file can be generated with build_iptc_tree.py and should be
        copied to this location.
        """
        return CONFIG_DIR / "iptc_categories.json"

    def ragTopics(self) -> list[RagTopic]:
        """Return the list of directories to watch for RAG ingestion.

        Reads from the ``rag.topics`` array in user-config.json::

            {
                "rag": {
                    "topics": [
                        {"name": "docs", "path": "/home/user/Documents/knowledge"},
                        {"name": "notes", "path": "/home/user/Notes"}
                    ]
                }
            }
        """
        raw = self._data.get("rag", {}).get("topics", [])
        topics = []
        for entry in raw:
            name = entry.get("name")
            path = entry.get("path")
            if name and path:
                topics.append(RagTopic(name=name, path=str(path)))
            else:
                logger.warning("Skipping invalid topic entry: %s", entry)
        return topics

    def projectTopics(self) -> list[ProjectTopic]:
        """Return the list of project directories.

        Reads from the ``projects`` array in user-config.json::

            {
                "projects": [
                    {
                        "name": "backend",
                        "path": "/home/user/work/backend",
                        "description": "REST API built with FastAPI"
                    }
                ]
            }
        """
        raw = self._data.get("projects", [])
        projects = []
        for entry in raw:
            name = entry.get("name")
            path = entry.get("path")
            description = entry.get("description", "")
            if name and path:
                projects.append(ProjectTopic(name=name, path=str(path), description=description))
            else:
                logger.warning("Skipping invalid project entry: %s", entry)
        return projects