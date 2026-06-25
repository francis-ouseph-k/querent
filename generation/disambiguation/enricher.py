"""
generation/disambiguation/enricher.py
─────────────────────────────────────

Schema Enricher to append DDL comments to options for better semantic disambiguation.
"""
import json
import re
from typing import Dict
from utils.logging_config import get_logger

logger = get_logger(__name__)

class SchemaEnricher:
    """Enriches semantic disambiguation options with DDL comments from the schema graph."""

    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.table_comments: Dict[str, str] = {}
        self._load_schema()

    def _load_schema(self) -> None:
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for node in data.get("nodes", []):
                node_id = node.get("id")
                comment = node.get("comment")
                if node_id and comment:
                    self.table_comments[node_id] = comment
            logger.debug("Loaded %d schema comments from %s", len(self.table_comments), self.schema_path)
        except Exception as e:
            logger.error("Failed to load schema from %s: %s", self.schema_path, e)

    def enrich(self, option: str) -> str:
        """
        Enriches an option string with DDL comments of any tables mentioned within it.
        """
        appended_comments = []
        for table, comment in self.table_comments.items():
            # Use word boundaries to avoid partial matches
            if re.search(rf"\b{re.escape(table)}\b", option, re.IGNORECASE):
                appended_comments.append(f"{table}: {comment}")
                
        if appended_comments:
            return option + " " + " ".join(appended_comments)
        return option