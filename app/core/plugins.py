from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("advanced_ai_agent.plugins")


class PluginManager:
    """Loads opt-in Python plugins from the project plugins directory.

    A plugin exposes:
      TOOL_SCHEMA = {OpenAI function tool schema}
      async def run(args, context) -> str  (sync is also accepted)
    """

    def __init__(self, directory: str | None = None):
        self.directory = Path(directory) if directory else settings.project_root / "plugins"
        if not self.directory.is_absolute():
            self.directory = settings.project_root / self.directory
        self.plugins: dict[str, tuple[dict[str, Any], Any]] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    @staticmethod
    def _validate_schema(schema: Any) -> str:
        if not isinstance(schema, dict) or schema.get("type") != "function":
            raise ValueError("TOOL_SCHEMA must be an OpenAI-style function tool schema")
        fn = schema.get("function")
        if not isinstance(fn, dict):
            raise ValueError("TOOL_SCHEMA.function must be an object")
        name = str(fn.get("name", "")).strip()
        if not name or len(name) > 64:
            raise ValueError("Plugin function name is missing or too long")
        if not isinstance(fn.get("parameters", {}), dict):
            raise ValueError("Plugin function parameters must be an object")
        return name

    def reload(self) -> None:
        self.plugins.clear()
        self.errors.clear()
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"agent_plugin_{path.stem}", path)
            if not spec or not spec.loader:
                self.errors[path.name] = "Could not create import spec"
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                schema = getattr(module, "TOOL_SCHEMA")
                runner = getattr(module, "run")
                if not callable(runner):
                    raise TypeError("run must be callable")
                name = self._validate_schema(schema)
                if name in self.plugins:
                    raise ValueError(f"Duplicate plugin tool name: {name}")
                self.plugins[name] = (schema, runner)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.errors[path.name] = message
                logger.warning("Plugin %s was skipped: %s", path, message)

    def schemas(self) -> list[dict[str, Any]]:
        return [schema for schema, _ in self.plugins.values()]

    async def execute(self, name: str, args: dict[str, Any], context: dict[str, Any]) -> str:
        if name not in self.plugins:
            raise KeyError(name)
        _, runner = self.plugins[name]
        value = runner(args, context)
        if inspect.isawaitable(value):
            value = await value
        return str(value)
