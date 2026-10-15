"""Load and validate the YAML/JSON configuration file."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from .models import AppConfig

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` so secrets can stay out of the file."""

    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")

    return _ENV_RE.sub(repl, text)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    text = _expand_env(p.read_text(encoding="utf-8"))
    data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    cfg = AppConfig.model_validate(data or {})
    # Relative database paths resolve next to the config file, not the CWD.
    db = Path(cfg.settings.database)
    if not db.is_absolute():
        cfg.settings.database = str((p.parent / db).resolve())
    return cfg
