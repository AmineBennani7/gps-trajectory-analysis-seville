"""Read KEY=VALUE lines from a .env file into the environment.

Only sets variables that are not already defined, so real environment
variables (CI, Docker) always win. Kept minimal to avoid another dependency.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> list[str]:
    """Load `path` if it exists. Returns the names of the variables that were set."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
