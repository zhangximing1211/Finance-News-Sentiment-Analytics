from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


def load_local_env(env_path: str | Path | None = None) -> Path | None:
    resolved_env_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if not resolved_env_path.exists():
        return None

    for raw_line in resolved_env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return resolved_env_path
