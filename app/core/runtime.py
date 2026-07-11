from __future__ import annotations

import sys
from collections.abc import Sequence

SUPPORTED_PYTHON = (3, 12)


def require_supported_python(version_info: Sequence[int] | None = None) -> None:
    version = tuple((version_info or sys.version_info)[:2])
    if version != SUPPORTED_PYTHON:
        current = ".".join(str(part) for part in version)
        expected = ".".join(str(part) for part in SUPPORTED_PYTHON)
        raise RuntimeError(
            f"Lyle AI Service requires Python {expected}.x; current interpreter is Python {current}. "
            "Create .venv312 with python3.12 and install requirements.lock."
        )
