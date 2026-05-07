"""Safety helpers for paths, model names, and local-only operation."""

from __future__ import annotations

import re
from pathlib import Path

MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,79}$")


class SafetyError(ValueError):
    """Raised when user-supplied input would escape the local project boundary."""


def validate_model_name(name: str) -> str:
    """Return a stripped safe model name or raise SafetyError.

    We intentionally allow spaces because the VV-GPT3 UI already used names like
    "Shakespeare final boss". Slashes, path traversal, shell metacharacter-heavy
    names, and empty names are rejected.
    """
    if not isinstance(name, str):
        raise SafetyError("Model name must be text")
    cleaned = name.strip()
    if not cleaned:
        raise SafetyError("Model name cannot be empty")
    if not MODEL_NAME_RE.match(cleaned):
        raise SafetyError(
            "Model name may only contain letters, numbers, spaces, dots, underscores, and hyphens"
        )
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise SafetyError("Model name cannot contain path separators or traversal")
    return cleaned


def ensure_within_directory(path: str | Path, root: str | Path) -> Path:
    """Resolve *path* and ensure it stays inside *root*."""
    root_path = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise SafetyError(f"Path escapes allowed directory: {resolved}") from exc
    return resolved


def model_checkpoint_path(model_name: str, variant: str = "final", models_dir: str | Path = "models") -> Path:
    """Build a safe checkpoint path for a model variant.

    Variants:
    - final -> models/<name>.pt
    - best -> models/<name>_best.pt
    - latest -> models/<name>_latest.pt
    """
    safe_name = validate_model_name(model_name)
    suffixes = {"final": ".pt", "best": "_best.pt", "latest": "_latest.pt"}
    if variant not in suffixes:
        raise SafetyError(f"Unknown model variant: {variant}")
    root = Path(models_dir).resolve()
    return ensure_within_directory(root / f"{safe_name}{suffixes[variant]}", root)


def safe_uploaded_file_path(file_path: str | Path, uploads_dir: str | Path = "uploads") -> Path:
    """Validate that an uploaded training file path points inside uploads/.

    The upload endpoint returns paths like ``uploads/input_123.txt`` while some
    callers may pass just ``input_123.txt``. Accept both forms without turning
    the former into ``uploads/uploads/input_123.txt``.
    """
    root = Path(uploads_dir).resolve()
    candidate = Path(file_path)
    if candidate.is_absolute():
        return ensure_within_directory(candidate, root)

    if candidate.parts and candidate.parts[0] == Path(uploads_dir).name:
        return ensure_within_directory(Path.cwd() / candidate, root)

    return ensure_within_directory(root / candidate, root)
