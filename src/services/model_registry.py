"""Model registry and metadata sidecars for VV-GPT3.

VV-GPT3 listed models by deserializing every .pt file with torch.load. That is
unsafe for untrusted checkpoints because .pt files are pickle-based. VV-GPT3 uses
JSON sidecars for metadata and only falls back to minimal file metadata when the
sidecar is missing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .safety import SafetyError, ensure_within_directory, validate_model_name, model_checkpoint_path

VARIANTS = {
    "final": ".pt",
    "best": "_best.pt",
    "latest": "_latest.pt",
}


def metadata_path_for_model(model_name: str, models_dir: str | Path = "models") -> Path:
    safe_name = validate_model_name(model_name)
    root = Path(models_dir).resolve()
    return ensure_within_directory(root / f"{safe_name}.metadata.json", root)


def family_name_from_checkpoint(path: str | Path) -> str:
    stem = Path(path).stem
    if stem.endswith("_best"):
        return stem[:-5]
    if stem.endswith("_latest"):
        return stem[:-7]
    return stem


def variant_from_checkpoint(path: str | Path) -> str:
    stem = Path(path).stem
    if stem.endswith("_best"):
        return "best"
    if stem.endswith("_latest"):
        return "latest"
    return "final"


def write_model_metadata(model_name: str, metadata: Dict[str, Any], models_dir: str | Path = "models") -> Path:
    """Write a safe JSON metadata sidecar for a model family."""
    safe_name = validate_model_name(model_name)
    path = metadata_path_for_model(safe_name, models_dir)
    payload = dict(metadata)
    payload.setdefault("app", "VV-GPT3")
    payload.setdefault("schema_version", 1)
    payload["name"] = safe_name
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_model_metadata(model_name: str, models_dir: str | Path = "models") -> Dict[str, Any]:
    path = metadata_path_for_model(model_name, models_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def model_paths_for_family(model_name: str, models_dir: str | Path = "models") -> Dict[str, str | None]:
    paths: Dict[str, str | None] = {}
    for variant in VARIANTS:
        path = model_checkpoint_path(model_name, variant, models_dir)
        paths[variant] = str(path) if path.exists() else None
    return paths


def resolve_model_for_chat(model_name: str, preferred_variant: str = "final", models_dir: str | Path = "models") -> Path:
    """Resolve a safe existing checkpoint for chat.

    Falls back final -> best -> latest so older training runs remain usable.
    """
    validate_model_name(model_name)
    ordered = [preferred_variant, "final", "best", "latest"]
    seen = []
    for variant in ordered:
        if variant in seen:
            continue
        seen.append(variant)
        path = model_checkpoint_path(model_name, variant, models_dir)
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found for model '{model_name}'")


def list_models(models_dir: str | Path = "models") -> List[Dict[str, Any]]:
    """List model families without loading checkpoint pickle payloads."""
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    families: Dict[str, Dict[str, Any]] = {}

    for checkpoint in sorted(root.glob("*.pt")):
        try:
            ensure_within_directory(checkpoint, root)
            name = validate_model_name(family_name_from_checkpoint(checkpoint))
        except SafetyError:
            continue
        variant = variant_from_checkpoint(checkpoint)
        entry = families.setdefault(
            name,
            {
                "name": name,
                "file": f"{name}.pt",
                "path": str(model_checkpoint_path(name, "final", root)),
                "variants": {},
                "size": 0,
                "modified": "",
                "params": "Unknown",
                "vocab_size": "Unknown",
                "training_metadata": None,
                "metadata_available": False,
            },
        )
        stat = checkpoint.stat()
        entry["variants"][variant] = {
            "path": str(checkpoint),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        entry["size"] += stat.st_size
        entry["modified"] = max(entry.get("modified") or "", entry["variants"][variant]["modified"])

    for name, entry in families.items():
        metadata = read_model_metadata(name, root)
        if metadata:
            entry["metadata_available"] = True
            entry["params"] = metadata.get("params", metadata.get("parameter_count", "Unknown"))
            entry["vocab_size"] = metadata.get("vocab_size", "Unknown")
            entry["training_metadata"] = metadata.get("training_metadata") or metadata.get("metrics")
            entry["model_size"] = metadata.get("model_size", "Unknown")
            entry["preflight"] = metadata.get("preflight")
        paths = model_paths_for_family(name, root)
        entry["paths"] = paths
        preferred = paths.get("final") or paths.get("best") or paths.get("latest")
        if preferred:
            entry["path"] = preferred
        if isinstance(entry.get("params"), (int, float)):
            entry["params"] = f"{entry['params']/1_000_000:.2f}M"

    return sorted(families.values(), key=lambda m: m.get("modified", ""), reverse=True)


def delete_model_family(model_name: str, models_dir: str | Path = "models") -> int:
    """Delete final/best/latest checkpoints and metadata for one model family."""
    deleted = 0
    for variant in VARIANTS:
        path = model_checkpoint_path(model_name, variant, models_dir)
        if path.exists():
            path.unlink()
            deleted += 1
    metadata_path = metadata_path_for_model(model_name, models_dir)
    if metadata_path.exists():
        metadata_path.unlink()
        deleted += 1
    return deleted
