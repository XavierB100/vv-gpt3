"""Training preflight analysis for VV-GPT3."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class PreflightResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    dataset_chars: int
    total_tokens: int
    train_tokens: int
    val_tokens: int
    block_size: int
    batch_size: int
    recommended_model_size: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recommend_model_size(total_tokens: int) -> str:
    if total_tokens < 25_000:
        return "nano"
    if total_tokens < 75_000:
        return "micro"
    if total_tokens < 300_000:
        return "tiny"
    if total_tokens < 1_000_000:
        return "small"
    return "medium"


def run_preflight(text: str, train_data, val_data, block_size: int, batch_size: int, model_size: str) -> PreflightResult:
    """Validate a proposed training run and generate human-readable guidance."""
    train_tokens = int(len(train_data))
    val_tokens = int(len(val_data))
    total_tokens = train_tokens + val_tokens
    errors: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []

    if not text or not text.strip():
        errors.append("Dataset is empty after preprocessing.")
    if total_tokens <= block_size + 1:
        errors.append(
            f"Dataset has {total_tokens:,} tokens, but block size is {block_size}. Upload more text or choose a smaller block size."
        )
    if train_tokens <= block_size + 1:
        errors.append(
            f"Training split has {train_tokens:,} tokens, too small for block size {block_size}."
        )
    if val_tokens <= block_size + 1:
        errors.append(
            f"Validation split has {val_tokens:,} tokens, too small for block size {block_size}."
        )
    if batch_size > 64:
        warnings.append("Large batch sizes can cause MPS/CUDA out-of-memory errors on laptops.")
    if total_tokens < 50_000:
        warnings.append("This is a small dataset; expect mimicry/repetition rather than robust language ability.")
    if model_size in {"medium", "large"} and total_tokens < 300_000:
        warnings.append("Selected model size is large for this dataset and may overfit quickly.")

    recommended = recommend_model_size(total_tokens)
    if model_size != recommended:
        recommendations.append(
            f"Recommended model size for this dataset is '{recommended}' based on token count."
        )
    if block_size >= 512 and total_tokens < 100_000:
        recommendations.append("Consider block size 128 or 256 for smaller datasets.")
    if batch_size >= 64:
        recommendations.append("If training crashes, retry with batch size 16 or 32.")

    return PreflightResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        recommendations=recommendations,
        dataset_chars=len(text),
        total_tokens=total_tokens,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        block_size=int(block_size),
        batch_size=int(batch_size),
        recommended_model_size=recommended,
    )
