import json

from src.services.model_registry import (
    write_model_metadata,
    read_model_metadata,
    list_models,
    resolve_model_for_chat,
)


def test_model_metadata_roundtrip(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    path = write_model_metadata("demo", {"params": 123, "vocab_size": 50257}, models)
    assert path.exists()
    data = read_model_metadata("demo", models)
    assert data["name"] == "demo"
    assert data["params"] == 123
    assert data["app"] == "VV-GPT3"


def test_list_models_uses_sidecar_without_loading_checkpoint(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "demo.pt").write_bytes(b"not a real checkpoint")
    write_model_metadata("demo", {"params": 123_000_000, "vocab_size": 50257}, models)

    listed = list_models(models)

    assert len(listed) == 1
    assert listed[0]["name"] == "demo"
    assert listed[0]["metadata_available"] is True
    assert listed[0]["params"] == "123.00M"


def test_resolve_model_for_chat_prefers_final_then_fallback(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    latest = models / "demo_latest.pt"
    latest.write_bytes(b"checkpoint")
    assert resolve_model_for_chat("demo", "final", models) == latest.resolve()
