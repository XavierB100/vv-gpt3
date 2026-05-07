import pytest

from src.services.safety import SafetyError, validate_model_name, ensure_within_directory, model_checkpoint_path, safe_uploaded_file_path


def test_validate_model_name_allows_existing_friendly_names():
    assert validate_model_name("Shakespeare final boss") == "Shakespeare final boss"
    assert validate_model_name("model_v3-1.test") == "model_v3-1.test"


@pytest.mark.parametrize("bad", ["", "../evil", "bad/name", "bad\\name", "*wild"])
def test_validate_model_name_rejects_unsafe_names(bad):
    with pytest.raises(SafetyError):
        validate_model_name(bad)


def test_ensure_within_directory_rejects_escape(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    with pytest.raises(SafetyError):
        ensure_within_directory(tmp_path / "outside.pt", root)


def test_model_checkpoint_path_builds_expected_variants(tmp_path):
    root = tmp_path / "models"
    path = model_checkpoint_path("demo", "best", root)
    assert path.name == "demo_best.pt"
    assert path.parent == root.resolve()


def test_safe_uploaded_file_path_accepts_upload_response_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    uploaded = uploads / "input_123.txt"
    uploaded.write_text("training text")

    resolved = safe_uploaded_file_path("uploads/input_123.txt", "uploads")

    assert resolved == uploaded.resolve()


def test_safe_uploaded_file_path_accepts_uploaded_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    uploaded = uploads / "input_123.txt"
    uploaded.write_text("training text")

    resolved = safe_uploaded_file_path("input_123.txt", "uploads")

    assert resolved == uploaded.resolve()
