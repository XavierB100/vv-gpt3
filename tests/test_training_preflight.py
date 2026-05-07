import torch

from src.services.training_preflight import run_preflight, recommend_model_size


def test_preflight_rejects_dataset_shorter_than_block_size():
    data = torch.arange(20)
    result = run_preflight("hello", data[:10], data[10:], block_size=16, batch_size=4, model_size="tiny")
    assert not result.ok
    assert result.errors


def test_preflight_accepts_reasonable_dataset_and_recommends_size():
    data = torch.arange(1000)
    result = run_preflight("x" * 1000, data[:900], data[900:], block_size=32, batch_size=8, model_size="nano")
    assert result.ok
    assert result.total_tokens == 1000
    assert result.recommended_model_size == "nano"


def test_recommend_model_size_scales_with_tokens():
    assert recommend_model_size(10_000) == "nano"
    assert recommend_model_size(50_000) == "micro"
    assert recommend_model_size(150_000) == "tiny"
