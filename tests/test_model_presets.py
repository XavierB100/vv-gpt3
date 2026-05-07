from src.models.enhanced_gpt import GPT, GPTConfig


def test_model_presets_instantiate_with_expected_ordering():
    sizes = ["nano", "micro", "tiny"]
    counts = []
    for size in sizes:
        cfg = GPTConfig.get_preset(size, vocab_size=50257, block_size=128)
        model = GPT(cfg)
        counts.append(model.get_num_params())
    assert counts == sorted(counts)
    assert counts[0] > 1_000_000
