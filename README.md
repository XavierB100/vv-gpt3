# VV-GPT3

**VV-GPT3** is a local AI model lab for macOS: upload your own text, train a small GPT-style Transformer from scratch, manage checkpoints, and chat with the result — all on your own machine.

This project is the upgraded successor to VV-GPT2. It keeps the original local training studio concept and adds safer checkpoint handling, model metadata sidecars, training preflight validation, cached chat model loading, clearer docs, and smoke tests.

## What it is — and what it is not

VV-GPT3 is a **GPT-2-style decoder-only causal Transformer training studio**. It uses PyTorch and `tiktoken` GPT-2 BPE tokenization, but it does **not** download or wrap OpenAI models. Models are trained from scratch on your data, so generated text tends to mimic your dataset rather than behave like ChatGPT.

## Key features

- Local Flask web UI for training/chat/model management.
- PyTorch GPT-style model implementation in `src/models/enhanced_gpt.py`.
- GPT-2 BPE tokenization via `tiktoken`.
- Apple Silicon MPS, CUDA, or CPU device selection.
- Training telemetry via Flask-SocketIO and Chart.js.
- Best/latest/final checkpoint saving.
- VV-GPT3 metadata sidecars: `models/<name>.metadata.json`.
- Safer model-name/path validation.
- Dataset preflight checks before training.
- Cached chat model loading for faster repeated messages.
- Tests for safety, preflight, registry, presets, and route smoke.

## Quick start

```bash
cd /Users/xavier/AI/hermes-test-workspace/vv-gpt3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 desktop_launcher.py
```

Open:

```text
http://127.0.0.1:5000
```

You can also double-click:

```text
VV_GPT3.command
```

If macOS blocks it, run:

```bash
chmod +x VV_GPT3.command
```

## Direct development run

```bash
source .venv/bin/activate
python3 web_app.py
```

`desktop_launcher.py` is recommended for everyday use because it runs without the Flask debug reloader.

## CLI training

```bash
python3 -m src.training.train \
  --data data/input.txt \
  --model_size tiny \
  --block_size 256 \
  --batch_size 32 \
  --max_iters 2000 \
  --learning_rate 0.0003 \
  --device auto
```

## CLI chat

```bash
python3 -m src.chat.chat --model "models/YOUR_MODEL.pt"
```

## Model presets

Approximate parameter counts with GPT-2 vocabulary and block size 256:

| Preset | Layers | Heads | Embedding | Params |
|---|---:|---:|---:|---:|
| Nano | 3 | 3 | 144 | ~8.0M |
| Micro | 4 | 4 | 192 | ~11.4M |
| Tiny | 6 | 6 | 384 | ~30.0M |
| Small | 8 | 8 | 512 | ~51.0M |
| Medium | 12 | 12 | 768 | ~123.7M |
| Large | 16 | 16 | 1024 | ~253.0M |

Because these train from scratch, bigger is not always better. Small datasets usually do better with Nano/Micro/Tiny.

## Safety notes

- Treat `.pt` checkpoint files as trusted local files only. PyTorch checkpoints are pickle-based and can be unsafe if downloaded from untrusted sources.
- VV-GPT3 lists models using JSON metadata sidecars instead of deserializing every checkpoint.
- Keep the app bound to `127.0.0.1` unless you add authentication and understand the risk.
- Do not commit `.pt` checkpoints, uploads, logs, or `.venv` to Git.

## Tests

```bash
source .venv/bin/activate
pytest -q
python3 -m compileall web_app.py src
```

## Project docs

- `docs/VV_GPT3_IMPLEMENTATION_PLAN.md` — upgrade plan.
- `docs/VV_GPT3_UPGRADE_LOG.md` — implementation log.
- `PROJECT_DEEP_DIVE_FINDINGS.md` — inherited VV-GPT2 deep-dive findings that motivated VV-GPT3.

## License

Personal project. Add a license before distributing broadly.
