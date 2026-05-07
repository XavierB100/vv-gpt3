# VV-GPT3 Upgrade Log

This document records the upgrade work done while converting VV-GPT3 into VV-GPT3.

## Goals

- Preserve the local-first custom GPT training studio concept.
- Improve safety around checkpoints, paths, and destructive operations.
- Make training setup smarter with preflight validation and recommendations.
- Make chat faster and richer by caching loaded models and exposing more generation controls.
- Rebrand docs/UI accurately as VV-GPT3.
- Add tests/smoke checks so the project is easier to maintain.

## Work log

### 2026-05-07

- Created fresh project copy at `/Users/xavier/AI/hermes-test-workspace/vv-gpt3`.
- Excluded `.git`, `.venv`, `__pycache__`, generated `.pyc`, local checkpoint `.pt` files, uploads, and logs from the copy.
- Initialized a new Git repository on `main`.
- Added this upgrade log as permanent project documentation.
- Added `src/services/safety.py` for safe model names and confined paths.
- Added `src/services/model_registry.py` for checkpoint metadata sidecars and listing without loading checkpoint pickles.
- Added `src/services/training_preflight.py` for dataset/block-size validation and recommendations.
- Added `src/services/chat_cache.py` so chat can reuse loaded checkpoints instead of reloading every request.
- Updated `web_app.py` to use safe model deletion, safe chat resolution, preflight checks, sidecar metadata writes, stricter Socket.IO origins, and an environment-backed secret key.
- Rebranded the copied app/docs/launcher to VV-GPT3 and corrected model parameter count labels.
- Added pytest coverage for safety helpers, training preflight, model registry, model presets, and Flask routes.
- Added `.github/workflows/smoke.yml` for CI smoke checks.
- Fixed upload/start-training path validation so both `uploads/<file>.txt` and `<file>.txt` resolve correctly inside `uploads/`; added regression tests.

Further implementation entries should be added as features land.
