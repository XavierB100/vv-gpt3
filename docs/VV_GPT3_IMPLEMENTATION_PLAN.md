# VV-GPT3 Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build VV-GPT3 as a safer, smarter, more polished successor to VV-GPT3.

**Architecture:** Keep the existing Flask/PyTorch core, but add service modules for safety, model registry, training preflight, and chat caching. Update the UI/docs to use these safer interfaces and add tests around the new boundaries.

**Tech Stack:** Python, Flask, Flask-SocketIO, PyTorch, tiktoken, pytest, GitHub Actions.

---

## Task 1: Project copy and rebrand foundation

- Create `/Users/xavier/AI/hermes-workspace/vv-gpt3` from VV-GPT3 excluding runtime artifacts.
- Initialize fresh git repository.
- Replace current-app branding with VV-GPT3 while preserving history references.
- Add upgrade log and plan docs.

## Task 2: Safety and model registry

- Add `src/services/safety.py` for safe model names and confined paths.
- Add `src/services/model_registry.py` for sidecar metadata and model listing without unsafe checkpoint metadata loads.
- Update Models page and delete/chat APIs to operate on safe model names/IDs instead of arbitrary paths.

## Task 3: Training preflight and metadata

- Add `src/services/training_preflight.py` for dataset/token/block-size validation and recommendations.
- Run preflight before starting training.
- Save metadata JSON sidecars when checkpoints/final models are saved.
- Fix resume best-loss continuity where possible.

## Task 4: Chat cache and smarter generation controls

- Add `src/services/chat_cache.py` to cache loaded `ChatBot` objects by model file and mtime.
- Expose top-k, top-p, repetition penalty-friendly API fields where supported.
- Return timing/tokens metadata to the UI.

## Task 5: UI/docs polish

- Update README, About page, launcher names, and UI labels.
- Correct model parameter counts.
- Add honest local/offline and checkpoint-trust notes.
- Improve training/chat explanatory text.

## Task 6: Tests and verification

- Add pytest tests for safety helpers, preflight, model registry metadata, route smoke, and model presets.
- Add GitHub Actions smoke workflow.
- Run AST/import/pytest checks.

## Task 7: GitHub publish

- Commit all work.
- Create `XavierB100/vv-gpt3` on GitHub if it does not exist.
- Push `main`.
