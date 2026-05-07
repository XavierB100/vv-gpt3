# VV-GPT3 — Full Project Changelog & Technical Overview
### Xavier's Local AI Training Command Center
**Last Updated:** 19 April 2026

---

## 🏗️ What Is VV-GPT3?

VV-GPT3 is a full-stack web application that lets you **train, monitor, and chat with** your own GPT-2 language model — entirely on your MacBook's Apple Silicon GPU (MPS). No cloud, no API keys, no cost.

**Tech Stack:**
- **Backend:** Python / Flask / Socket.IO (real-time WebSocket events)
- **ML Engine:** PyTorch with MPS acceleration
- **Frontend:** Jinja2 templates, Bootstrap 5, Chart.js
- **Tokenizer:** Tiktoken BPE (GPT-2's 50,257-token vocabulary)
- **Architecture:** Transformer with Flash Attention, Weight Tying, KV Caching

---

## 📜 Version History (Chronological)

### v0.0 — Initial Clone (`ed58451`)
- Cloned from `vv-gpt`, the original character-level GPT project
- Basic training loop and model architecture

---

### v1.0 — Core Architecture Upgrade (`a55debf` → `7fa2bb7`)

**What changed:**
- Migrated from character-level tokenization to **Tiktoken BPE** (same tokenizer as OpenAI's GPT-2)
- Vocabulary jumped from ~65 characters to **50,257 subword tokens**
- Implemented **Weight Tying** between input embeddings and output projection (reduces parameters ~30%)
- Added **Flash Attention** for faster, memory-efficient self-attention
- Implemented **KV Caching** for dramatically faster inference (each new token only computes attention for itself, not the entire sequence)
- Renamed project to `vv-gpt3` across all build scripts and launchers

**Model Sizes Available:**
| Scale | Parameters | Use Case |
|-------|-----------|----------|
| Nano | 0.8M | Quick experiments |
| Micro | 1.5M | Testing |
| Tiny | 10M | Good balance (your default) |
| Small | 25M | Better quality |
| Medium | 70M | High quality, slower |
| Large | 150M | Maximum quality |

---

### v2.0 — Premium UI Overhaul (`1b6bace` → `2a1c52b`)

**What changed:**
The entire frontend was rebuilt from scratch with a premium glassmorphic "Nexus" design language.

- **Dashboard** (`index.html`): Three animated hover cards — Train Model, Inference Chat, Models Hub — with radial gradient effects, the Outfit/Inter/JetBrains Mono font stack, and a deep space dark theme
- **Training Page** (`train.html`): Full mission-control layout with a left config panel and right telemetry HUD. Fake terminal with green/yellow/red dots. Real-time progress bar with animated stripes
- **Inference Chat** (`chat.html`): Side-by-side layout with a settings panel (model selector, Temperature/Max Length sliders) and a modern chat interface with typing indicator dots
- **Models Hub** (`models.html`): Card grid with parameter counts, file sizes, vocab stats, and the "Initialize" button that takes you directly to chat with that model
- **About Page** (`about.html`): System information display

**UI Bugs Fixed:**
- Model deletion route collision (delete was killing the wrong model)
- `system_info` dict wasn't being passed to the homepage (caused 500 errors)
- Typography inconsistencies across all views

---

### v3.0 — Architecture Documentation (`2d6204a`)
- Comprehensive architectural write-up documenting all Python source files
- Mapped the full data flow from upload → tokenization → training → checkpoint → inference

---

### v3.1 — Live Telemetry & Intelligence System (`0af34e4` → `15cf20f`)

This was the biggest feature drop. The training page became a real-time command center.

#### Chart.js Divergence Graph
- **Dual-line chart** tracking Training Loss (indigo) and Validation Loss (emerald) in real-time
- Y-axis with loss values, legend with labeled lines
- `spanGaps: true` so the validation line renders continuously even with sparse data points

#### Overfit Detection Tripwire
- Server-side `val_loss_history` array tracks every evaluation result
- When validation loss increases for **3 consecutive evaluations**, the system fires:
  - `⚠️ OVERFITTING DETECTED` warning in the terminal
  - `💡 Recommendation` message telling you the exact step and val_loss of your best model
- Originally auto-paused training; changed to advisory-only per user request

#### Delta Readouts on Checkpoints
- Every checkpoint save now shows the **change from the previous evaluation**:
  - `🔽 -0.0942 better` (green) when val_loss improves
  - `🔼 +0.0201 worse` (red) when val_loss degrades
- Removed redundant file paths (`→ models/testing_latest.pt`) from log lines
- Added one-time startup message: `📁 Checkpoints continuously saving to models/testing_[best/latest].pt`

#### Chat Parameter Context Tokens
- When you adjust Temperature or Max Length in the Inference Chat, a system message appears:
  - `[System] Parameter Updated: Temperature changed from 0.8 to 1.2`
- Shows the "before and after" values so you always know what changed

---

### v3.1.1 — Critical Bug Fixes (`310997b` → `e85aeef`)

#### Missing Validation Line Fix
**Root Cause:** The Python backend fired TWO `training_progress` events per eval step — the first without `val_loss`, the second with it. The chart's `includes()` guard accepted the first event (with `null` validation) then silently rejected the second event carrying the actual data. The emerald line was being thrown away every time.

**Fix:** Replaced the `includes()` guard with an `indexOf()` lookup that updates existing data points in-place when the second event arrives.

#### Other Fixes
- Graph enlarged from 120px to **200px** with proper container sizing
- Y-axis ticks enabled so you can read actual loss values
- Chart legend enabled and labeled
- Smart log labels: `📈 Improving!` → `📊 Converging...` → `⚠️ Overfitting` (state machine that reacts to the tripwire)
- `overfitDetected` flag properly resets when a new training session starts
- Log buffer increased from 200 → **1000** entries (was truncating early logs)
- Fixed an inconsistent 50-entry buffer on system messages

---

### v3.2.0 — UX Intelligence Upgrade (`759096d`) ⭐ LATEST

8 features shipped in one commit:

#### 1. Parameter Help Tooltips
- Every training parameter (Scale, Iterations, Learning Rate, Eval Interval, Block Size, Batch Size, Dropout) now has a small `ⓘ` icon
- Hover/click to see a plain-English explanation of what it does and how it affects training
- Uses Bootstrap tooltips, initialized on DOMContentLoaded

#### 2. Overfit Zone Shading on Chart
- Custom Chart.js plugin (`overfitZonePlugin`) draws a **vertical dashed white line** at the step where best val_loss was achieved
- Everything to the right of that line gets a **faint red background tint** (`rgba(244,63,94,0.06)`)
- Small label reads `Best: Step 650` above the line
- `bestValStep` is tracked via `checkpoint_saved` events with `type: 'best'`

#### 3. Export Training Logs
- New `Export` button in the terminal header bar (next to the fake macOS dots)
- Downloads the entire terminal log as `{modelName}_training_log.txt`
- Uses Blob + URL.createObjectURL for client-side file generation

#### 4. Training Page Persistence
- **Problem:** Navigating away from the training page during a run lost all logs and chart data
- **Solution:** Server-side `training_event_history` list records every socket event
- New `GET /training_history` endpoint returns the full event buffer as JSON
- On page load, if training is active, the frontend fetches and **replays** all events into the DOM
- Logs, chart, telemetry values, and overfit state all rebuild perfectly

#### 5. Active Parameter Status Bar (Chat)
- Persistent glassmorphic strip between the chat header and messages:
  - `🌡️ Temp: 0.8 · 📏 Max Tokens: 200 · 🧠 Model: testing`
- Updates in real-time when sliders change or a new model is selected
- Hidden until a model is loaded

#### 6. Chat Conversation Persistence
- Chat history now saves to `localStorage` keyed by model path
- Refreshing the page restores the full conversation
- Selecting a model loads any existing chat history
- "Clear Memory" button properly wipes both DOM and localStorage

#### 7. Training History Cards (Models Hub)
- Training metadata is now embedded directly into saved `.pt` checkpoint files
- Metadata includes: model size, learning rate, batch size, block size, dropout, best val loss, best step, total training time, time to reach best, dataset size, and the full `val_loss_history` array
- Models Hub cards now display a **Training History** section with all metrics in a 2-column grid
- Mini **sparkline chart** (Chart.js) renders the val_loss curve inline on each card

#### 8. Dashboard Health Strip
- Subtle muted text below the 3 navigation cards:
  - `3 models trained · MPS · 1.2 GB used`
- Shows model count, active GPU device, and total disk usage

---

## 🧠 How It All Works (Under the Hood)

### Training Flow
```
Upload .txt → Tiktoken BPE tokenizer → 50,257 token IDs
    → Train/Val split (90/10)
    → GPT-2 transformer processes random chunks
    → Cross-entropy loss measures prediction accuracy
    → Adam optimizer adjusts 10M+ weights
    → Every 50 steps: evaluate on validation data, save checkpoint
    → Best checkpoint saved separately as {name}_best.pt
    → Overfit tripwire monitors val_loss trajectory
    → On completion: final {name}.pt = copy of best checkpoint
```

### Inference Flow
```
User types prompt → Tiktoken encodes to token IDs
    → Feed through 6 transformer layers (attention → FFN)
    → Final layer outputs 50,257 probabilities
    → Temperature scaling adjusts distribution
    → Top-k (50) + Top-p (0.9) sampling picks next token
    → KV Cache stores previous computations
    → Repeat up to Max Length times
    → Tiktoken decodes IDs back to text
    → Response appears in chat bubble
```

### Key Files
| File | Purpose |
|------|---------|
| `web_app.py` | Flask server, all routes, training loop, socket events |
| `templates/train.html` | Training page with Chart.js, terminal, and telemetry |
| `templates/chat.html` | Inference chat with parameter sliders and localStorage |
| `templates/models.html` | Models Hub with training history cards and sparklines |
| `templates/index.html` | Dashboard with 3 navigation cards and health strip |
| `templates/base.html` | Shared layout, Nexus design system, sidebar navigation |
| `src/models/gpt.py` | GPT-2 architecture (Flash Attention, Weight Tying, KV Cache) |
| `src/training/train.py` | Training utilities (loss estimation, learning rate scheduling) |
| `src/data/data_loader.py` | Data processing and Tiktoken tokenization |
| `src/inference/chatbot.py` | ChatBot class for text generation with sampling |

---

## 📊 Training Results (Your Test Runs)

| Metric | Value |
|--------|-------|
| Model | Tiny (10M params) |
| Dataset | Shakespeare (~1.2M characters) |
| Best Val Loss | ~4.1082 (step 650) |
| Training Loss at Stop | ~2.60 |
| Overfitting Onset | ~Step 500 (val_loss plateaus, training loss keeps dropping) |
| Training Time | ~1 hour for 1000 steps on M1 Max |

---

*Generated by Antigravity AI · Claude Opus 4.6*
